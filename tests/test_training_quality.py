import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'flux-kontext-training/src')]
try:
    import numpy as np
    import torch
    from PIL import Image
    from flux_kontext_training.trainer import FluxKontextTrainer, _build_lr_scheduler, OneStepReconstructionLoss, ReconstructionLossWeights
    from flux_kontext_training.visual_experiment import FloorPlanImageMetrics, ProgressController, VisualEvaluator, write_image
    from flux_kontext_training.schemas import TrainingConfig
    from flux_kontext_training.remote_cli import CLITrainingConfig
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


@unittest.skipUnless(AVAILABLE, 'CPU PyTorch/Pillow are required for training quality tests')
class QualityTests(unittest.TestCase):
    def trainer(self, cancelled=False):
        config = TrainingConfig(dataset='test', batch_size=2, validation_noise_repeats=2)
        return FluxKontextTrainer(config, lambda m: None, None, None, lambda: cancelled)

    def test_validation_is_fixed_noise_weighted_flow_and_preserves_rng(self):
        trainer = self.trainer()
        pipe = types.SimpleNamespace(transformer=torch.nn.Linear(1, 1), text_encoder=torch.nn.Linear(1, 1), text_encoder_2=None)
        pipe.text_encoder.eval()
        samples = []
        def step(pipe, batch, *args, **kwargs):
            samples.append(float(torch.rand(())))
            return types.SimpleNamespace(flow=torch.tensor(1.0 if len(batch) == 2 else 9.0), total=torch.tensor(1000.0))
        state = torch.get_rng_state().clone()
        with patch.object(trainer, '_training_step', side_effect=step):
            first = trainer._validate_epoch(pipe, [1, 2, 3], torch.device('cpu'), torch.float32, 0.1)
            second = trainer._validate_epoch(pipe, [1, 2, 3], torch.device('cpu'), torch.float32, 0.9)
        self.assertAlmostEqual(first, 11 / 3, places=5)
        self.assertEqual(first, second)
        self.assertEqual(samples[:4], samples[4:])
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertTrue(pipe.transformer.training)
        self.assertFalse(pipe.text_encoder.training)

    def test_validation_restores_eval_modes_even_on_error(self):
        trainer = self.trainer()
        pipe = types.SimpleNamespace(transformer=torch.nn.Linear(1, 1), text_encoder=None, text_encoder_2=None)
        with patch.object(trainer, '_training_step', side_effect=RuntimeError('failed')):
            with self.assertRaises(RuntimeError):
                trainer._validate_epoch(pipe, [1], torch.device('cpu'), torch.float32, 0.5)
        self.assertTrue(pipe.transformer.training)

    def test_cancelled_validation_is_not_a_perfect_zero_loss(self):
        trainer = self.trainer(True)
        pipe = types.SimpleNamespace(transformer=torch.nn.Linear(1, 1), text_encoder=None, text_encoder_2=None)
        self.assertIsNone(trainer._validate_epoch(pipe, [1], torch.device('cpu'), torch.float32, 0.5))

    def test_cosine_learning_rate_does_not_restart_at_the_final_step(self):
        parameter = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.AdamW([parameter], lr=1.0)
        scheduler = _build_lr_scheduler(optimizer, 'cosine', 10, 0.0, 1, cosine_final_ratio=0.1)
        rates = [optimizer.param_groups[0]['lr']]
        for _ in range(10):
            optimizer.step()
            scheduler.step()
            rates.append(optimizer.param_groups[0]['lr'])
        self.assertAlmostEqual(rates[-1], 0.1)
        self.assertTrue(all(left >= right for left, right in zip(rates, rates[1:])))

    def test_identical_structure_is_better_than_a_blank_or_shifted_image(self):
        target = np.full((32, 32, 3), 255, dtype=np.uint8)
        target[5:8, 4:27] = 0
        target[5:27, 4:7] = 0
        teacher = Image.fromarray(target)
        identical = FloorPlanImageMetrics.measure(teacher, teacher)
        blank = FloorPlanImageMetrics.measure(Image.new('RGB', (32, 32), 'white'), teacher)
        shifted = FloorPlanImageMetrics.measure(Image.fromarray(np.roll(target, 8, axis=0)), teacher)
        self.assertEqual(identical['wall_iou'], 1.0)
        self.assertEqual(blank['wall_iou'], 0.0)
        self.assertLess(shifted['wall_iou'], identical['wall_iou'])

    def test_latent_reconstruction_is_sigma_weighted_velocity_error(self):
        target = torch.randn(2, 4, 3)
        noise = torch.randn_like(target)
        sigma = torch.tensor([0.2, 0.8]).reshape(2, 1, 1)
        prediction = torch.randn_like(target)
        noisy = (1 - sigma) * target + sigma * noise
        actual, pixel = OneStepReconstructionLoss(0.5)(None, prediction, noisy, sigma, target,
                            torch.zeros(2, 3, 4, 4), 4, 4, ReconstructionLossWeights(latent=1.0, pixel=0.0))
        expected = (sigma.square() * (prediction - (noise - target)).square()).mean()
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
        self.assertIsNone(pixel)

    def test_training_loop_records_real_optimizer_updates_and_gradients(self):
        from flux_kontext_training import trainer as module
        config = TrainingConfig(dataset='test', epochs=1, batch_size=2, gradient_accumulation_steps=2,
                                optimizer='adamw', lr_scheduler='constant', lr_warmup_fraction=0,
                                augment=False, rotation_mode='none', use_buckets=False)
        metrics=[]
        trainer = FluxKontextTrainer(config, metrics.append, None, None, lambda: False)
        parameter=torch.nn.Parameter(torch.tensor([1.0]))
        trainer._trainable_params=[parameter]
        optimizer=torch.optim.SGD([parameter],lr=0.1)
        def loss(*args,**kwargs):
            value=parameter.square().mean()
            return module.TrainingLossResult(value,value,None,None,module.ReconstructionLossWeights(0,0))
        with patch.object(module, '_validate_instruction_token_length'), \
             patch.object(module, 'resolve_dataset', return_value=Path('test')), \
             patch.object(module, 'list_training_pairs', side_effect=lambda p,s,*a:list(range(4)) if s=='train' else [0]), \
             patch.object(trainer, '_preload_images'), \
             patch.object(trainer, '_load_pipeline', return_value=(object(),torch.device('cpu'),torch.float32)), \
             patch.object(trainer, '_validate_pair_prompt_token_lengths'), \
             patch.object(trainer, '_load_source_checkpoint'), patch.object(trainer, '_build_prompt_embed_cache'), \
             patch.object(trainer, '_publish_preview'), patch.object(trainer, '_configure_trainable_parameters', return_value=optimizer), \
             patch.object(trainer, '_training_step', side_effect=loss), patch.object(trainer, '_validate_epoch', return_value=0.5), \
             patch.object(trainer, '_save_best_trainable_parameters'), patch.object(trainer, '_release_trainable_tensors'), \
             patch.object(module, 'release_cuda_memory'):
            trainer.run()
        self.assertEqual([m.optimizer_step for m in metrics],[0,1])
        self.assertIsNone(metrics[0].gradient_norm)
        self.assertGreater(metrics[1].gradient_norm,0)
        self.assertLess(parameter.item(),1)

    def test_adapter_evaluation_default_covers_all_pairs(self):
        from flux_kontext_training import adapter_loss_evaluator as module
        evaluator = module.AdapterLossEvaluator(TrainingConfig(dataset='test'), Path('dummy'), include_preview=False)
        measurement = module.BatchLossMeasurement(torch.tensor(1.0), torch.tensor(1.0), 0.1)
        with patch.object(module, 'resolve_dataset', return_value=Path('dataset')), \
             patch.object(module, 'list_training_pairs', return_value=list(range(5))), \
             patch.object(evaluator, '_prepare_loss_inputs', return_value={}), \
             patch.object(evaluator, '_offload_non_transformer_for_loss'), \
             patch.object(evaluator, '_measure_loss', return_value=measurement), \
             patch.object(evaluator, '_measure_base_loss', return_value=None):
            summary = evaluator._evaluate_split(None, 'val', torch.device('cpu'), torch.float32)
        self.assertEqual(summary.pair_count, 5)

    def test_early_stop_and_plateau_lr_are_bounded_and_visual_progress_resets_stop(self):
        with tempfile.TemporaryDirectory() as temp:
            config = TrainingConfig(dataset='test', early_stopping_min_epochs=4, early_stopping_patience=3,
                                    plateau_lr_patience=2)
            controller = ProgressController(config, Path(temp))
            self.assertIsNone(controller.observe_flow(1, 0.1, 1e-4))
            self.assertIsNone(controller.observe_flow(2, 0.2, 1e-4))
            self.assertEqual(controller.observe_flow(3, 0.2, 1e-4), 5e-5)
            controller.observe_flow(4, 0.2, 5e-5)
            self.assertTrue(controller.stop_requested)
            self.assertTrue(controller.observe_visual(4, 0.7))
            self.assertFalse(controller.stop_requested)

    def test_visual_experiment_cannot_silently_exceed_30_epochs(self):
        with self.assertRaises(ValueError):
            CLITrainingConfig(dataset='test', epochs=31, visual_evaluation_enabled=True)
        self.assertEqual(CLITrainingConfig(dataset='test', epochs=30, visual_evaluation_enabled=True).epochs, 30)

    def test_visual_experiment_rejects_unsupported_text_encoder_before_remote_start(self):
        for field in ['train_text_encoder', 'train_text_encoder_2']:
            with self.subTest(field=field), self.assertRaises(ValueError):
                CLITrainingConfig(dataset='test', visual_evaluation_enabled=True, **{field: True})
        with self.assertRaises(ValueError):
            CLITrainingConfig(dataset='test', visual_evaluation_enabled=True, method='full')

    def test_immutable_image_is_published_only_with_complete_extension(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'epoch_0002.jpg'
            write_image(path, Image.new('RGB', (16, 16), 'white'))
            self.assertTrue(path.exists())
            self.assertFalse(path.with_name(path.name + '.tmp').exists())
            with Image.open(path) as image:
                self.assertEqual(image.size, (16, 16))

    def test_visual_sample_selection_covers_validation_set(self):
        self.assertEqual(VisualEvaluator.select(list(range(10)), 4), [0, 3, 6, 9])

    def test_encoder_reload_uses_the_adapter_created_by_diffusers(self):
        from flux_kontext_training import trainer as module
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / 'text_encoder_lora.pt').touch()
            encoder = types.SimpleNamespace(peft_config={'default_0': object()}, requires_grad_=Mock())
            setter = Mock()
            peft = types.SimpleNamespace(set_peft_model_state_dict=setter)
            weights = {'lora_B.weight': torch.ones(1)}
            with patch.dict(sys.modules, {'peft': peft}), patch.object(module, '_torch_load', return_value=weights):
                self.trainer()._load_text_encoder_component(encoder, path, 'text_encoder')
            setter.assert_called_once_with(encoder, weights, adapter_name='default_0')
            encoder.requires_grad_.assert_called_once_with(False)

    def test_visual_comparison_reuses_the_same_seed_and_publishes_boards(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            source, target = directory / '0290_start.png', directory / '0290_end.png'
            Image.new('RGB', (32, 32), 'white').save(source)
            teacher = np.full((32, 32, 3), 255, dtype=np.uint8)
            teacher[5:9, 4:27] = 0
            Image.fromarray(teacher).save(target)
            seeds = []
            class Pipe:
                transformer = torch.nn.Linear(1, 1)
                def __call__(self, **kwargs):
                    seeds.append(kwargs['generator'].initial_seed())
                    return types.SimpleNamespace(images=[Image.fromarray(teacher).resize((kwargs['width'], kwargs['height']))])
            class Helper:
                def _restore_text_encoders(self, *args): pass
                def _offload_text_encoders(self, *args): pass
                def _encode_prompts(self, *args): return torch.zeros(1, 2, 4), torch.zeros(1, 4), None
            pair = types.SimpleNamespace(input_image_path=str(source), target_image_path=str(target))
            evaluator = VisualEvaluator(TrainingConfig(dataset='test', resolution=256), directory / 'evaluation')
            first = evaluator.render(Pipe(), Helper(), [pair], 0, 0, torch.device('cpu'), stage='base')
            second = evaluator.render(Pipe(), Helper(), [pair], 2, 10, torch.device('cpu'))
            self.assertEqual(seeds[0], seeds[1])
            self.assertEqual(first['wall_iou'], second['wall_iou'])
            self.assertTrue((directory / 'evaluation/epoch_0002.jpg').exists())
            self.assertTrue((directory / 'evaluation/timeline.json').exists())
            # Train images get their own baseline and never reuse a val image
            # with the same filename when assessing memorization.
            train_base = evaluator.render(Pipe(), Helper(), [pair], 0, 0, torch.device('cpu'), stage='train_base', split='train')
            train_final = evaluator.render(Pipe(), Helper(), [pair], 2, 10, torch.device('cpu'), stage='train_final_selected', split='train')
            self.assertEqual(train_base['split'], 'train')
            self.assertEqual(train_final['split'], 'train')
            self.assertTrue((directory / 'evaluation/train_base/0290_target.png').exists())
            self.assertTrue((directory / 'evaluation/train_final_selected.jpg').exists())


if __name__ == '__main__':
    unittest.main()
