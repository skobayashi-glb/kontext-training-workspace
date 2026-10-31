"""A single bounded training run with fixed visual comparisons and early stopping."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch

from .augmentation import load_rgba_as_rgb, nearest_bucket, RESOLUTION_BUCKETS
from .datasets import list_training_pairs, resolve_dataset
from .job_store import JobStore
from .paths import RUNS_DIR
from .schemas import JobStatus, TrainingConfig
from .trainer import FluxKontextTrainer, release_cuda_memory, _hugging_face_token, _training_device, _training_dtype


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def write_image(path: Path, image: Image.Image):
    temporary = path.with_name(path.name + '.tmp')
    image.save(temporary, format='PNG' if path.suffix.lower() == '.png' else 'JPEG', **({} if path.suffix.lower() == '.png' else {'quality': 94}))
    temporary.replace(path)


class FloorPlanImageMetrics:
    """Auxiliary geometry/cleanup scores; never a substitute for visual review."""
    @staticmethod
    def measure(prediction: Image.Image, target: Image.Image) -> dict:
        predicted = np.asarray(prediction.convert('RGB'), dtype=np.float32) / 255
        expected = np.asarray(target.convert('RGB'), dtype=np.float32) / 255
        foreground = predicted.mean(axis=2) < 0.80
        reference = expected.mean(axis=2) < 0.80
        walls = predicted.max(axis=2) < 0.25
        reference_walls = expected.max(axis=2) < 0.25
        def iou(a, b):
            union = np.logical_or(a, b).sum()
            return float(np.logical_and(a, b).sum() / union) if union else 1.0
        correct = np.logical_and(foreground, reference).sum()
        return {'wall_iou': iou(walls, reference_walls), 'foreground_iou': iou(foreground, reference),
                'foreground_precision': float(correct / max(1, foreground.sum())),
                'foreground_recall': float(correct / max(1, reference.sum())),
                'background_false_positive_rate': float(np.logical_and(foreground, ~reference).sum() / max(1, (~reference).sum())),
                'pixel_mae': float(np.abs(predicted - expected).mean())}


class ProgressController:
    def __init__(self, config: TrainingConfig, directory: Path):
        self.config, self.directory = config, directory
        self.best_flow = float('inf')
        self.best_visual = -1.0
        self.best_flow_epoch = self.best_visual_epoch = 0
        self.last_lr_reduction_epoch = 0
        self.epoch = 0
        self.stop_requested = False
        self.reason = ''
        self.remote = Path(os.environ['KONTEXT_REMOTE_JOB_DIR']) if os.getenv('KONTEXT_REMOTE_JOB_DIR') else None

    def record(self, action, **values):
        with (self.directory / 'decisions.jsonl').open('a', encoding='utf-8') as output:
            output.write(json.dumps({'time': time.time(), 'epoch': self.epoch, 'action': action, **values}, ensure_ascii=False) + '\n')

    def feedback(self):
        path = self.remote / 'control.json' if self.remote else None
        if path and path.exists():
            return {**json.loads(path.read_text(encoding='utf-8')), '_revision': path.stat().st_mtime_ns}
        return {}

    def observe_flow(self, epoch, value, learning_rate):
        self.epoch = epoch
        if value is not None and math.isfinite(value) and value < self.best_flow - self.config.early_stopping_min_delta:
            self.best_flow, self.best_flow_epoch = value, epoch
        last_progress = max(self.best_flow_epoch, self.best_visual_epoch)
        stagnant = epoch - last_progress
        self.stop_requested = epoch >= self.config.early_stopping_min_epochs and stagnant >= self.config.early_stopping_patience
        if self.stop_requested:
            self.reason = f'No validation/visual improvement for {stagnant} epochs'
        if stagnant >= self.config.plateau_lr_patience and epoch - self.last_lr_reduction_epoch >= self.config.plateau_lr_patience:
            new_rate = max(self.config.minimum_learning_rate, learning_rate * self.config.plateau_lr_factor)
            if new_rate < learning_rate:
                self.last_lr_reduction_epoch = epoch
                self.record('reduce_learning_rate', old=learning_rate, new=new_rate, reason='validation plateau')
                return new_rate
        return None

    def observe_visual(self, epoch, score):
        if epoch > 0 and score > self.best_visual + 0.002:
            self.best_visual, self.best_visual_epoch = score, epoch
            self.stop_requested = False
            self.reason = ''
            self.record('new_visual_checkpoint', wall_iou=score)
            return True
        return False


class VisualEvaluator:
    def __init__(self, config, output: Path, baseline_root: Path | None = None):
        self.config, self.output = config, output
        self.baseline_root = baseline_root or output / 'base'
        self.output.mkdir(parents=True, exist_ok=True)
        self.records = []

    @staticmethod
    def select(pairs, count):
        if len(pairs) <= count:
            return list(pairs)
        return [pairs[round(index * (len(pairs) - 1) / (count - 1))] for index in range(count)] if count > 1 else [pairs[0]]

    @staticmethod
    def panel(image, title, width=384, height=420):
        panel = Image.new('RGB', (width, height), 'white')
        copy = ImageOps.contain(image, (width - 8, height - 35), Image.Resampling.LANCZOS)
        panel.paste(copy, ((width - copy.width) // 2, 30 + (height - 35 - copy.height) // 2))
        ImageDraw.Draw(panel).text((8, 8), title, fill='black')
        return panel

    def history_boards(self, project: Path):
        for run in sorted((project / 'runs').glob('*')):
            previews = []
            for split, count in [('val', 4), ('train', 2)]:
                previews.extend(self.select(sorted((run / 'previews').glob(split + '_step_*')), count))
            for directory in previews:
                images = []
                for name in ['input', 'target', 'prediction']:
                    path = directory / (name + '.png')
                    if path.is_file():
                        with Image.open(path) as source:
                            images.append(self.panel(source.convert('RGB'), name))
                if len(images) == 3:
                    board = Image.new('RGB', (3 * 384, 420), 'white')
                    for index, image in enumerate(images):
                        board.paste(image, (index * 384, 0))
                    write_image(self.output / f'history_{run.name[:8]}_{directory.name}.jpg', board)

    @torch.no_grad()
    def render(self, pipe, helper, pairs, epoch, step, device, stage=None, split='val'):
        stage = stage or f'epoch_{epoch:04d}'
        stage_dir = self.output / stage
        stage_dir.mkdir(exist_ok=True)
        panels, measurements = [], []
        mode = pipe.transformer.training
        pipe.transformer.eval()
        try:
            for pair in pairs:
                key = Path(pair.input_image_path).stem.removesuffix('_start')
                with load_rgba_as_rgb(pair.input_image_path) as source, load_rgba_as_rgb(pair.target_image_path) as reference:
                    width, height = nearest_bucket(reference.size, RESOLUTION_BUCKETS)
                    scale = self.config.resolution / 1024
                    width, height = max(16, int(width * scale) // 16 * 16), max(16, int(height * scale) // 16 * 16)
                    target = reference.resize((width, height), Image.Resampling.LANCZOS)
                    dtype = next(pipe.transformer.parameters()).dtype
                    helper._restore_text_encoders(pipe, device, dtype)
                    embeds, pooled, _ = helper._encode_prompts(pipe, (pair,), device)
                    seed = self.config.seed + int(hashlib.sha256(key.encode()).hexdigest()[:6], 16)
                    prediction = pipe(image=source, prompt_embeds=embeds.detach(), pooled_prompt_embeds=pooled.detach(),
                                      height=height, width=width, num_inference_steps=self.config.evaluation_inference_steps,
                                      guidance_scale=self.config.evaluation_guidance_scale,
                                      generator=torch.Generator(device=device).manual_seed(seed), output_type='pil').images[0].convert('RGB')
                    helper._offload_text_encoders(pipe)
                    write_image(stage_dir / (key + '_prediction.png'), prediction)
                    scores = FloorPlanImageMetrics.measure(prediction, target)
                    measurements.append({'key': key, 'seed': seed, 'width': width, 'height': height, **scores})
                    baseline_root = self.output / 'train_base' if split == 'train' else self.baseline_root
                    base_file = baseline_root / (key + '_prediction.png')
                    if epoch == 0 and stage in {'base', 'train_base'}:
                        write_image(stage_dir / (key + '_input.png'), source)
                        write_image(stage_dir / (key + '_target.png'), target)
                    baseline = Image.open(base_file).convert('RGB') if base_file.exists() else Image.new('RGB', prediction.size, '#eeeeee')
                    for image, title in [(source, key + ' input'), (target, 'target'), (baseline, 'base / untrained'),
                                         (prediction, f'{stage} IoU={scores["wall_iou"]:.3f}')]:
                        panels.append(self.panel(image, title))
                    baseline.close()
                    prediction.close()
                    target.close()
        finally:
            pipe.transformer.train(mode)
        board = Image.new('RGB', (384 * 4, 420 * len(pairs)), 'white')
        for index, panel in enumerate(panels):
            board.paste(panel, ((index % 4) * 384, (index // 4) * 420))
        write_image(self.output / (stage + '.jpg'), board)
        record = {'stage': stage, 'split': split, 'epoch': epoch, 'step': step, 'wall_iou': float(np.mean([m['wall_iou'] for m in measurements])),
                  'cases': measurements, 'board': stage + '.jpg', 'inference_steps': self.config.evaluation_inference_steps,
                  'guidance_scale': self.config.evaluation_guidance_scale, 'caption': self.config.caption,
                  'instruction': self.config.instruction, 'model_name': self.config.model_name}
        write_json(stage_dir / 'metrics.json', record)
        self.records.append(record)
        write_json(self.output / 'timeline.json', {'fixed_seed': True, 'records': self.records})
        print(f'VISUAL_EVAL {stage}: wall_iou={record["wall_iou"]:.4f}', flush=True)
        return record


class MonitoredTrainer(FluxKontextTrainer):
    def __init__(self, *args, evaluator, controller, store, job, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluator, self.controller, self.store, self.job = evaluator, controller, store, job
        self.optimizer = None
        self._last_control_revision = None
        dataset = resolve_dataset(self.config.dataset)
        self.cases = evaluator.select(list_training_pairs(dataset, 'val', self.config.caption, self.config.instruction), self.config.evaluation_samples)
        self.train_cases = evaluator.select(list_training_pairs(dataset, 'train', self.config.caption, self.config.instruction), 2)

    def _configure_trainable_parameters(self, pipe):
        self.optimizer = super()._configure_trainable_parameters(pipe)
        return self.optimizer

    def _save_best_trainable_parameters(self, pipe, val_loss):
        super()._save_best_trainable_parameters(pipe, val_loss)
        epoch = self.controller.epoch + 1
        current_rate = self.optimizer.param_groups[0]['lr']
        rate = self.controller.observe_flow(epoch, val_loss, current_rate)
        feedback = self.controller.feedback()
        if feedback.get('learning_rate') and feedback.get('_revision') != self._last_control_revision:
            rate = float(feedback['learning_rate'])
            self._last_control_revision = feedback['_revision']
            self.controller.record('visual_feedback_learning_rate', learning_rate=rate, reason=feedback.get('reason', ''))
        if rate is not None and rate != self.store.learning_rate(self.job.id):
            self.store.update_learning_rate(self.job.id, rate)
        write_json(Path(self.job.run_dir) / 'progress.json', {'epoch': epoch, 'val_flow_loss': val_loss,
                   'best_flow_epoch': self.controller.best_flow_epoch, 'best_visual_epoch': self.controller.best_visual_epoch,
                   'early_stop_requested': self.controller.stop_requested, 'reason': self.controller.reason})

    def _publish_preview(self, pipe, pair, epoch, step, device, source='val'):
        if self.store.is_cancelled(self.job.id):
            return
        if source == 'train':
            if epoch == 0:
                self.evaluator.render(pipe, self, self.train_cases, epoch, step, device, stage='train_base', split='train')
            return
        if source != 'val':
            return
        if epoch and epoch % self.config.evaluation_interval and not self.controller.stop_requested and epoch != self.config.epochs:
            return
        record = self.evaluator.render(pipe, self, self.cases, epoch, step, device, stage='base' if epoch == 0 else None)
        self.controller.observe_visual(epoch, record['wall_iou'])
        if epoch > 0:
            original = self.output_dir
            try:
                self.output_dir = Path(self.job.run_dir) / 'checkpoints' / f'epoch_{epoch:04d}'
                self._save_trainable_parameters(pipe)
                write_json(self.output_dir / 'selection.json', record)
            finally:
                self.output_dir = original


class VisualExperiment:
    def __init__(self, config: TrainingConfig):
        if config.epochs > 30:
            raise ValueError('This experiment is limited to 30 epochs.')
        if config.train_text_encoder or config.train_text_encoder_2 or config.method != 'lora':
            raise ValueError('The controlled visual experiment uses transformer LoRA with frozen text encoders.')
        self.config = config

    def run(self):
        from .remote_cli import TrainingCommand
        TrainingCommand(self.config).validate()
        store = JobStore(runs_dir=RUNS_DIR, load_existing=False)
        job = store.create(self.config)
        directory = Path(job.run_dir)
        write_json(directory / 'initial_config.json', self.config.model_dump())
        remote = Path(os.environ['KONTEXT_REMOTE_JOB_DIR']) if os.getenv('KONTEXT_REMOTE_JOB_DIR') else directory.parent
        write_json(remote / 'training.json', {'training_id': job.id, 'run_dir': job.run_dir})
        controller = ProgressController(self.config, directory)
        evaluator = VisualEvaluator(self.config, directory / 'evaluation')
        request = json.loads((remote / 'request.json').read_text()) if (remote / 'request.json').exists() else {}
        project = Path(request.get('remote_project', directory.parent.parent))
        evaluator.history_boards(project)
        print(f'training_id={job.id}; max_epochs={self.config.epochs}; visual_every={self.config.evaluation_interval}', flush=True)
        def cancel(signum, frame):
            store.cancel(job.id)
        previous = {sig: signal.signal(sig, cancel) for sig in (signal.SIGTERM, signal.SIGINT)}
        trainer = None
        store.set_status(job.id, JobStatus.RUNNING)
        try:
            trainer = MonitoredTrainer(config=job.config, on_metric=lambda metric: store.add_metric(job.id, metric),
                on_preview=lambda preview: None, get_learning_rate=lambda: store.learning_rate(job.id),
                is_cancelled=lambda: store.is_cancelled(job.id) or controller.stop_requested or bool(controller.feedback().get('stop')),
                output_dir=directory, evaluator=evaluator, controller=controller, store=store, job=job)
            trainer.run()
            if store.is_cancelled(job.id):
                store.set_status(job.id, JobStatus.CANCELLED)
                return 130
            feedback = controller.feedback()
            preferred = feedback.get('preferred_epoch') or controller.best_visual_epoch
            checkpoint = directory / 'checkpoints' / f'epoch_{int(preferred):04d}' / 'weights'
            if not checkpoint.is_dir():
                checkpoint = directory / 'weights'
            write_json(directory / 'selected_checkpoint.json', {'checkpoint': str(checkpoint), 'epoch': preferred,
                       'best_flow_epoch': controller.best_flow_epoch, 'early_stopped': controller.stop_requested,
                       'reason': feedback.get('reason') or controller.reason, 'validation_set_used_for_selection': True})
            del trainer
            trainer = None
            release_cuda_memory()
            self.final_evaluation(evaluator, checkpoint, store, job, int(preferred or 0))
            store.set_status(job.id, JobStatus.COMPLETED)
            return 0
        except BaseException as error:
            store.set_status(job.id, JobStatus.FAILED, error=str(error))
            raise
        finally:
            del trainer
            release_cuda_memory()
            for sig, handler in previous.items():
                signal.signal(sig, handler)

    def final_evaluation(self, evaluator, checkpoint, store, job, selected_epoch):
        from diffusers import FluxKontextPipeline
        device = _training_device()
        pipe = None
        try:
            pipe = FluxKontextPipeline.from_pretrained(self.config.model_name, torch_dtype=_training_dtype(device),
                                                       token=_hugging_face_token(), local_files_only=True)
            pipe.to(device)
            pipe.set_progress_bar_config(disable=True)
            for module in [pipe.transformer, pipe.vae, pipe.text_encoder, pipe.text_encoder_2]:
                if module is not None:
                    module.requires_grad_(False)
                    module.eval()
            pipe.load_lora_weights(str(checkpoint))
            helper = FluxKontextTrainer(self.config, on_metric=lambda metric: None, on_preview=None,
                                        get_learning_rate=None, is_cancelled=lambda: store.is_cancelled(job.id))
            dataset = resolve_dataset(self.config.dataset)
            pairs = list_training_pairs(dataset, 'val', self.config.caption, self.config.instruction)
            # Extend base comparison to every validation image only after
            # training selection has finished; no further fitting uses these images.
            pipe.disable_lora()
            evaluator.render(pipe, helper, pairs, 0, 0, device, stage='base_final')
            for pair in pairs:
                key = Path(pair.input_image_path).stem.removesuffix('_start')
                source = evaluator.output / 'base_final' / (key + '_prediction.png')
                target = evaluator.output / 'base' / (key + '_prediction.png')
                if not target.exists():
                    target.parent.mkdir(exist_ok=True)
                    temporary = target.with_name(target.name + '.tmp')
                    temporary.write_bytes(source.read_bytes())
                    temporary.replace(target)
            pipe.enable_lora()
            evaluator.render(pipe, helper, pairs, selected_epoch, 0, device, stage='final_selected')
            train_pairs = evaluator.select(list_training_pairs(dataset, 'train', self.config.caption, self.config.instruction), 2)
            evaluator.render(pipe, helper, train_pairs, selected_epoch, 0, device, stage='train_final_selected', split='train')
        finally:
            release_cuda_memory(pipe)

    def evaluate_historical(self, project, output):
        from diffusers import FluxKontextPipeline
        device = _training_device()
        errors = []
        for directory in sorted((project / 'runs').iterdir()):
            if not (directory / 'config.json').exists() or not (directory / 'weights').is_dir():
                continue
            pipe = None
            try:
                original = json.loads((directory / 'config.json').read_text())
                config = TrainingConfig.model_validate({**original,
                    'evaluation_inference_steps': self.config.evaluation_inference_steps,
                    'evaluation_guidance_scale': self.config.evaluation_guidance_scale})
                pipe = FluxKontextPipeline.from_pretrained(config.model_name, torch_dtype=_training_dtype(device),
                                                           token=_hugging_face_token(), local_files_only=True)
                pipe.to(device)
                pipe.set_progress_bar_config(disable=True)
                helper = FluxKontextTrainer(config, lambda metric: None, None, None, lambda: False)
                helper._load_lora_component(pipe, directory / 'weights')
                for module in [pipe.transformer, pipe.vae, pipe.text_encoder, pipe.text_encoder_2]:
                    if module is not None:
                        module.requires_grad_(False)
                        module.eval()
                pairs = list_training_pairs(resolve_dataset(config.dataset), 'val', config.caption, config.instruction)
                legacy = VisualEvaluator(config, output / ('legacy_' + directory.name[:8]), baseline_root=output / 'base')
                legacy.render(pipe, helper, legacy.select(pairs, self.config.evaluation_samples), 0, 0, device,
                              stage='historical_selected')
            except Exception as error:
                errors.append({'run': directory.name, 'error': str(error)})
                print('Historical evaluation unavailable: ' + str(error), flush=True)
            finally:
                release_cuda_memory(pipe)
                del pipe
                if 'helper' in locals():
                    del helper
                release_cuda_memory()
        write_json(output / 'historical_evaluation_errors.json', errors)
