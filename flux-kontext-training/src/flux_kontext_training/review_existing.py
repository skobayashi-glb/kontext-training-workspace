"""Evaluate historical adapters and the base; never creates an optimizer."""
import argparse
import json
import os
from pathlib import Path

from .datasets import list_training_pairs, resolve_dataset
from .job_store import JobStore
from .paths import RUNS_DIR
from .remote_cli import CLITrainingConfig
from .schemas import JobStatus
from .visual_experiment import VisualEvaluator, VisualExperiment, write_json
from .trainer import FluxKontextTrainer, _training_device, _training_dtype, _hugging_face_token, release_cuda_memory


class ExistingCheckpointReview:
    def __init__(self, config):
        self.config = config

    def run(self):
        from diffusers import FluxKontextPipeline
        store = JobStore(runs_dir=RUNS_DIR, load_existing=False)
        job = store.create(self.config)
        output = Path(job.run_dir)
        remote = Path(os.environ['KONTEXT_REMOTE_JOB_DIR'])
        project = Path(json.loads((remote / 'request.json').read_text())['remote_project'])
        write_json(remote / 'training.json', {'training_id': job.id, 'run_dir': job.run_dir, 'mode': 'evaluation_only'})
        write_json(output / 'review_mode.json', {'training_performed': False, 'optimizer_created': False})
        evaluator = VisualEvaluator(self.config, output / 'evaluation')
        evaluator.history_boards(project)
        store.set_status(job.id, JobStatus.RUNNING)
        pipe = helper = None
        try:
            print('EVALUATION_ONLY: no training or optimizer updates', flush=True)
            device = _training_device()
            pipe = FluxKontextPipeline.from_pretrained(self.config.model_name, torch_dtype=_training_dtype(device),
                                                       token=_hugging_face_token(), local_files_only=True)
            pipe.to(device)
            pipe.set_progress_bar_config(disable=True)
            for module in [pipe.transformer, pipe.vae, pipe.text_encoder, pipe.text_encoder_2]:
                if module is not None:
                    module.requires_grad_(False)
                    module.eval()
            helper = FluxKontextTrainer(self.config, lambda metric: None, None, None, lambda: store.is_cancelled(job.id))
            pairs = list_training_pairs(resolve_dataset(self.config.dataset), 'val', self.config.caption, self.config.instruction)
            evaluator.render(pipe, helper, evaluator.select(pairs, self.config.evaluation_samples), 0, 0, device, stage='base')
            release_cuda_memory(pipe)
            del pipe, helper
            pipe = helper = None
            release_cuda_memory()
            VisualExperiment(self.config).evaluate_historical(project, evaluator.output)
            store.set_status(job.id, JobStatus.COMPLETED)
            return 0
        except BaseException as error:
            store.set_status(job.id, JobStatus.FAILED, error=str(error))
            raise
        finally:
            release_cuda_memory(pipe)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    return ExistingCheckpointReview(CLITrainingConfig.model_validate_json(args.config.read_text(encoding='utf-8-sig'))).run()


if __name__ == '__main__':
    raise SystemExit(main())
