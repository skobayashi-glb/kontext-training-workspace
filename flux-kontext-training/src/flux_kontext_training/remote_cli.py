"""Finite training command for EC2 jobs; importing this module never loads CUDA."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys

from pydantic import ConfigDict, model_validator

from .checkpoints import TrainingCheckpointRegistry
from .datasets import list_training_pairs, resolve_dataset
from .job_store import JobStore
from .paths import RUNS_DIR
from .schemas import JobStatus, TrainingConfig


class CLITrainingConfig(TrainingConfig):
    # A misspelled "epochs" must not silently run the default ten epochs.
    model_config = ConfigDict(extra='forbid')

    @model_validator(mode='after')
    def bounded_visual_run(self):
        if self.visual_evaluation_enabled and self.epochs > 30:
            raise ValueError('Visual experiments are limited to 30 epochs.')
        if self.visual_evaluation_enabled and (self.method != 'lora' or self.train_text_encoder or self.train_text_encoder_2):
            raise ValueError('Visual experiments currently require transformer LoRA and frozen text encoders.')
        return self


class TrainingCommand:
    def __init__(self, config: TrainingConfig, trainer_factory=None, release_memory=None) -> None:
        self.config = config
        self.trainer_factory = trainer_factory
        self.release_memory = release_memory

    def validate(self) -> None:
        dataset = resolve_dataset(self.config.dataset)
        for split in ('train', 'val'):
            if not list_training_pairs(dataset, split, self.config.caption, self.config.instruction):
                raise ValueError(f"Dataset '{self.config.dataset}' has no paired {split} images.")
        if self.config.source_checkpoint_id:
            if self.config.method == 'full':
                raise ValueError('Full training must start from the base model.')
            checkpoint = TrainingCheckpointRegistry().get(self.config.source_checkpoint_id)
            if checkpoint is None or checkpoint.model_name != self.config.model_name or 'full' in checkpoint.components:
                raise ValueError('Source checkpoint is missing or incompatible.')

    def run(self) -> int:
        self.validate()
        # Only load the GPU stack after validation. Never load/mutate past jobs.
        if self.trainer_factory is None:
            from .trainer import FluxKontextTrainer, release_cuda_memory
            self.trainer_factory = FluxKontextTrainer
            self.release_memory = release_cuda_memory
        store = JobStore(runs_dir=RUNS_DIR, load_existing=False)
        job = store.create(self.config)
        remote_dir = os.getenv('KONTEXT_REMOTE_JOB_DIR')
        if remote_dir:
            (Path(remote_dir) / 'training.json').write_text(
                json.dumps({'training_id': job.id, 'run_dir': job.run_dir}), encoding='utf-8')
        print(f'training_id={job.id}', flush=True)
        print(f'output_dir={job.run_dir}', flush=True)

        def cancel(signum, frame):
            store.cancel(job.id)

        previous = {sig: signal.signal(sig, cancel) for sig in (signal.SIGTERM, signal.SIGINT)}
        trainer = None
        try:
            store.set_status(job.id, JobStatus.RUNNING)
            trainer = self.trainer_factory(
                config=job.config,
                on_metric=lambda metric: store.add_metric(job.id, metric),
                on_preview=lambda preview: store.set_preview(job.id, preview),
                get_learning_rate=lambda: store.learning_rate(job.id),
                is_cancelled=lambda: store.is_cancelled(job.id),
                output_dir=Path(job.run_dir),
            )
            trainer.run()
            cancelled = store.is_cancelled(job.id)
            store.set_status(job.id, JobStatus.CANCELLED if cancelled else JobStatus.COMPLETED)
            return 130 if cancelled else 0
        except BaseException as error:
            store.set_status(job.id, JobStatus.FAILED, error=str(error))
            raise
        finally:
            del trainer
            if self.release_memory:
                self.release_memory()
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = CLITrainingConfig.model_validate_json(args.config.read_text(encoding='utf-8-sig'))
    if config.visual_evaluation_enabled:
        from .visual_experiment import VisualExperiment
        return VisualExperiment(config).run()
    return TrainingCommand(config).run()


if __name__ == '__main__':
    sys.exit(main())
