from __future__ import annotations

import json
from pathlib import Path

from .paths import RUNS_DIR
from .schemas import (
    CheckpointComponent,
    CheckpointManifest,
    TrainingCheckpointSummary,
    TrainingConfig,
    TrainingMethod,
)


class TrainingCheckpointRegistry:
    def __init__(self, runs_dir: Path = RUNS_DIR) -> None:
        self.runs_dir = runs_dir

    def list(self) -> list[TrainingCheckpointSummary]:
        if not self.runs_dir.exists():
            return []
        checkpoints = [
            checkpoint
            for run_dir in sorted(self.runs_dir.iterdir())
            if run_dir.is_dir()
            for checkpoint in [self._summary_for_run(run_dir)]
            if checkpoint is not None
        ]
        return sorted(checkpoints, key=lambda checkpoint: checkpoint.created_at, reverse=True)

    def get(self, checkpoint_id: str) -> TrainingCheckpointSummary | None:
        run_dir = self.runs_dir / checkpoint_id
        if not run_dir.is_dir():
            return None
        return self._summary_for_run(run_dir)

    def resolve_manifest(self, checkpoint_id: str) -> CheckpointManifest:
        checkpoint = self.get(checkpoint_id)
        if checkpoint is None:
            raise ValueError(f"Checkpoint '{checkpoint_id}' does not exist or has no saved weights.")
        return self.read_manifest(Path(checkpoint.path))

    def read_manifest(self, weights_dir: Path) -> CheckpointManifest:
        manifest_path = weights_dir / "manifest.json"
        if manifest_path.is_file():
            with manifest_path.open(encoding="utf-8") as file:
                data = json.load(file)
            return CheckpointManifest.model_validate(data)

        config = self._read_config(weights_dir.parent)
        if config is None:
            raise ValueError(f"Checkpoint '{weights_dir}' does not contain a readable config.")
        component = self._infer_component(weights_dir, config.method)
        if component is None:
            raise ValueError(f"Checkpoint '{weights_dir}' does not contain supported weights.")
        return CheckpointManifest(model_name=config.model_name, components=[component])

    def write_manifest(
        self,
        weights_dir: Path,
        model_name: str,
        components: list[CheckpointComponent],
    ) -> None:
        manifest = CheckpointManifest(model_name=model_name, components=components)
        with (weights_dir / "manifest.json").open("w", encoding="utf-8") as file:
            json.dump(manifest.model_dump(), file, indent=2)

    def _summary_for_run(self, run_dir: Path) -> TrainingCheckpointSummary | None:
        config = self._read_config(run_dir)
        if config is None:
            return None
        weights_dir = run_dir / "weights"
        if not weights_dir.is_dir():
            return None
        try:
            manifest = self.read_manifest(weights_dir)
        except ValueError:
            return None
        components = [component.method for component in manifest.components]
        if not components:
            return None
        created_at = self._read_created_at(run_dir)
        component_label = " + ".join(_method_label(method) for method in components)
        return TrainingCheckpointSummary(
            id=run_dir.name,
            label=f"{config.dataset} / {component_label} / {run_dir.name[:8]}",
            run_id=run_dir.name,
            created_at=created_at,
            dataset=config.dataset,
            model_name=manifest.model_name,
            method=config.method,
            components=components,
            path=str(weights_dir),
        )

    def _read_config(self, run_dir: Path) -> TrainingConfig | None:
        config_path = run_dir / "config.json"
        if not config_path.is_file():
            return None
        with config_path.open(encoding="utf-8") as file:
            data = json.load(file)
        if "epochs" not in data and "max_steps" in data:
            data["epochs"] = data["max_steps"]
        return TrainingConfig.model_validate(data)

    def _read_created_at(self, run_dir: Path) -> str:
        job_path = run_dir / "job.json"
        if job_path.is_file():
            with job_path.open(encoding="utf-8") as file:
                data = json.load(file)
            created_at = data.get("created_at")
            if isinstance(created_at, str):
                return created_at
        return str(run_dir.stat().st_ctime)

    def _infer_component(self, weights_dir: Path, method: TrainingMethod) -> CheckpointComponent | None:
        if method == "full":
            transformer_dir = weights_dir / "transformer"
            if transformer_dir.is_dir():
                return CheckpointComponent(method="full", path=str(transformer_dir))
            return None
        if method == "prompt_tuning":
            prompt_path = weights_dir / "prompt_tuning.pt"
            if prompt_path.is_file():
                return CheckpointComponent(method="prompt_tuning", path=str(prompt_path))
            return None
        if any(path.is_file() for path in weights_dir.iterdir() if path.name != "manifest.json"):
            return CheckpointComponent(method="lora", path=str(weights_dir))
        return None


def _method_label(method: TrainingMethod) -> str:
    if method == "prompt_tuning":
        return "Prompt-tuning"
    if method == "full":
        return "Full"
    return "LoRA"
