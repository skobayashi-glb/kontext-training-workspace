from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from uuid import uuid4

from .paths import RUNS_DIR
from .schemas import EvaluationPreview, JobStatus, MetricEvent, PreviewHistoryEntry, PreviewSource, TrainingConfig, TrainingJob


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    def __init__(self, runs_dir: Path = RUNS_DIR, load_existing: bool = True) -> None:
        self.runs_dir = runs_dir
        self._jobs: dict[str, TrainingJob] = {}
        self._metrics: dict[str, list[MetricEvent]] = {}
        self._previews: dict[str, dict[str, EvaluationPreview]] = {}
        self._cancelled: set[str] = set()
        self._manual_learning_rate: set[str] = set()
        self._lock = Lock()
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        if load_existing:
            self._load_runs()

    def create(self, config: TrainingConfig) -> TrainingJob:
        job_id = uuid4().hex
        run_dir = self.runs_dir / job_id
        run_dir.mkdir(parents=True, exist_ok=True)
        now = utc_now()
        job = TrainingJob(
            id=job_id,
            status=JobStatus.QUEUED,
            config=config,
            created_at=now,
            updated_at=now,
            run_dir=str(run_dir),
        )
        with self._lock:
            self._jobs[job_id] = job
            self._metrics[job_id] = []
        self._write_config(run_dir, config)
        self._write_job(run_dir, job)
        return job

    def list(self) -> list[TrainingJob]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda job: job.created_at, reverse=True)

    def get(self, job_id: str) -> TrainingJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def metrics(self, job_id: str) -> list[MetricEvent]:
        with self._lock:
            return list(self._metrics.get(job_id, []))

    def preview(self, job_id: str, source: PreviewSource = "val") -> EvaluationPreview | None:
        with self._lock:
            return self._previews.get(job_id, {}).get(source)

    def preview_history(self, job_id: str, source: PreviewSource = "val") -> list[PreviewHistoryEntry]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return []
            run_dir = Path(job.run_dir)
        manifest_path = run_dir / "previews.jsonl"
        if not manifest_path.exists():
            return []
        entries = []
        with manifest_path.open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                entry = PreviewHistoryEntry.model_validate_json(line)
                if entry.source == source:
                    entries.append(entry)
        return entries

    def preview_history_image_path(self, job_id: str, source: PreviewSource, step: int, kind: str) -> Path | None:
        if kind not in {"input", "prediction", "target"}:
            return None
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            run_dir = Path(job.run_dir)
        path = run_dir / "previews" / f"{source}_step_{step:06d}" / f"{kind}.png"
        return path if path.exists() else None

    def learning_rate(self, job_id: str) -> float | None:
        with self._lock:
            if job_id not in self._manual_learning_rate:
                return None
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return job.config.learning_rate

    def set_status(self, job_id: str, status: JobStatus, error: str | None = None) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = status
            job.updated_at = utc_now()
            job.error = error
            run_dir = Path(job.run_dir)
        self._write_job(run_dir, job)

    def add_metric(self, job_id: str, metric: MetricEvent) -> None:
        with self._lock:
            self._metrics[job_id].append(metric)
            job = self._jobs[job_id]
            job.latest_metric = metric
            job.updated_at = utc_now()
            run_dir = Path(job.run_dir)
        with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as file:
            file.write(metric.model_dump_json() + "\n")
        self._write_job(run_dir, job)

    def set_preview(self, job_id: str, preview: EvaluationPreview) -> None:
        with self._lock:
            self._previews.setdefault(job_id, {})[preview.source] = preview
            job = self._jobs[job_id]
            job.updated_at = utc_now()

    def update_learning_rate(self, job_id: str, learning_rate: float) -> TrainingJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job.status not in {JobStatus.QUEUED, JobStatus.RUNNING}:
                raise ValueError("Learning rate can only be changed while a job is queued or running.")
            job.config.learning_rate = learning_rate
            job.updated_at = utc_now()
            self._manual_learning_rate.add(job_id)
            run_dir = Path(job.run_dir)
        self._write_config(run_dir, job.config)
        self._write_job(run_dir, job)
        return job

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            if job_id not in self._jobs:
                return False
            self._cancelled.add(job_id)
            job = self._jobs[job_id]
            if job.status == JobStatus.QUEUED:
                job.status = JobStatus.CANCELLED
                job.updated_at = utc_now()
                self._write_job(Path(job.run_dir), job)
            return True

    def is_cancelled(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._cancelled

    def delete(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            if job.status in {JobStatus.QUEUED, JobStatus.RUNNING}:
                raise ValueError("Running jobs must be stopped before deletion.")
            run_dir = Path(job.run_dir).resolve()
            runs_dir = self.runs_dir.resolve()
            if runs_dir not in run_dir.parents:
                raise ValueError("Invalid run directory.")
            del self._jobs[job_id]
            self._metrics.pop(job_id, None)
            self._previews.pop(job_id, None)
            self._cancelled.discard(job_id)
        if run_dir.exists():
            shutil.rmtree(run_dir)
        return True

    def _write_config(self, run_dir: Path, config: TrainingConfig) -> None:
        with (run_dir / "config.json").open("w", encoding="utf-8") as file:
            json.dump(config.model_dump(), file, indent=2)

    def _write_job(self, run_dir: Path, job: TrainingJob) -> None:
        with (run_dir / "job.json").open("w", encoding="utf-8") as file:
            json.dump(job.model_dump(mode="json"), file, indent=2)

    def _load_runs(self) -> None:
        for run_dir in sorted(self.runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            job = self._read_job(run_dir)
            if job is None:
                continue
            metrics = self._read_metrics(run_dir)
            if metrics:
                job.latest_metric = metrics[-1]
            if job.status in {JobStatus.QUEUED, JobStatus.RUNNING}:
                job.status = JobStatus.CANCELLED
                job.updated_at = utc_now()
                job.error = job.error or "Job was interrupted before the server restarted."
                self._write_job(run_dir, job)
            self._jobs[job.id] = job
            self._metrics[job.id] = metrics

    def _read_job(self, run_dir: Path) -> TrainingJob | None:
        job_path = run_dir / "job.json"
        if job_path.exists():
            with job_path.open(encoding="utf-8") as file:
                data = json.load(file)
            if "config" in data and isinstance(data["config"], dict):
                data["config"] = self._read_config(data["config"])
            return TrainingJob.model_validate(data)

        config_path = run_dir / "config.json"
        if not config_path.exists():
            return None
        with config_path.open(encoding="utf-8") as file:
            config = self._read_config(json.load(file))
        created_at = datetime.fromtimestamp(run_dir.stat().st_ctime, timezone.utc).isoformat()
        return TrainingJob(
            id=run_dir.name,
            status=JobStatus.COMPLETED,
            config=config,
            created_at=created_at,
            updated_at=created_at,
            run_dir=str(run_dir),
        )

    def _read_metrics(self, run_dir: Path) -> list[MetricEvent]:
        metrics_path = run_dir / "metrics.jsonl"
        if not metrics_path.exists():
            return []
        metrics: list[MetricEvent] = []
        with metrics_path.open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                metrics.append(MetricEvent.model_validate_json(line))
        return metrics

    def _read_config(self, data: dict[str, object]) -> TrainingConfig:
        if "epochs" not in data and "max_steps" in data:
            data["epochs"] = data["max_steps"]
        max_sequence_length = data.get("max_sequence_length")
        if isinstance(max_sequence_length, int) and max_sequence_length > 512:
            data["max_sequence_length"] = 512
        return TrainingConfig.model_validate(data)
