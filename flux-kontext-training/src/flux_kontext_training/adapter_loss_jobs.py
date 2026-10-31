from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from uuid import uuid4

from .adapter_loss_evaluator import AdapterLossEvaluator
from .job_store import utc_now
from .paths import MODELS_DIR, RUNS_DIR
from .schemas import AdapterLossJob, AdapterLossRequest, AdapterLossResult, AdapterSummary, JobStatus


class AdapterRegistry:
    """Lists local LoRA adapter files that can be evaluated."""

    def list(self) -> list[AdapterSummary]:
        if not MODELS_DIR.exists():
            return []
        adapters: list[AdapterSummary] = []
        for path in sorted(MODELS_DIR.glob("*.safetensors")):
            if not path.is_file():
                continue
            adapters.append(
                AdapterSummary(
                    name=path.name,
                    path=str(path),
                    size_bytes=path.stat().st_size,
                )
            )
        return adapters

    def resolve(self, adapter_path: str) -> Path:
        path = Path(adapter_path)
        if not path.is_absolute():
            path = MODELS_DIR / path
        resolved = path.resolve()
        models_dir = MODELS_DIR.resolve()
        if resolved != models_dir and models_dir not in resolved.parents:
            raise ValueError("Invalid adapter path.")
        if not resolved.is_file() or resolved.suffix != ".safetensors":
            raise ValueError(f"Adapter '{adapter_path}' does not exist.")
        return resolved


class AdapterLossJobStore:
    """Persistent store for adapter loss evaluation jobs."""

    def __init__(self) -> None:
        self._jobs: dict[str, AdapterLossJob] = {}
        self._lock = Lock()
        self._runs_dir = RUNS_DIR / "adapter_loss"
        self._runs_dir.mkdir(parents=True, exist_ok=True)
        self._load_jobs()

    def create(self, request: AdapterLossRequest) -> AdapterLossJob:
        job_id = uuid4().hex
        run_dir = self._runs_dir / job_id
        run_dir.mkdir(parents=True, exist_ok=True)
        now = utc_now()
        job = AdapterLossJob(
            id=job_id,
            status=JobStatus.QUEUED,
            request=request,
            created_at=now,
            updated_at=now,
            run_dir=str(run_dir),
        )
        with self._lock:
            self._jobs[job.id] = job
        self._write_request(run_dir, request)
        self._write_job(run_dir, job)
        return job

    def list(self) -> list[AdapterLossJob]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda job: job.created_at, reverse=True)

    def get(self, job_id: str) -> AdapterLossJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def set_running(self, job_id: str) -> None:
        self._set(job_id, status=JobStatus.RUNNING)

    def set_completed(self, job_id: str, results: list[AdapterLossResult]) -> None:
        self._set(job_id, status=JobStatus.COMPLETED, results=results)

    def set_failed(self, job_id: str, error: str) -> None:
        self._set(job_id, status=JobStatus.FAILED, error=error)

    def _set(
        self,
        job_id: str,
        status: JobStatus,
        results: list[AdapterLossResult] | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = status
            job.updated_at = utc_now()
            if results is not None:
                job.results = results
            job.error = error
            run_dir = Path(job.run_dir)
        self._write_job(run_dir, job)

    def _write_request(self, run_dir: Path, request: AdapterLossRequest) -> None:
        with (run_dir / "request.json").open("w", encoding="utf-8") as file:
            json.dump(request.model_dump(mode="json"), file, indent=2)

    def _write_job(self, run_dir: Path, job: AdapterLossJob) -> None:
        with (run_dir / "job.json").open("w", encoding="utf-8") as file:
            json.dump(job.model_dump(mode="json"), file, indent=2)

    def _load_jobs(self) -> None:
        for run_dir in sorted(self._runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            job_path = run_dir / "job.json"
            if not job_path.is_file():
                continue
            with job_path.open(encoding="utf-8") as file:
                data = json.load(file)
            data["run_dir"] = data.get("run_dir") or str(run_dir)
            job = AdapterLossJob.model_validate(data)
            if job.status in {JobStatus.QUEUED, JobStatus.RUNNING}:
                job.status = JobStatus.CANCELLED
                job.updated_at = utc_now()
                job.error = job.error or "Adapter loss job was interrupted before the server restarted."
                self._write_job(run_dir, job)
            self._jobs[job.id] = job


class AdapterLossJobRunner:
    """Runs adapter loss evaluation and publishes the result to a store."""

    def __init__(self, store: AdapterLossJobStore, registry: AdapterRegistry) -> None:
        self.store = store
        self.registry = registry

    def run(self, job_id: str) -> None:
        job = self.store.get(job_id)
        if job is None:
            return
        self.store.set_running(job_id)
        try:
            adapter_path = self.registry.resolve(job.request.adapter_path)
            splits = ("train", "val") if job.request.split == "both" else (job.request.split,)
            evaluator = AdapterLossEvaluator(
                config=job.request.config,
                adapter_path=adapter_path,
                batch_size=job.request.batch_size,
                max_pairs=job.request.max_pairs,
                noise_repeats=job.request.noise_repeats,
                rollout_steps=job.request.rollout_steps,
            )
            summaries = evaluator.evaluate(splits)
            results = [AdapterLossResult.model_validate(summary.as_dict()) for summary in summaries]
            self.store.set_completed(job_id, results)
        except Exception as exc:
            self.store.set_failed(job_id, str(exc))
