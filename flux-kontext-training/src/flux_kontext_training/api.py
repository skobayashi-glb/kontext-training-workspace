from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from threading import Thread

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .adapter_loss_jobs import AdapterLossJobRunner, AdapterLossJobStore, AdapterRegistry
from .checkpoints import TrainingCheckpointRegistry
from .datasets import list_datasets, list_training_pairs, resolve_dataset
from .job_store import JobStore
from .paths import DATASETS_DIR, UI_DIST_DIR
from .schemas import AdapterLossRequest, JobStatus, LearningRateUpdate, PreviewSource, TrainingConfig
from .trainer import FluxKontextTrainer, release_cuda_memory


app = FastAPI(title="Flux Kontext Training")
store = JobStore()
checkpoints = TrainingCheckpointRegistry()
adapter_registry = AdapterRegistry()
adapter_loss_jobs = AdapterLossJobStore()
adapter_loss_runner = AdapterLossJobRunner(adapter_loss_jobs, adapter_registry)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/datasets")
def datasets() -> dict[str, object]:
    return {"datasets": list_datasets()}


@app.get("/api/prompts")
def prompt_presets() -> dict[str, object]:
    prompts_file = DATASETS_DIR / "default_prompts.json"
    if not prompts_file.is_file():
        return {"presets": []}
    data = json.loads(prompts_file.read_text(encoding="utf-8"))
    presets = [
        {"id": key, "label": key, "task": entry.get("task", "")}
        for key, entry in data.get("prompts", {}).items()
    ]
    return {"presets": presets}


@app.get("/api/training/modes")
def training_modes() -> dict[str, object]:
    return {
        "modes": [
            {"id": "lora", "label": "LoRA", "enabled": True},
            {"id": "prompt_tuning", "label": "Prompt-tuning", "enabled": True},
            {"id": "full", "label": "Full", "enabled": True},
        ]
    }


@app.get("/api/checkpoints")
def training_checkpoints() -> dict[str, object]:
    return {"checkpoints": checkpoints.list()}


@app.get("/api/adapters")
def adapters() -> dict[str, object]:
    return {"adapters": adapter_registry.list()}


@app.post("/api/adapter-loss/jobs")
def create_adapter_loss_job(request: AdapterLossRequest):
    try:
        adapter_registry.resolve(request.adapter_path)
        resolve_dataset(request.config.dataset)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job = adapter_loss_jobs.create(request)
    worker = Thread(target=adapter_loss_runner.run, args=(job.id,), daemon=True)
    worker.start()
    return job


@app.get("/api/adapter-loss/jobs")
def list_adapter_loss_jobs():
    return {"jobs": adapter_loss_jobs.list()}


@app.get("/api/adapter-loss/jobs/{job_id}")
def adapter_loss_job(job_id: str):
    job = adapter_loss_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Adapter loss job not found.")
    return job


@app.post("/api/jobs")
def create_job(config: TrainingConfig):
    try:
        dataset = resolve_dataset(config.dataset)
        train_pairs = list_training_pairs(dataset, "train", config.caption, config.instruction)
        val_pairs = list_training_pairs(dataset, "val", config.caption, config.instruction)
        if not train_pairs:
            raise ValueError(f"Dataset '{config.dataset}' must contain paired *_start/*_end train images.")
        if not val_pairs:
            raise ValueError(f"Dataset '{config.dataset}' must contain paired *_start/*_end val images.")
        if config.method == "full" and config.source_checkpoint_id:
            raise ValueError("Full-parameter training must start from the base model, not an adapter checkpoint.")
        if config.source_checkpoint_id:
            checkpoint = checkpoints.get(config.source_checkpoint_id)
            if checkpoint is None:
                raise ValueError(f"Checkpoint '{config.source_checkpoint_id}' does not exist or has no saved weights.")
            if checkpoint.model_name != config.model_name:
                raise ValueError(
                    "Selected checkpoint was trained from a different base model. "
                    f"Checkpoint model: '{checkpoint.model_name}', current model: '{config.model_name}'."
                )
            if "full" in checkpoint.components:
                raise ValueError("Full-parameter checkpoints cannot be used as adapter training sources.")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job = store.create(config)
    worker = Thread(target=_run_job, args=(job.id,), daemon=True)
    worker.start()
    return job


@app.get("/api/jobs")
def jobs():
    return {"jobs": store.list()}


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    training_job = store.get(job_id)
    if training_job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return training_job


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    if not store.cancel(job_id):
        raise HTTPException(status_code=404, detail="Job not found.")
    return {"ok": True}


@app.patch("/api/jobs/{job_id}/learning-rate")
def update_job_learning_rate(job_id: str, update: LearningRateUpdate):
    try:
        training_job = store.update_learning_rate(job_id, update.learning_rate)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if training_job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return training_job


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    try:
        if not store.delete(job_id):
            raise HTTPException(status_code=404, detail="Job not found.")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True}


@app.get("/api/jobs/{job_id}/metrics")
def job_metrics(job_id: str):
    if store.get(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {"metrics": store.metrics(job_id)}


@app.get("/api/jobs/{job_id}/preview")
def job_preview(job_id: str, source: PreviewSource = "val"):
    if store.get(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {"preview": store.preview(job_id, source)}


@app.get("/api/jobs/{job_id}/previews")
def job_preview_history(job_id: str, source: PreviewSource = "val"):
    if store.get(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {"previews": store.preview_history(job_id, source)}


@app.get("/api/jobs/{job_id}/previews/{source}/{step}/{kind}")
def job_preview_history_image(job_id: str, source: PreviewSource, step: int, kind: str):
    if store.get(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    path = store.preview_history_image_path(job_id, source, step, kind)
    if path is None:
        raise HTTPException(status_code=404, detail="Preview image not found.")
    return FileResponse(path)


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str):
    if store.get(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found.")

    async def stream():
        sent = 0
        last_preview_step: dict[str, int] = {"train": -1, "val": -1}
        while True:
            metrics = store.metrics(job_id)
            for metric in metrics[sent:]:
                yield _sse("metric", metric.model_dump())
            sent = len(metrics)
            for source in ("train", "val"):
                preview = store.preview(job_id, source)
                if preview is not None and preview.step != last_preview_step[source]:
                    yield _sse("preview", preview.model_dump())
                    last_preview_step[source] = preview.step
            current = store.get(job_id)
            if current is None:
                yield _sse("error", {"message": "Job not found."})
                return
            yield _sse("status", current.model_dump(mode="json"))
            if current.status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}:
                return
            await asyncio.sleep(1)

    return StreamingResponse(stream(), media_type="text/event-stream")


if UI_DIST_DIR.exists():
    app.mount("/assets", StaticFiles(directory=UI_DIST_DIR / "assets"), name="assets")


@app.get("/{path:path}", include_in_schema=False)
def spa(path: str):
    index = UI_DIST_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    raise HTTPException(
        status_code=404,
        detail="UI build not found. Run `npm install` and `npm run build` in the ui directory.",
    )


def _run_job(job_id: str) -> None:
    job = store.get(job_id)
    if job is None:
        return
    store.set_status(job_id, JobStatus.RUNNING)
    trainer = None
    try:
        trainer = FluxKontextTrainer(
            config=job.config,
            on_metric=lambda metric: store.add_metric(job_id, metric),
            on_preview=lambda preview: store.set_preview(job_id, preview),
            get_learning_rate=lambda: store.learning_rate(job_id),
            is_cancelled=lambda: store.is_cancelled(job_id),
            output_dir=Path(job.run_dir),
        )
        trainer.run()
        status = JobStatus.CANCELLED if store.is_cancelled(job_id) else JobStatus.COMPLETED
        store.set_status(job_id, status)
    except Exception as exc:
        store.set_status(job_id, JobStatus.FAILED, error=str(exc))
    finally:
        del trainer
        release_cuda_memory()


def _sse(event: str, data: dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def run() -> None:
    import uvicorn

    reload_enabled = os.getenv("FLUX_KONTEXT_RELOAD", "").lower() in {"1", "true", "yes"}
    uvicorn.run("flux_kontext_training.api:app", host="127.0.0.1", port=8000, reload=reload_enabled)
