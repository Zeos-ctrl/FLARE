"""Automated experiments: durable train->eval runs an agent (or the UI) can drive.

An *experiment* is a durable record (``src/core/experiments.py``) plus one queued
job that trains a design and then benchmarks it, recording ``mean_match`` as the
comparable metric. Jobs flow through the shared ``job_manager`` so existing SSE
streaming (``/api/jobs/{job_id}/stream``) and the one-at-a-time queue apply.
"""
from __future__ import annotations

import re
import time
from typing import Optional

from fastapi import APIRouter, HTTPException

from src.api import state
from src.api.helpers import config_from_state
from src.api.jobs import JobCancelled, JobHandle, job_manager
from src.api.schemas import ExperimentRequest, ExperimentSpec, SettingsModel
from src.core import experiments

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


def _slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_") or "exp"


def _job_exists(job_id: Optional[str]) -> bool:
    return bool(job_id) and job_manager.get(job_id) is not None


def _view(record: dict) -> dict:
    """Reconcile a stored record and overlay live job progress for running rows."""
    record = experiments.reconcile(record, _job_exists)
    job = job_manager.get(record.get("job_id")) if record.get("job_id") else None
    if job and record.get("status") == "running":
        record = {**record, "progress": {**record.get("progress", {}), **job.progress}}
    return record


# ----------------------------------------------------------------------
# Harness target: train -> (optional) eval, recording metrics as it goes.
# ----------------------------------------------------------------------
def _run_experiment_target(handle: JobHandle, exp_id: str, spec: ExperimentSpec) -> dict:
    from src.core.trainer import Trainer

    try:
        record = experiments.load(exp_id)
        project_name = record["project_name"]
        experiments.update(exp_id, status="running", started_at=time.time())

        merged = {**state.load_settings().model_dump(), **spec.settings_overrides}
        settings = SettingsModel(**merged)
        design = spec.model_design or state.load_model_design(spec.model_name or "default")
        config = config_from_state(project_name, settings, design)

        def on_epoch_end(metrics: dict):
            handle.check_cancel()
            handle.emit({"phase": "train", **metrics})
            key = "best_val_loss_amp" if metrics.get("model_type") == "amp" else "best_val_loss_phase"
            experiments.update(
                exp_id,
                progress={k: metrics.get(k) for k in
                          ("model_type", "epoch", "total_epochs", "val_loss")},
                metrics={key: metrics.get("best_val_loss")},
            )

        handle.emit({"phase": "data", "message": f"Training {project_name}"})
        Trainer(config).run_training(on_epoch_end=on_epoch_end)

        metrics_out: dict = {}
        if spec.evaluate:
            from src.core.evaluator import Evaluator

            handle.check_cancel()
            handle.emit({"phase": "eval", "message": "Benchmarking against ground truth"})
            evaluator = Evaluator(config.project_path, device=config.device)
            results = evaluator.benchmark(n_samples=spec.eval_n_samples, batch_size=32, plot=True)
            metrics_out["mean_match"] = float(results.mean_match)
            handle.emit({"phase": "eval", "mean_match": float(results.mean_match)})

        experiments.update(exp_id, status="completed", finished_at=time.time(),
                           metrics=metrics_out, progress={"phase": "done"})
        handle.emit({"phase": "done", "message": "Experiment complete", **metrics_out})
        return {"experiment_id": exp_id, "project_name": project_name, **metrics_out}

    except JobCancelled:
        experiments.update(exp_id, status="cancelled", finished_at=time.time())
        raise
    except Exception as exc:  # noqa: BLE001
        experiments.update(exp_id, status="failed", finished_at=time.time(),
                           error=f"{type(exc).__name__}: {exc}")
        raise


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------
def submit_experiment_spec(spec: ExperimentSpec) -> dict:
    """Create a durable experiment record and queue its train->eval job.

    Shared by the HTTP endpoint and the in-process agent auto-runner so both
    enqueue experiments the same way (one durable record, one queued job).
    """
    exp_id = experiments.new_id()
    name = spec.name or (f"{spec.group}-{exp_id[:6]}" if spec.group else f"exp-{exp_id[:6]}")
    project_name = f"{_slugify(name)}_{exp_id[:6]}"

    record = {
        "id": exp_id,
        "name": name,
        "status": "pending",
        "created_at": time.time(),
        "group": spec.group,
        "tags": spec.tags,
        "source": spec.source,
        "job_id": None,
        "project_name": project_name,
        "spec": spec.model_dump(),
        "progress": {},
        "metrics": {},
        "error": None,
    }
    experiments.create(record)

    job = job_manager.submit(
        "experiment", _run_experiment_target,
        params={"experiment_id": exp_id, "project_name": project_name},
        exp_id=exp_id, spec=spec,
    )
    return experiments.update(exp_id, job_id=job.id)


@router.post("")
def create_experiment(req: ExperimentRequest):
    return submit_experiment_spec(ExperimentSpec(**req.model_dump()))


@router.get("")
def list_experiments(group: Optional[str] = None, status: Optional[str] = None):
    records = [_view(r) for r in experiments.list_all()]
    if group:
        records = [r for r in records if r.get("group") == group]
    if status:
        records = [r for r in records if r.get("status") == status]
    return records


@router.get("/{exp_id}")
def get_experiment(exp_id: str):
    record = experiments.load(exp_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return _view(record)


@router.post("/{exp_id}/cancel")
def cancel_experiment(exp_id: str):
    record = experiments.load(exp_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    if record.get("status") in experiments.ACTIVE:
        job_manager.cancel(record.get("job_id"))
        record = experiments.update(exp_id, status="cancelled")
    return _view(record)


@router.delete("/{exp_id}")
def delete_experiment(exp_id: str):
    if not experiments.delete(exp_id):
        raise HTTPException(status_code=404, detail="Experiment not found")
    return {"deleted": exp_id}
