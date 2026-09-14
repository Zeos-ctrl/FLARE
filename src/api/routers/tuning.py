"""Launch and monitor hyperparameter tuning jobs."""
from __future__ import annotations

import json
import os

from fastapi import APIRouter, HTTPException

from src.api import state
from src.api.helpers import config_from_state
from src.api.jobs import JobHandle, job_manager
from src.api.schemas import TuneRequest

router = APIRouter(prefix="/api/tuning", tags=["tuning"])


def _tune_target(handle: JobHandle, config, model_type: str, n_trials) -> dict:
    from src.core.tuner import Tuner

    handle.emit({"phase": "data", "message": "Generating HPO dataset"})
    tuner = Tuner(config)

    def callback(study, trial):
        handle.check_cancel()
        handle.emit({
            "phase": "tune",
            "trial": trial.number,
            "value": trial.value,
            "best_value": study.best_value,
            "params": trial.params,
        })

    tuner.run_tuning(model_type=model_type, n_trials=n_trials, callback=callback)
    handle.emit({"phase": "done", "message": "Tuning complete"})
    return {"project_name": config.project_name}


@router.post("")
def start_tuning(req: TuneRequest):
    try:
        design = state.load_model_design(req.model_name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model design '{req.model_name}' not found")
    config = config_from_state(req.project_name, state.load_settings(), design)
    job = job_manager.submit(
        "tuning", _tune_target,
        params={"project_name": config.project_name},
        config=config, model_type=req.model_type, n_trials=req.n_trials,
    )
    return job.to_dict()


@router.get("/{project_name}/best_params")
def get_best_params(project_name: str):
    """Return the best amp/phase hyperparameters found for a project."""
    base = os.path.join("checkpoints", project_name)
    out = {}
    for kind in ("amp", "phase"):
        path = os.path.join(base, f"{kind}_params.json")
        if os.path.exists(path):
            with open(path) as f:
                out[kind] = json.load(f)
    if not out:
        raise HTTPException(status_code=404, detail="No tuned params found")
    return out
