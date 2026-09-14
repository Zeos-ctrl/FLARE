"""Launch and monitor training jobs."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.api import state
from src.api.helpers import config_from_state
from src.api.jobs import JobHandle, job_manager
from src.api.schemas import TrainRequest

router = APIRouter(prefix="/api/training", tags=["training"])


def _train_target(handle: JobHandle, config) -> dict:
    from src.core.trainer import Trainer

    handle.emit({"phase": "data", "message": "Preparing dataset"})
    trainer = Trainer(config)
    dataset = trainer.prepare_data()
    handle.emit({"phase": "data", "message": f"Dataset ready: {dataset.inputs.shape[0]} rows"})

    def on_epoch_end(metrics: dict):
        handle.check_cancel()
        handle.emit({"phase": "train", **metrics})

    handle.emit({"phase": "train", "message": "Training models"})
    trainer.run_training(dataset, on_epoch_end=on_epoch_end)
    handle.emit({"phase": "done", "message": "Training complete"})
    return {"project_name": config.project_name, "checkpoint_path": config.project_path}


@router.post("")
def start_training(req: TrainRequest):
    try:
        design = state.load_model_design(req.model_name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model design '{req.model_name}' not found")
    config = config_from_state(req.project_name, state.load_settings(), design)
    job = job_manager.submit(
        "training", _train_target,
        params={"project_name": config.project_name, "model": req.model_name},
        config=config,
    )
    return job.to_dict()
