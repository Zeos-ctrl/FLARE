"""Launch evaluation jobs and serve benchmark results/plots."""
from __future__ import annotations

import json
import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from src.api.jobs import JobHandle, job_manager
from src.api.schemas import EvaluateRequest

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])

CHECKPOINT_DIR = os.environ.get("FLARE_CHECKPOINTS", "checkpoints")


def _evaluate_target(handle: JobHandle, project_name, n_samples, batch_size, device) -> dict:
    from src.core.evaluator import Evaluator

    path = os.path.join(CHECKPOINT_DIR, project_name)
    handle.emit({"phase": "eval", "message": f"Benchmarking {project_name}"})
    evaluator = Evaluator(path, device=device)
    results = evaluator.benchmark(n_samples=n_samples, batch_size=batch_size, plot=True)
    handle.emit({"phase": "done", "message": "Evaluation complete",
                 "mean_match": results.mean_match})
    return {"project_name": project_name, "results": results.to_dict()}


@router.post("")
def start_evaluation(req: EvaluateRequest):
    job = job_manager.submit(
        "evaluation", _evaluate_target,
        params={"project_name": req.project_name},
        project_name=req.project_name, n_samples=req.n_samples,
        batch_size=req.batch_size, device=req.device,
    )
    return job.to_dict()


@router.get("/{project_name}/results")
def get_results(project_name: str):
    path = os.path.join(CHECKPOINT_DIR, project_name, "evaluation", "benchmark_results.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="No evaluation results")
    with open(path) as f:
        return json.load(f)


@router.get("/{project_name}/plots")
def list_plots(project_name: str):
    eval_dir = os.path.join(CHECKPOINT_DIR, project_name, "evaluation")
    if not os.path.isdir(eval_dir):
        return []
    return [f for f in os.listdir(eval_dir) if f.endswith(".png")]


@router.get("/{project_name}/plots/{plot_name}")
def get_plot(project_name: str, plot_name: str):
    if "/" in plot_name or ".." in plot_name:
        raise HTTPException(status_code=400, detail="Invalid plot name")
    path = os.path.join(CHECKPOINT_DIR, project_name, "evaluation", plot_name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Plot not found")
    return FileResponse(path, media_type="image/png")
