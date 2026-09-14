"""Project (checkpoint) discovery and management."""
from __future__ import annotations

import json
import os
import shutil

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/projects", tags=["projects"])

CHECKPOINT_DIR = os.environ.get("FLARE_CHECKPOINTS", "checkpoints")


def _project_info(name: str) -> dict:
    path = os.path.join(CHECKPOINT_DIR, name)
    info = {
        "name": name,
        "path": path,
        "trained": os.path.exists(os.path.join(path, "amp_model.pt")),
        "evaluated": os.path.exists(os.path.join(path, "evaluation", "benchmark_results.json")),
        "has_dataset": os.path.exists(os.path.join(path, "dataset.npz")),
        "config": None,
        "meta": None,
    }
    cfg_path = os.path.join(path, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            info["config"] = json.load(f)
    meta_path = os.path.join(path, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            info["meta"] = json.load(f)
    return info


@router.get("")
def list_projects():
    if not os.path.isdir(CHECKPOINT_DIR):
        return []
    names = [d for d in os.listdir(CHECKPOINT_DIR)
             if os.path.isdir(os.path.join(CHECKPOINT_DIR, d))]
    return [_project_info(n) for n in sorted(names)]


@router.get("/{name}")
def get_project(name: str):
    path = os.path.join(CHECKPOINT_DIR, name)
    if not os.path.isdir(path):
        raise HTTPException(status_code=404, detail="Project not found")
    return _project_info(name)


@router.delete("/{name}")
def delete_project(name: str):
    path = os.path.join(CHECKPOINT_DIR, name)
    if not os.path.isdir(path):
        raise HTTPException(status_code=404, detail="Project not found")
    shutil.rmtree(path)
    return {"deleted": name}
