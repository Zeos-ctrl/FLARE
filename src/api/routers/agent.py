"""Agent auto-run: start/stop/inspect an autonomous experiment search.

Thin HTTP surface over ``src/api/agent_runner.py``. The runner loops
read-leaderboard -> propose (Claude or random) -> submit -> wait, on its own
thread; these routes let the dashboard drive and watch it.
"""
from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException

from src.agent.proposer import (
    LOCAL_DEFAULT_URL,
    claude_available,
    list_local_models,
)
from src.api import agent_runner
from src.api.schemas import AutoRunConfig

router = APIRouter(prefix="/api/agent", tags=["agent"])


@router.get("/available")
def available():
    """Whether the Claude proposer can run (SDK + credentials present)."""
    return {"claude": claude_available()}


@router.get("/local/models")
def local_models(base_url: str = LOCAL_DEFAULT_URL, api_key: str = ""):
    """List models a local OpenAI-compatible server exposes (the GGUF selector).

    ``base_url`` points at llama.cpp's ``llama-server`` (default :8080), LM Studio
    (:1234), Ollama (:11434), or any other OpenAI-compatible endpoint. ``api_key``
    is only needed if that server (or a proxy in front of it) requires one.
    """
    try:
        models = list_local_models(base_url, api_key=api_key or None)
        return {"ok": True, "base_url": base_url, "models": models}
    except Exception as exc:  # noqa: BLE001 — server down / bad URL / bad key
        return {"ok": False, "base_url": base_url, "models": [], "error": str(exc)}


@router.post("/auto")
def start_auto(cfg: AutoRunConfig):
    group = cfg.group or f"auto-{time.strftime('%Y%m%d-%H%M%S')}"
    try:
        runner = agent_runner.start_runner(
            group=group,
            budget=cfg.budget,
            proposer=cfg.proposer,
            objective=cfg.objective,
            local_base_url=cfg.local_base_url,
            local_model=cfg.local_model,
            local_api_key=cfg.local_api_key,
            num_epochs=cfg.num_epochs,
            eval_n_samples=cfg.eval_n_samples,
            evaluate=cfg.evaluate,
            settings_overrides=cfg.settings_overrides,
            notes=cfg.notes,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return runner.to_dict()


@router.get("/auto")
def list_auto():
    return [r.to_dict() for r in agent_runner.list_runners()]


@router.get("/auto/active")
def active_auto():
    runner = agent_runner.active_runner()
    return runner.to_dict() if runner else None


@router.get("/auto/{runner_id}")
def get_auto(runner_id: str):
    runner = agent_runner.get_runner(runner_id)
    if runner is None:
        raise HTTPException(status_code=404, detail="Auto-run not found")
    return runner.to_dict()


@router.post("/auto/{runner_id}/stop")
def stop_auto(runner_id: str):
    runner = agent_runner.get_runner(runner_id)
    if runner is None:
        raise HTTPException(status_code=404, detail="Auto-run not found")
    runner.stop()
    return runner.to_dict()
