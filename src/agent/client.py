"""FlareClient — a thin HTTP client an agent uses to run experiments.

The dashboard API is the single entry point: submitting through it means every
run is queued (one at a time), streamed to the GUI, and recorded durably. An
agent's loop is just:

    client = FlareClient()
    for _ in range(budget):
        board = client.leaderboard(group="my_sweep")      # what worked so far
        spec = propose_next(board, client.search_space())  # your logic / an LLM
        rec = client.submit_experiment(spec)
        rec = client.wait(rec["id"])                        # blocks until done
        print(rec["name"], rec["metrics"].get("mean_match"))

Requires ``requests`` (already a dependency of the project's HTTP tests).
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import requests

# Fields of a model design an agent may vary, with suggested ranges. Mirrors
# ModelDesignModel; see src/api/schemas.py. Purely advisory — the API accepts
# any valid design.
SEARCH_SPACE: Dict[str, Any] = {
    "amp_hidden_size": {"type": "int", "choices": [64, 128, 256, 512]},
    "amp_layers": {"type": "int", "range": [2, 6]},
    "amp_banks": {"type": "int", "range": [1, 6]},
    "amp_dropout": {"type": "float", "range": [0.0, 0.5]},
    "amp_lr": {"type": "float", "range": [1e-5, 1e-2], "log": True},
    "phase_hidden_size": {"type": "int", "choices": [64, 128, 256, 512]},
    "phase_layers": {"type": "int", "range": [2, 6]},
    "phase_banks": {"type": "int", "range": [1, 8]},
    "phase_dropout": {"type": "float", "range": [0.0, 0.5]},
    "phase_lr": {"type": "float", "range": [1e-5, 1e-2], "log": True},
    "fourier_bands": {"type": "int", "range": [4, 32]},
    "fourier_max_freq": {"type": "float", "range": [1.0, 50.0]},
    "num_epochs": {"type": "int", "range": [10, 400], "note": "training budget"},
}

TERMINAL = {"completed", "failed", "cancelled", "interrupted"}


class FlareClient:
    """Drive FLARE experiments over the dashboard HTTP API."""

    def __init__(self, base_url: str = "http://localhost:8420", timeout: float = 30.0):
        self.base = base_url.rstrip("/")
        self.timeout = timeout

    # --- low-level -----------------------------------------------------
    def _get(self, path: str, **params) -> Any:
        r = requests.get(f"{self.base}{path}", params=params or None, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, body: Optional[dict] = None) -> Any:
        r = requests.post(f"{self.base}{path}", json=body, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    # --- read helpers --------------------------------------------------
    def health(self) -> dict:
        return self._get("/api/health")

    def get_settings(self) -> dict:
        return self._get("/api/settings")

    def list_designs(self) -> List[dict]:
        return self._get("/api/models")

    def list_custom_models(self) -> dict:
        return self._get("/api/custom-models")

    def search_space(self) -> Dict[str, Any]:
        """Design fields an agent may vary, with suggested ranges."""
        return SEARCH_SPACE

    # --- experiments ---------------------------------------------------
    def submit_experiment(self, spec: dict) -> dict:
        """Queue an experiment. ``spec`` is an ExperimentSpec (see schemas.py).

        Vary architecture by passing an inline ``model_design`` dict, or start
        from a saved design with ``model_name``. Patch data/system settings via
        ``settings_overrides`` (e.g. ``{"num_samples": 2000}``).
        """
        return self._post("/api/experiments", spec)

    def list_experiments(self, group: Optional[str] = None,
                         status: Optional[str] = None) -> List[dict]:
        params = {}
        if group:
            params["group"] = group
        if status:
            params["status"] = status
        return self._get("/api/experiments", **params)

    def get_experiment(self, exp_id: str) -> dict:
        return self._get(f"/api/experiments/{exp_id}")

    def cancel(self, exp_id: str) -> dict:
        return self._post(f"/api/experiments/{exp_id}/cancel")

    def wait(self, exp_id: str, poll: float = 5.0,
             timeout: Optional[float] = None) -> dict:
        """Block until an experiment reaches a terminal state; return its record."""
        start = time.time()
        while True:
            rec = self.get_experiment(exp_id)
            if rec.get("status") in TERMINAL:
                return rec
            if timeout is not None and time.time() - start > timeout:
                return rec
            time.sleep(poll)

    def leaderboard(self, group: Optional[str] = None,
                    metric: str = "mean_match", descending: bool = True) -> List[dict]:
        """Completed experiments ranked by ``metric`` (default eval mean_match)."""
        rows = [r for r in self.list_experiments(group=group)
                if r.get("status") == "completed" and r.get("metrics", {}).get(metric) is not None]
        rows.sort(key=lambda r: r["metrics"][metric], reverse=descending)
        return rows
