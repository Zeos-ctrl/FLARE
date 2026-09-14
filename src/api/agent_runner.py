"""In-process agent auto-runner: an autonomous train->eval search loop.

The manual experiment loop (``agent/AGENT.md``) is "read leaderboard -> propose
next design -> submit -> wait -> repeat". This module runs that loop *inside the
dashboard process* so the UI can start it with one click and watch it live.

Concurrency note: experiments run through the shared ``job_manager``, which has a
single worker by default (``FLARE_MAX_CONCURRENT``). The auto-runner therefore
runs on its **own** daemon thread — NOT as a queued job — so it never occupies
the worker it depends on. It submits experiments (which queue normally) and
polls their durable records until each finishes.

The decision of *what to try next* is delegated to a pluggable proposer
(``src/agent/proposer.py``): a local OpenAI-compatible model by default (falls
back to random search if no local server answers), or Claude / random.
"""
from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from typing import Any, Deque, Dict, List, Optional

from src.agent.client import SEARCH_SPACE
from src.agent.proposer import make_proposer
from src.api import state
from src.api.jobs import job_manager
from src.api.routers.experiments import submit_experiment_spec
from src.api.schemas import ExperimentSpec, ModelDesignModel
from src.core import experiments

TERMINAL = {"completed", "failed", "cancelled", "interrupted"}
_POLL_SECONDS = 3.0

_lock = threading.Lock()
_runners: "Dict[str, AutoRunner]" = {}


def _leaderboard(group: Optional[str], metric: str = "mean_match") -> List[dict]:
    """Completed experiments in ``group`` ranked by ``metric`` (best first)."""
    rows = [r for r in experiments.list_all()
            if (group is None or r.get("group") == group)
            and r.get("status") == "completed"
            and r.get("metrics", {}).get(metric) is not None]
    rows.sort(key=lambda r: r["metrics"][metric], reverse=True)
    return rows


class AutoRunner:
    """Drives one autonomous search campaign on a background thread."""

    def __init__(self, group: str, budget: int, proposer: str, num_epochs: int,
                 eval_n_samples: int, evaluate: bool,
                 settings_overrides: Dict[str, Any], notes: Optional[str] = None,
                 objective: Optional[str] = None,
                 local_base_url: Optional[str] = None, local_model: Optional[str] = None,
                 local_api_key: Optional[str] = None):
        self.id = uuid.uuid4().hex[:12]
        self.group = group
        self.budget = budget
        self.requested_proposer = proposer
        self.objective = objective
        self.local_base_url = local_base_url
        self.local_model = local_model
        self.local_api_key = local_api_key
        self.num_epochs = num_epochs
        self.eval_n_samples = eval_n_samples
        self.evaluate = evaluate
        self.settings_overrides = settings_overrides
        self.notes = notes

        self.status = "pending"          # pending -> running -> completed|failed|stopped
        self.iteration = 0
        self.current_experiment: Optional[str] = None
        self.history: List[dict] = []    # per-run {experiment_id, name, design, mean_match, status}
        self.error: Optional[str] = None
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self._log: Deque[str] = deque(maxlen=200)

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"auto-{self.id}", daemon=True)
        # Resolve the proposer up front so the UI shows immediately whether the
        # Claude path was available or fell back to random.
        self._proposer, self.proposer = make_proposer(
            proposer, base_url=local_base_url, model=local_model, api_key=local_api_key)
        if self.proposer != proposer:
            self._emit(f"proposer '{proposer}' unavailable — falling back to '{self.proposer}'")
        self._base_design = ModelDesignModel().model_dump()

    # --- lifecycle -----------------------------------------------------
    def start(self):
        self._thread.start()

    def stop(self):
        """Request a graceful stop; cancels the in-flight experiment if any."""
        self._stop.set()
        self._emit("stop requested")
        if self.current_experiment:
            rec = experiments.load(self.current_experiment)
            if rec and rec.get("job_id"):
                job_manager.cancel(rec["job_id"])

    # --- the loop ------------------------------------------------------
    def _run(self):
        self.status = "running"
        self.started_at = time.time()
        self._emit(f"auto-run started: group={self.group} budget={self.budget} proposer={self.proposer}")
        try:
            for i in range(self.budget):
                if self._stop.is_set():
                    break
                self.iteration = i + 1
                board = _leaderboard(self.group)
                context = {"base_design": self._base_design,
                           "objective": self.objective or "maximize evaluation mean_match (0..1)"}
                design = self._proposer(board, SEARCH_SPACE, context)
                self._emit(f"[{self.iteration}/{self.budget}] proposed {design}")

                # num_epochs is a campaign-level budget owned by the config, so
                # it overrides whatever the proposer suggested; record the design
                # that actually ran.
                effective = {**design, "num_epochs": self.num_epochs}
                spec = self._build_spec(effective, i)
                rec = submit_experiment_spec(spec)
                self.current_experiment = rec["id"]
                self._emit(f"[{self.iteration}/{self.budget}] submitted {rec['name']} ({rec['id']})")

                rec = self._wait(rec["id"])
                match = (rec.get("metrics") or {}).get("mean_match")
                self.history.append({
                    "experiment_id": rec["id"], "name": rec["name"],
                    "design": effective, "status": rec.get("status"), "mean_match": match,
                })
                self._emit(f"[{self.iteration}/{self.budget}] {rec.get('status')} "
                           f"mean_match={match if match is None else round(match, 4)}")
                self.current_experiment = None
                if self._stop.is_set():
                    break

            self.status = "stopped" if self._stop.is_set() else "completed"
            self._emit(f"auto-run {self.status}")
        except Exception as exc:  # noqa: BLE001
            self.status = "failed"
            self.error = f"{type(exc).__name__}: {exc}"
            self._emit(f"auto-run failed: {self.error}")
        finally:
            self.finished_at = time.time()

    def _build_spec(self, design: Dict[str, Any], i: int) -> ExperimentSpec:
        merged = {**self._base_design, **design, "num_epochs": self.num_epochs}
        return ExperimentSpec(
            name=f"{self.group}-{i}",
            group=self.group,
            source="agent",
            model_design=ModelDesignModel(**merged),
            settings_overrides=self.settings_overrides,
            evaluate=self.evaluate,
            eval_n_samples=self.eval_n_samples,
            notes=self.notes,
        )

    def _wait(self, exp_id: str) -> dict:
        """Poll the durable record until the experiment reaches a terminal state."""
        cancelled = False
        while True:
            rec = experiments.load(exp_id) or {"id": exp_id, "status": "interrupted"}
            if rec.get("status") in TERMINAL:
                return rec
            if self._stop.is_set() and not cancelled and rec.get("job_id"):
                job_manager.cancel(rec["job_id"])
                cancelled = True
            time.sleep(_POLL_SECONDS)

    # --- reporting -----------------------------------------------------
    def _emit(self, msg: str):
        self._log.append(f"{time.strftime('%H:%M:%S')}  {msg}")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "group": self.group,
            "budget": self.budget,
            "iteration": self.iteration,
            "proposer": self.proposer,
            "requested_proposer": self.requested_proposer,
            "objective": self.objective,
            "local_base_url": self.local_base_url,
            "local_model": self.local_model,
            "num_epochs": self.num_epochs,
            "current_experiment": self.current_experiment,
            "history": self.history,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "log": list(self._log),
        }


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------
def active_runner() -> Optional[AutoRunner]:
    with _lock:
        for r in _runners.values():
            if r.status in ("pending", "running"):
                return r
    return None


def start_runner(**kwargs) -> AutoRunner:
    """Start one auto-run. Only one may be active at a time (single GPU worker)."""
    with _lock:
        existing = next((r for r in _runners.values() if r.status in ("pending", "running")), None)
        if existing is not None:
            raise RuntimeError(f"an auto-run is already active ({existing.id})")
        runner = AutoRunner(**kwargs)
        _runners[runner.id] = runner
    runner.start()
    return runner


def get_runner(runner_id: str) -> Optional[AutoRunner]:
    with _lock:
        return _runners.get(runner_id)


def list_runners() -> List[AutoRunner]:
    with _lock:
        return sorted(_runners.values(), key=lambda r: r.created_at, reverse=True)
