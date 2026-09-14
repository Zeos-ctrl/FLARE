"""Durable registry of automated experiments.

Each experiment is one JSON file under ``FLARE_EXPERIMENTS`` (default
``experiments/``). Unlike the in-memory job manager, these records survive a
server restart, so the dashboard and an agent can see the full history of runs,
their status, and their metrics.

Records are plain dicts here (no Pydantic / API imports) so this module stays in
the core layer; the API router converts to/from ``ExperimentRecord``.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from typing import Callable, Dict, List, Optional

EXPERIMENTS_DIR = os.environ.get("FLARE_EXPERIMENTS", "experiments")

_lock = threading.Lock()

# Statuses considered "still owned by a live job".
ACTIVE = ("pending", "running")


def _ensure_dir():
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def _path(exp_id: str) -> str:
    return os.path.join(EXPERIMENTS_DIR, f"{exp_id}.json")


def _write_atomic(exp_id: str, record: Dict):
    """Write via a temp file + rename so concurrent readers never see a partial
    (truncated) file — reads always return either the old or new full record."""
    _ensure_dir()
    tmp = _path(exp_id) + f".{os.getpid()}.{threading.get_ident()}.tmp"
    with open(tmp, "w") as f:
        json.dump(record, f, indent=2)
    os.replace(tmp, _path(exp_id))


def create(record: Dict) -> Dict:
    """Persist a new experiment record and return it."""
    record.setdefault("created_at", time.time())
    with _lock:
        _write_atomic(record["id"], record)
    return record


def load(exp_id: str) -> Optional[Dict]:
    path = _path(exp_id)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def update(exp_id: str, **fields) -> Optional[Dict]:
    """Read-modify-write an experiment record.

    ``metrics`` and ``progress`` are shallow-merged into the existing dicts;
    every other field is replaced. Safe under concurrent workers via ``_lock``.
    """
    with _lock:
        record = load(exp_id)
        if record is None:
            return None
        for key, value in fields.items():
            if key in ("metrics", "progress") and isinstance(value, dict):
                merged = {**record.get(key, {}), **value}
                record[key] = merged
            else:
                record[key] = value
        _write_atomic(exp_id, record)
    return record


def list_all() -> List[Dict]:
    if not os.path.isdir(EXPERIMENTS_DIR):
        return []
    records = []
    for fname in os.listdir(EXPERIMENTS_DIR):
        if fname.endswith(".json"):
            try:
                with open(os.path.join(EXPERIMENTS_DIR, fname)) as f:
                    records.append(json.load(f))
            except (json.JSONDecodeError, OSError):
                continue
    records.sort(key=lambda r: r.get("created_at", 0), reverse=True)
    return records


def delete(exp_id: str) -> bool:
    path = _path(exp_id)
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


def reconcile(record: Dict, job_exists: Callable[[Optional[str]], bool]) -> Dict:
    """Return a view of ``record`` with dead active runs marked ``interrupted``.

    A record left ``pending``/``running`` whose job is no longer known to the job
    manager (e.g. the server restarted mid-run) is reported as ``interrupted``.
    Does not mutate the stored file — status is corrected lazily on read.
    """
    if record.get("status") in ACTIVE and not job_exists(record.get("job_id")):
        return {**record, "status": "interrupted"}
    return record
