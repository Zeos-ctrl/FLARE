"""
In-process background job manager for long-running FLARE tasks.

Training, tuning, evaluation and parameter estimation all run for minutes and
must not block the API event loop. Each is submitted here as a daemon thread
that reports progress into a bounded event buffer, which the routers expose via
Server-Sent Events so the dashboard can stream live curves.

This is deliberately single-process (suitable for a research dashboard with one
user); it is not a distributed task queue.

Jobs are serialized through a bounded worker pool: ``FLARE_MAX_CONCURRENT``
(default 1) worker threads pull from a FIFO queue, so an agent enqueuing many
experiments cannot spawn unbounded GPU-fighting threads. Raise the env var when
more hardware is available.
"""
from __future__ import annotations

import logging
import os
import queue
import threading
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    id: str
    type: str
    status: JobStatus = JobStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    params: Dict[str, Any] = field(default_factory=dict)
    progress: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    events: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=2000))
    _cancel: threading.Event = field(default_factory=threading.Event)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "params": self.params,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
        }


class JobHandle:
    """Passed to job targets so they can emit progress and check for cancel."""

    def __init__(self, job: Job):
        self._job = job

    def emit(self, event: Dict[str, Any]):
        """Record a progress event (also stored as the job's latest progress)."""
        event = {"t": time.time(), **event}
        self._job.events.append(event)
        self._job.progress = {**self._job.progress, **event}

    @property
    def cancelled(self) -> bool:
        return self._job._cancel.is_set()

    def check_cancel(self):
        if self.cancelled:
            raise JobCancelled()


class JobCancelled(Exception):
    """Raised inside a job target when cancellation is requested."""


class JobManager:
    """Thread-based registry of background jobs with a serialized worker pool.

    Submitted jobs stay ``PENDING`` in a FIFO queue until a worker is free; with
    the default of one worker they run strictly one at a time.
    """

    def __init__(self, max_concurrent: Optional[int] = None):
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self._queue: "queue.Queue[Tuple[Job, Callable[..., Any], Dict[str, Any]]]" = queue.Queue()
        if max_concurrent is None:
            max_concurrent = int(os.environ.get("FLARE_MAX_CONCURRENT", "1"))
        self.max_concurrent = max(1, max_concurrent)
        for i in range(self.max_concurrent):
            threading.Thread(target=self._worker, daemon=True,
                             name=f"job-worker-{i}").start()

    def submit(self, job_type: str, target: Callable[..., Any],
               params: Optional[Dict[str, Any]] = None, **kwargs) -> Job:
        """Enqueue ``target(handle, **kwargs)`` to run on a worker thread.

        Returns immediately with a ``PENDING`` job; the target's return value
        (if a dict) is stored as ``job.result`` when it completes.
        """
        job = Job(id=uuid.uuid4().hex[:12], type=job_type, params=params or {})
        with self._lock:
            self._jobs[job.id] = job
        self._queue.put((job, target, kwargs))
        return job

    def _worker(self):
        while True:
            job, target, kwargs = self._queue.get()
            try:
                self._run(job, target, kwargs)
            finally:
                self._queue.task_done()

    def _run(self, job: Job, target: Callable[..., Any], kwargs: Dict[str, Any]):
        # A job cancelled while still queued never starts.
        if job._cancel.is_set():
            job.status = JobStatus.CANCELLED
            job.finished_at = time.time()
            return
        job.status = JobStatus.RUNNING
        job.started_at = time.time()
        handle = JobHandle(job)
        try:
            result = target(handle, **kwargs)
            if job._cancel.is_set():
                job.status = JobStatus.CANCELLED
            else:
                job.status = JobStatus.COMPLETED
                if isinstance(result, dict):
                    job.result = result
        except JobCancelled:
            job.status = JobStatus.CANCELLED
        except Exception as exc:  # noqa: BLE001
            job.status = JobStatus.FAILED
            job.error = f"{type(exc).__name__}: {exc}"
            job.events.append({"t": time.time(), "level": "error",
                               "message": job.error})
            logger.exception("Job %s (%s) failed", job.id, job.type)
            logger.debug(traceback.format_exc())
        finally:
            job.finished_at = time.time()

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def list(self, job_type: Optional[str] = None) -> List[Job]:
        jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        if job_type:
            jobs = [j for j in jobs if j.type == job_type]
        return jobs

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job and job.status in (JobStatus.PENDING, JobStatus.RUNNING):
            job._cancel.set()
            return True
        return False


# Shared singleton used by the routers.
job_manager = JobManager()
