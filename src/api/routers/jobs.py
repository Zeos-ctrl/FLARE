"""Job status, listing, live event streaming (SSE) and cancellation."""
from __future__ import annotations

import asyncio
import json
from typing import Optional

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from src.api.jobs import JobStatus, job_manager

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.get("")
def list_jobs(type: Optional[str] = None):
    return [j.to_dict() for j in job_manager.list(job_type=type)]


@router.get("/{job_id}")
def get_job(job_id: str):
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()


@router.post("/{job_id}/cancel")
def cancel_job(job_id: str):
    if not job_manager.cancel(job_id):
        raise HTTPException(status_code=409, detail="Job not cancellable")
    return {"cancelled": True}


@router.get("/{job_id}/stream")
async def stream_job(job_id: str):
    """Stream a job's progress events as Server-Sent Events.

    Replays buffered events then tails new ones until the job finishes.
    """
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator():
        sent = 0
        while True:
            events = list(job.events)
            for event in events[sent:]:
                yield {"event": "progress", "data": json.dumps(event)}
            sent = len(events)

            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                yield {"event": "done", "data": json.dumps(job.to_dict())}
                break
            await asyncio.sleep(0.5)

    return EventSourceResponse(event_generator())
