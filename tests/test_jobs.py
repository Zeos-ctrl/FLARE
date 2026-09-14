"""Tests for the background job manager."""
import time

from src.api.jobs import JobManager, JobStatus


def _wait(job, timeout=5.0):
    start = time.time()
    while job.status in (JobStatus.PENDING, JobStatus.RUNNING):
        if time.time() - start > timeout:
            raise TimeoutError("job did not finish")
        time.sleep(0.02)


def test_job_completes_and_stores_result():
    mgr = JobManager()

    def target(handle):
        handle.emit({"message": "hello"})
        return {"answer": 42}

    job = mgr.submit("test", target)
    _wait(job)
    assert job.status == JobStatus.COMPLETED
    assert job.result == {"answer": 42}
    assert any(e.get("message") == "hello" for e in job.events)


def test_job_failure_is_captured():
    mgr = JobManager()

    def target(handle):
        raise ValueError("boom")

    job = mgr.submit("test", target)
    _wait(job)
    assert job.status == JobStatus.FAILED
    assert "boom" in job.error


def test_job_cancellation():
    mgr = JobManager()

    def target(handle):
        for _ in range(200):
            handle.check_cancel()
            time.sleep(0.01)

    job = mgr.submit("test", target)
    time.sleep(0.05)
    assert mgr.cancel(job.id)
    _wait(job)
    assert job.status == JobStatus.CANCELLED


def test_list_filters_by_type():
    mgr = JobManager()
    mgr.submit("a", lambda h: {"x": 1})
    mgr.submit("b", lambda h: {"x": 2})
    time.sleep(0.1)
    assert len(mgr.list(job_type="a")) == 1
    assert len(mgr.list()) == 2
