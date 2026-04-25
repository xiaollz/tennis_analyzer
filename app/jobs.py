"""Simple in-process background job queue.

Jobs run on a thread pool and persist their state to disk (one JSON file per
job). Frontend polls `GET /api/jobs/{job_id}` to get progress.

Not a distributed queue — good enough for a local-first app.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Optional

from app import storage


# ── Job state ───────────────────────────────────────────────────────

@dataclass
class JobState:
    job_id: str
    kind: str                            # "segment" | "diagnose"
    status: str = "queued"              # queued | running | done | error
    progress: float = 0.0               # 0.0 - 1.0
    message: str = ""
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    error: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)   # e.g. {video_id, clip_id}
    result: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Manager ─────────────────────────────────────────────────────────

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="baseline-job")
_jobs: Dict[str, JobState] = {}
_lock = threading.Lock()
# Throttle disk writes: progress-only updates flush at most every PERSIST_MIN_S.
# Status / message / error / result changes ALWAYS flush immediately so the
# poller never observes stale terminal state.
_last_persist_ts: Dict[str, float] = {}
PERSIST_MIN_S = 0.4


def _persist(job: JobState) -> None:
    storage.write_json(storage.job_path(job.job_id), job.to_dict())
    _last_persist_ts[job.job_id] = time.time()


def _update(job_id: str, **fields) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            return
        # Detect "important" changes that must flush to disk immediately.
        force_persist = False
        for k, v in fields.items():
            if k in ("status", "error", "result", "started_at", "ended_at"):
                if getattr(job, k, None) != v:
                    force_persist = True
            setattr(job, k, v)
        if force_persist:
            _persist(job)
            return
        # Otherwise (progress/message tick), throttle.
        last = _last_persist_ts.get(job_id, 0.0)
        if (time.time() - last) >= PERSIST_MIN_S or job.progress >= 1.0:
            _persist(job)


def create_job(kind: str, payload: Dict[str, Any]) -> JobState:
    job = JobState(job_id=uuid.uuid4().hex[:12], kind=kind, payload=payload)
    with _lock:
        _jobs[job.job_id] = job
    _persist(job)
    return job


def get_job(job_id: str) -> Optional[JobState]:
    with _lock:
        job = _jobs.get(job_id)
    if job:
        return job
    # Fallback: read from disk (survives restarts of the polling client,
    # though running jobs don't survive a server restart)
    data = storage.read_json(storage.job_path(job_id))
    if not data:
        return None
    try:
        return JobState(**data)
    except TypeError:
        return None


def submit(
    kind: str,
    fn: Callable[..., Dict[str, Any]],
    payload: Dict[str, Any],
) -> JobState:
    """Submit a job. `fn(progress_cb, **payload) -> dict` runs in background.

    `progress_cb(pct: float, msg: str)` can be called by `fn` to update status.
    """
    job = create_job(kind, payload)

    def _progress_cb(pct: float, msg: str) -> None:
        _update(
            job.job_id,
            status="running",
            progress=float(max(0.0, min(1.0, pct))),
            message=str(msg)[:200],
        )

    def _wrap():
        _update(job.job_id, status="running", started_at=time.time(), message="开始")
        try:
            result = fn(progress_cb=_progress_cb, **payload)
            _update(
                job.job_id,
                status="done",
                progress=1.0,
                ended_at=time.time(),
                result=result if isinstance(result, dict) else {"value": result},
                message="完成",
            )
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            _update(
                job.job_id,
                status="error",
                ended_at=time.time(),
                error=err[:4000],
                message="失败",
            )

    _executor.submit(_wrap)
    return job
