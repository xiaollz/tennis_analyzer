"""HTTP endpoints for Baseline.

All routes are mounted under /api/. The frontend is a separate SPA (built by
Claude desktop) that talks to this backend via fetch.
"""

from __future__ import annotations

import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from app import jobs, services, storage


router = APIRouter(prefix="/api")


# ── Health ──────────────────────────────────────────────────────────

@router.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "service": "baseline", "ts": time.time()}


# ── Video upload + segmentation ─────────────────────────────────────

@router.post("/videos")
async def upload_video(
    file: UploadFile = File(...),
    stroke: str = Form("forehand"),
) -> Dict[str, Any]:
    """Accept a video upload, persist it, and kick off segmentation.

    Parameters (multipart form):
      file:   video file (mp4/mov/m4v/mkv)
      stroke: "forehand" | "backhand" | "auto"  (default forehand —
              the auto classifier is unreliable on short clips)

    Returns:
      {video_id, job_id, stroke}
    """
    ext = Path(file.filename or "video.mp4").suffix.lower() or ".mp4"
    if ext not in (".mp4", ".mov", ".m4v", ".mkv"):
        raise HTTPException(status_code=400, detail=f"unsupported extension: {ext}")

    if stroke not in services.VALID_STROKES:
        stroke = "forehand"

    video_id = uuid.uuid4().hex[:12]
    storage.ensure_storage()
    vdir = storage.video_dir(video_id)
    vdir.mkdir(parents=True, exist_ok=True)

    original = storage.video_original_path(video_id)
    with open(original, "wb") as out:
        shutil.copyfileobj(file.file, out)

    meta = {
        "video_id": video_id,
        "filename": file.filename,
        "original_path": str(original),
        "uploaded_at": time.time(),
        "stroke": stroke,
    }
    storage.write_json(storage.video_meta_path(video_id), meta)

    job = jobs.submit(
        kind="segment",
        fn=services.run_segmentation,
        payload={"video_id": video_id, "video_path": str(original)},
    )

    return {"video_id": video_id, "job_id": job.job_id, "stroke": stroke}


@router.get("/videos")
def list_videos() -> Dict[str, Any]:
    return {"videos": storage.list_videos()}


@router.get("/videos/{video_id}")
def get_video(video_id: str) -> Dict[str, Any]:
    meta = storage.read_json(storage.video_meta_path(video_id))
    if not meta:
        raise HTTPException(status_code=404, detail="video not found")
    manifest = storage.read_json(storage.video_manifest_path(video_id)) or {}
    return {
        "meta": meta,
        "clips": manifest.get("clips", []),
        "total_onsets": manifest.get("total_onsets", 0),
        "segmented": bool(manifest),
        "error": manifest.get("error"),
    }


@router.delete("/videos/{video_id}")
def delete_video(video_id: str) -> Dict[str, Any]:
    vdir = storage.video_dir(video_id)
    if not vdir.exists():
        raise HTTPException(status_code=404, detail="video not found")
    # Also remove diagnoses tied to its clips
    manifest = storage.read_json(storage.video_manifest_path(video_id)) or {}
    for c in manifest.get("clips", []):
        ddir = storage.diagnosis_dir(c["clip_id"])
        if ddir.exists():
            shutil.rmtree(ddir, ignore_errors=True)
    shutil.rmtree(vdir, ignore_errors=True)
    return {"deleted": video_id}


# ── Jobs ────────────────────────────────────────────────────────────

@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> Dict[str, Any]:
    job = jobs.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job.to_dict()


# ── Clips ───────────────────────────────────────────────────────────

@router.get("/clips/{clip_id}")
def get_clip(clip_id: str) -> Dict[str, Any]:
    info = storage.find_clip(clip_id)
    if not info:
        raise HTTPException(status_code=404, detail="clip not found")
    # Attach diagnosis availability
    diag = storage.read_json(storage.diagnosis_result_path(clip_id))
    return {
        "clip_id": clip_id,
        "video_id": info["video_id"],
        "clip": info["clip"],
        "has_diagnosis": diag is not None,
        "urls": {
            "video": f"/api/clips/{clip_id}/video",
            "thumbnail": f"/api/clips/{clip_id}/thumbnail",
            "diagnose": f"/api/clips/{clip_id}/diagnose",
            "diagnosis": f"/api/diagnoses/{clip_id}",
        },
    }


@router.get("/clips/{clip_id}/video")
def get_clip_video(clip_id: str):
    info = storage.find_clip(clip_id)
    if not info:
        raise HTTPException(status_code=404, detail="clip not found")
    path = Path(info["clip"]["clip_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="clip video missing")
    return FileResponse(path, media_type="video/mp4", filename=path.name)


@router.get("/clips/{clip_id}/thumbnail")
def get_clip_thumbnail(clip_id: str):
    info = storage.find_clip(clip_id)
    if not info:
        raise HTTPException(status_code=404, detail="clip not found")
    path = Path(info["clip"]["thumbnail_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="thumbnail missing")
    return FileResponse(path, media_type="image/jpeg")


@router.post("/clips/{clip_id}/diagnose")
def start_diagnose(clip_id: str, stroke: str | None = None) -> Dict[str, Any]:
    """Start diagnosis on a clip.

    Query param:
      stroke: "forehand" | "backhand" | "auto"
        Falls back to the parent video's saved preference (set at upload),
        which itself defaults to "forehand".
    """
    info = storage.find_clip(clip_id)
    if not info:
        raise HTTPException(status_code=404, detail="clip not found")

    # Resolve stroke: query → video meta → default
    if stroke not in services.VALID_STROKES:
        video_meta = info.get("video_meta") or {}
        stroke = video_meta.get("stroke", "forehand")
        if stroke not in services.VALID_STROKES:
            stroke = "forehand"

    # If already done, return idempotently
    if storage.diagnosis_result_path(clip_id).exists():
        return {"clip_id": clip_id, "job_id": None, "status": "already_done", "stroke": stroke}

    job = jobs.submit(
        kind="diagnose",
        fn=services.run_diagnosis,
        payload={"clip_id": clip_id, "stroke": stroke},
    )
    return {"clip_id": clip_id, "job_id": job.job_id, "status": "queued", "stroke": stroke}


@router.get("/clips/{clip_id}/annotated")
def get_clip_annotated(clip_id: str):
    path = storage.diagnosis_annotated_path(clip_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="annotated video not yet generated")
    return FileResponse(path, media_type="video/mp4", filename=path.name)


@router.get("/clips/{clip_id}/keyframes")
def get_clip_keyframes(clip_id: str):
    path = storage.diagnosis_keyframe_grid_path(clip_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="keyframe grid not yet generated")
    return FileResponse(path, media_type="image/png")


@router.get("/clips/{clip_id}/report.md")
def get_clip_report(clip_id: str):
    path = storage.diagnosis_report_path(clip_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="report not yet generated")
    return FileResponse(path, media_type="text/markdown")


# ── Diagnosis ───────────────────────────────────────────────────────

@router.get("/diagnoses/{clip_id}")
def get_diagnosis(clip_id: str) -> Dict[str, Any]:
    result = storage.read_json(storage.diagnosis_result_path(clip_id))
    status = storage.read_json(storage.diagnosis_status_path(clip_id))
    if not result and not status:
        raise HTTPException(status_code=404, detail="diagnosis not found (never started)")
    if not result:
        return {"clip_id": clip_id, "status": status.get("status", "unknown"), "ready": False}
    return {
        "clip_id": clip_id,
        "status": "done",
        "ready": True,
        "result": result,
    }
