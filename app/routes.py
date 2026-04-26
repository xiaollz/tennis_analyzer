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

@router.post("/clips/upload")
async def upload_clip(
    file: UploadFile = File(...),
    stroke: str = Form("forehand"),
) -> Dict[str, Any]:
    """Accept a pre-cut clip and register it directly — no segmentation.

    User cuts the clip in another app (e.g. Photos / a dedicated trimmer),
    then uploads only the small piece they want analyzed. We treat the
    uploaded file as both the 'video' (for storage layout compatibility)
    and the only clip in that video.

    Returns:
      {clip_id, video_id, stroke}

    The frontend then calls POST /api/clips/{clip_id}/diagnose to run pose
    + VLM. We don't kick that off automatically — the frontend's Loading
    screen owns the diagnose lifecycle (resume across app restart, etc.)
    so we keep the same contract as the legacy flow.
    """
    import cv2

    ext = Path(file.filename or "clip.mp4").suffix.lower() or ".mp4"
    if ext not in (".mp4", ".mov", ".m4v", ".mkv"):
        raise HTTPException(status_code=400, detail=f"unsupported extension: {ext}")
    if stroke not in services.VALID_STROKES:
        stroke = "forehand"

    video_id = uuid.uuid4().hex[:12]
    clip_id = f"{video_id}_c000"
    storage.ensure_storage()
    vdir = storage.video_dir(video_id)
    vdir.mkdir(parents=True, exist_ok=True)
    clips_dir = storage.video_clips_dir(video_id)
    clips_dir.mkdir(parents=True, exist_ok=True)

    # Save upload as both the original AND the clip — they're the same
    # file. clip_path points at original.mp4 so we don't double the disk.
    original = storage.video_original_path(video_id)
    with open(original, "wb") as out:
        shutil.copyfileobj(file.file, out)

    # Probe duration / fps via cv2
    cap = cv2.VideoCapture(str(original))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_s = total_frames / fps if total_frames > 0 else 0.0

    # Thumbnail at midpoint — pose pipeline will detect real impact later
    thumb_path = clips_dir / f"{clip_id}.jpg"
    if total_frames > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames // 2))
        ok, frame = cap.read()
        if ok:
            h, w = frame.shape[:2]
            scale = 720.0 / max(h, w) if max(h, w) > 720 else 1.0
            if scale < 1.0:
                frame = cv2.resize(
                    frame, (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            cv2.imwrite(str(thumb_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    cap.release()

    midpoint = duration_s / 2 if duration_s > 0 else 0.0
    storage.write_json(storage.video_meta_path(video_id), {
        "video_id": video_id,
        "filename": file.filename,
        "original_path": str(original),
        "uploaded_at": time.time(),
        "stroke": stroke,
        "duration_s": duration_s,
        "fps": fps,
        "clip_count": 1,
        "segmented_at": time.time(),
        "direct_upload": True,                     # marks the new flow
    })
    storage.write_json(storage.video_manifest_path(video_id), {
        "video_id": video_id,
        "video_path": str(original),
        "fps": fps,
        "duration_s": duration_s,
        "total_onsets": 1,
        "clips": [{
            "clip_id": clip_id,
            "video_id": video_id,
            "index": 0,
            "start_s": 0.0,
            "end_s": duration_s,
            "impact_times_s": [midpoint],          # placeholder — pose finds real impact
            "onset_strength": 1.0,
            "clip_path": str(original),
            "thumbnail_path": str(thumb_path),
            "duration_s": duration_s,
        }],
    })

    return {"clip_id": clip_id, "video_id": video_id, "stroke": stroke}


@router.post("/videos")
async def upload_video(
    file: UploadFile = File(...),
    stroke: str = Form("forehand"),
) -> Dict[str, Any]:
    """[Legacy] Accept a video upload and run audio segmentation.

    Kept for backward compat with previously uploaded videos. New uploads
    should use POST /clips/upload and pre-cut on-device.
    """
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
def delete_video(video_id: str, keep_clips: bool = False) -> Dict[str, Any]:
    """Physically remove video files from disk.

    Query params:
      keep_clips: if true, only delete the original.mp4 (saves space but
                  keeps the segmented clips + diagnoses). Default false.
    """
    result = storage.delete_video_artifacts(video_id, keep_clips=keep_clips)
    if not result.get("deleted"):
        raise HTTPException(status_code=404, detail=result.get("reason", "delete failed"))
    return {
        "deleted": video_id,
        "kept_clips": result.get("kept_clips", False),
        "bytes_freed": result.get("bytes_freed", 0),
        "items": result.get("items", []),
    }


# ── Storage management ─────────────────────────────────────────────

@router.get("/storage")
def get_storage_usage() -> Dict[str, Any]:
    """Return disk usage breakdown so the UI can render a storage panel."""
    return storage.storage_usage()


@router.delete("/storage")
def cleanup_storage(older_than_days: float = 7.0) -> Dict[str, Any]:
    """Cleanup stale jobs files older than N days."""
    n = storage.cleanup_jobs(older_than_days=older_than_days)
    return {"jobs_cleaned": n}


@router.post("/storage/wipe")
def wipe_storage() -> Dict[str, Any]:
    """Nuclear: wipe ALL videos / clips / diagnoses / jobs. Returns bytes freed."""
    return storage.cleanup_all()


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
