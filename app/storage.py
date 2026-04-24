"""Filesystem storage layout for the Baseline app.

Layout (all under storage/):

    videos/{video_id}/
        meta.json              uploaded video metadata
        original.mp4           original uploaded video
        manifest.json          segmentation result
        clips/
            {clip_id}.mp4      extracted clip
            {clip_id}.jpg      thumbnail

    diagnoses/{clip_id}/
        status.json            diagnosis status snapshot
        result.json            structured diagnosis result (frontend-ready)
        keyframe_grid.png      6-frame grid used by VLM
        annotated.mp4          clip with skeleton overlay
        report.md              long-form markdown report

    jobs/
        {job_id}.json          job state

The layout is append-only and gitignored.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


# ── Root ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STORAGE_ROOT = PROJECT_ROOT / "storage"


def ensure_storage() -> None:
    """Create top-level storage directories if missing."""
    for sub in ("videos", "diagnoses", "jobs"):
        (STORAGE_ROOT / sub).mkdir(parents=True, exist_ok=True)


# ── Path helpers ────────────────────────────────────────────────────

def video_dir(video_id: str) -> Path:
    return STORAGE_ROOT / "videos" / video_id


def video_meta_path(video_id: str) -> Path:
    return video_dir(video_id) / "meta.json"


def video_original_path(video_id: str) -> Path:
    return video_dir(video_id) / "original.mp4"


def video_manifest_path(video_id: str) -> Path:
    return video_dir(video_id) / "manifest.json"


def video_clips_dir(video_id: str) -> Path:
    return video_dir(video_id) / "clips"


def clip_file_for(video_id: str, clip_id: str) -> Path:
    return video_clips_dir(video_id) / f"{clip_id}.mp4"


def clip_thumb_for(video_id: str, clip_id: str) -> Path:
    return video_clips_dir(video_id) / f"{clip_id}.jpg"


def diagnosis_dir(clip_id: str) -> Path:
    return STORAGE_ROOT / "diagnoses" / clip_id


def diagnosis_status_path(clip_id: str) -> Path:
    return diagnosis_dir(clip_id) / "status.json"


def diagnosis_result_path(clip_id: str) -> Path:
    return diagnosis_dir(clip_id) / "result.json"


def diagnosis_keyframe_grid_path(clip_id: str) -> Path:
    return diagnosis_dir(clip_id) / "keyframe_grid.png"


def diagnosis_annotated_path(clip_id: str) -> Path:
    return diagnosis_dir(clip_id) / "annotated.mp4"


def diagnosis_report_path(clip_id: str) -> Path:
    return diagnosis_dir(clip_id) / "report.md"


def job_path(job_id: str) -> Path:
    return STORAGE_ROOT / "jobs" / f"{job_id}.json"


# ── JSON I/O ────────────────────────────────────────────────────────

def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)


# ── Manifest helpers ────────────────────────────────────────────────

def list_videos() -> list[Dict[str, Any]]:
    """Return a list of all uploaded videos (meta + clip_count)."""
    root = STORAGE_ROOT / "videos"
    if not root.exists():
        return []
    out = []
    for vd in sorted(root.iterdir(), reverse=True):
        if not vd.is_dir():
            continue
        meta = read_json(vd / "meta.json")
        if not meta:
            continue
        manifest = read_json(vd / "manifest.json")
        meta["clip_count"] = len(manifest.get("clips", [])) if manifest else 0
        meta["has_segments"] = manifest is not None
        out.append(meta)
    return out


def find_clip(clip_id: str) -> Optional[Dict[str, Any]]:
    """Given a clip_id like `abc123_c002`, find its manifest entry.

    Returns a dict with {video_id, clip} or None.
    """
    # clip_id prefix == video_id
    if "_c" in clip_id:
        video_id = clip_id.rsplit("_c", 1)[0]
    else:
        return None
    manifest = read_json(video_manifest_path(video_id))
    if not manifest:
        return None
    for c in manifest.get("clips", []):
        if c.get("clip_id") == clip_id:
            return {"video_id": video_id, "clip": c, "video_meta": read_json(video_meta_path(video_id))}
    return None
