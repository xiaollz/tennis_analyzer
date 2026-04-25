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


# ── Disk usage ──────────────────────────────────────────────────────

def _dir_size(path: Path) -> int:
    """Total bytes in a directory tree (best-effort)."""
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except (OSError, FileNotFoundError):
            pass
    return total


def storage_usage() -> Dict[str, Any]:
    """Compute current storage usage by category.

    Returns:
        {
          "videos": {bytes, count},
          "clips": {bytes, count},
          "diagnoses": {bytes, count},
          "jobs": {bytes, count},
          "total_bytes": int,
          "total_human": "12.3 MB",
          "by_video": [{video_id, filename, bytes, has_original, has_clips, has_diagnoses}, ...]
        }
    """
    root = STORAGE_ROOT
    if not root.exists():
        return {"videos": {"bytes": 0, "count": 0}, "clips": {"bytes": 0, "count": 0},
                "diagnoses": {"bytes": 0, "count": 0}, "jobs": {"bytes": 0, "count": 0},
                "total_bytes": 0, "total_human": "0 B", "by_video": []}

    def fmt(b: int) -> str:
        for unit in ("B", "KB", "MB", "GB"):
            if b < 1024 or unit == "GB":
                return f"{b:.1f} {unit}" if unit != "B" else f"{b} B"
            b /= 1024.0
        return f"{b:.1f} GB"

    videos_dir = root / "videos"
    diagnoses_dir = root / "diagnoses"
    jobs_dir = root / "jobs"

    total_video_bytes = 0
    total_clip_bytes = 0
    total_diag_bytes = 0
    by_video: list[dict] = []
    video_count = 0
    clip_count = 0
    diag_count = 0

    if videos_dir.exists():
        for vd in videos_dir.iterdir():
            if not vd.is_dir():
                continue
            video_count += 1
            meta = read_json(vd / "meta.json") or {}
            original_path = vd / "original.mp4"
            clips_dir_p = vd / "clips"

            v_bytes = original_path.stat().st_size if original_path.exists() else 0
            c_bytes = _dir_size(clips_dir_p) if clips_dir_p.exists() else 0
            cnt_clips = sum(1 for p in clips_dir_p.glob("*.mp4")) if clips_dir_p.exists() else 0
            clip_count += cnt_clips
            manifest_bytes = (vd / "manifest.json").stat().st_size if (vd / "manifest.json").exists() else 0
            total_video_bytes += v_bytes
            total_clip_bytes += c_bytes + manifest_bytes

            # Diagnoses tied to this video
            d_bytes = 0
            d_count = 0
            manifest = read_json(vd / "manifest.json") or {}
            for c in manifest.get("clips", []):
                ddir = diagnoses_dir / c["clip_id"]
                if ddir.exists():
                    d_bytes += _dir_size(ddir)
                    d_count += 1
            total_diag_bytes += d_bytes
            diag_count += d_count

            by_video.append({
                "video_id": meta.get("video_id", vd.name),
                "filename": meta.get("filename", vd.name),
                "uploaded_at": meta.get("uploaded_at"),
                "bytes": v_bytes + c_bytes + manifest_bytes + d_bytes,
                "human": fmt(v_bytes + c_bytes + manifest_bytes + d_bytes),
                "has_original": original_path.exists(),
                "original_bytes": v_bytes,
                "clips_bytes": c_bytes,
                "diagnoses_bytes": d_bytes,
                "clip_count": cnt_clips,
                "diagnosis_count": d_count,
            })

    jobs_bytes = _dir_size(jobs_dir) if jobs_dir.exists() else 0
    jobs_count = sum(1 for _ in jobs_dir.glob("*.json")) if jobs_dir.exists() else 0

    total = total_video_bytes + total_clip_bytes + total_diag_bytes + jobs_bytes

    return {
        "videos":    {"bytes": total_video_bytes, "count": video_count, "human": fmt(total_video_bytes)},
        "clips":     {"bytes": total_clip_bytes, "count": clip_count, "human": fmt(total_clip_bytes)},
        "diagnoses": {"bytes": total_diag_bytes, "count": diag_count, "human": fmt(total_diag_bytes)},
        "jobs":      {"bytes": jobs_bytes, "count": jobs_count, "human": fmt(jobs_bytes)},
        "total_bytes": total,
        "total_human": fmt(total),
        "by_video": sorted(by_video, key=lambda v: -v["bytes"]),
    }


def cleanup_jobs(older_than_days: float = 7.0) -> int:
    """Remove jobs/*.json older than N days. Returns count deleted."""
    import time
    jobs_dir = STORAGE_ROOT / "jobs"
    if not jobs_dir.exists():
        return 0
    cutoff = time.time() - older_than_days * 86400
    n = 0
    for p in jobs_dir.glob("*.json"):
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink()
                n += 1
        except OSError:
            pass
    return n


def delete_video_artifacts(video_id: str, keep_clips: bool = False) -> Dict[str, Any]:
    """Physically delete a video and its derived artifacts.

    Args:
        keep_clips: if True, only the original.mp4 is removed; clips and
                    diagnoses remain. Useful for "free space but keep results".

    Returns a summary dict of what was deleted.
    """
    import shutil
    vdir = video_dir(video_id)
    if not vdir.exists():
        return {"deleted": False, "reason": "video not found"}

    bytes_freed = 0
    items: list[str] = []

    if keep_clips:
        # Only remove original.mp4
        original = video_original_path(video_id)
        if original.exists():
            bytes_freed += original.stat().st_size
            original.unlink()
            items.append("original.mp4")
            # mark in meta
            meta = read_json(video_meta_path(video_id)) or {}
            meta["original_deleted"] = True
            write_json(video_meta_path(video_id), meta)
        return {"deleted": True, "kept_clips": True, "bytes_freed": bytes_freed, "items": items}

    # Full delete: video + clips + diagnoses
    manifest = read_json(video_manifest_path(video_id)) or {}
    for c in manifest.get("clips", []):
        ddir = diagnosis_dir(c["clip_id"])
        if ddir.exists():
            bytes_freed += _dir_size(ddir)
            shutil.rmtree(ddir, ignore_errors=True)
            items.append(f"diagnosis:{c['clip_id']}")

    bytes_freed += _dir_size(vdir)
    shutil.rmtree(vdir, ignore_errors=True)
    items.append(f"video:{video_id}")

    return {"deleted": True, "kept_clips": False, "bytes_freed": bytes_freed, "items": items}


def cleanup_all() -> Dict[str, Any]:
    """Nuclear option: delete EVERYTHING in storage. Returns summary."""
    import shutil
    if not STORAGE_ROOT.exists():
        return {"freed": 0, "human": "0 B"}
    freed = _dir_size(STORAGE_ROOT)
    for sub in ("videos", "diagnoses", "jobs"):
        d = STORAGE_ROOT / sub
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
    ensure_storage()
    return {"freed": freed}
