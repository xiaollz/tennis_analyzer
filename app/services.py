"""Service layer — wraps segmentation + TennisAnalysisPipeline for the API.

Two primary functions:

- `run_segmentation(video_id, video_path, progress_cb)`:
    Runs SoundBasedSegmenter and persists the manifest.

- `run_diagnosis(clip_id, progress_cb)`:
    Runs TennisAnalysisPipeline on a single clip and persists a
    frontend-friendly JSON result.

Both are designed to be called from `app.jobs.submit()` so they execute in
the background with progress reporting.
"""

from __future__ import annotations

import time
import shutil
import traceback
from dataclasses import is_dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from app import storage
from segmentation import SoundBasedSegmenter


# ── Segmentation ─────────────────────────────────────────────────────

VALID_STROKES = ("forehand", "backhand", "auto")


def run_segmentation(
    *,
    progress_cb: Callable[[float, str], None],
    video_id: str,
    video_path: str,
) -> Dict[str, Any]:
    """Run audio-based segmentation on a video. Persist manifest.json.

    Returns a dict with {video_id, clip_count, manifest_path}.
    """
    seg = SoundBasedSegmenter()
    out_dir = storage.video_dir(video_id)
    result = seg.segment(
        video_path=video_path,
        video_id=video_id,
        output_dir=str(out_dir),
        progress_cb=progress_cb,
    )

    manifest = result.to_dict()
    storage.write_json(storage.video_manifest_path(video_id), manifest)

    # Patch the video meta with duration/fps
    meta = storage.read_json(storage.video_meta_path(video_id)) or {}
    meta.update({
        "duration_s": result.duration_s,
        "fps": result.fps,
        "total_onsets": result.total_onsets,
        "clip_count": len(result.clips),
        "segmented_at": time.time(),
    })
    storage.write_json(storage.video_meta_path(video_id), meta)

    return {
        "video_id": video_id,
        "clip_count": len(result.clips),
        "error": result.error,
    }


# ── Diagnosis ────────────────────────────────────────────────────────

def run_diagnosis(
    *,
    progress_cb: Callable[[float, str], None],
    clip_id: str,
    stroke: str = "forehand",
) -> Dict[str, Any]:
    """Run the full tennis analysis pipeline on a single clip.

    Parameters
    ----------
    stroke : "forehand" | "backhand" | "auto"
        Force evaluator path. "auto" lets the stroke classifier decide,
        but it's unreliable on short / side-angle clips, so the default
        is "forehand" (the user's primary focus).

    Persists:
      - storage/diagnoses/{clip_id}/result.json   (frontend-ready JSON)
      - storage/diagnoses/{clip_id}/annotated.mp4 (skeleton overlay)
      - storage/diagnoses/{clip_id}/keyframe_grid.png
      - storage/diagnoses/{clip_id}/report.md
      - storage/diagnoses/{clip_id}/status.json

    Returns a summary dict.
    """
    if stroke not in VALID_STROKES:
        stroke = "forehand"

    info = storage.find_clip(clip_id)
    if not info:
        raise RuntimeError(f"clip not found: {clip_id}")

    clip = info["clip"]
    clip_path = clip["clip_path"]
    diag_dir = storage.diagnosis_dir(clip_id)
    diag_dir.mkdir(parents=True, exist_ok=True)

    # Write initial status
    storage.write_json(storage.diagnosis_status_path(clip_id), {
        "clip_id": clip_id,
        "status": "running",
        "started_at": time.time(),
        "stroke": stroke,
    })

    # Import pipeline lazily (heavy import: torch, YOLO, etc.)
    progress_cb(0.02, "加载模型")
    from main import TennisAnalysisPipeline

    pipeline = TennisAnalysisPipeline(
        stroke_mode=stroke,
        output_dir=str(diag_dir / "_pipeline_out"),
    )

    def _inner_progress(current: int, total: int, message: str):
        # pipeline reports stage progress as (current, total, message)
        frac = 0.05 + 0.85 * (current / max(total, 1))
        progress_cb(min(frac, 0.9), message)

    progress_cb(0.05, "开始分析")
    result = pipeline.run(clip_path, progress_callback=_inner_progress)

    # ── serialize into frontend JSON ──────────────────────────────
    progress_cb(0.92, "生成结果 JSON")
    frontend_result = _serialize_result(clip_id, info, result)

    # ── copy artifacts into stable paths ──────────────────────────
    progress_cb(0.95, "保存产物")

    # Annotated video
    if result.get("annotated_video_path"):
        src = Path(result["annotated_video_path"])
        if src.exists():
            shutil.copy(src, storage.diagnosis_annotated_path(clip_id))
            frontend_result["artifacts"]["annotated_video"] = f"/api/clips/{clip_id}/annotated"

    # Keyframe grid (first swing's)
    for vlm in result.get("vlm_results", []):
        if vlm and vlm.get("keyframe_grid_path"):
            src = Path(vlm["keyframe_grid_path"])
            if src.exists():
                shutil.copy(src, storage.diagnosis_keyframe_grid_path(clip_id))
                frontend_result["artifacts"]["keyframe_grid"] = f"/api/clips/{clip_id}/keyframes"
                break

    # Report md
    if result.get("report_path"):
        src = Path(result["report_path"])
        if src.exists():
            shutil.copy(src, storage.diagnosis_report_path(clip_id))
            frontend_result["artifacts"]["report_md"] = f"/api/clips/{clip_id}/report.md"

    # Write final result
    storage.write_json(storage.diagnosis_result_path(clip_id), frontend_result)
    storage.write_json(storage.diagnosis_status_path(clip_id), {
        "clip_id": clip_id,
        "status": "done",
        "ended_at": time.time(),
    })

    # Best-effort cleanup of pipeline's temp output
    try:
        shutil.rmtree(diag_dir / "_pipeline_out", ignore_errors=True)
    except Exception:
        pass

    progress_cb(1.0, "完成")
    return {"clip_id": clip_id, "overall_score": frontend_result.get("overall_score")}


# ── Serialization helpers ────────────────────────────────────────────

def _to_serializable(obj: Any) -> Any:
    """Recursively convert dataclasses / numpy values to JSON-safe primitives."""
    import numpy as np

    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if is_dataclass(obj):
        return _to_serializable(asdict(obj))
    # Fall back to str for unknown objects
    try:
        return str(obj)
    except Exception:
        return None


def _serialize_kpi(kpi) -> Dict[str, Any]:
    """Turn a KPIResult dataclass into a frontend-friendly dict."""
    return {
        "name": getattr(kpi, "name", ""),
        "label": getattr(kpi, "label", getattr(kpi, "name", "")),
        "value": _to_serializable(getattr(kpi, "value", None)),
        "unit": getattr(kpi, "unit", ""),
        "score": _to_serializable(getattr(kpi, "score", None)),
        "level": getattr(kpi, "level", ""),
        "description": getattr(kpi, "description", ""),
        "phase": getattr(kpi, "phase", ""),
    }


def _serialize_swing(swing) -> Dict[str, Any]:
    """Turn a SwingEvaluation into a frontend-friendly dict."""
    phase_scores = {}
    for phase_name, ps in (swing.phase_scores or {}).items():
        phase_scores[phase_name] = {
            "phase": getattr(ps, "phase", phase_name),
            "score": _to_serializable(getattr(ps, "score", 0)),
            "kpis": [_serialize_kpi(k) for k in getattr(ps, "kpis", [])],
        }
    return {
        "swing_index": swing.swing_index,
        "impact_frame": getattr(swing.swing_event, "impact_frame", None),
        "overall_score": _to_serializable(swing.overall_score),
        "arm_style": swing.arm_style,
        "phase_scores": phase_scores,
        "kpi_results": [_serialize_kpi(k) for k in (swing.kpi_results or [])],
        "raw_metrics": _to_serializable(swing.raw_metrics or {}),
    }


def _serialize_result(
    clip_id: str,
    clip_info: Dict[str, Any],
    pipeline_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Collapse TennisAnalysisPipeline.run() output into frontend-ready JSON."""
    report = pipeline_result.get("report")
    stroke_type = pipeline_result.get("stroke_type", "forehand")
    vlm_results = pipeline_result.get("vlm_results", []) or []

    swings = []
    if report and getattr(report, "swing_evaluations", None):
        for sw in report.swing_evaluations:
            swings.append(_serialize_swing(sw))

    overall_score = _to_serializable(getattr(report, "average_score", None)) if report else None

    # Grab the first non-None VLM result as the primary one
    primary_vlm = next((v for v in vlm_results if v), None)
    vlm_payload = _to_serializable(primary_vlm) if primary_vlm else None

    # Extract diagnosis sub-sections from VLM result (diagnose() attaches them)
    diagnosis_payload: Optional[Dict[str, Any]] = None
    if primary_vlm:
        diagnosis_payload = {
            "root_cause": _to_serializable(primary_vlm.get("root_cause")),
            "causal_chain": _to_serializable(primary_vlm.get("causal_chain", [])),
            "drill_recommendation": _to_serializable(primary_vlm.get("drill_recommendation")),
            "narrative": primary_vlm.get("narrative") or primary_vlm.get("diagnosis_narrative"),
            "contradictions": _to_serializable(primary_vlm.get("contradictions", [])),
            "arbitration": _to_serializable(primary_vlm.get("arbitration", [])),
            "issues": _to_serializable(primary_vlm.get("issues", [])),
        }

    return {
        "clip_id": clip_id,
        "video_id": clip_info["video_id"],
        "clip": clip_info["clip"],
        "stroke_type": stroke_type,
        "overall_score": overall_score,
        "swings": swings,
        "vlm": vlm_payload,
        "diagnosis": diagnosis_payload,
        "artifacts": {
            "clip_video": f"/api/clips/{clip_id}/video",
            "thumbnail": f"/api/clips/{clip_id}/thumbnail",
            "annotated_video": None,
            "keyframe_grid": None,
            "report_md": None,
        },
        "created_at": time.time(),
    }
