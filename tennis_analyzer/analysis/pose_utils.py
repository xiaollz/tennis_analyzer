"""Pose geometry utilities shared across analysis modules.

This project runs on 2D COCO keypoints. Many metrics need a rough pixel->cm
conversion. Shoulder width works well for behind-view footage, but is heavily
foreshortened in side-view footage. Torso height (shoulder-center to hip-center)
is usually more stable in side view.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config.keypoints import KEYPOINT_NAMES


# ---------------------------------------------------------------------------
# Camera view conventions
# ---------------------------------------------------------------------------

CAMERA_VIEW_SIDE = "side"
CAMERA_VIEW_BACK = "back"
CAMERA_VIEW_UNKNOWN = "unknown"


# Rough anthropometric references (used only for approximate deltas).
SHOULDER_WIDTH_REF_CM = 41.0
# Shoulder-center to hip-center distance for an average adult male (approx).
TORSO_HEIGHT_REF_CM = 55.0


@dataclass(frozen=True)
class PoseScale:
    shoulder_width_px: Optional[float]
    torso_height_px: Optional[float]
    pixels_per_cm_shoulder: Optional[float]
    pixels_per_cm_torso: Optional[float]


def _conf_ok(confidence: np.ndarray, idx: int, thr: float) -> bool:
    try:
        return float(confidence[idx]) >= float(thr)
    except Exception:
        return False


def estimate_shoulder_width_px(
    keypoints: np.ndarray, confidence: np.ndarray, *, min_conf: float = 0.35
) -> Optional[float]:
    l_sh = KEYPOINT_NAMES["left_shoulder"]
    r_sh = KEYPOINT_NAMES["right_shoulder"]
    if not (_conf_ok(confidence, l_sh, min_conf) and _conf_ok(confidence, r_sh, min_conf)):
        return None
    return float(np.linalg.norm(np.asarray(keypoints[r_sh]) - np.asarray(keypoints[l_sh])))


def estimate_torso_height_px(
    keypoints: np.ndarray, confidence: np.ndarray, *, min_conf: float = 0.35
) -> Optional[float]:
    l_sh = KEYPOINT_NAMES["left_shoulder"]
    r_sh = KEYPOINT_NAMES["right_shoulder"]
    l_hip = KEYPOINT_NAMES["left_hip"]
    r_hip = KEYPOINT_NAMES["right_hip"]
    if not (
        _conf_ok(confidence, l_sh, min_conf)
        and _conf_ok(confidence, r_sh, min_conf)
        and _conf_ok(confidence, l_hip, min_conf)
        and _conf_ok(confidence, r_hip, min_conf)
    ):
        return None
    sh = 0.5 * (np.asarray(keypoints[l_sh], dtype=np.float32) + np.asarray(keypoints[r_sh], dtype=np.float32))
    hip = 0.5 * (np.asarray(keypoints[l_hip], dtype=np.float32) + np.asarray(keypoints[r_hip], dtype=np.float32))
    return float(np.linalg.norm(sh - hip))


def estimate_scale(
    keypoints: np.ndarray,
    confidence: np.ndarray,
    *,
    shoulder_width_cm: float = SHOULDER_WIDTH_REF_CM,
    torso_height_cm: float = TORSO_HEIGHT_REF_CM,
    min_conf: float = 0.35,
    min_shoulder_width_px: float = 12.0,
    min_torso_height_px: float = 20.0,
) -> PoseScale:
    """Estimate pose scale in pixels/cm from shoulder width and torso height."""
    sw_px = estimate_shoulder_width_px(keypoints, confidence, min_conf=min_conf)
    th_px = estimate_torso_height_px(keypoints, confidence, min_conf=min_conf)

    ppcm_sw: Optional[float] = None
    if sw_px is not None and sw_px >= float(min_shoulder_width_px) and shoulder_width_cm > 1e-6:
        ppcm_sw = float(sw_px) / float(shoulder_width_cm)

    ppcm_torso: Optional[float] = None
    if th_px is not None and th_px >= float(min_torso_height_px) and torso_height_cm > 1e-6:
        ppcm_torso = float(th_px) / float(torso_height_cm)

    return PoseScale(
        shoulder_width_px=float(sw_px) if sw_px is not None else None,
        torso_height_px=float(th_px) if th_px is not None else None,
        pixels_per_cm_shoulder=ppcm_sw,
        pixels_per_cm_torso=ppcm_torso,
    )


def choose_pixels_per_cm(scale: PoseScale, *, camera_view: Optional[str]) -> Optional[float]:
    """Choose a reasonable pixels/cm conversion based on camera view."""
    view = (camera_view or CAMERA_VIEW_UNKNOWN).lower()

    # Side view: shoulder width is foreshortened; torso height is more stable.
    if view == CAMERA_VIEW_SIDE:
        return scale.pixels_per_cm_torso or scale.pixels_per_cm_shoulder

    # Back view: shoulder width is usually reliable.
    if view == CAMERA_VIEW_BACK:
        return scale.pixels_per_cm_shoulder or scale.pixels_per_cm_torso

    # Unknown: be conservative but still try to provide a usable scale.
    return scale.pixels_per_cm_shoulder or scale.pixels_per_cm_torso


def estimate_view_ratio(
    keypoints: np.ndarray, confidence: np.ndarray, *, min_conf: float = 0.35
) -> Optional[float]:
    """Return shoulder_width_px / torso_height_px (lower => more side-view)."""
    scale = estimate_scale(keypoints, confidence, min_conf=min_conf)
    if scale.shoulder_width_px is None or scale.torso_height_px is None:
        return None
    if scale.torso_height_px < 1e-6:
        return None
    return float(scale.shoulder_width_px / scale.torso_height_px)

