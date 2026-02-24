"""Kinematic calculator: angles, rotations, and body-plane geometry.

All functions are pure (stateless) and operate on single-frame keypoint
arrays.  Higher-level temporal analysis (e.g. angular velocity over time)
is built on top of these primitives in the evaluation layer.
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from config.keypoints import KEYPOINT_NAMES


# ── Low-level geometry helpers ────────────────────────────────────────

def _vec_angle(v1: np.ndarray, v2: np.ndarray) -> Optional[float]:
    """Angle in degrees between two 2-D vectors, or None if degenerate."""
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))


def _signed_angle_2d(v1: np.ndarray, v2: np.ndarray) -> Optional[float]:
    """Signed angle from *v1* to *v2* (positive = counter-clockwise in image coords)."""
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cross = float(v1[0] * v2[1] - v1[1] * v2[0])
    dot = float(np.dot(v1, v2))
    return float(np.degrees(np.arctan2(cross, dot)))


def _conf_ok(confidence: np.ndarray, idx: int, thr: float = 0.3) -> bool:
    try:
        return float(confidence[idx]) >= thr
    except (IndexError, TypeError):
        return False


# ── Joint angle calculators ──────────────────────────────────────────

def joint_angle(
    keypoints: np.ndarray,
    confidence: np.ndarray,
    a_idx: int,
    vertex_idx: int,
    b_idx: int,
    min_conf: float = 0.3,
) -> Optional[float]:
    """Angle at *vertex* formed by segments vertex→a and vertex→b (degrees)."""
    if not all(_conf_ok(confidence, i, min_conf) for i in (a_idx, vertex_idx, b_idx)):
        return None
    v1 = keypoints[a_idx] - keypoints[vertex_idx]
    v2 = keypoints[b_idx] - keypoints[vertex_idx]
    return _vec_angle(v1, v2)


def elbow_angle(keypoints: np.ndarray, confidence: np.ndarray, right: bool = True) -> Optional[float]:
    """Elbow angle (shoulder-elbow-wrist)."""
    side = "right" if right else "left"
    return joint_angle(
        keypoints, confidence,
        KEYPOINT_NAMES[f"{side}_shoulder"],
        KEYPOINT_NAMES[f"{side}_elbow"],
        KEYPOINT_NAMES[f"{side}_wrist"],
    )


def knee_angle(keypoints: np.ndarray, confidence: np.ndarray, right: bool = True) -> Optional[float]:
    """Knee angle (hip-knee-ankle)."""
    side = "right" if right else "left"
    return joint_angle(
        keypoints, confidence,
        KEYPOINT_NAMES[f"{side}_hip"],
        KEYPOINT_NAMES[f"{side}_knee"],
        KEYPOINT_NAMES[f"{side}_ankle"],
    )


def min_knee_angle(keypoints: np.ndarray, confidence: np.ndarray) -> Optional[float]:
    """Return the smaller (more bent) knee angle of both legs."""
    left = knee_angle(keypoints, confidence, right=False)
    right = knee_angle(keypoints, confidence, right=True)
    angles = [a for a in (left, right) if a is not None]
    return min(angles) if angles else None


# ── Body-plane geometry ──────────────────────────────────────────────

def shoulder_line_vector(keypoints: np.ndarray, confidence: np.ndarray) -> Optional[np.ndarray]:
    """Vector from left_shoulder to right_shoulder (2D)."""
    ls, rs = KEYPOINT_NAMES["left_shoulder"], KEYPOINT_NAMES["right_shoulder"]
    if not (_conf_ok(confidence, ls) and _conf_ok(confidence, rs)):
        return None
    return keypoints[rs].astype(np.float64) - keypoints[ls].astype(np.float64)


def hip_line_vector(keypoints: np.ndarray, confidence: np.ndarray) -> Optional[np.ndarray]:
    """Vector from left_hip to right_hip (2D)."""
    lh, rh = KEYPOINT_NAMES["left_hip"], KEYPOINT_NAMES["right_hip"]
    if not (_conf_ok(confidence, lh) and _conf_ok(confidence, rh)):
        return None
    return keypoints[rh].astype(np.float64) - keypoints[lh].astype(np.float64)


def shoulder_hip_angle(keypoints: np.ndarray, confidence: np.ndarray) -> Optional[float]:
    """Angle between shoulder line and hip line (X-Factor proxy, degrees)."""
    sv = shoulder_line_vector(keypoints, confidence)
    hv = hip_line_vector(keypoints, confidence)
    if sv is None or hv is None:
        return None
    return _vec_angle(sv, hv)


def shoulder_rotation_signed(keypoints: np.ndarray, confidence: np.ndarray) -> Optional[float]:
    """Signed rotation of shoulder line relative to hip line.

    Positive = shoulders rotated counter-clockwise relative to hips (in image coords).
    """
    sv = shoulder_line_vector(keypoints, confidence)
    hv = hip_line_vector(keypoints, confidence)
    if sv is None or hv is None:
        return None
    return _signed_angle_2d(hv, sv)


# ── Body centres and scale ───────────────────────────────────────────

def shoulder_center(keypoints: np.ndarray, confidence: np.ndarray) -> Optional[np.ndarray]:
    ls, rs = KEYPOINT_NAMES["left_shoulder"], KEYPOINT_NAMES["right_shoulder"]
    if not (_conf_ok(confidence, ls) and _conf_ok(confidence, rs)):
        return None
    return 0.5 * (keypoints[ls].astype(np.float64) + keypoints[rs].astype(np.float64))


def hip_center(keypoints: np.ndarray, confidence: np.ndarray) -> Optional[np.ndarray]:
    lh, rh = KEYPOINT_NAMES["left_hip"], KEYPOINT_NAMES["right_hip"]
    if not (_conf_ok(confidence, lh) and _conf_ok(confidence, rh)):
        return None
    return 0.5 * (keypoints[lh].astype(np.float64) + keypoints[rh].astype(np.float64))


def torso_height_px(keypoints: np.ndarray, confidence: np.ndarray) -> Optional[float]:
    """Shoulder-center to hip-center distance in pixels."""
    sc = shoulder_center(keypoints, confidence)
    hc = hip_center(keypoints, confidence)
    if sc is None or hc is None:
        return None
    return float(np.linalg.norm(sc - hc))


def shoulder_width_px(keypoints: np.ndarray, confidence: np.ndarray) -> Optional[float]:
    sv = shoulder_line_vector(keypoints, confidence)
    if sv is None:
        return None
    return float(np.linalg.norm(sv))


# ── Spine angle ──────────────────────────────────────────────────────

def spine_angle_from_vertical(keypoints: np.ndarray, confidence: np.ndarray) -> Optional[float]:
    """Angle of the spine (shoulder-center to hip-center) from vertical (degrees).

    0° = perfectly upright.  Positive = leaning.
    """
    sc = shoulder_center(keypoints, confidence)
    hc = hip_center(keypoints, confidence)
    if sc is None or hc is None:
        return None
    spine_vec = sc - hc  # points upward (shoulder above hip in image = negative dy)
    vertical = np.array([0.0, -1.0])  # "up" in image coords
    return _vec_angle(spine_vec, vertical)


# ── Contact-point geometry ───────────────────────────────────────────

def wrist_forward_distance(
    keypoints: np.ndarray,
    confidence: np.ndarray,
    is_right_handed: bool = True,
    forward_sign: float = 1.0,
) -> Optional[float]:
    """Horizontal distance of the dominant wrist in front of hip center (pixels).

    Positive = in front.  Uses *forward_sign* to handle left/right ambiguity.
    """
    side = "right" if is_right_handed else "left"
    wrist_idx = KEYPOINT_NAMES[f"{side}_wrist"]
    if not _conf_ok(confidence, wrist_idx):
        return None
    hc = hip_center(keypoints, confidence)
    if hc is None:
        return None
    dx = float(keypoints[wrist_idx][0]) - float(hc[0])
    return dx * forward_sign


def wrist_forward_normalised(
    keypoints: np.ndarray,
    confidence: np.ndarray,
    is_right_handed: bool = True,
    forward_sign: float = 1.0,
) -> Optional[float]:
    """Wrist forward distance normalised by torso height."""
    dist = wrist_forward_distance(keypoints, confidence, is_right_handed, forward_sign)
    th = torso_height_px(keypoints, confidence)
    if dist is None or th is None or th < 1e-6:
        return None
    return dist / th


# ── Head stability helper ────────────────────────────────────────────

def nose_position(keypoints: np.ndarray, confidence: np.ndarray) -> Optional[np.ndarray]:
    idx = KEYPOINT_NAMES["nose"]
    if not _conf_ok(confidence, idx, 0.3):
        return None
    return keypoints[idx].astype(np.float64).copy()
