"""Kinematic calculator: angles, rotations, and body-plane geometry.

All functions are pure (stateless) and operate on single-frame keypoint
arrays.  Higher-level temporal analysis (e.g. angular velocity over time)
is built on top of these primitives in the evaluation layer.

v3: Added Slot preparation, ground-force proxy, lag quality,
    elbow tuck, SIR proxy, and windshield-wiper angle helpers.
"""

from __future__ import annotations

from typing import Optional, Tuple, List
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
    right_a = knee_angle(keypoints, confidence, right=True)
    angles = [a for a in (left, right_a) if a is not None]
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


# =====================================================================
# v3: NEW BIOMECHANICS HELPERS
# =====================================================================

# ── Slot Preparation (Phase 2) ──────────────────────────────────────

def elbow_behind_torso_distance(
    keypoints: np.ndarray,
    confidence: np.ndarray,
    is_right_handed: bool = True,
    forward_sign: float = 1.0,
) -> Optional[float]:
    """How far the dominant elbow is behind the torso center (pixels).

    Positive = elbow is behind (good for Slot Preparation).
    This measures the "Elbow Back" position that Rick Macci emphasizes.
    """
    side = "right" if is_right_handed else "left"
    elbow_idx = KEYPOINT_NAMES[f"{side}_elbow"]
    if not _conf_ok(confidence, elbow_idx):
        return None
    sc = shoulder_center(keypoints, confidence)
    if sc is None:
        return None
    # "Behind" is opposite to forward direction
    dx = float(sc[0]) - float(keypoints[elbow_idx][0])
    return dx * forward_sign


def elbow_behind_torso_normalised(
    keypoints: np.ndarray,
    confidence: np.ndarray,
    is_right_handed: bool = True,
    forward_sign: float = 1.0,
) -> Optional[float]:
    """Elbow behind torso distance normalised by shoulder width."""
    dist = elbow_behind_torso_distance(keypoints, confidence, is_right_handed, forward_sign)
    sw = shoulder_width_px(keypoints, confidence)
    if dist is None or sw is None or sw < 1e-6:
        return None
    return dist / sw


def elbow_height_relative(
    keypoints: np.ndarray,
    confidence: np.ndarray,
    is_right_handed: bool = True,
) -> Optional[float]:
    """Height of dominant elbow relative to shoulder (normalised by torso height).

    Positive = elbow above shoulder.  Negative = elbow below shoulder.
    In image coords, y increases downward, so we invert.
    """
    side = "right" if is_right_handed else "left"
    elbow_idx = KEYPOINT_NAMES[f"{side}_elbow"]
    shoulder_idx = KEYPOINT_NAMES[f"{side}_shoulder"]
    if not (_conf_ok(confidence, elbow_idx) and _conf_ok(confidence, shoulder_idx)):
        return None
    th = torso_height_px(keypoints, confidence)
    if th is None or th < 1e-6:
        return None
    # Invert because y increases downward in image coords
    dy = float(keypoints[shoulder_idx][1]) - float(keypoints[elbow_idx][1])
    return dy / th


def wrist_below_elbow_distance(
    keypoints: np.ndarray,
    confidence: np.ndarray,
    is_right_handed: bool = True,
) -> Optional[float]:
    """Vertical distance of wrist below elbow (normalised by torso height).

    Positive = wrist is below elbow (good for racket drop / "Pat the Dog").
    """
    side = "right" if is_right_handed else "left"
    wrist_idx = KEYPOINT_NAMES[f"{side}_wrist"]
    elbow_idx = KEYPOINT_NAMES[f"{side}_elbow"]
    if not (_conf_ok(confidence, wrist_idx) and _conf_ok(confidence, elbow_idx)):
        return None
    th = torso_height_px(keypoints, confidence)
    if th is None or th < 1e-6:
        return None
    # In image coords, y increases downward, so wrist_y > elbow_y means wrist is below
    dy = float(keypoints[wrist_idx][1]) - float(keypoints[elbow_idx][1])
    return dy / th


# ── Leg Drive & Ground Force Proxy (Phase 3) ────────────────────────

def hip_center_vertical_position(
    keypoints: np.ndarray,
    confidence: np.ndarray,
) -> Optional[float]:
    """Vertical position of hip center in pixels (y-coordinate).

    Used for tracking hip vertical movement over time to compute
    ground-force proxy (vertical acceleration).
    """
    hc = hip_center(keypoints, confidence)
    if hc is None:
        return None
    return float(hc[1])


def compute_vertical_acceleration(
    positions: List[float],
    fps: float,
) -> Optional[float]:
    """Compute peak upward acceleration from a series of vertical positions.

    Uses central difference for velocity, then forward difference for acceleration.
    Returns peak upward acceleration (positive = upward in real world).

    In image coords, y increases downward, so upward movement = decreasing y.
    """
    if len(positions) < 3 or fps <= 0:
        return None
    dt = 1.0 / fps
    positions_arr = np.array(positions, dtype=np.float64)

    # Velocity via central difference (negate because image y is inverted)
    velocity = np.zeros(len(positions_arr))
    for i in range(1, len(positions_arr) - 1):
        velocity[i] = -(positions_arr[i + 1] - positions_arr[i - 1]) / (2 * dt)

    # Acceleration via forward difference
    accel = np.zeros(len(velocity) - 1)
    for i in range(len(velocity) - 1):
        accel[i] = (velocity[i + 1] - velocity[i]) / dt

    if len(accel) == 0:
        return None
    return float(np.max(accel))  # Peak upward acceleration


# ── Lag & Elbow Drive (Phase 5) ──────────────────────────────────────

def elbow_to_torso_distance(
    keypoints: np.ndarray,
    confidence: np.ndarray,
    is_right_handed: bool = True,
) -> Optional[float]:
    """Distance from dominant elbow to torso center line (normalised by torso height).

    Measures how close the elbow is "tucked" to the body during forward swing.
    Smaller = better tuck (elbow driving close to body).
    """
    side = "right" if is_right_handed else "left"
    elbow_idx = KEYPOINT_NAMES[f"{side}_elbow"]
    if not _conf_ok(confidence, elbow_idx):
        return None

    sc = shoulder_center(keypoints, confidence)
    hc = hip_center(keypoints, confidence)
    if sc is None or hc is None:
        return None

    th = float(np.linalg.norm(sc - hc))
    if th < 1e-6:
        return None

    # Torso center line: from hip_center to shoulder_center
    # Project elbow onto this line and measure perpendicular distance
    elbow_pos = keypoints[elbow_idx].astype(np.float64)
    torso_vec = sc - hc
    torso_len = np.linalg.norm(torso_vec)
    if torso_len < 1e-6:
        return None

    # Vector from hip_center to elbow
    he_vec = elbow_pos - hc
    # Perpendicular distance = |cross product| / |torso_vec|
    cross = abs(float(torso_vec[0] * he_vec[1] - torso_vec[1] * he_vec[0]))
    perp_dist = cross / torso_len
    return perp_dist / th


# ── SIR Proxy (Phase 6) ─────────────────────────────────────────────

def forearm_angle(
    keypoints: np.ndarray,
    confidence: np.ndarray,
    is_right_handed: bool = True,
) -> Optional[float]:
    """Angle of the forearm (elbow→wrist vector) relative to horizontal.

    Used to track forearm rotation over time as a proxy for
    Shoulder Internal Rotation (SIR).
    Returns angle in degrees: 0° = horizontal, 90° = vertical.
    """
    side = "right" if is_right_handed else "left"
    elbow_idx = KEYPOINT_NAMES[f"{side}_elbow"]
    wrist_idx = KEYPOINT_NAMES[f"{side}_wrist"]
    if not (_conf_ok(confidence, elbow_idx) and _conf_ok(confidence, wrist_idx)):
        return None

    forearm_vec = keypoints[wrist_idx].astype(np.float64) - keypoints[elbow_idx].astype(np.float64)
    horizontal = np.array([1.0, 0.0])
    return _vec_angle(forearm_vec, horizontal)


def compute_angular_velocity(
    angles: List[float],
    fps: float,
) -> Optional[float]:
    """Compute peak angular velocity from a series of angles (degrees).

    Returns peak angular velocity in degrees/second.
    """
    if len(angles) < 2 or fps <= 0:
        return None
    dt = 1.0 / fps
    velocities = []
    for i in range(1, len(angles)):
        dtheta = abs(angles[i] - angles[i - 1])
        # Handle angle wrapping
        if dtheta > 180:
            dtheta = 360 - dtheta
        velocities.append(dtheta / dt)
    return float(max(velocities)) if velocities else None


# ── Windshield Wiper Follow-Through (Phase 7) ───────────────────────

def wrist_lateral_displacement(
    keypoints: np.ndarray,
    confidence: np.ndarray,
    is_right_handed: bool = True,
    forward_sign: float = 1.0,
) -> Optional[float]:
    """Lateral (horizontal) position of dominant wrist relative to shoulder center.

    Used to track the windshield-wiper motion after contact.
    Positive = wrist is on the non-dominant side (follow-through direction).
    """
    side = "right" if is_right_handed else "left"
    wrist_idx = KEYPOINT_NAMES[f"{side}_wrist"]
    if not _conf_ok(confidence, wrist_idx):
        return None
    sc = shoulder_center(keypoints, confidence)
    if sc is None:
        return None
    # For right-hander, follow-through goes to the left (negative x in image)
    # forward_sign helps disambiguate camera angle
    dx = float(keypoints[wrist_idx][0]) - float(sc[0])
    # For windshield wiper, we want to measure how far the wrist crosses to the other side
    # For right-hander facing left: follow-through = wrist moves further left = more negative dx
    return -dx * forward_sign


def compute_wiper_sweep_angle(
    wrist_positions: List[np.ndarray],
    shoulder_centers: List[np.ndarray],
) -> Optional[float]:
    """Compute the angular sweep of the wrist around the shoulder center.

    Measures the "windshield wiper" arc in degrees.
    """
    if len(wrist_positions) < 2 or len(shoulder_centers) < 2:
        return None

    angles = []
    for wp, sc in zip(wrist_positions, shoulder_centers):
        if wp is None or sc is None:
            continue
        vec = wp - sc
        angle = float(np.degrees(np.arctan2(vec[1], vec[0])))
        angles.append(angle)

    if len(angles) < 2:
        return None

    # Total angular sweep
    total_sweep = 0.0
    for i in range(1, len(angles)):
        diff = abs(angles[i] - angles[i - 1])
        if diff > 180:
            diff = 360 - diff
        total_sweep += diff

    return total_sweep


# ── Hip rotation speed (Phase 3 & 4) ────────────────────────────────

def hip_line_angle(keypoints: np.ndarray, confidence: np.ndarray) -> Optional[float]:
    """Angle of the hip line relative to horizontal (degrees).

    Used to track hip rotation speed over time.
    """
    hv = hip_line_vector(keypoints, confidence)
    if hv is None:
        return None
    horizontal = np.array([1.0, 0.0])
    return _signed_angle_2d(horizontal, hv)


def shoulder_line_angle(keypoints: np.ndarray, confidence: np.ndarray) -> Optional[float]:
    """Angle of the shoulder line relative to horizontal (degrees).

    Used to track shoulder rotation speed and compute hip-shoulder separation timing.
    """
    sv = shoulder_line_vector(keypoints, confidence)
    if sv is None:
        return None
    horizontal = np.array([1.0, 0.0])
    return _signed_angle_2d(horizontal, sv)


def compute_rotation_speed(
    angles: List[Optional[float]],
    fps: float,
) -> List[float]:
    """Compute rotation speed (degrees/second) from a series of angles.

    Handles None values by interpolation.
    Returns a list of speeds (same length as input, first element is 0).
    """
    if len(angles) < 2 or fps <= 0:
        return [0.0] * len(angles)

    dt = 1.0 / fps
    speeds = [0.0]

    for i in range(1, len(angles)):
        if angles[i] is not None and angles[i - 1] is not None:
            diff = angles[i] - angles[i - 1]
            # Handle angle wrapping
            if diff > 180:
                diff -= 360
            elif diff < -180:
                diff += 360
            speeds.append(abs(diff) / dt)
        else:
            speeds.append(0.0)

    return speeds


def compute_peak_rotation_speed(
    angles: List[Optional[float]],
    fps: float,
) -> Optional[float]:
    """Compute peak rotation speed from a series of angles."""
    speeds = compute_rotation_speed(angles, fps)
    valid = [s for s in speeds if s > 0]
    return max(valid) if valid else None


# ── Hip-Shoulder Separation Timing (Phase 3→4) ──────────────────────

def find_peak_rotation_frame(
    angles: List[Optional[float]],
    fps: float,
    frame_indices: List[int],
) -> Optional[int]:
    """Find the frame index where rotation speed peaks.

    Returns the frame index (from frame_indices) at peak rotation speed.
    """
    speeds = compute_rotation_speed(angles, fps)
    if not speeds or max(speeds) == 0:
        return None
    peak_idx = int(np.argmax(speeds))
    if peak_idx < len(frame_indices):
        return frame_indices[peak_idx]
    return None


def hip_shoulder_separation_timing(
    hip_angles: List[Optional[float]],
    shoulder_angles: List[Optional[float]],
    fps: float,
    frame_indices: List[int],
) -> Optional[float]:
    """Time delay (seconds) between hip peak rotation and shoulder peak rotation.

    Positive = hip peaks first (correct kinetic chain).
    """
    hip_peak_frame = find_peak_rotation_frame(hip_angles, fps, frame_indices)
    shoulder_peak_frame = find_peak_rotation_frame(shoulder_angles, fps, frame_indices)

    if hip_peak_frame is None or shoulder_peak_frame is None:
        return None

    return (shoulder_peak_frame - hip_peak_frame) / fps
