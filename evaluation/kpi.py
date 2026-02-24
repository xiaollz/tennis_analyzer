"""Key Performance Indicator (KPI) definitions for Modern Forehand evaluation.

Each KPI is a self-contained class that:
    1. Receives relevant data (keypoints, trajectories, frame indices).
    2. Computes a raw metric value.
    3. Maps it to a 0-100 score using the thresholds from ``FrameworkConfig``.
    4. Returns a ``KPIResult`` with the score, raw value, rating, and feedback.

KPIs are grouped by swing phase:
    Phase 1 – Preparation: ShoulderRotationKPI, KneeBendKPI, SpineAngleKPI
    Phase 3 – Kinetic Chain: KineticChainSequenceKPI, HipShoulderSeparationKPI, HandPathLinearityKPI
    Phase 4 – Contact: ContactPointKPI, ElbowAngleAtContactKPI, BodyFreezeKPI, HeadStabilityKPI
    Phase 5 – Extension: ForwardExtensionKPI, FollowThroughPathKPI
    Phase 6 – Balance: OverallHeadStabilityKPI, SpineConsistencyKPI
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import numpy as np

from config.framework_config import FrameworkConfig, DEFAULT_CONFIG
from analysis.kinematic_calculator import (
    shoulder_hip_angle,
    min_knee_angle,
    spine_angle_from_vertical,
    elbow_angle,
    wrist_forward_normalised,
    nose_position,
    torso_height_px,
    shoulder_rotation_signed,
    hip_center,
    shoulder_center,
)


# ── Result container ─────────────────────────────────────────────────

@dataclass
class KPIResult:
    """Result of a single KPI evaluation."""
    kpi_id: str
    name: str
    phase: str
    raw_value: Optional[float]
    unit: str
    score: float          # 0-100
    rating: str           # "excellent", "good", "fair", "poor", "n/a"
    feedback: str         # human-readable coaching tip
    details: Dict[str, Any] = None  # optional extra data

    def __post_init__(self):
        if self.details is None:
            self.details = {}


def _linear_score(value: float, poor: float, good: float, excellent: float) -> float:
    """Map a value to 0-100 using a piecewise-linear scale.

    For metrics where *larger is better* (e.g. shoulder rotation):
        poor < good < excellent  →  score rises with value.
    For metrics where *smaller is better* (e.g. head displacement):
        poor > good > excellent  →  score rises as value decreases.
    """
    if excellent > poor:  # larger is better
        if value >= excellent:
            return 100.0
        if value <= poor:
            return max(0.0, 20.0 * value / max(poor, 1e-6))
        if value >= good:
            return 70.0 + 30.0 * (value - good) / max(excellent - good, 1e-6)
        return 20.0 + 50.0 * (value - poor) / max(good - poor, 1e-6)
    else:  # smaller is better
        if value <= excellent:
            return 100.0
        if value >= poor:
            return max(0.0, 20.0 * (1.0 - value / max(poor, 1e-6)))
        if value <= good:
            return 70.0 + 30.0 * (good - value) / max(good - excellent, 1e-6)
        return 20.0 + 50.0 * (poor - value) / max(poor - good, 1e-6)


def _rating_from_score(score: float) -> str:
    if score >= 85:
        return "excellent"
    if score >= 65:
        return "good"
    if score >= 40:
        return "fair"
    return "poor"


# ── Abstract base ────────────────────────────────────────────────────

class BaseKPI(ABC):
    """Abstract base for all KPIs."""

    kpi_id: str = ""
    name: str = ""
    phase: str = ""
    unit: str = ""

    def __init__(self, cfg: FrameworkConfig = DEFAULT_CONFIG):
        self.cfg = cfg

    @abstractmethod
    def evaluate(self, **kwargs) -> KPIResult:
        ...


# =====================================================================
# Phase 1: Preparation
# =====================================================================

class ShoulderRotationKPI(BaseKPI):
    """P1.1 – Maximum shoulder rotation (X-Factor) during preparation."""
    kpi_id = "P1.1"
    name = "Shoulder Rotation (X-Factor)"
    phase = "preparation"
    unit = "degrees"

    def evaluate(self, *, shoulder_rotation_values: List[float], **kw) -> KPIResult:
        if not shoulder_rotation_values:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "n/a", "Insufficient data for shoulder rotation.")

        max_rot = float(max(shoulder_rotation_values))
        c = self.cfg.preparation
        score = _linear_score(max_rot, c.shoulder_rotation_poor, c.shoulder_rotation_good, c.shoulder_rotation_excellent)
        rating = _rating_from_score(score)

        if max_rot >= c.shoulder_rotation_excellent:
            fb = f"Excellent shoulder turn of {max_rot:.0f}°. Full coil achieved."
        elif max_rot >= c.shoulder_rotation_good:
            fb = f"Good shoulder turn ({max_rot:.0f}°). Try to reach 90°+ for maximum coil."
        else:
            fb = f"Shoulder turn is only {max_rot:.0f}°. Focus on a full unit turn — shoulders should rotate ≥90° relative to hips."

        return KPIResult(self.kpi_id, self.name, self.phase, max_rot, self.unit, score, rating, fb)


class KneeBendKPI(BaseKPI):
    """P1.4 – Knee bend on the loaded leg during preparation."""
    kpi_id = "P1.4"
    name = "Knee Bend (Loading)"
    phase = "preparation"
    unit = "degrees"

    def evaluate(self, *, knee_angle_values: List[float], **kw) -> KPIResult:
        if not knee_angle_values:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "n/a", "Insufficient data for knee bend.")

        min_angle = float(min(knee_angle_values))
        c = self.cfg.preparation
        # Smaller angle = more bend = better
        score = _linear_score(min_angle, c.knee_bend_poor, c.knee_bend_good, c.knee_bend_excellent)
        rating = _rating_from_score(score)

        if min_angle <= c.knee_bend_excellent:
            fb = f"Great knee bend ({min_angle:.0f}°). Strong lower-body loading."
        elif min_angle <= c.knee_bend_good:
            fb = f"Decent knee bend ({min_angle:.0f}°). Try to sink a bit lower for more ground-force."
        else:
            fb = f"Legs too straight ({min_angle:.0f}°). Bend your knees more to load the legs — aim for ≤140°."

        return KPIResult(self.kpi_id, self.name, self.phase, min_angle, self.unit, score, rating, fb)


class SpineAngleKPI(BaseKPI):
    """P1.3 – Spine posture (deviation from vertical)."""
    kpi_id = "P1.3"
    name = "Spine Posture"
    phase = "preparation"
    unit = "degrees"

    def evaluate(self, *, spine_angle_values: List[float], **kw) -> KPIResult:
        if not spine_angle_values:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "n/a", "Insufficient data for spine angle.")

        avg_lean = float(np.mean(spine_angle_values))
        c = self.cfg.preparation
        # Smaller lean = better
        score = _linear_score(avg_lean, c.spine_lean_warning, c.spine_lean_good_max, 5.0)
        rating = _rating_from_score(score)

        if avg_lean <= c.spine_lean_good_max:
            fb = f"Good upright posture (avg lean {avg_lean:.1f}°)."
        else:
            fb = f"Excessive forward lean ({avg_lean:.1f}°). Keep spine straighter — aim for <{c.spine_lean_good_max:.0f}° from vertical."

        return KPIResult(self.kpi_id, self.name, self.phase, avg_lean, self.unit, score, rating, fb)


# =====================================================================
# Phase 3: Kinetic Chain
# =====================================================================

class KineticChainSequenceKPI(BaseKPI):
    """KC3.1 – Sequential peak-speed ordering: hip → shoulder → elbow → wrist."""
    kpi_id = "KC3.1"
    name = "Kinetic Chain Sequence"
    phase = "kinetic_chain"
    unit = "sequence_score"

    def evaluate(
        self,
        *,
        hip_peak_frame: Optional[int] = None,
        shoulder_peak_frame: Optional[int] = None,
        elbow_peak_frame: Optional[int] = None,
        wrist_peak_frame: Optional[int] = None,
        fps: float = 30.0,
        **kw,
    ) -> KPIResult:
        peaks = [
            ("hip", hip_peak_frame),
            ("shoulder", shoulder_peak_frame),
            ("elbow", elbow_peak_frame),
            ("wrist", wrist_peak_frame),
        ]
        available = [(name, f) for name, f in peaks if f is not None]
        if len(available) < 3:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "n/a",
                             "Not enough joint data to evaluate kinetic chain sequence.")

        # Check if the order is correct (each peak should come after the previous)
        correct_pairs = 0
        total_pairs = 0
        delays = []
        for i in range(len(available) - 1):
            total_pairs += 1
            delay_frames = available[i + 1][1] - available[i][1]
            delay_s = delay_frames / fps
            delays.append(delay_s)
            if delay_frames > 0:
                correct_pairs += 1

        seq_ratio = correct_pairs / max(total_pairs, 1)
        score = seq_ratio * 100.0
        rating = _rating_from_score(score)

        order_str = " → ".join(f"{n}(f{f})" for n, f in available)
        if score >= 85:
            fb = f"Excellent kinetic chain sequence: {order_str}. Proper proximal-to-distal firing."
        elif score >= 50:
            fb = f"Partially correct sequence: {order_str}. Some segments fire simultaneously — work on sequential hip-then-shoulder rotation."
        else:
            fb = f"Kinetic chain out of order: {order_str}. Focus on legs → hips → torso → arm sequence."

        return KPIResult(self.kpi_id, self.name, self.phase, score, self.unit, score, rating, fb,
                         details={"order": order_str, "delays_s": delays})


class HipShoulderSeparationKPI(BaseKPI):
    """KC3.2 – Maximum hip-shoulder separation angle during forward swing."""
    kpi_id = "KC3.2"
    name = "Hip-Shoulder Separation"
    phase = "kinetic_chain"
    unit = "degrees"

    def evaluate(self, *, hip_shoulder_sep_values: List[float], **kw) -> KPIResult:
        if not hip_shoulder_sep_values:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "n/a",
                             "Insufficient data for hip-shoulder separation.")

        max_sep = float(max(hip_shoulder_sep_values))
        c = self.cfg.kinetic_chain
        score = _linear_score(max_sep, 10.0, c.hip_shoulder_separation_good, c.hip_shoulder_separation_excellent)
        rating = _rating_from_score(score)

        if max_sep >= c.hip_shoulder_separation_excellent:
            fb = f"Excellent hip-shoulder separation ({max_sep:.0f}°). Great X-Factor stretch."
        elif max_sep >= c.hip_shoulder_separation_good:
            fb = f"Good separation ({max_sep:.0f}°). Hips are leading the shoulders."
        else:
            fb = f"Low hip-shoulder separation ({max_sep:.0f}°). Let your hips rotate first — create a 'coil' before the shoulders follow."

        return KPIResult(self.kpi_id, self.name, self.phase, max_sep, self.unit, score, rating, fb)


class HandPathLinearityKPI(BaseKPI):
    """KC3.4 – Linearity of the hand path through the contact zone."""
    kpi_id = "KC3.4"
    name = "Hand Path Linearity"
    phase = "kinetic_chain"
    unit = "R²"

    def evaluate(self, *, wrist_positions_contact_zone: Optional[np.ndarray] = None, **kw) -> KPIResult:
        if wrist_positions_contact_zone is None or len(wrist_positions_contact_zone) < 3:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "n/a",
                             "Insufficient data for hand path linearity.")

        pts = np.asarray(wrist_positions_contact_zone, dtype=np.float64)
        # Fit a line and compute R²
        x = pts[:, 0]
        y = pts[:, 1]
        # Use total least squares (PCA) for direction-agnostic fit
        centroid = pts.mean(axis=0)
        centered = pts - centroid
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
        if s.sum() < 1e-6:
            r_squared = 1.0
        else:
            r_squared = float(s[0] ** 2 / (s ** 2).sum())

        c = self.cfg.kinetic_chain
        score = _linear_score(r_squared, 0.5, c.hand_path_linearity_good, c.hand_path_linearity_excellent)
        rating = _rating_from_score(score)

        if r_squared >= c.hand_path_linearity_excellent:
            fb = f"Very linear hand path (R²={r_squared:.2f}). Clean swing through the contact zone."
        elif r_squared >= c.hand_path_linearity_good:
            fb = f"Reasonably linear hand path (R²={r_squared:.2f}). Minor arc detected."
        else:
            fb = f"Hand path is too curved (R²={r_squared:.2f}). Aim for a straighter path through the ball — swing from inside-out."

        return KPIResult(self.kpi_id, self.name, self.phase, r_squared, self.unit, score, rating, fb)


# =====================================================================
# Phase 4: Contact
# =====================================================================

class ContactPointKPI(BaseKPI):
    """C4.1 – Contact point position (wrist forward of hip, normalised)."""
    kpi_id = "C4.1"
    name = "Contact Point Position"
    phase = "contact"
    unit = "torso_heights"

    def evaluate(self, *, contact_forward_norm: Optional[float] = None, **kw) -> KPIResult:
        if contact_forward_norm is None:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "n/a",
                             "Could not measure contact point position.")

        c = self.cfg.contact
        val = contact_forward_norm
        if val < c.contact_forward_poor_min:
            score = max(0.0, 20.0 * val / max(c.contact_forward_poor_min, 1e-6))
        elif val < c.contact_forward_good_min:
            score = 20.0 + 50.0 * (val - c.contact_forward_poor_min) / max(c.contact_forward_good_min - c.contact_forward_poor_min, 1e-6)
        elif val <= c.contact_forward_good_max:
            score = 85.0
        else:
            # Too far in front
            overshoot = val - c.contact_forward_good_max
            score = max(40.0, 85.0 - overshoot * 100.0)

        score = float(np.clip(score, 0, 100))
        rating = _rating_from_score(score)

        if score >= 70:
            fb = f"Good contact point ({val:.2f} torso-heights in front). Ball met well in front of body."
        elif val < c.contact_forward_good_min:
            fb = f"Contact too close to body ({val:.2f}). Hit the ball further in front — extend your arm forward."
        else:
            fb = f"Contact too far in front ({val:.2f}). You may be over-reaching — adjust timing."

        return KPIResult(self.kpi_id, self.name, self.phase, val, self.unit, score, rating, fb)


class ElbowAngleAtContactKPI(BaseKPI):
    """C4.2 – Elbow angle at the moment of contact."""
    kpi_id = "C4.2"
    name = "Elbow Angle at Contact"
    phase = "contact"
    unit = "degrees"

    def evaluate(self, *, elbow_angle_at_contact: Optional[float] = None, **kw) -> KPIResult:
        if elbow_angle_at_contact is None:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "n/a",
                             "Could not measure elbow angle at contact.")

        c = self.cfg.contact
        angle = elbow_angle_at_contact

        # Determine which style and score accordingly
        in_straight = c.straight_arm_min <= angle <= c.straight_arm_max
        in_double_bend = c.double_bend_min <= angle <= c.double_bend_max

        if in_straight:
            score = 90.0
            style = "straight-arm (Gordon Type 3)"
            fb = f"Straight-arm contact ({angle:.0f}°) — {style} style. Excellent extension."
        elif in_double_bend:
            score = 90.0
            style = "double-bend"
            fb = f"Double-bend contact ({angle:.0f}°) — {style} style. Good compact structure."
        elif c.double_bend_max < angle < c.straight_arm_min:
            # In between: acceptable but not optimal
            score = 60.0
            style = "intermediate"
            fb = f"Elbow angle at contact is {angle:.0f}° — between double-bend and straight-arm. Commit to one style for consistency."
        elif angle < c.double_bend_min:
            score = 30.0
            style = "too bent"
            fb = f"Arm too bent at contact ({angle:.0f}°). Extend more — you're losing power from a cramped position."
        else:
            score = 70.0
            style = "hyper-extended"
            fb = f"Arm angle at contact ({angle:.0f}°) is fine."

        rating = _rating_from_score(score)
        return KPIResult(self.kpi_id, self.name, self.phase, angle, self.unit, score, rating, fb,
                         details={"style": style})


class BodyFreezeKPI(BaseKPI):
    """C4.3 – Torso angular velocity at contact (should be near zero = 'blocking')."""
    kpi_id = "C4.3"
    name = "Body Freeze at Contact"
    phase = "contact"
    unit = "degrees/s"

    def evaluate(self, *, torso_angular_velocity_at_contact: Optional[float] = None, **kw) -> KPIResult:
        if torso_angular_velocity_at_contact is None:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "n/a",
                             "Could not measure torso angular velocity at contact.")

        c = self.cfg.contact
        val = abs(torso_angular_velocity_at_contact)
        score = _linear_score(val, c.body_freeze_warning, c.body_freeze_good_max, 10.0)
        rating = _rating_from_score(score)

        if val <= c.body_freeze_good_max:
            fb = f"Good body freeze at contact ({val:.0f}°/s). Chest stops rotating to create a stable platform."
        else:
            fb = f"Body still rotating at contact ({val:.0f}°/s). Try to 'block' your torso at impact — chest should face the target and stop."

        return KPIResult(self.kpi_id, self.name, self.phase, val, self.unit, score, rating, fb)


class HeadStabilityAtContactKPI(BaseKPI):
    """C4.4 – Head stability around the contact point."""
    kpi_id = "C4.4"
    name = "Head Stability at Contact"
    phase = "contact"
    unit = "normalised_displacement"

    def evaluate(self, *, head_displacement_norm: Optional[float] = None, **kw) -> KPIResult:
        if head_displacement_norm is None:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "n/a",
                             "Could not measure head stability.")

        c = self.cfg.contact
        val = head_displacement_norm
        score = _linear_score(val, c.head_stability_warning, c.head_stability_good_max, 0.02)
        rating = _rating_from_score(score)

        if val <= c.head_stability_good_max:
            fb = f"Excellent head stability ({val:.3f}). Eyes stayed on the ball."
        else:
            fb = f"Head moved too much around contact ({val:.3f}). Keep your eyes on the contact point — head should be the last thing to move."

        return KPIResult(self.kpi_id, self.name, self.phase, val, self.unit, score, rating, fb)


# =====================================================================
# Phase 5: Extension & Follow-Through
# =====================================================================

class ForwardExtensionKPI(BaseKPI):
    """E5.1 – Forward extension distance after contact (normalised by torso height)."""
    kpi_id = "E5.1"
    name = "Forward Extension"
    phase = "extension"
    unit = "torso_heights"

    def evaluate(self, *, forward_extension_norm: Optional[float] = None, **kw) -> KPIResult:
        if forward_extension_norm is None:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "n/a",
                             "Could not measure forward extension.")

        c = self.cfg.extension
        val = forward_extension_norm
        score = _linear_score(val, 0.1, c.forward_extension_good, c.forward_extension_excellent)
        rating = _rating_from_score(score)

        if val >= c.forward_extension_excellent:
            fb = f"Excellent forward extension ({val:.2f} torso-heights). Great penetration through the ball."
        elif val >= c.forward_extension_good:
            fb = f"Good extension ({val:.2f}). Continue pushing through the ball toward the target."
        else:
            fb = f"Limited forward extension ({val:.2f}). Extend your arm further through the ball after contact — aim for 2-3 feet of forward travel."

        return KPIResult(self.kpi_id, self.name, self.phase, val, self.unit, score, rating, fb)


class FollowThroughPathKPI(BaseKPI):
    """E5.2 – Follow-through path: upward-to-forward ratio."""
    kpi_id = "E5.2"
    name = "Follow-Through Path"
    phase = "extension"
    unit = "ratio"

    def evaluate(self, *, upward_forward_ratio: Optional[float] = None, **kw) -> KPIResult:
        if upward_forward_ratio is None:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "n/a",
                             "Could not measure follow-through path.")

        c = self.cfg.extension
        val = upward_forward_ratio
        score = _linear_score(val, c.followthrough_upward_forward_warning, c.followthrough_upward_forward_good_max, 0.5)
        rating = _rating_from_score(score)

        if val <= c.followthrough_upward_forward_good_max:
            fb = f"Good follow-through balance (up/fwd ratio: {val:.2f}). Forward extension before upward finish."
        else:
            fb = f"Follow-through goes up too quickly (ratio: {val:.2f}). Extend forward first, then let the racket rise naturally via shoulder rotation."

        return KPIResult(self.kpi_id, self.name, self.phase, val, self.unit, score, rating, fb)


# =====================================================================
# Phase 6: Balance
# =====================================================================

class OverallHeadStabilityKPI(BaseKPI):
    """B6.2 – Head vertical stability over the entire swing."""
    kpi_id = "B6.1"
    name = "Overall Head Stability"
    phase = "balance"
    unit = "normalised_std"

    def evaluate(self, *, head_y_std_norm: Optional[float] = None, **kw) -> KPIResult:
        if head_y_std_norm is None:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "n/a",
                             "Could not measure overall head stability.")

        c = self.cfg.balance
        val = head_y_std_norm
        score = _linear_score(val, c.head_vertical_stability_warning, c.head_vertical_stability_good, 0.01)
        rating = _rating_from_score(score)

        if val <= c.head_vertical_stability_good:
            fb = f"Excellent head stability throughout the swing (std: {val:.3f})."
        else:
            fb = f"Head bounces vertically during the swing (std: {val:.3f}). Maintain a consistent head height — avoid dipping or rising."

        return KPIResult(self.kpi_id, self.name, self.phase, val, self.unit, score, rating, fb)


class SpineConsistencyKPI(BaseKPI):
    """B6.2 – Spine angle consistency throughout the swing."""
    kpi_id = "B6.2"
    name = "Spine Consistency"
    phase = "balance"
    unit = "degrees_std"

    def evaluate(self, *, spine_angle_std: Optional[float] = None, **kw) -> KPIResult:
        if spine_angle_std is None:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "n/a",
                             "Could not measure spine consistency.")

        c = self.cfg.balance
        val = spine_angle_std
        score = _linear_score(val, c.spine_consistency_warning, c.spine_consistency_good, 2.0)
        rating = _rating_from_score(score)

        if val <= c.spine_consistency_good:
            fb = f"Consistent spine posture (std: {val:.1f}°). Good body control."
        else:
            fb = f"Spine angle varies too much (std: {val:.1f}°). Keep your torso stable — avoid hunching or leaning during the swing."

        return KPIResult(self.kpi_id, self.name, self.phase, val, self.unit, score, rating, fb)


# =====================================================================
# KPI Registry
# =====================================================================

ALL_KPIS = [
    ShoulderRotationKPI,
    KneeBendKPI,
    SpineAngleKPI,
    KineticChainSequenceKPI,
    HipShoulderSeparationKPI,
    HandPathLinearityKPI,
    ContactPointKPI,
    ElbowAngleAtContactKPI,
    BodyFreezeKPI,
    HeadStabilityAtContactKPI,
    ForwardExtensionKPI,
    FollowThroughPathKPI,
    OverallHeadStabilityKPI,
    SpineConsistencyKPI,
]
