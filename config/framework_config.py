"""Modern Forehand evaluation framework configuration.

All thresholds, weights, and scoring parameters are centralised here so that
tuning the evaluation criteria never requires touching analysis code.

Sources:
    - Dr. Brian Gordon (Type 3 forehand biomechanics)
    - Rick Macci (compact unit turn, elbow extension, "the flip")
    - Tennis Doctor (four non-negotiables, kinetic chain sequence)
    - Feel Tennis (modern forehand 8-step model)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple


# =====================================================================
# Phase 1: Preparation & Unit Turn
# =====================================================================

@dataclass(frozen=True)
class PreparationConfig:
    """Thresholds for the preparation / unit-turn phase."""

    # P1.1 Shoulder rotation (X-Factor) relative to hip line
    shoulder_rotation_excellent: float = 90.0   # degrees
    shoulder_rotation_good: float = 70.0
    shoulder_rotation_poor: float = 45.0

    # P1.4 Knee bend on the loaded leg
    knee_bend_excellent: float = 120.0   # degrees (smaller = more bend)
    knee_bend_good: float = 140.0
    knee_bend_poor: float = 155.0

    # P1.3 Spine angle (deviation from vertical)
    spine_lean_good_max: float = 15.0    # degrees from vertical
    spine_lean_warning: float = 25.0

    # P1.6 Elbow elevation: elbow-y should be above wrist-y
    elbow_above_wrist_required: bool = True


# =====================================================================
# Phase 2: Loading & Racket Drop
# =====================================================================

@dataclass(frozen=True)
class LoadingConfig:
    """Thresholds for the loading / lag phase."""

    # L2.1 Wrist laid-back angle (forearm-to-hand)
    wrist_layback_good_min: float = 70.0   # degrees
    wrist_layback_good_max: float = 120.0

    # L2.2 Elbow-high-hand-low: vertical difference (normalised by torso height)
    elbow_hand_drop_good: float = 0.15     # elbow at least 15% torso-height above wrist


# =====================================================================
# Phase 3: Forward Swing & Kinetic Chain
# =====================================================================

@dataclass(frozen=True)
class KineticChainConfig:
    """Thresholds for kinetic-chain sequencing."""

    # KC3.1 Acceptable time window for sequential peak ordering (seconds)
    # hip_peak -> shoulder_peak -> elbow_peak -> wrist_peak
    max_segment_delay_s: float = 0.20      # each segment should peak within 200 ms of prior
    min_segment_delay_s: float = 0.01      # must be > 0 (truly sequential)

    # KC3.2 Hip-shoulder separation (X-Factor stretch)
    hip_shoulder_separation_good: float = 25.0   # degrees
    hip_shoulder_separation_excellent: float = 40.0

    # KC3.4 Hand path linearity (R² of linear fit through contact zone)
    hand_path_linearity_good: float = 0.85
    hand_path_linearity_excellent: float = 0.95


# =====================================================================
# Phase 4: Contact Point
# =====================================================================

@dataclass(frozen=True)
class ContactConfig:
    """Thresholds for the contact-point phase."""

    # C4.1 Contact point: wrist should be in front of hip (normalised by torso height)
    contact_forward_good_min: float = 0.3   # 30% of torso height in front
    contact_forward_good_max: float = 0.8
    contact_forward_poor_min: float = 0.1

    # C4.2 Elbow angle at contact
    # Straight-arm (Gordon Type 3): 165-180°
    straight_arm_min: float = 165.0
    straight_arm_max: float = 180.0
    # Double-bend: 120-145°
    double_bend_min: float = 120.0
    double_bend_max: float = 145.0

    # C4.3 Body freeze: torso angular velocity at contact (degrees/s)
    body_freeze_good_max: float = 60.0     # near-zero rotation
    body_freeze_warning: float = 120.0

    # C4.4 Head stability: nose displacement (normalised by torso height) over ±5 frames
    head_stability_good_max: float = 0.05
    head_stability_warning: float = 0.10

    # C4.5 Wrist stability: angle change through contact zone (degrees)
    wrist_stability_good_max: float = 15.0
    wrist_stability_warning: float = 30.0


# =====================================================================
# Phase 5: Extension & Follow-Through
# =====================================================================

@dataclass(frozen=True)
class ExtensionConfig:
    """Thresholds for the extension / follow-through phase."""

    # E5.1 Forward extension distance (normalised by torso height)
    forward_extension_good: float = 0.4
    forward_extension_excellent: float = 0.6

    # E5.4 Follow-through: upward-to-forward ratio (should be moderate)
    followthrough_upward_forward_good_max: float = 1.5
    followthrough_upward_forward_warning: float = 2.5

    # Post-contact analysis window (seconds)
    post_contact_window_s: float = 0.30


# =====================================================================
# Phase 6: Balance & Recovery
# =====================================================================

@dataclass(frozen=True)
class BalanceConfig:
    """Thresholds for balance and recovery."""

    # B6.1 Head vertical stability (normalised std-dev over entire swing)
    head_vertical_stability_good: float = 0.03
    head_vertical_stability_warning: float = 0.06

    # B6.2 Spine angle consistency (std-dev in degrees)
    spine_consistency_good: float = 5.0
    spine_consistency_warning: float = 10.0


# =====================================================================
# Impact Detection
# =====================================================================

@dataclass(frozen=True)
class ImpactDetectionConfig:
    """Parameters for automatic impact-frame detection."""

    min_wrist_conf: float = 0.5
    cooldown_frames: int = 25
    min_peak_speed_sw_s: float = 3.0
    min_peak_speed_px_s: float = 450.0
    min_peak_speed_px_s_floor: float = 80.0
    peak_over_baseline_ratio: float = 1.25
    max_frame_gap: int = 2


# =====================================================================
# Scoring Weights
# =====================================================================

@dataclass(frozen=True)
class ScoringWeights:
    """Weights for computing the overall Modern Forehand Score."""

    preparation: float = 0.15
    loading: float = 0.10
    kinetic_chain: float = 0.20
    contact: float = 0.25
    extension: float = 0.15
    balance: float = 0.15

    def as_dict(self) -> Dict[str, float]:
        return {
            "preparation": self.preparation,
            "loading": self.loading,
            "kinetic_chain": self.kinetic_chain,
            "contact": self.contact,
            "extension": self.extension,
            "balance": self.balance,
        }


# =====================================================================
# Anthropometric References
# =====================================================================

@dataclass(frozen=True)
class AnthropometricConfig:
    """Reference body dimensions for pixel-to-cm conversion."""

    shoulder_width_cm: float = 41.0
    torso_height_cm: float = 55.0    # shoulder-center to hip-center
    arm_length_cm: float = 65.0
    min_shoulder_width_px: float = 12.0
    min_torso_height_px: float = 20.0


# =====================================================================
# Master Configuration
# =====================================================================

@dataclass(frozen=True)
class FrameworkConfig:
    """Top-level configuration aggregating all sub-configs."""

    preparation: PreparationConfig = field(default_factory=PreparationConfig)
    loading: LoadingConfig = field(default_factory=LoadingConfig)
    kinetic_chain: KineticChainConfig = field(default_factory=KineticChainConfig)
    contact: ContactConfig = field(default_factory=ContactConfig)
    extension: ExtensionConfig = field(default_factory=ExtensionConfig)
    balance: BalanceConfig = field(default_factory=BalanceConfig)
    impact_detection: ImpactDetectionConfig = field(default_factory=ImpactDetectionConfig)
    scoring: ScoringWeights = field(default_factory=ScoringWeights)
    anthropometry: AnthropometricConfig = field(default_factory=AnthropometricConfig)


# Default configuration instance
DEFAULT_CONFIG = FrameworkConfig()
