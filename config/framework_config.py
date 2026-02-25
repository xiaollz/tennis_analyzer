"""Modern Forehand evaluation framework configuration — v3 (8-phase model).

All thresholds, weights, and scoring parameters are centralised here so that
tuning the evaluation criteria never requires touching analysis code.

v3 upgrade: 6-phase → 8-phase model aligned with biomechanics deep-dive.

Sources:
    - Dr. Brian Gordon (Type 3 forehand biomechanics)
    - Rick Macci (compact unit turn, elbow extension, "the flip")
    - Tennis Doctor (four non-negotiables, kinetic chain sequence)
    - Feel Tennis (modern forehand 8-step model)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict


# =====================================================================
# Phase 1: Unit Turn (一体化转体)
# =====================================================================

@dataclass(frozen=True)
class UnitTurnConfig:
    """Thresholds for the unit-turn phase."""

    # P1.1 Shoulder rotation (X-Factor) relative to hip line
    shoulder_rotation_excellent: float = 90.0   # degrees
    shoulder_rotation_good: float = 70.0
    shoulder_rotation_poor: float = 45.0

    # P1.2 Knee bend on the loaded leg
    knee_bend_excellent: float = 120.0   # degrees (smaller = more bend)
    knee_bend_good: float = 140.0
    knee_bend_poor: float = 155.0

    # P1.3 Spine angle (deviation from vertical)
    spine_lean_good_max: float = 15.0    # degrees from vertical
    spine_lean_warning: float = 25.0


# =====================================================================
# Phase 2: Slot Preparation (槽位准备)
# =====================================================================

@dataclass(frozen=True)
class SlotPrepConfig:
    """Thresholds for the slot-preparation phase.

    The "Slot" is the position where the elbow is behind the body,
    the racket head has dropped below the wrist, and the arm is relaxed.
    Rick Macci calls this "elbow back, racket drop".
    """

    # SP2.1 Elbow behind torso (normalised by shoulder width)
    elbow_behind_good: float = 0.3       # 30% of shoulder width behind
    elbow_behind_excellent: float = 0.5
    elbow_behind_poor: float = 0.1

    # SP2.2 Racket drop: wrist below elbow (normalised by torso height)
    # Positive = wrist is below elbow (good)
    racket_drop_good: float = 0.15       # 15% of torso height
    racket_drop_excellent: float = 0.25
    racket_drop_poor: float = 0.05

    # SP2.3 Elbow height relative to shoulder (normalised by torso height)
    # Slightly below shoulder is ideal (~0 to -0.1)
    elbow_height_good_min: float = -0.15
    elbow_height_good_max: float = 0.05


# =====================================================================
# Phase 3: Leg Drive & Hip Fire (蹬转与髋部启动)
# =====================================================================

@dataclass(frozen=True)
class LegDriveConfig:
    """Thresholds for the leg-drive and hip-fire phase."""

    # LD3.1 Ground force proxy: peak upward hip acceleration (px/s²)
    # Higher = more explosive leg drive
    ground_force_good: float = 3000.0     # px/s²
    ground_force_excellent: float = 6000.0
    ground_force_poor: float = 1000.0

    # LD3.2 Hip rotation speed (degrees/s)
    hip_rotation_speed_good: float = 200.0
    hip_rotation_speed_excellent: float = 400.0
    hip_rotation_speed_poor: float = 100.0


# =====================================================================
# Phase 4: Torso & Shoulder Pull (躯干与肩部牵引)
# =====================================================================

@dataclass(frozen=True)
class TorsoPullConfig:
    """Thresholds for the torso-pull phase.

    This is where the hip-shoulder separation (X-Factor stretch) peaks.
    """

    # TP4.1 Hip-shoulder separation (X-Factor stretch)
    hip_shoulder_separation_good: float = 25.0   # degrees
    hip_shoulder_separation_excellent: float = 40.0
    hip_shoulder_separation_poor: float = 10.0

    # TP4.2 Hip-shoulder timing delay (seconds)
    # Hip should peak rotation before shoulder
    timing_delay_good: float = 0.03      # at least 30ms
    timing_delay_excellent: float = 0.08  # 80ms is ideal
    timing_delay_poor: float = 0.0       # simultaneous = bad


# =====================================================================
# Phase 5: Lag & Elbow Drive (滞后与肘部驱动)
# =====================================================================

@dataclass(frozen=True)
class LagDriveConfig:
    """Thresholds for the lag and elbow-drive phase.

    This is where the elbow tucks close to the body and drives forward,
    while the racket head lags behind due to relaxed wrist.
    """

    # LE5.1 Elbow tuck distance (normalised by torso height)
    # Smaller = closer to body = better
    elbow_tuck_good: float = 0.25
    elbow_tuck_excellent: float = 0.15
    elbow_tuck_poor: float = 0.40

    # LE5.2 Hand path linearity (R² of linear fit through contact zone)
    hand_path_linearity_good: float = 0.85
    hand_path_linearity_excellent: float = 0.95


# =====================================================================
# Phase 6: Contact & SIR (击球与肩内旋)
# =====================================================================

@dataclass(frozen=True)
class ContactConfig:
    """Thresholds for the contact-point phase."""

    # C6.1 Contact point: wrist should be in front of hip (normalised by torso height)
    contact_forward_good_min: float = 0.3
    contact_forward_good_max: float = 0.8
    contact_forward_poor_min: float = 0.1

    # C6.2 Elbow angle at contact
    straight_arm_min: float = 165.0
    straight_arm_max: float = 180.0
    double_bend_min: float = 120.0
    double_bend_max: float = 145.0

    # C6.3 Body freeze: torso angular velocity at contact (degrees/s)
    body_freeze_good_max: float = 60.0
    body_freeze_warning: float = 120.0

    # C6.4 Head stability: nose displacement (normalised by torso height) over ±5 frames
    head_stability_good_max: float = 0.05
    head_stability_warning: float = 0.10

    # C6.5 SIR proxy: forearm angular velocity at contact (degrees/s)
    sir_proxy_good: float = 300.0
    sir_proxy_excellent: float = 600.0
    sir_proxy_poor: float = 100.0


# =====================================================================
# Phase 7: Windshield Wiper Follow-Through (雨刷式随挥)
# =====================================================================

@dataclass(frozen=True)
class WiperConfig:
    """Thresholds for the windshield-wiper follow-through phase."""

    # WW7.1 Forward extension distance (normalised by torso height)
    forward_extension_good: float = 0.4
    forward_extension_excellent: float = 0.6

    # WW7.2 Wiper sweep angle (degrees)
    wiper_sweep_good: float = 60.0
    wiper_sweep_excellent: float = 90.0
    wiper_sweep_poor: float = 30.0

    # Post-contact analysis window (seconds)
    post_contact_window_s: float = 0.30


# =====================================================================
# Phase 8: Deceleration & Balance (减速与平衡)
# =====================================================================

@dataclass(frozen=True)
class BalanceConfig:
    """Thresholds for balance and recovery."""

    # B8.1 Head vertical stability (normalised std-dev over entire swing)
    head_vertical_stability_good: float = 0.03
    head_vertical_stability_warning: float = 0.06

    # B8.2 Spine angle consistency (std-dev in degrees)
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
# Scoring Weights (8-phase model)
# =====================================================================

@dataclass(frozen=True)
class ScoringWeights:
    """Weights for computing the overall Modern Forehand Score.

    8-phase model:
        1. Unit Turn          10%
        2. Slot Preparation   10%
        3. Leg Drive          15%
        4. Torso Pull         15%
        5. Lag & Elbow Drive  10%
        6. Contact & SIR      20%
        7. Wiper Follow-Thru  10%
        8. Balance            10%
    """

    unit_turn: float = 0.10
    slot_prep: float = 0.10
    leg_drive: float = 0.15
    torso_pull: float = 0.15
    lag_drive: float = 0.10
    contact: float = 0.20
    wiper: float = 0.10
    balance: float = 0.10

    def as_dict(self) -> Dict[str, float]:
        return {
            "unit_turn": self.unit_turn,
            "slot_prep": self.slot_prep,
            "leg_drive": self.leg_drive,
            "torso_pull": self.torso_pull,
            "lag_drive": self.lag_drive,
            "contact": self.contact,
            "wiper": self.wiper,
            "balance": self.balance,
        }


# =====================================================================
# Anthropometric References
# =====================================================================

@dataclass(frozen=True)
class AnthropometricConfig:
    """Reference body dimensions for pixel-to-cm conversion."""

    shoulder_width_cm: float = 41.0
    torso_height_cm: float = 55.0
    arm_length_cm: float = 65.0
    min_shoulder_width_px: float = 12.0
    min_torso_height_px: float = 20.0


# =====================================================================
# Master Configuration
# =====================================================================

@dataclass(frozen=True)
class FrameworkConfig:
    """Top-level configuration aggregating all sub-configs (v3, 8-phase)."""

    unit_turn: UnitTurnConfig = field(default_factory=UnitTurnConfig)
    slot_prep: SlotPrepConfig = field(default_factory=SlotPrepConfig)
    leg_drive: LegDriveConfig = field(default_factory=LegDriveConfig)
    torso_pull: TorsoPullConfig = field(default_factory=TorsoPullConfig)
    lag_drive: LagDriveConfig = field(default_factory=LagDriveConfig)
    contact: ContactConfig = field(default_factory=ContactConfig)
    wiper: WiperConfig = field(default_factory=WiperConfig)
    balance: BalanceConfig = field(default_factory=BalanceConfig)
    impact_detection: ImpactDetectionConfig = field(default_factory=ImpactDetectionConfig)
    scoring: ScoringWeights = field(default_factory=ScoringWeights)
    anthropometry: AnthropometricConfig = field(default_factory=AnthropometricConfig)

    # ── Legacy compatibility aliases ──
    @property
    def preparation(self):
        """Alias for backward compatibility."""
        return self.unit_turn

    @property
    def kinetic_chain(self):
        """Alias for backward compatibility."""
        return self.torso_pull

    @property
    def extension(self):
        """Alias for backward compatibility."""
        return self.wiper


# Default configuration instance
DEFAULT_CONFIG = FrameworkConfig()
