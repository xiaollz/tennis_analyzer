"""Fault-tolerant forehand evaluation configuration.

This version follows the principle hierarchy from
``The Fault Tolerant Forehand`` and only keeps indicators that can be
estimated with reasonable stability from body-only COCO-17 keypoints.

Core scoring layers:
    1. Unit turn / setup
    2. Hip-led chain
    3. Contact structure
    4. Out / through path
    5. Stability
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class UnitTurnConfig:
    """Preparation quality before the forward swing."""

    shoulder_rotation_excellent: float = 55.0
    shoulder_rotation_good: float = 35.0
    shoulder_rotation_poor: float = 18.0

    knee_bend_excellent: float = 118.0
    knee_bend_good: float = 138.0
    knee_bend_poor: float = 156.0

    spine_lean_good_max: float = 16.0
    spine_lean_warning: float = 28.0


@dataclass(frozen=True)
class ChainConfig:
    """Hip-led chain quality without relying on absolute swing speed."""

    hip_shoulder_separation_excellent: float = 32.0
    hip_shoulder_separation_good: float = 20.0
    hip_shoulder_separation_poor: float = 8.0

    hip_lead_timing_excellent: float = 0.050
    hip_lead_timing_good: float = 0.000
    hip_lead_timing_poor: float = -0.035


@dataclass(frozen=True)
class ContactConfig:
    """Contact geometry and arm structure."""

    contact_forward_good_min: float = 0.18
    contact_forward_good_max: float = 0.90
    contact_forward_poor_min: float = 0.04
    contact_forward_excellent_min: float = 0.25
    contact_forward_excellent_max: float = 0.75

    contact_spacing_good_min: float = 0.32
    contact_spacing_good_max: float = 0.75
    contact_spacing_poor_min: float = 0.12
    contact_spacing_poor_max: float = 0.95
    contact_spacing_excellent_min: float = 0.42
    contact_spacing_excellent_max: float = 0.65

    elbow_angle_good_min: float = 115.0
    elbow_angle_good_max: float = 165.0
    elbow_angle_poor_min: float = 95.0
    elbow_angle_poor_max: float = 178.0
    elbow_angle_excellent_min: float = 125.0
    elbow_angle_excellent_max: float = 155.0


@dataclass(frozen=True)
class ThroughConfig:
    """How the hand travels through and beyond contact."""

    hand_path_linearity_poor: float = 0.60
    hand_path_linearity_good: float = 0.82
    hand_path_linearity_excellent: float = 0.94

    forward_extension_poor: float = 0.08
    forward_extension_good: float = 0.25
    forward_extension_excellent: float = 0.40

    outside_extension_poor: float = 0.03
    outside_extension_good: float = 0.12
    outside_extension_excellent: float = 0.22

    post_contact_window_s: float = 0.25


@dataclass(frozen=True)
class StabilityConfig:
    """Head and trunk stability proxies."""

    head_contact_good: float = 0.055
    head_contact_warning: float = 0.100

    head_vertical_stability_good: float = 0.070
    head_vertical_stability_warning: float = 0.140

    spine_consistency_good: float = 4.5
    spine_consistency_warning: float = 8.0


@dataclass(frozen=True)
class ImpactDetectionConfig:
    """Parameters for automatic impact-frame detection."""

    min_wrist_conf: float = 0.5
    cooldown_frames: int = 25
    min_peak_speed_sw_s: float = 2.5
    min_peak_speed_px_s: float = 180.0
    min_peak_speed_px_s_floor: float = 80.0
    peak_over_baseline_ratio: float = 1.25
    max_frame_gap: int = 2


@dataclass(frozen=True)
class ScoringWeights:
    """Phase weights for the fault-tolerant forehand model."""

    unit_turn: float = 0.20
    chain: float = 0.15
    contact: float = 0.30
    through: float = 0.20
    stability: float = 0.15

    def as_dict(self) -> Dict[str, float]:
        return {
            "unit_turn": self.unit_turn,
            "chain": self.chain,
            "contact": self.contact,
            "through": self.through,
            "stability": self.stability,
        }


@dataclass(frozen=True)
class AnthropometricConfig:
    shoulder_width_cm: float = 41.0
    torso_height_cm: float = 55.0
    arm_length_cm: float = 65.0
    min_shoulder_width_px: float = 12.0
    min_torso_height_px: float = 20.0


@dataclass(frozen=True)
class FrameworkConfig:
    """Top-level configuration for the forehand evaluator."""

    unit_turn: UnitTurnConfig = field(default_factory=UnitTurnConfig)
    chain: ChainConfig = field(default_factory=ChainConfig)
    contact: ContactConfig = field(default_factory=ContactConfig)
    through: ThroughConfig = field(default_factory=ThroughConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    impact_detection: ImpactDetectionConfig = field(default_factory=ImpactDetectionConfig)
    scoring: ScoringWeights = field(default_factory=ScoringWeights)
    anthropometry: AnthropometricConfig = field(default_factory=AnthropometricConfig)

    @property
    def preparation(self):
        return self.unit_turn

    @property
    def kinetic_chain(self):
        return self.chain

    @property
    def extension(self):
        return self.through

    @property
    def balance(self):
        return self.stability


DEFAULT_CONFIG = FrameworkConfig()
