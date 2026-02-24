"""Stroke-specific monitor profiles.

These profiles map the `tennis_coach` skill guidance to concrete thresholds.
They are intentionally small and opinionated; fine-tune per camera angle later.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StrokeProfile:
    """Thresholds for a stroke type."""

    name: str

    # Contact point (in front of body). Skill baseline: 20-40cm for forehand.
    contact_good_min_cm: float = 20.0
    contact_good_max_cm: float = 40.0
    contact_ok_min_cm: float = 10.0
    contact_good_threshold_px: float = 50.0
    contact_ok_threshold_px: float = 0.0

    # Contact zone (forward + upward after impact).
    contact_zone_good_forward_cm: float = 15.0
    contact_zone_good_upward_cm: float = 10.0
    contact_zone_good_forward_px: float = 20.0
    contact_zone_good_upward_px: float = 20.0
    # How long after contact we evaluate "contact zone / follow-through".
    # Broddfelt-style cues focus on the early follow-through, not the final finish.
    # 0.25s ~= 8 frames @30fps.
    contact_zone_window_s: float = 0.25

    # Weight transfer (combined ankle lift + hip forward). Heuristic thresholds.
    weight_transfer_good_cm: float = 8.0
    weight_transfer_ok_cm: float = 4.0
    weight_transfer_good_px: float = 30.0
    weight_transfer_ok_px: float = 10.0

    # Unit turn timing: X-Factor max should happen at least this long before contact.
    unit_turn_early_s: float = 0.25


FOREHAND_PROFILE = StrokeProfile(
    name="forehand",
    contact_good_min_cm=20.0,
    contact_good_max_cm=40.0,
    contact_ok_min_cm=10.0,
    unit_turn_early_s=0.25,
)

# One-handed backhand typically needs contact further in front than forehand.
BACKHAND_1H_PROFILE = StrokeProfile(
    name="backhand_1h",
    contact_good_min_cm=25.0,
    contact_good_max_cm=55.0,
    contact_ok_min_cm=15.0,
    unit_turn_early_s=0.25,
)

# Serve is special: Big3 "contact point" as wrist-vs-hip doesn't apply.
SERVE_PROFILE = StrokeProfile(
    name="serve",
)
