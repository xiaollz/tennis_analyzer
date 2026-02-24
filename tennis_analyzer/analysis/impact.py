"""Impact (contact) detection utilities.

This module centralizes "impact frame" detection so that all monitors use the
same logic and thresholds.

Design goals:
- Work in real-time (single pass over frames)
- Be robust to noise by detecting a *speed peak* (accel -> decel)
- Prefer scale-aware thresholds (normalize by shoulder width) when possible
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Deque, Optional, Tuple, List

import numpy as np

from ..config.keypoints import KEYPOINT_NAMES


@dataclass(frozen=True)
class ImpactEvent:
    """An estimated impact (contact) event."""

    # Estimated contact frame index (0-based, aligned to the detected speed peak).
    impact_frame_idx: int

    # Frame index when the detector fired (typically impact_frame_idx + 1).
    trigger_frame_idx: int

    # Wrist speed at the detected peak.
    peak_speed_px_s: float

    # Same speed normalized by shoulder width (shoulder-widths per second), if available.
    peak_speed_sw_s: Optional[float]

    # Unit direction of the wrist velocity at the detected peak (dx, dy in image coords).
    # Useful as a best-effort "forward swing" axis for other monitors.
    peak_velocity_unit: Tuple[float, float]

    # Optional: inferred bounce frame index (audio onset), when available in
    # hybrid (two-pass) impact detection. Best-effort only.
    bounce_frame_idx: Optional[int] = None


class WristSpeedImpactDetector:
    """Detect impact as a wrist-speed local maximum (peak) with cooldown.

    Notes on indexing:
    - At frame t we compute speed from wrist(t-1) -> wrist(t).
    - A local maximum at speed(t-1) is detected when speed(t-2) < speed(t-1) > speed(t).
    - When we detect such a peak while processing frame t, we report:
        impact_frame_idx = t-1
        trigger_frame_idx = t
    """

    def __init__(
        self,
        fps: float = 30.0,
        is_right_handed: bool = True,
        *,
        min_wrist_conf: float = 0.5,
        cooldown_frames: int = 18,
        history_size: int = 7,
        # Thresholds (use normalized when possible, else px/s).
        min_peak_speed_sw_s: float = 3.0,
        min_peak_speed_px_s: float = 450.0,
        # When using normalized speeds, still require a minimal absolute px/s
        # to avoid triggering on keypoint jitter for tiny players.
        min_peak_speed_px_s_floor: float = 80.0,
        # Require the peak to be clearly above recent baseline.
        peak_over_baseline_ratio: float = 1.25,
        # If we lose the wrist for too long, reset velocity to avoid giant jumps.
        max_frame_gap: int = 2,
        # Guard against bad shoulder detection producing tiny shoulder widths
        # (which would explode normalized speeds).
        min_shoulder_width_px: float = 12.0,
        # Use a rolling median of shoulder width to stabilize normalization and
        # allow side-view/far-camera footage where shoulders are small in pixels.
        shoulder_width_history_size: int = 60,
        shoulder_width_history_min_samples: int = 5,
    ):
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.is_right_handed = is_right_handed

        self.min_wrist_conf = float(min_wrist_conf)
        self.cooldown_frames = int(cooldown_frames)
        self.history_size = int(history_size)
        self.min_peak_speed_sw_s = float(min_peak_speed_sw_s)
        self.min_peak_speed_px_s = float(min_peak_speed_px_s)
        self.min_peak_speed_px_s_floor = float(min_peak_speed_px_s_floor)
        self.peak_over_baseline_ratio = float(peak_over_baseline_ratio)
        self.max_frame_gap = int(max_frame_gap)
        self.min_shoulder_width_px = float(min_shoulder_width_px)
        self.shoulder_width_history_size = int(shoulder_width_history_size)
        self.shoulder_width_history_min_samples = int(shoulder_width_history_min_samples)

        self.wrist_idx = (
            KEYPOINT_NAMES["right_wrist"] if is_right_handed else KEYPOINT_NAMES["left_wrist"]
        )

        self._prev_wrist: Optional[np.ndarray] = None
        self._prev_frame_idx: Optional[int] = None

        self._speed_px_s: Deque[float] = deque(maxlen=self.history_size)
        self._speed_sw_s: Deque[Optional[float]] = deque(maxlen=self.history_size)
        self._vel_px_s: Deque[np.ndarray] = deque(maxlen=self.history_size)
        self._sw_px_ref: Deque[Optional[float]] = deque(maxlen=self.history_size)

        # Longer rolling shoulder-width history for scale stabilization.
        self._sw_px_hist: Deque[float] = deque(maxlen=self.shoulder_width_history_size)

        self._cooldown_left = 0

    def reset(self) -> None:
        self._prev_wrist = None
        self._prev_frame_idx = None
        self._speed_px_s.clear()
        self._speed_sw_s.clear()
        self._vel_px_s.clear()
        self._sw_px_ref.clear()
        self._sw_px_hist.clear()
        self._cooldown_left = 0

    def update(
        self,
        frame_idx: int,
        keypoints: np.ndarray,
        confidence: np.ndarray,
    ) -> Tuple[Optional[ImpactEvent], float]:
        """Update detector with the current frame pose.

        Returns:
            (event, wrist_speed_px_s)
            event is None when no impact is detected.
        """
        if self._cooldown_left > 0:
            self._cooldown_left -= 1

        if confidence[self.wrist_idx] < self.min_wrist_conf:
            # Don't update prev_wrist; missing data would create a giant jump later.
            return None, 0.0

        wrist = np.asarray(keypoints[self.wrist_idx], dtype=np.float32)

        # Reset if frames are not consecutive (or near-consecutive).
        if self._prev_frame_idx is not None:
            gap = int(frame_idx) - int(self._prev_frame_idx)
            if gap <= 0:
                # Non-monotonic inputs: reset state.
                self._prev_wrist = wrist.copy()
                self._prev_frame_idx = int(frame_idx)
                self._speed_px_s.clear()
                self._speed_sw_s.clear()
                self._vel_px_s.clear()
                return None, 0.0
            if gap > self.max_frame_gap:
                self._prev_wrist = wrist.copy()
                self._prev_frame_idx = int(frame_idx)
                self._speed_px_s.clear()
                self._speed_sw_s.clear()
                self._vel_px_s.clear()
                return None, 0.0

        if self._prev_wrist is None:
            self._prev_wrist = wrist.copy()
            self._prev_frame_idx = int(frame_idx)
            return None, 0.0

        # Speed/velocity (px/s). For non-consecutive frames we use the actual frame gap.
        prev_idx = int(self._prev_frame_idx) if self._prev_frame_idx is not None else int(frame_idx) - 1
        gap = max(1, int(frame_idx) - prev_idx)
        dt_s = gap / self.fps

        delta_px = wrist - self._prev_wrist
        speed_px_s = float(np.linalg.norm(delta_px)) / max(1e-6, dt_s)
        vel_px_s = delta_px / max(1e-6, dt_s)

        # Shoulder-width normalized speed (sw/s), using a rolling median shoulder width
        # to handle far-camera side views and temporary shoulder occlusions.
        sw_px_ref = self._shoulder_width_px_estimate(keypoints, confidence)
        speed_sw_s = (speed_px_s / sw_px_ref) if (sw_px_ref is not None and sw_px_ref > 1e-6) else None

        self._speed_px_s.append(speed_px_s)
        self._speed_sw_s.append(speed_sw_s)
        self._vel_px_s.append(vel_px_s.astype(np.float32))
        self._sw_px_ref.append(sw_px_ref)

        self._prev_wrist = wrist.copy()
        self._prev_frame_idx = int(frame_idx)

        event = self._maybe_detect_peak(trigger_frame_idx=int(frame_idx))
        return event, speed_px_s

    def _shoulder_width_px_instant(
        self, keypoints: np.ndarray, confidence: np.ndarray, min_conf: float = 0.35
    ) -> Optional[float]:
        l = KEYPOINT_NAMES["left_shoulder"]
        r = KEYPOINT_NAMES["right_shoulder"]
        if confidence[l] < min_conf or confidence[r] < min_conf:
            return None
        sw = float(np.linalg.norm(np.asarray(keypoints[r]) - np.asarray(keypoints[l])))
        if sw < self.min_shoulder_width_px:
            return None
        return sw

    def _shoulder_width_px_estimate(self, keypoints: np.ndarray, confidence: np.ndarray) -> Optional[float]:
        """Return a stable shoulder-width estimate in pixels.

        Uses a rolling median when enough samples exist. This lets us normalize
        wrist speed even when the player is small in the frame (side view) or
        shoulders are briefly occluded.
        """
        sw = self._shoulder_width_px_instant(keypoints, confidence)
        if sw is not None:
            if len(self._sw_px_hist) >= self.shoulder_width_history_min_samples:
                med = float(np.median(np.asarray(self._sw_px_hist, dtype=np.float32)))
                # Ignore extreme outliers (pose glitches).
                if med > 1e-6:
                    if sw < 0.4 * med or sw > 2.5 * med:
                        sw = None
            if sw is not None:
                self._sw_px_hist.append(float(sw))

        if not self._sw_px_hist:
            return sw
        # Median is more stable than the instantaneous width.
        return float(np.median(np.asarray(self._sw_px_hist, dtype=np.float32)))

    def _maybe_detect_peak(self, trigger_frame_idx: int) -> Optional[ImpactEvent]:
        if self._cooldown_left > 0:
            return None

        # Need at least 3 samples to see a peak at the middle one.
        if len(self._speed_px_s) < 3:
            return None

        # Prefer normalized speeds if available for the latest samples.
        # If any of the last 3 are None, fall back to px/s.
        use_norm = (
            self._speed_sw_s[-1] is not None
            and self._speed_sw_s[-2] is not None
            and self._speed_sw_s[-3] is not None
        )

        if use_norm:
            # NOTE: `_speed_sw_s` may contain None for earlier frames where shoulder
            # width is not reliable. We only require the last 3 samples to be
            # non-None for peak detection; baseline stats should ignore Nones.
            speeds = [
                float(x) if x is not None else float("nan")
                for x in self._speed_sw_s
            ]
            peak_threshold = self.min_peak_speed_sw_s
        else:
            speeds = list(self._speed_px_s)
            peak_threshold = self.min_peak_speed_px_s

        # Peak candidate is the previous sample (index -2), because we only know it's a
        # peak once we see the current sample (-1) drop.
        prev_prev = speeds[-3]
        prev = speeds[-2]
        curr = speeds[-1]

        is_local_max = prev > prev_prev and prev > curr
        if not is_local_max:
            return None

        peak_px_s = float(self._speed_px_s[-2])
        if use_norm:
            # Use a small absolute floor to prevent false positives from jitter.
            if peak_px_s < self.min_peak_speed_px_s_floor:
                return None
        else:
            # When we can't normalize, fall back to a conservative absolute px/s threshold.
            if peak_px_s < self.min_peak_speed_px_s:
                return None

        if prev < peak_threshold:
            return None

        # Baseline from earlier samples (exclude the last 2 samples).
        baseline_window = speeds[:-2]
        if use_norm:
            baseline_vals = [v for v in baseline_window if np.isfinite(v)]
        else:
            baseline_vals = baseline_window
        baseline = float(np.median(baseline_vals)) if baseline_vals else 0.0
        if baseline > 1e-6 and (prev / baseline) < self.peak_over_baseline_ratio:
            return None

        # Fire!
        self._cooldown_left = self.cooldown_frames

        impact_frame_idx = trigger_frame_idx - 1

        # Always store px/s peak from the px/s deque (aligned by index).
        peak_sw_s = float(self._speed_sw_s[-2]) if self._speed_sw_s[-2] is not None else None
        peak_vel = np.asarray(self._vel_px_s[-2], dtype=np.float32) if len(self._vel_px_s) >= 2 else np.zeros(2, dtype=np.float32)
        peak_vel_norm = float(np.linalg.norm(peak_vel))
        if peak_vel_norm > 1e-6:
            peak_vel_unit = (float(peak_vel[0] / peak_vel_norm), float(peak_vel[1] / peak_vel_norm))
        else:
            peak_vel_unit = (0.0, 0.0)

        return ImpactEvent(
            impact_frame_idx=impact_frame_idx,
            trigger_frame_idx=trigger_frame_idx,
            peak_speed_px_s=peak_px_s,
            peak_speed_sw_s=peak_sw_s,
            peak_velocity_unit=peak_vel_unit,
        )
