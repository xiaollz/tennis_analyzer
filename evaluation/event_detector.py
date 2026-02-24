"""Automatic swing-event detection.

Detects key events in a forehand swing:
    - **Impact frame**: the moment of ball contact, identified as a wrist-speed
      local maximum (peak-then-decelerate pattern).
    - **Preparation start**: estimated as the frame where shoulder rotation
      begins increasing significantly before impact.
    - **Follow-through end**: estimated as the frame where wrist speed drops
      below a baseline after impact.

The impact detector is ported and improved from the v1 ``WristSpeedImpactDetector``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Optional, Tuple, List, Deque

import numpy as np

from config.keypoints import KEYPOINT_NAMES
from config.framework_config import ImpactDetectionConfig, DEFAULT_CONFIG


@dataclass(frozen=True)
class ImpactEvent:
    """Detected ball-contact event."""
    impact_frame_idx: int
    trigger_frame_idx: int
    peak_speed_px_s: float
    peak_speed_sw_s: Optional[float]
    peak_velocity_unit: Tuple[float, float]


@dataclass
class SwingEvent:
    """A complete swing event with preparation, impact, and follow-through."""
    prep_start_frame: Optional[int] = None
    impact_frame: Optional[int] = None
    followthrough_end_frame: Optional[int] = None
    impact_event: Optional[ImpactEvent] = None


class ImpactDetector:
    """Real-time impact detection via wrist-speed peak analysis.

    At frame *t* we compute speed from wrist(t-1) → wrist(t).
    A local maximum at speed(t-1) is detected when speed(t-2) < speed(t-1) > speed(t).
    When detected at frame *t*, we report impact_frame_idx = t-1.
    """

    def __init__(
        self,
        fps: float = 30.0,
        is_right_handed: bool = True,
        cfg: Optional[ImpactDetectionConfig] = None,
    ):
        c = cfg or DEFAULT_CONFIG.impact_detection
        self.fps = max(fps, 1.0)
        self.is_right_handed = is_right_handed

        self.min_wrist_conf = c.min_wrist_conf
        self.cooldown_frames = c.cooldown_frames
        self.min_peak_speed_sw_s = c.min_peak_speed_sw_s
        self.min_peak_speed_px_s = c.min_peak_speed_px_s
        self.min_peak_speed_px_s_floor = c.min_peak_speed_px_s_floor
        self.peak_over_baseline_ratio = c.peak_over_baseline_ratio
        self.max_frame_gap = c.max_frame_gap

        self.wrist_idx = KEYPOINT_NAMES["right_wrist" if is_right_handed else "left_wrist"]

        self._prev_wrist: Optional[np.ndarray] = None
        self._prev_frame_idx: Optional[int] = None
        self._history_size = 7

        self._speed_px_s: Deque[float] = deque(maxlen=self._history_size)
        self._speed_sw_s: Deque[Optional[float]] = deque(maxlen=self._history_size)
        self._vel_px_s: Deque[np.ndarray] = deque(maxlen=self._history_size)

        self._sw_px_hist: Deque[float] = deque(maxlen=60)
        self._cooldown_left = 0

        # Accumulated events
        self.events: List[ImpactEvent] = []

    def reset(self):
        self._prev_wrist = None
        self._prev_frame_idx = None
        self._speed_px_s.clear()
        self._speed_sw_s.clear()
        self._vel_px_s.clear()
        self._sw_px_hist.clear()
        self._cooldown_left = 0
        self.events.clear()

    def update(
        self,
        frame_idx: int,
        keypoints: np.ndarray,
        confidence: np.ndarray,
    ) -> Tuple[Optional[ImpactEvent], float]:
        """Process one frame. Returns (event_or_None, current_wrist_speed_px_s)."""
        if self._cooldown_left > 0:
            self._cooldown_left -= 1

        if confidence[self.wrist_idx] < self.min_wrist_conf:
            return None, 0.0

        wrist = np.asarray(keypoints[self.wrist_idx], dtype=np.float64)

        # Reset on non-consecutive frames
        if self._prev_frame_idx is not None:
            gap = int(frame_idx) - int(self._prev_frame_idx)
            if gap <= 0 or gap > self.max_frame_gap:
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

        prev_idx = int(self._prev_frame_idx) if self._prev_frame_idx is not None else int(frame_idx) - 1
        gap = max(1, int(frame_idx) - prev_idx)
        dt_s = gap / self.fps

        delta = wrist - self._prev_wrist
        speed_px_s = float(np.linalg.norm(delta)) / max(1e-6, dt_s)
        vel_px_s = delta / max(1e-6, dt_s)

        sw_px = self._shoulder_width_estimate(keypoints, confidence)
        speed_sw_s = (speed_px_s / sw_px) if (sw_px is not None and sw_px > 1e-6) else None

        self._speed_px_s.append(speed_px_s)
        self._speed_sw_s.append(speed_sw_s)
        self._vel_px_s.append(vel_px_s.astype(np.float64))

        self._prev_wrist = wrist.copy()
        self._prev_frame_idx = int(frame_idx)

        event = self._maybe_detect_peak(int(frame_idx))
        if event is not None:
            self.events.append(event)
        return event, speed_px_s

    def _shoulder_width_estimate(self, keypoints: np.ndarray, confidence: np.ndarray) -> Optional[float]:
        ls = KEYPOINT_NAMES["left_shoulder"]
        rs = KEYPOINT_NAMES["right_shoulder"]
        if confidence[ls] < 0.35 or confidence[rs] < 0.35:
            sw = None
        else:
            sw = float(np.linalg.norm(keypoints[rs] - keypoints[ls]))
            if sw < 12.0:
                sw = None

        if sw is not None:
            if len(self._sw_px_hist) >= 5:
                med = float(np.median(list(self._sw_px_hist)))
                if med > 1e-6 and (sw < 0.4 * med or sw > 2.5 * med):
                    sw = None
            if sw is not None:
                self._sw_px_hist.append(sw)

        if not self._sw_px_hist:
            return sw
        return float(np.median(list(self._sw_px_hist)))

    def _maybe_detect_peak(self, trigger_frame_idx: int) -> Optional[ImpactEvent]:
        if self._cooldown_left > 0 or len(self._speed_px_s) < 3:
            return None

        use_norm = all(s is not None for s in list(self._speed_sw_s)[-3:])

        if use_norm:
            speeds = [float(x) if x is not None else float("nan") for x in self._speed_sw_s]
            peak_threshold = self.min_peak_speed_sw_s
        else:
            speeds = list(self._speed_px_s)
            peak_threshold = self.min_peak_speed_px_s

        prev_prev, prev, curr = speeds[-3], speeds[-2], speeds[-1]

        if not (prev > prev_prev and prev > curr):
            return None

        peak_px_s = float(self._speed_px_s[-2])
        if use_norm and peak_px_s < self.min_peak_speed_px_s_floor:
            return None
        if not use_norm and peak_px_s < self.min_peak_speed_px_s:
            return None
        if prev < peak_threshold:
            return None

        baseline_vals = [v for v in speeds[:-2] if np.isfinite(v)]
        baseline = float(np.median(baseline_vals)) if baseline_vals else 0.0
        if baseline > 1e-6 and (prev / baseline) < self.peak_over_baseline_ratio:
            return None

        self._cooldown_left = self.cooldown_frames
        impact_frame_idx = trigger_frame_idx - 1

        peak_sw_s = float(self._speed_sw_s[-2]) if self._speed_sw_s[-2] is not None else None
        peak_vel = self._vel_px_s[-2] if len(self._vel_px_s) >= 2 else np.zeros(2)
        norm = float(np.linalg.norm(peak_vel))
        if norm > 1e-6:
            unit = (float(peak_vel[0] / norm), float(peak_vel[1] / norm))
        else:
            unit = (0.0, 0.0)

        return ImpactEvent(
            impact_frame_idx=impact_frame_idx,
            trigger_frame_idx=trigger_frame_idx,
            peak_speed_px_s=peak_px_s,
            peak_speed_sw_s=peak_sw_s,
            peak_velocity_unit=unit,
        )


class SwingPhaseEstimator:
    """Estimate preparation-start and follow-through-end around each impact.

    Uses simple heuristics on the wrist-speed and shoulder-rotation time series.
    """

    def __init__(self, fps: float = 30.0):
        self.fps = max(fps, 1.0)

    def estimate_phases(
        self,
        impact_frame: int,
        wrist_speeds: np.ndarray,
        frame_indices: List[int],
    ) -> SwingEvent:
        """Given an impact frame and the full wrist-speed series, estimate phases."""
        event = SwingEvent(impact_frame=impact_frame)

        if len(wrist_speeds) == 0 or len(frame_indices) == 0:
            return event

        # Map frame indices to array positions
        frame_to_pos = {f: i for i, f in enumerate(frame_indices)}
        impact_pos = frame_to_pos.get(impact_frame)
        if impact_pos is None:
            return event

        # ── Preparation start: walk backward from impact until speed drops below 20% of peak
        peak_speed = float(wrist_speeds[impact_pos]) if impact_pos < len(wrist_speeds) else 0.0
        threshold = peak_speed * 0.20
        prep_pos = impact_pos
        for i in range(impact_pos - 1, -1, -1):
            if i < len(wrist_speeds) and wrist_speeds[i] < threshold:
                prep_pos = i
                break
        # Go back a bit more to capture the start of the turn
        prep_pos = max(0, prep_pos - int(0.3 * self.fps))
        event.prep_start_frame = frame_indices[prep_pos]

        # ── Follow-through end: walk forward from impact until speed drops below 25% of peak
        ft_threshold = peak_speed * 0.25
        ft_pos = min(impact_pos + 1, len(wrist_speeds) - 1)
        for i in range(impact_pos + 1, len(wrist_speeds)):
            if wrist_speeds[i] < ft_threshold:
                ft_pos = i
                break
        # Add a small buffer for the full follow-through
        ft_pos = min(len(frame_indices) - 1, ft_pos + int(0.2 * self.fps))
        event.followthrough_end_frame = frame_indices[ft_pos]

        return event
