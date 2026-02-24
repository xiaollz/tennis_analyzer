#!/usr/bin/env python3
"""
Big 3 Real-time Monitors
========================
Simplified monitors focused only on the Big 3 Checkpoints:
1. Contact Point - Wrist vs Hip (in-front distance)
2. Weight Transfer - Back-foot release + hip/torso shift (side view; conservative in back view)
3. Contact Zone - Early follow-through after contact (penetration + brush), measured in body coordinates

Design references (forehand):
- `docs/learn_ytb/broddfelt_complete_25_videos.md`
- `docs/learn_ytb/网球学习指南_v2_综合版.md` (正手部分)

Priority order for coaching feedback:
Contact Point -> Weight Transfer -> Contact Zone
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum
from collections import deque

from ..config.keypoints import KEYPOINT_NAMES
from .impact import WristSpeedImpactDetector, ImpactEvent
from .pose_utils import (
    CAMERA_VIEW_BACK,
    CAMERA_VIEW_SIDE,
    TORSO_HEIGHT_REF_CM,
    choose_pixels_per_cm,
    estimate_scale,
)


class Status(Enum):
    """Quality status for monitors."""
    UNKNOWN = "unknown"
    GOOD = "good"
    OK = "ok"
    BAD = "bad"


@dataclass
class ContactPointStatus:
    """Status for contact point checkpoint."""
    status: Status
    # Projection of (wrist - hip) onto the inferred forward axis.
    # Always in pixels; optional normalized/scaled versions are provided too.
    delta_px: float
    delta_sw: Optional[float]  # in shoulder-widths
    delta_cm: Optional[float]  # approximate, using shoulder_width_cm
    message: str
    # Optional prep-timing diagnostics (best-effort; may be None).
    prep_reference: Optional[str] = None  # "bounce" | "impact" | None
    prep_lead_s: Optional[float] = None   # how long before the reference the racket was "set"


@dataclass
class WeightTransferStatus:
    """Status for weight transfer checkpoint."""
    status: Status
    # Combined score, always in pixels plus optional normalized/scaled versions.
    score_px: float
    score_sw: Optional[float]
    score_cm: Optional[float]
    # Components.
    ankle_lift_px: float
    ankle_lift_sw: Optional[float]
    ankle_lift_cm: Optional[float]
    hip_forward_px: float
    hip_forward_sw: Optional[float]
    hip_forward_cm: Optional[float]
    message: str


@dataclass
class ContactZoneStatus:
    """Status for contact zone checkpoint."""
    status: Status
    # Displacement from impact to the end of the window.
    # Always in pixels; optional normalized/scaled versions are provided too.
    forward_px: float
    forward_sw: Optional[float]
    forward_cm: Optional[float]
    upward_px: float
    upward_sw: Optional[float]
    upward_cm: Optional[float]
    message: str


class ContactPointMonitor:
    """
    Monitor 1: Contact Point
    Tracks how far in front of the body the wrist is at any moment.
    """
    
    def __init__(
        self,
        is_right_handed: bool = True,
        *,
        shoulder_width_cm: float = 41.0,
        # Side-view/far-camera footage often yields small shoulder pixel widths.
        min_shoulder_width_px: float = 12.0,
        good_min_cm: float = 20.0,
        good_max_cm: float = 40.0,
        ok_min_cm: float = 10.0,
        good_threshold_px: float = 50.0,
        ok_threshold_px: float = 0.0,
        # Prep timing (best-effort). We don't track the ball, so "bounce" can
        # only be inferred in hybrid mode from audio onsets.
        prep_window_s: float = 0.8,
        prep_exclude_tail_frames: int = 1,
        prep_min_samples: int = 6,
        # When bounce is unknown, we fallback to "racket set before impact".
        prep_impact_good_lead_s: float = 0.18,
        prep_impact_ok_lead_s: float = 0.08,
        # When bounce is inferred, allow a small "after bounce" tolerance.
        prep_bounce_ok_after_s: float = 0.10,
    ):
        self.is_right_handed = is_right_handed
        self.wrist_idx = KEYPOINT_NAMES["right_wrist"] if is_right_handed else KEYPOINT_NAMES["left_wrist"]
        # Use hip-center when possible; fall back to dominant hip.
        self.hip_idx = KEYPOINT_NAMES["right_hip"] if is_right_handed else KEYPOINT_NAMES["left_hip"]
        self.left_hip_idx = KEYPOINT_NAMES["left_hip"]
        self.right_hip_idx = KEYPOINT_NAMES["right_hip"]
        
        # Scale/thresholds (defaults align with tennis_coach forehand guidance: 20-40cm).
        self.shoulder_width_cm = float(shoulder_width_cm)
        # Guard against occasional pose glitches where shoulder points collapse
        # (would explode cm/sw conversions).
        self.min_shoulder_width_px = float(min_shoulder_width_px)
        self.good_min_cm = float(good_min_cm)
        self.good_max_cm = float(good_max_cm)
        self.ok_min_cm = float(ok_min_cm)
        # Fallback thresholds when scale is unknown (px).
        self.good_threshold_px = float(good_threshold_px)
        self.ok_threshold_px = float(ok_threshold_px)

        # Prep timing configuration.
        self.prep_window_s = float(prep_window_s)
        self.prep_exclude_tail_frames = int(prep_exclude_tail_frames)
        self.prep_min_samples = int(prep_min_samples)
        self.prep_impact_good_lead_s = float(prep_impact_good_lead_s)
        self.prep_impact_ok_lead_s = float(prep_impact_ok_lead_s)
        self.prep_bounce_ok_after_s = float(prep_bounce_ok_after_s)
        
        # Direction inference (side-view ambiguity): use recent wrist dx to infer "forward".
        self._prev_wrist_x: Optional[float] = None
        self._dx_history: deque[float] = deque(maxlen=7)

        # Pre-impact wrist-x history (frame_idx, wrist_x). Used to estimate the
        # "racket set" timing (deepest takeback) before the contact frame.
        self._wrist_x_hist: deque[tuple[int, float]] = deque(maxlen=240)
        self._frame_count = 0

        self.current_delta_cm = 0.0
        self.max_delta_cm = -float("inf")  # Track best contact this swing

    def _reset_swing_state(self) -> None:
        """Reset per-swing history without touching external frame indices."""
        self.max_delta_cm = -float("inf")
        self.current_delta_cm = 0.0
        self._prev_wrist_x = None
        self._dx_history.clear()
        self._wrist_x_hist.clear()
        
    def update(
        self,
        keypoints: np.ndarray,
        confidence: np.ndarray,
        *,
        frame_idx: Optional[int] = None,
        is_impact: bool = False,
        impact_frame_idx: Optional[int] = None,
        bounce_frame_idx: Optional[int] = None,
        fps: Optional[float] = None,
        forward_sign: Optional[float] = None,
        forward_axis_unit: Optional[Tuple[float, float]] = None,
        camera_view: Optional[str] = None,
    ) -> ContactPointStatus:
        """Update with new frame data."""
        if frame_idx is None:
            self._frame_count += 1
            frame_idx = self._frame_count
        else:
            self._frame_count = int(frame_idx)

        if confidence[self.wrist_idx] < 0.5:
            return ContactPointStatus(
                status=Status.UNKNOWN,
                delta_px=0.0,
                delta_sw=None,
                delta_cm=None,
                message="检测不到手腕",
            )

        # Direction inference from wrist movement.
        wrist_x = float(keypoints[self.wrist_idx][0])
        self._wrist_x_hist.append((int(frame_idx), float(wrist_x)))
        if self._prev_wrist_x is not None:
            self._dx_history.append(wrist_x - self._prev_wrist_x)
        self._prev_wrist_x = wrist_x
        if forward_sign is None:
            forward_sign = 1.0
            if len(self._dx_history) >= 3:
                dx_med = float(np.median(np.asarray(self._dx_history, dtype=np.float32)))
                if abs(dx_med) > 1e-3:
                    forward_sign = 1.0 if dx_med > 0 else -1.0
        
        hip_xy: Optional[np.ndarray] = None
        if confidence[self.left_hip_idx] >= 0.35 and confidence[self.right_hip_idx] >= 0.35:
            hip_xy = (np.asarray(keypoints[self.left_hip_idx], dtype=np.float32) + np.asarray(keypoints[self.right_hip_idx], dtype=np.float32)) / 2.0
        elif confidence[self.hip_idx] >= 0.5:
            hip_xy = np.asarray(keypoints[self.hip_idx], dtype=np.float32)
        else:
            return ContactPointStatus(
                status=Status.UNKNOWN,
                delta_px=0.0,
                delta_sw=None,
                delta_cm=None,
                message="检测不到髋部",
            )
        wrist_xy = np.asarray(keypoints[self.wrist_idx], dtype=np.float32)
        delta_vec = wrist_xy - np.asarray(hip_xy, dtype=np.float32)

        # Project onto a best-effort "forward swing" axis.
        # - Prefer the axis from the impact wrist-velocity direction if provided.
        # - Fall back to image X axis with inferred sign.
        if forward_axis_unit is not None:
            # NOTE: For side-view footage, the velocity direction can be nearly
            # vertical at the peak (dy >> dx). Using the full 2D direction then
            # makes the "forward" projection unstable and can flip sign. We
            # therefore only use the *horizontal* component to infer forward.
            dx = float(forward_axis_unit[0])
            if abs(dx) > 1e-3:
                delta_px_forward = float(delta_vec[0]) * (1.0 if dx > 0 else -1.0)
            else:
                delta_px_forward = float(delta_vec[0]) * float(forward_sign)
        else:
            delta_px_forward = float(delta_vec[0]) * float(forward_sign)

        # Scale conversion:
        # - Behind-view: shoulder-width is usually stable.
        # - Side-view: shoulder-width is foreshortened; torso height is more stable.
        scale = estimate_scale(
            keypoints,
            confidence,
            shoulder_width_cm=self.shoulder_width_cm,
            torso_height_cm=TORSO_HEIGHT_REF_CM,
            min_shoulder_width_px=self.min_shoulder_width_px,
        )
        sw_px = scale.shoulder_width_px
        pixels_per_cm = choose_pixels_per_cm(scale, camera_view=camera_view)

        delta_px = float(delta_px_forward)
        delta_sw = (delta_px / sw_px) if (sw_px is not None and sw_px > 1e-6) else None
        delta_cm = (delta_px / pixels_per_cm) if (pixels_per_cm is not None and pixels_per_cm > 1e-6) else None
        
        # Keep internal fields for quick debugging/plots (unit depends on availability).
        self.current_delta_cm = float(delta_cm) if delta_cm is not None else delta_px
        self.max_delta_cm = max(self.max_delta_cm, self.current_delta_cm)
        
        # Determine status
        prep_reference: Optional[str] = None
        prep_lead_s: Optional[float] = None
        if delta_cm is None:
            # Fallback when scale is unknown.
            if delta_px >= self.good_threshold_px:
                status = Status.GOOD
                message = "击球点OK ✓"
            elif delta_px >= self.ok_threshold_px:
                status = Status.OK
                message = "略偏晚：再往身前一些"
            else:
                status = Status.BAD
                message = "偏晚：再往身前一些"
        else:
            if self.good_min_cm <= delta_cm <= self.good_max_cm:
                status = Status.GOOD
                message = "击球点OK ✓"
            elif delta_cm < self.good_min_cm:
                need_cm = max(0.0, float(self.good_min_cm - delta_cm))
                # Below ok_min is considered "bad" (late contact).
                if delta_cm < self.ok_min_cm:
                    status = Status.BAD
                    message = f"偏晚：再往身前 {need_cm:.0f}cm"
                else:
                    status = Status.OK
                    message = f"略偏晚：再往身前 {need_cm:.0f}cm"
            else:
                need_cm = max(0.0, float(delta_cm - self.good_max_cm))
                status = Status.OK
                message = f"过靠前：回收 {need_cm:.0f}cm"

        # Best-effort prep timing (only meaningful when caller signals impact).
        if bool(is_impact):
            fps_eff = float(fps) if (fps and float(fps) > 1e-6) else 30.0
            imp_idx = int(impact_frame_idx) if impact_frame_idx is not None else int(frame_idx)

            # Prefer bounce reference when available (hybrid mode).
            if bounce_frame_idx is not None and int(bounce_frame_idx) < int(imp_idx):
                prep_reference = "bounce"
                ref_idx = int(bounce_frame_idx)
            else:
                prep_reference = "impact"
                ref_idx = int(imp_idx)

            # Find the deepest takeback (minimum wrist_x along the forward axis)
            # within a short pre-impact window.
            win = int(round(max(0.2, float(self.prep_window_s)) * fps_eff))
            start = int(imp_idx) - int(win)
            end = int(imp_idx) - int(self.prep_exclude_tail_frames)
            samples = [(fi, x) for (fi, x) in self._wrist_x_hist if int(start) <= int(fi) <= int(end)]

            set_frame: Optional[int] = None
            if len(samples) >= int(self.prep_min_samples):
                fs = float(forward_sign) if (forward_sign is not None and abs(float(forward_sign)) > 1e-6) else 1.0
                set_frame = int(min(samples, key=lambda t: float(t[1]) * fs)[0])

            if set_frame is not None:
                prep_lead_s = float(ref_idx - int(set_frame)) / fps_eff

                # Classify prep quality and produce a delta-to-goal note.
                prep_note = ""
                if prep_reference == "bounce":
                    # Goal: racket set BEFORE bounce (lead_s >= 0). Allow a small grace window.
                    if prep_lead_s >= 0.0:
                        prep_note = ""
                    elif prep_lead_s >= -float(self.prep_bounce_ok_after_s):
                        need_s = max(0.0, -prep_lead_s)
                        prep_note = f"准备略晚：再提前约 {need_s:.1f}s"
                    else:
                        need_s = max(0.0, -prep_lead_s)
                        prep_note = f"准备偏晚：再提前约 {need_s:.1f}s"
                else:
                    # Fallback: goal is to complete takeback some lead time before impact.
                    if prep_lead_s >= float(self.prep_impact_good_lead_s):
                        prep_note = ""
                    elif prep_lead_s >= float(self.prep_impact_ok_lead_s):
                        need_s = max(0.0, float(self.prep_impact_good_lead_s) - prep_lead_s)
                        prep_note = f"准备略急：再提前约 {need_s:.1f}s"
                    else:
                        need_s = max(0.0, float(self.prep_impact_good_lead_s) - prep_lead_s)
                        prep_note = f"准备偏晚：再提前约 {need_s:.1f}s"

                if prep_note:
                    # Keep Big3 priority: contact point message first, then root-cause hint.
                    if status == Status.GOOD:
                        status = Status.OK
                        message = f"击球点OK，但{prep_note}"
                    else:
                        message = f"{message}（{prep_note}）"

            # Start fresh for the next swing.
            self._reset_swing_state()
        
        return ContactPointStatus(
            status=status,
            delta_px=delta_px,
            delta_sw=delta_sw,
            delta_cm=delta_cm,
            message=message,
            prep_reference=prep_reference,
            prep_lead_s=prep_lead_s,
        )
    
    def reset(self):
        """Reset for new swing."""
        self._reset_swing_state()
        self._frame_count = 0


class WeightTransferMonitor:
    """
    Monitor 2: Weight Transfer
    Tracks back foot rotation and hip engagement.
    """
    
    def __init__(
        self,
        is_right_handed: bool = True,
        *,
        shoulder_width_cm: float = 41.0,
        # Side-view/far-camera footage often yields small shoulder pixel widths.
        min_shoulder_width_px: float = 12.0,
        min_ankle_conf: float = 0.2,
        min_hip_conf: float = 0.35,
        baseline_window_frames: int = 20,
        baseline_exclude_tail_frames: int = 2,
        min_baseline_samples: int = 6,
        good_threshold_px: float = 30.0,
        ok_threshold_px: float = 10.0,
        good_threshold_cm: float = 8.0,
        ok_threshold_cm: float = 4.0,
        # Sequence/dynamic heuristics (best-effort, side-view preferred).
        sequence_window_frames: int = 12,  # ~0.4s @30fps
        sequence_min_samples: int = 4,
        active_hip_velocity_cm_s: float = 15.0,
    ):
        self.is_right_handed = is_right_handed
        self.back_ankle_idx = KEYPOINT_NAMES["right_ankle"] if is_right_handed else KEYPOINT_NAMES["left_ankle"]
        self.back_knee_idx = KEYPOINT_NAMES["right_knee"] if is_right_handed else KEYPOINT_NAMES["left_knee"]
        self.back_hip_idx = KEYPOINT_NAMES["right_hip"] if is_right_handed else KEYPOINT_NAMES["left_hip"]
        self.left_hip_idx = KEYPOINT_NAMES["left_hip"]
        self.right_hip_idx = KEYPOINT_NAMES["right_hip"]
        
        # Thresholds
        self.good_threshold_px = float(good_threshold_px)
        self.ok_threshold_px = float(ok_threshold_px)
        # Typical "good" values in side-view videos are single-digit centimeters.
        self.good_threshold_cm = float(good_threshold_cm)
        self.ok_threshold_cm = float(ok_threshold_cm)

        self.shoulder_width_cm = float(shoulder_width_cm)
        self.min_shoulder_width_px = float(min_shoulder_width_px)
        self.min_ankle_conf = float(min_ankle_conf)
        self.min_hip_conf = float(min_hip_conf)

        # Baseline window configuration (pre-impact).
        self.baseline_window_frames = int(baseline_window_frames)
        self.baseline_exclude_tail_frames = int(baseline_exclude_tail_frames)
        self.min_baseline_samples = int(min_baseline_samples)

        self.sequence_window_frames = int(sequence_window_frames)
        self.sequence_min_samples = int(sequence_min_samples)
        self.active_hip_velocity_cm_s = float(active_hip_velocity_cm_s)

        # Direction inference (same ambiguity as contact point).
        self._prev_wrist_x: Optional[float] = None
        self._dx_history: deque[float] = deque(maxlen=7)
        
        # Pre-impact baseline samples (collected independently; ankles are often low-confidence).
        self._ankle_samples: deque[tuple[int, float]] = deque(maxlen=240)
        self._knee_samples: deque[tuple[int, float]] = deque(maxlen=240)
        self._hip_samples: deque[tuple[int, float]] = deque(maxlen=240)
        self._last_result: Optional[WeightTransferStatus] = None

        # Frame counter fallback when caller doesn't provide frame_idx.
        self._frame_count = 0
        
    def update(
        self,
        keypoints: np.ndarray,
        confidence: np.ndarray,
        *,
        frame_idx: Optional[int] = None,
        is_impact: bool = False,
        impact_frame_idx: Optional[int] = None,
        impact_keypoints: Optional[np.ndarray] = None,
        impact_confidence: Optional[np.ndarray] = None,
        forward_sign: Optional[float] = None,
        fps: Optional[float] = None,
        camera_view: Optional[str] = None,
    ) -> WeightTransferStatus:
        """Update with new frame data."""
        self._frame_count += 1
        if frame_idx is None:
            frame_idx = self._frame_count

        # Collect baseline samples (pre-impact window). Keep them independent so we can
        # still compute something even when ankles are missing.
        if confidence[self.back_ankle_idx] >= self.min_ankle_conf:
            ankle_y = float(keypoints[self.back_ankle_idx][1])
            self._ankle_samples.append((int(frame_idx), float(ankle_y)))
        if confidence[self.back_knee_idx] >= 0.25:
            knee_x = float(keypoints[self.back_knee_idx][0])
            self._knee_samples.append((int(frame_idx), float(knee_x)))
        hip_x: Optional[float] = None
        if (
            confidence[self.left_hip_idx] >= self.min_hip_conf
            and confidence[self.right_hip_idx] >= self.min_hip_conf
        ):
            hip_x = float(
                0.5
                * (
                    float(keypoints[self.left_hip_idx][0])
                    + float(keypoints[self.right_hip_idx][0])
                )
            )
        elif confidence[self.back_hip_idx] >= self.min_hip_conf:
            hip_x = float(keypoints[self.back_hip_idx][0])
        if hip_x is not None:
            self._hip_samples.append((int(frame_idx), float(hip_x)))

        # Direction inference from dominant wrist.
        wrist_idx = KEYPOINT_NAMES["right_wrist"] if self.is_right_handed else KEYPOINT_NAMES["left_wrist"]
        if confidence[wrist_idx] > 0.35:
            wrist_x = float(keypoints[wrist_idx][0])
            if self._prev_wrist_x is not None:
                self._dx_history.append(wrist_x - self._prev_wrist_x)
            self._prev_wrist_x = wrist_x
        if forward_sign is None:
            forward_sign = 1.0
            if len(self._dx_history) >= 3:
                dx_med = float(np.median(np.asarray(self._dx_history, dtype=np.float32)))
                if abs(dx_med) > 1e-3:
                    forward_sign = 1.0 if dx_med > 0 else -1.0

        # Only evaluate weight transfer on impact. Otherwise return the last frozen
        # result (if any) while still collecting baseline samples for the next swing.
        if not is_impact:
            if self._last_result is not None:
                return self._last_result
            return WeightTransferStatus(
                status=Status.UNKNOWN,
                score_px=0.0,
                score_sw=None,
                score_cm=None,
                ankle_lift_px=0.0,
                ankle_lift_sw=None,
                ankle_lift_cm=None,
                hip_forward_px=0.0,
                hip_forward_sw=None,
                hip_forward_cm=None,
                message="等待击球..."
            )

        # Use impact pose if provided (KineticChainManager impact peak alignment).
        if impact_keypoints is not None and impact_confidence is not None:
            kps_i = impact_keypoints
            conf_i = impact_confidence
        else:
            kps_i = keypoints
            conf_i = confidence

        # Resolve impact frame idx for baseline slicing.
        if impact_frame_idx is None:
            impact_frame_idx = int(frame_idx)

        ankle_y_i: Optional[float] = (
            float(kps_i[self.back_ankle_idx][1])
            if conf_i[self.back_ankle_idx] >= self.min_ankle_conf
            else None
        )
        hip_x_i: Optional[float] = None
        if (
            conf_i[self.left_hip_idx] >= self.min_hip_conf
            and conf_i[self.right_hip_idx] >= self.min_hip_conf
        ):
            hip_x_i = float(
                0.5
                * (
                    float(kps_i[self.left_hip_idx][0])
                    + float(kps_i[self.right_hip_idx][0])
                )
            )
        elif conf_i[self.back_hip_idx] >= self.min_hip_conf:
            hip_x_i = float(kps_i[self.back_hip_idx][0])

        # Build baseline window from recent samples strictly before impact.
        start = int(impact_frame_idx) - int(self.baseline_window_frames)
        end = int(impact_frame_idx) - int(self.baseline_exclude_tail_frames)
        ankle_base = [s for s in self._ankle_samples if start <= s[0] <= end]
        knee_base = [s for s in self._knee_samples if start <= s[0] <= end]
        hip_base = [s for s in self._hip_samples if start <= s[0] <= end]

        base_ankle_y: Optional[float] = (
            float(np.median([v[1] for v in ankle_base])) if len(ankle_base) >= self.min_baseline_samples else None
        )
        base_hip_x: Optional[float] = (
            float(np.median([v[1] for v in hip_base])) if len(hip_base) >= self.min_baseline_samples else None
        )
        base_knee_x: Optional[float] = (
            float(np.median([v[1] for v in knee_base])) if len(knee_base) >= self.sequence_min_samples else None
        )

        ankle_lift_px_opt: Optional[float] = None
        if base_ankle_y is not None and ankle_y_i is not None:
            ankle_lift_px_opt = float(base_ankle_y - ankle_y_i)  # lower Y = higher in screen coords

        hip_forward_px_opt: Optional[float] = None
        if base_hip_x is not None and hip_x_i is not None:
            hip_forward_px_opt = float((hip_x_i - base_hip_x) * float(forward_sign))

        # Behind-view: hip "forward" translation is mostly depth and not observable in 2D.
        # Keep weight transfer signal conservative: use ankle lift only.
        # Hip "forward" translation is only reliably observable in side view.
        # - Back view: "forward" is mostly depth => disable hip-forward.
        # - Unknown view: be conservative and disable hip-forward too.
        view = (camera_view or CAMERA_VIEW_SIDE).lower()
        if view != CAMERA_VIEW_SIDE:
            hip_forward_px_opt = None

        # Scale at impact (same rationale as ContactPointMonitor).
        scale = estimate_scale(
            kps_i,
            conf_i,
            shoulder_width_cm=self.shoulder_width_cm,
            torso_height_cm=TORSO_HEIGHT_REF_CM,
            min_shoulder_width_px=self.min_shoulder_width_px,
        )
        sw_px = scale.shoulder_width_px
        pixels_per_cm = choose_pixels_per_cm(scale, camera_view=camera_view)

        if ankle_lift_px_opt is None and hip_forward_px_opt is None:
            return WeightTransferStatus(
                status=Status.UNKNOWN,
                score_px=0.0,
                score_sw=None,
                score_cm=None,
                ankle_lift_px=0.0,
                ankle_lift_sw=None,
                ankle_lift_cm=None,
                hip_forward_px=0.0,
                hip_forward_sw=None,
                hip_forward_cm=None,
                message="重心关键点不足（脚踝/髋部）"
            )

        hip_weight = 0.5 if ankle_lift_px_opt is not None else 1.0
        ankle_lift_px = float(ankle_lift_px_opt or 0.0)
        hip_forward_px = float(hip_forward_px_opt or 0.0)
        score_px = float(ankle_lift_px + hip_weight * hip_forward_px)

        ankle_lift_cm = float(ankle_lift_px / pixels_per_cm) if (pixels_per_cm and pixels_per_cm > 1e-6 and ankle_lift_px_opt is not None) else None
        ankle_lift_sw = float(ankle_lift_px / sw_px) if (sw_px and sw_px > 1e-6 and ankle_lift_px_opt is not None) else None
        hip_forward_cm = float(hip_forward_px / pixels_per_cm) if (pixels_per_cm and pixels_per_cm > 1e-6 and hip_forward_px_opt is not None) else None
        hip_forward_sw = float(hip_forward_px / sw_px) if (sw_px and sw_px > 1e-6 and hip_forward_px_opt is not None) else None

        score_cm = float(score_px / pixels_per_cm) if (pixels_per_cm and pixels_per_cm > 1e-6) else None
        score_sw = float(score_px / sw_px) if (sw_px and sw_px > 1e-6) else None

        # Determine status.
        if pixels_per_cm is None:
            good_thr = self.good_threshold_px
            ok_thr = self.ok_threshold_px
            score_for_threshold = score_px
        else:
            good_thr = self.good_threshold_cm
            ok_thr = self.ok_threshold_cm
            score_for_threshold = float(score_cm or 0.0)

        if score_for_threshold >= good_thr:
            status = Status.GOOD
            message = "重心转移充分 ✓"
        elif score_for_threshold >= ok_thr:
            status = Status.OK
            need_to_good = max(0.0, float(good_thr - score_for_threshold))
            if pixels_per_cm is not None:
                message = f"再多释放 {need_to_good:.0f}cm"
            else:
                message = "重心有转移，再充分些"
        else:
            status = Status.BAD
            need_to_good = max(0.0, float(good_thr - score_for_threshold))
            if pixels_per_cm is not None:
                message = f"重心不足：再多释放 {need_to_good:.0f}cm"
            else:
                message = "重心不足：后脚再释放些"

        if ankle_lift_px_opt is None and hip_forward_px_opt is not None:
            message = f"{message}（仅看髋部）"
        elif hip_forward_px_opt is None and ankle_lift_px_opt is not None:
            message = f"{message}（仅看脚踝）"

        # ------------------------------------------------------------------
        # Sequence/dynamic hints (message-only; keep the core score stable).
        # ------------------------------------------------------------------
        # Try to detect "heel → (knee) → hip" order and whether hip is still
        # moving at contact. This is only meaningful in side view.
        view = (camera_view or "").lower()
        if view != CAMERA_VIEW_BACK:
            seq_start = int(impact_frame_idx) - int(self.sequence_window_frames)
            seq_end = int(impact_frame_idx)
            ankle_seq = [s for s in self._ankle_samples if seq_start <= s[0] <= seq_end]
            knee_seq = [s for s in self._knee_samples if seq_start <= s[0] <= seq_end]
            hip_seq = [s for s in self._hip_samples if seq_start <= s[0] <= seq_end]

            # Include the impact pose sample itself when we evaluated on an
            # earlier frame (trigger_frame != impact_frame).
            if ankle_y_i is not None:
                ankle_seq.append((int(impact_frame_idx), float(ankle_y_i)))
            if hip_x_i is not None:
                hip_seq.append((int(impact_frame_idx), float(hip_x_i)))
            knee_x_i: Optional[float] = None
            if conf_i[self.back_knee_idx] >= 0.25:
                knee_x_i = float(kps_i[self.back_knee_idx][0])
                knee_seq.append((int(impact_frame_idx), float(knee_x_i)))

            heel_thr_px = max(6.0, 0.6 * float(self.ok_threshold_px))
            hip_thr_px = max(6.0, 0.6 * float(self.ok_threshold_px))
            knee_thr_px = 5.0

            heel_lift_frame: Optional[int] = None
            if base_ankle_y is not None and ankle_seq:
                for fi, y in sorted(ankle_seq, key=lambda t: int(t[0])):
                    if float(base_ankle_y - float(y)) >= float(heel_thr_px):
                        heel_lift_frame = int(fi)
                        break

            hip_forward_frame: Optional[int] = None
            if base_hip_x is not None and hip_seq:
                for fi, x in sorted(hip_seq, key=lambda t: int(t[0])):
                    if heel_lift_frame is not None and int(fi) < int(heel_lift_frame):
                        continue
                    if float((float(x) - float(base_hip_x)) * float(forward_sign)) >= float(hip_thr_px):
                        hip_forward_frame = int(fi)
                        break

            knee_forward_frame: Optional[int] = None
            if base_knee_x is not None and knee_seq:
                for fi, x in sorted(knee_seq, key=lambda t: int(t[0])):
                    if heel_lift_frame is not None and int(fi) < int(heel_lift_frame):
                        continue
                    if float((float(x) - float(base_knee_x)) * float(forward_sign)) >= float(knee_thr_px):
                        knee_forward_frame = int(fi)
                        break

            active_transfer: Optional[bool] = None
            if pixels_per_cm is not None and pixels_per_cm > 1e-6 and len(hip_seq) >= 2:
                fps_eff = float(fps) if (fps and float(fps) > 1e-6) else 30.0
                hip_sorted = sorted(hip_seq, key=lambda t: int(t[0]))
                # Prefer a short window right before impact (dynamic transfer),
                # not the whole sequence window.
                hip_recent = [(fi, x) for (fi, x) in hip_sorted if int(fi) >= int(impact_frame_idx) - 3]
                if len(hip_recent) >= 2:
                    fi0, x0 = hip_recent[0]
                    fi1, x1 = hip_recent[-1]
                else:
                    fi0, x0 = hip_sorted[-2]
                    fi1, x1 = hip_sorted[-1]
                dt_s = max(1e-6, float(int(fi1) - int(fi0)) / fps_eff)
                vel_px_s = float((float(x1) - float(x0)) * float(forward_sign)) / dt_s
                vel_cm_s = float(vel_px_s / float(pixels_per_cm))
                active_transfer = vel_cm_s >= float(self.active_hip_velocity_cm_s)

            # Append at most ONE short hint (avoid verbose strings in Big3 UI).
            hint = ""
            if heel_lift_frame is None and ankle_lift_px_opt is not None:
                hint = "先释放后脚跟"
            elif hip_forward_frame is None and hip_forward_px_opt is not None:
                hint = "髋部推进要更早"
            elif active_transfer is False:
                hint = "击球时还要继续送"
            if hint:
                message = f"{message}（{hint}）"

        result = WeightTransferStatus(
            status=status,
            score_px=score_px,
            score_sw=score_sw,
            score_cm=score_cm,
            ankle_lift_px=ankle_lift_px,
            ankle_lift_sw=ankle_lift_sw,
            ankle_lift_cm=ankle_lift_cm,
            hip_forward_px=hip_forward_px,
            hip_forward_sw=hip_forward_sw,
            hip_forward_cm=hip_forward_cm,
            message=message,
        )

        # Freeze until reset / next swing.
        self._last_result = result
        # Clear baseline so the next swing starts fresh.
        self._ankle_samples.clear()
        self._knee_samples.clear()
        self._hip_samples.clear()
        return result
    
    def reset(self):
        """Reset for new swing."""
        self._ankle_samples.clear()
        self._knee_samples.clear()
        self._hip_samples.clear()
        self._last_result = None
        self._frame_count = 0
        self._prev_wrist_x = None
        self._dx_history.clear()


class ContactZoneMonitor:
    """
    Monitor 3: Contact Zone
    Tracks early follow-through after impact (penetration + brush).

    Important:
    - We measure wrist motion RELATIVE to the torso (hip/shoulder center) so body
      translation doesn't get mis-counted as "penetration".
    - This aligns better with Broddfelt's teaching: "send it through first,
      then brush up", and makes the metric more stable across footwork styles.
    """
    
    def __init__(
        self,
        is_right_handed: bool = True,
        window_size: int = 5,
        *,
        shoulder_width_cm: float = 41.0,
        # Side-view/far-camera footage often yields small shoulder pixel widths.
        min_shoulder_width_px: float = 12.0,
        good_forward_cm: float = 15.0,
        good_upward_cm: float = 10.0,
        good_forward_px: float = 20.0,
        good_upward_px: float = 20.0,
    ):
        self.is_right_handed = is_right_handed
        self.wrist_idx = KEYPOINT_NAMES["right_wrist"] if is_right_handed else KEYPOINT_NAMES["left_wrist"]
        self.window_size = window_size

        self.left_hip_idx = KEYPOINT_NAMES["left_hip"]
        self.right_hip_idx = KEYPOINT_NAMES["right_hip"]
        self.dom_hip_idx = KEYPOINT_NAMES["right_hip"] if is_right_handed else KEYPOINT_NAMES["left_hip"]
        self.left_sh_idx = KEYPOINT_NAMES["left_shoulder"]
        self.right_sh_idx = KEYPOINT_NAMES["right_shoulder"]
        
        # Thresholds
        self.good_forward_cm = float(good_forward_cm)
        self.good_upward_cm = float(good_upward_cm)
        self.good_forward_px = float(good_forward_px)
        self.good_upward_px = float(good_upward_px)
        self.shoulder_width_cm = float(shoulder_width_cm)
        self.min_shoulder_width_px = float(min_shoulder_width_px)

        self._prev_wrist_x: Optional[float] = None
        self._dx_history: deque[float] = deque(maxlen=7)
        
        # Body-relative wrist history (wrist - torso_center) starting at impact.
        self._rel_history: List[np.ndarray] = []
        # Freeze scale at (approx) impact for consistent px->cm conversion.
        self._pixels_per_cm_ref: Optional[float] = None
        self._sw_px_ref: Optional[float] = None
        self.is_post_impact = False
        self.impact_frame = 0
        self._frozen_result: Optional[ContactZoneStatus] = None

    def _torso_center(self, keypoints: np.ndarray, confidence: np.ndarray) -> Optional[np.ndarray]:
        # Prefer hip center; if missing, fall back to shoulder center.
        if confidence[self.left_hip_idx] >= 0.35 and confidence[self.right_hip_idx] >= 0.35:
            return (
                np.asarray(keypoints[self.left_hip_idx], dtype=np.float32)
                + np.asarray(keypoints[self.right_hip_idx], dtype=np.float32)
            ) / 2.0
        if confidence[self.dom_hip_idx] >= 0.35:
            return np.asarray(keypoints[self.dom_hip_idx], dtype=np.float32)
        if confidence[self.left_sh_idx] >= 0.35 and confidence[self.right_sh_idx] >= 0.35:
            return (
                np.asarray(keypoints[self.left_sh_idx], dtype=np.float32)
                + np.asarray(keypoints[self.right_sh_idx], dtype=np.float32)
            ) / 2.0
        return None
        
    def signal_impact(self, frame_num: int):
        """Call this when impact is detected."""
        self.is_post_impact = True
        self.impact_frame = frame_num
        self._rel_history.clear()
        self._pixels_per_cm_ref = None
        self._sw_px_ref = None
        self._frozen_result = None
        
    def update(
        self,
        keypoints: np.ndarray,
        confidence: np.ndarray,
        *,
        forward_sign: Optional[float] = None,
        forward_axis_unit: Optional[Tuple[float, float]] = None,
        camera_view: Optional[str] = None,
    ) -> ContactZoneStatus:
        """Update with new frame data."""
        # If we've already computed a post-impact result for the current swing,
        # keep returning it until the next impact signal.
        if self._frozen_result is not None and not self.is_post_impact:
            return self._frozen_result

        if confidence[self.wrist_idx] < 0.5:
            return ContactZoneStatus(
                status=Status.UNKNOWN,
                forward_px=0,
                forward_sw=None,
                forward_cm=None,
                upward_px=0,
                upward_sw=None,
                upward_cm=None,
                message="检测不到手腕"
            )
        
        wrist_pos = keypoints[self.wrist_idx].copy()

        # Direction inference.
        wrist_x = float(wrist_pos[0])
        if self._prev_wrist_x is not None:
            self._dx_history.append(wrist_x - self._prev_wrist_x)
        self._prev_wrist_x = wrist_x
        if forward_sign is None:
            forward_sign = 1.0
            if len(self._dx_history) >= 3:
                dx_med = float(np.median(np.asarray(self._dx_history, dtype=np.float32)))
                if abs(dx_med) > 1e-3:
                    forward_sign = 1.0 if dx_med > 0 else -1.0

        # Only track post-impact
        if not self.is_post_impact:
            return self._frozen_result or ContactZoneStatus(
                status=Status.UNKNOWN,
                forward_px=0,
                forward_sw=None,
                forward_cm=None,
                upward_px=0,
                upward_sw=None,
                upward_cm=None,
                message="等待击球..."
            )

        # Scale (same rationale as ContactPointMonitor). Freeze once for the
        # post-impact window so cm conversions stay consistent.
        if self._pixels_per_cm_ref is None:
            scale = estimate_scale(
                keypoints,
                confidence,
                shoulder_width_cm=self.shoulder_width_cm,
                torso_height_cm=TORSO_HEIGHT_REF_CM,
                min_shoulder_width_px=self.min_shoulder_width_px,
            )
            self._sw_px_ref = scale.shoulder_width_px
            self._pixels_per_cm_ref = choose_pixels_per_cm(scale, camera_view=camera_view)
        sw_px = self._sw_px_ref
        pixels_per_cm = self._pixels_per_cm_ref

        torso = self._torso_center(keypoints, confidence)
        if torso is None:
            return ContactZoneStatus(
                status=Status.UNKNOWN,
                forward_px=0,
                forward_sw=None,
                forward_cm=None,
                upward_px=0,
                upward_sw=None,
                upward_cm=None,
                message="检测不到躯干"
            )

        # Store wrist motion relative to torso to remove body translation.
        self._rel_history.append(np.asarray(wrist_pos - torso, dtype=np.float32))
        
        if len(self._rel_history) < 2:
            return ContactZoneStatus(
                status=Status.UNKNOWN,
                forward_px=0,
                forward_sw=None,
                forward_cm=None,
                upward_px=0,
                upward_sw=None,
                upward_cm=None,
                message="追踪手腕轨迹..."
            )

        # Wait until we have enough frames after impact to evaluate.
        # window_size is interpreted as "number of frames AFTER impact".
        if len(self._rel_history) < (self.window_size + 1):
            return ContactZoneStatus(
                status=Status.UNKNOWN,
                forward_px=0,
                forward_sw=None,
                forward_cm=None,
                upward_px=0,
                upward_sw=None,
                upward_cm=None,
                message="追踪随挥..."
            )
        
        start = self._rel_history[0]
        deltas = [np.asarray(v - start, dtype=np.float32) for v in self._rel_history]

        # Determine forward sign: prefer impact velocity horizontal direction.
        axis_sign = float(forward_sign)
        if forward_axis_unit is not None:
            dx = float(forward_axis_unit[0])
            if abs(dx) > 1e-3:
                axis_sign = 1.0 if dx > 0 else -1.0

        # Use the *maximum* penetration / brush within the window (more robust than
        # using only the last frame).
        forward_series = [float(d[0]) * axis_sign for d in deltas]
        upward_series = [float(-d[1]) for d in deltas]  # start_y - y
        forward_px = max(0.0, float(np.max(np.asarray(forward_series, dtype=np.float32))))
        upward_px = max(0.0, float(np.max(np.asarray(upward_series, dtype=np.float32))))

        # End-of-window deltas (same timestamp, useful for balance/shape checks).
        forward_end_px = max(0.0, float(forward_series[-1])) if forward_series else 0.0
        upward_end_px = max(0.0, float(upward_series[-1])) if upward_series else 0.0

        ratio_end: Optional[float] = None
        if upward_end_px > 1e-6:
            ratio_end = float(forward_end_px / upward_end_px)
        elif forward_end_px > 1e-6:
            ratio_end = float("inf")

        # "Vertical duration" proxy: how long after impact the wrist keeps a
        # meaningful forward component before becoming too steep upward.
        # We use cumulative deltas to reduce per-frame jitter sensitivity.
        drive_frames = 0
        max_drive_angle_deg = 60.0
        for t in range(1, len(forward_series)):
            f = max(0.0, float(forward_series[t]))
            u = max(0.0, float(upward_series[t]))
            if f < 1e-3 and u < 1e-3:
                continue
            ang = float(np.degrees(np.arctan2(u, f + 1e-6)))
            if ang <= max_drive_angle_deg:
                drive_frames += 1
            else:
                break

        forward_cm = float(forward_px / pixels_per_cm) if (pixels_per_cm and pixels_per_cm > 1e-6) else None
        upward_cm = float(upward_px / pixels_per_cm) if (pixels_per_cm and pixels_per_cm > 1e-6) else None

        forward_sw = float(forward_px / sw_px) if (sw_px and sw_px > 1e-6) else None
        upward_sw = float(upward_px / sw_px) if (sw_px and sw_px > 1e-6) else None
        
        # Determine status
        if pixels_per_cm is None:
            # Fallback to px thresholds.
            forward_ok = forward_px >= self.good_forward_px
            upward_ok = upward_px >= self.good_upward_px
        else:
            forward_ok = (forward_cm or 0.0) >= self.good_forward_cm
            upward_ok = (upward_cm or 0.0) >= self.good_upward_cm

        if forward_ok and upward_ok:
            status = Status.GOOD
            message = "随挥OK：穿透+上刷 ✓"
        elif forward_ok:
            status = Status.OK
            if pixels_per_cm is not None and upward_cm is not None:
                need_up = max(0.0, float(self.good_upward_cm - float(upward_cm)))
                prefix = "偏平：" if (ratio_end is not None and ratio_end > 2.0) else ""
                message = f"{prefix}上刷再多 {need_up:.0f}cm"
            else:
                message = "上刷再多一点"
        elif upward_ok:
            # If the ratio indicates "too sharp" (upward dominates), treat as bad.
            status = Status.BAD if (ratio_end is not None and ratio_end < 0.5 and drive_frames < 2) else Status.OK
            if pixels_per_cm is not None and forward_cm is not None:
                need_fwd = max(0.0, float(self.good_forward_cm - float(forward_cm)))
                prefix = "上升过陡：" if (ratio_end is not None and ratio_end < 0.5 and drive_frames < 2) else ""
                message = f"{prefix}穿透再多 {need_fwd:.0f}cm"
            else:
                message = "穿透再多一点"
        else:
            status = Status.BAD
            if pixels_per_cm is not None and forward_cm is not None and upward_cm is not None:
                need_fwd = max(0.0, float(self.good_forward_cm - float(forward_cm)))
                need_up = max(0.0, float(self.good_upward_cm - float(upward_cm)))
                prefix = ""
                if ratio_end is not None and ratio_end < 0.5 and drive_frames < 2:
                    prefix = "过陡："
                elif ratio_end is not None and ratio_end > 2.0:
                    prefix = "过平："
                message = f"{prefix}再多穿透 {need_fwd:.0f}cm + 上刷 {need_up:.0f}cm"
            else:
                message = "随挥不足：再多穿透+上刷"
        
        result = ContactZoneStatus(
            status=status,
            forward_px=float(forward_px),
            forward_sw=forward_sw,
            forward_cm=forward_cm,
            upward_px=float(upward_px),
            upward_sw=upward_sw,
            upward_cm=upward_cm,
            message=message
        )

        # Freeze result for this swing and stop tracking until the next impact.
        self._frozen_result = result
        self.is_post_impact = False
        return result
    
    def reset(self):
        """Reset for new swing."""
        self._rel_history.clear()
        self.is_post_impact = False
        self.impact_frame = 0
        self._frozen_result = None
        self._pixels_per_cm_ref = None
        self._sw_px_ref = None
        self._prev_wrist_x = None
        self._dx_history.clear()


class ImpactDetector:
    """
    Detect impact moment using wrist velocity peak.
    Simplified version of the original ImpactDetector.
    """
    
    def __init__(self, cooldown_frames: int = 18, is_right_handed: bool = True, fps: float = 30.0):
        self._det = WristSpeedImpactDetector(
            fps=fps,
            is_right_handed=is_right_handed,
            cooldown_frames=cooldown_frames,
        )
        self._last_speed = 0.0

    def update(
        self, frame_idx: int, keypoints: np.ndarray, confidence: np.ndarray
    ) -> Tuple[Optional[ImpactEvent], float]:
        event, speed = self._det.update(frame_idx, keypoints, confidence)
        self._last_speed = speed
        return event, speed

    def reset(self):
        self._det.reset()
        self._last_speed = 0.0


class Big3MonitorSet:
    """
    Unified container for all Big 3 monitors.
    """
    
    def __init__(self, is_right_handed: bool = True, fps: float = 30.0):
        self.contact_point = ContactPointMonitor(is_right_handed)
        self.weight_transfer = WeightTransferMonitor(is_right_handed)
        self.contact_zone = ContactZoneMonitor(is_right_handed)
        self.impact_detector = ImpactDetector(is_right_handed=is_right_handed, fps=fps)
        
        self.is_right_handed = is_right_handed
        self.last_impact = False

        # Keep previous frame so we can align Big3 evaluation to the estimated
        # impact frame (speed peak occurs one frame before detector triggers).
        self._prev_keypoints: Optional[np.ndarray] = None
        self._prev_confidence: Optional[np.ndarray] = None
        self._prev_frame_idx: Optional[int] = None
        
    def update(self, frame_idx: int, keypoints: np.ndarray, confidence: np.ndarray) -> dict:
        """
        Update all monitors with new frame.
        Returns dict with all statuses.
        """
        # Check for impact
        impact_event, speed = self.impact_detector.update(frame_idx, keypoints, confidence)
        is_impact = impact_event is not None

        forward_sign: Optional[float] = None
        forward_axis_unit: Optional[Tuple[float, float]] = None
        impact_idx = None
        impact_kps = None
        impact_conf = None
        if impact_event is not None:
            impact_idx = int(impact_event.impact_frame_idx)
            # Prefer using the pose at the estimated impact frame (usually prev frame).
            if (
                self._prev_keypoints is not None
                and self._prev_confidence is not None
                and self._prev_frame_idx == impact_idx
            ):
                impact_kps = self._prev_keypoints
                impact_conf = self._prev_confidence
            else:
                impact_kps = keypoints
                impact_conf = confidence

            dx = float(impact_event.peak_velocity_unit[0])
            if abs(dx) > 1e-3:
                forward_sign = 1.0 if dx > 0 else -1.0
            forward_axis_unit = impact_event.peak_velocity_unit
        
        if is_impact:
            # Seed contact-zone tracking at the estimated impact frame.
            if impact_idx is not None:
                self.contact_zone.signal_impact(int(impact_idx))
                if impact_kps is not None and impact_conf is not None:
                    self.contact_zone.update(
                        impact_kps, impact_conf, forward_sign=forward_sign, forward_axis_unit=forward_axis_unit
                    )
            else:
                self.contact_zone.signal_impact(frame_idx)
            self.last_impact = True
        
        # Update all monitors
        if is_impact and impact_kps is not None and impact_conf is not None:
            cp_status = self.contact_point.update(
                impact_kps, impact_conf, forward_sign=forward_sign, forward_axis_unit=forward_axis_unit
            )
        else:
            cp_status = self.contact_point.update(
                keypoints, confidence, forward_sign=forward_sign, forward_axis_unit=forward_axis_unit
            )

        wt_status = self.weight_transfer.update(
            keypoints,
            confidence,
            frame_idx=int(frame_idx),
            is_impact=is_impact,
            impact_frame_idx=impact_idx,
            impact_keypoints=impact_kps,
            impact_confidence=impact_conf,
            forward_sign=forward_sign,
        )
        cz_status = self.contact_zone.update(
            keypoints, confidence, forward_sign=forward_sign, forward_axis_unit=forward_axis_unit
        )

        # Save previous-frame snapshots.
        self._prev_keypoints = np.asarray(keypoints).copy()
        self._prev_confidence = np.asarray(confidence).copy()
        self._prev_frame_idx = int(frame_idx)
        
        return {
            "contact_point": cp_status,
            "weight_transfer": wt_status,
            "contact_zone": cz_status,
            "is_impact": is_impact,
            "wrist_speed": speed
        }
    
    def reset(self):
        """Reset all monitors for new swing."""
        self.contact_point.reset()
        self.weight_transfer.reset()
        self.contact_zone.reset()
        self.impact_detector.reset()
        self.last_impact = False
        self._prev_keypoints = None
        self._prev_confidence = None
        self._prev_frame_idx = None
