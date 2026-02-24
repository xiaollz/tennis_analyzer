"""
Forehand Kinetic Chain Monitoring System
正手动力链监控系统 (v2.0 - Double Bend Edition)

This module is now aligned with the `tennis_coach` skill:
- Primary checkpoints ("Big 3"): Contact Point, Weight Transfer, Contact Zone
- Secondary checkpoints (6): Extension, Knee Load, Unit Turn, Spacing, Wrist Structure, Balance

Important: Camera-view gating (侧面 vs 背面)
- Side view (侧面) is reliable for: Contact Point, Contact Zone (penetration/brush), Wrist Structure.
- Back view (背面) is reliable for: Unit Turn (approx), Spacing/Crowding.
- View-irrelevant (both): Weight Transfer (conservative), Extension, Knee Load, Balance.

Modules:
1. ExtensionMonitor - 手臂延伸监控 (双弯曲流派适配)
2. KneeLoadMonitor - 下肢加载监控（膝角近似）
3. SpacingMonitor - 击球空间/拥挤监测（背面）
4. XFactorMonitor - 转体幅度/时机（背面，肩宽压缩近似）
5. WristLaidbackMonitor - 手腕锁定检测（侧面）
6. BalanceMonitor - 平衡/头部稳定

Deprecated (kept for reference, not wired by the manager):
- LinearizationMonitor
- ShoulderAlignmentMonitor
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
from enum import Enum

from .base_monitor import BaseMonitor
from ..config.keypoints import KEYPOINT_NAMES
from .impact import WristSpeedImpactDetector, ImpactEvent
from .big3_monitors import (
    ContactPointMonitor as Big3ContactPointMonitor,
    WeightTransferMonitor as Big3WeightTransferMonitor,
    ContactZoneMonitor as Big3ContactZoneMonitor,
    ContactPointStatus as Big3ContactPointStatus,
    ContactZoneStatus as Big3ContactZoneStatus,
    Status as Big3Status,
)
from .pose_utils import (
    CAMERA_VIEW_BACK,
    CAMERA_VIEW_SIDE,
    CAMERA_VIEW_UNKNOWN,
    estimate_scale,
)
from .stroke_profiles import StrokeProfile, FOREHAND_PROFILE


# =============================================================================
# Global Physical Constants (全局物理参数)
# =============================================================================

USER_HEIGHT_CM = 175.0
ARM_LENGTH_REF_CM = 65.0
SHOULDER_WIDTH_REF_CM = 41.0
MIN_SPACE_THRESHOLD_CM = 45.5  # 70% of arm length or 1.2x shoulder width

class QualityStatus(Enum):
    """Quality status for all monitors."""
    UNKNOWN = "unknown"
    GOOD = "good"
    WARNING = "warning"


# =============================================================================
# Impact Detection (击球点检测)
# =============================================================================
#
# Impact detection is centralized in:
#   - `tennis_analyzer.analysis.impact.WristSpeedImpactDetector` (pose-only, real-time)
#   - `tennis_analyzer.analysis.audio_impact.TwoPassImpactDetector` (audio + pose, offline)
#
# Monitors in this file should NOT own their own impact detectors; they should
# accept an external `is_impact` signal from the manager.


# =============================================================================
# Module 1: Extension Monitor (手臂延伸监控)
# =============================================================================

class ExtensionMonitor(BaseMonitor):
    """
    Monitor arm extension during stroke - Double Bend Edition.
    检测挥拍时手臂弯曲角度（双弯曲流派适配）

    判定标准：
    - 🟢 绿区 (Good): 120° ≤ θ ≤ 145° (主动弯曲，动力链通透)
    - 🟡 黄区 (Warning): 115° ≤ θ < 120° 或 145° < θ < 155°
    - 🔴 红区 (Fail): θ < 115° (被动缩手，锁死动力链)
    """

    def __init__(
        self,
        angle_green_min: float = 120.0,
        angle_green_max: float = 145.0,
        angle_yellow_min: float = 115.0,
        angle_yellow_max: float = 155.0,
        is_right_handed: bool = True
    ):
        super().__init__(is_right_handed)
        self.angle_green_min = angle_green_min
        self.angle_green_max = angle_green_max
        self.angle_yellow_min = angle_yellow_min
        self.angle_yellow_max = angle_yellow_max

        # Track angles around impact
        self.angle_buffer: deque = deque(maxlen=10)
        self.current_angle = 0.0
        self.impact_angle = 0.0

    def calculate_arm_angle(
        self,
        keypoints: np.ndarray,
        confidence: np.ndarray
    ) -> Optional[float]:
        """Calculate elbow angle (shoulder-elbow-wrist)."""
        min_conf = 0.3
        if (confidence[self.shoulder_idx] < min_conf or
            confidence[self.elbow_idx] < min_conf or
            confidence[self.wrist_idx] < min_conf):
            return None

        shoulder = keypoints[self.shoulder_idx]
        elbow = keypoints[self.elbow_idx]
        wrist = keypoints[self.wrist_idx]

        # Vectors from elbow
        v1 = shoulder - elbow
        v2 = wrist - elbow

        dot = np.dot(v1, v2)
        mag1 = np.linalg.norm(v1)
        mag2 = np.linalg.norm(v2)

        if mag1 < 1e-6 or mag2 < 1e-6:
            return None

        cos_angle = np.clip(dot / (mag1 * mag2), -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    def update(
        self,
        keypoints: np.ndarray,
        confidence: np.ndarray,
        is_forehand: bool,
        **kwargs
    ) -> Dict[str, Any]:
        # Calculate current angle
        angle = self.calculate_arm_angle(keypoints, confidence)
        if angle is not None:
            self.current_angle = angle
            # Only add reasonable angles to buffer (90° ~ 180°)
            if 90 <= angle <= 180:
                self.angle_buffer.append(angle)

        # Evaluate on external impact signal (provided by manager).
        is_impact = bool(kwargs.get("is_impact", False))
        if is_impact and len(self.angle_buffer) >= 3:
            recent_angles = list(self.angle_buffer)[-5:]
            self.impact_angle = float(np.median(recent_angles))
            self._evaluate()

        return {
            "angle": self.current_angle,
            "impact_angle": self.impact_angle,
            "status": self.quality_status,
            "message": self.feedback_message
        }

    def _evaluate(self):
        # 🔴 Red: θ < 115° (被动缩手，锁死动力链)
        if self.impact_angle < self.angle_yellow_min:
            need = max(0.0, float(self.angle_green_min - self.impact_angle))
            self.set_feedback(f"[伸展] 缩手：再打开 {need:.0f}°", "warning")
        # 🟢 Green: 120° ≤ θ ≤ 145° (主动弯曲，动力链通透)
        elif self.angle_green_min <= self.impact_angle <= self.angle_green_max:
            self.set_feedback("[伸展] 好!", "good")
        # 🟡 Yellow: 115° ≤ θ < 120° or 145° < θ < 155°
        elif self.impact_angle < self.angle_green_min:
            need = max(0.0, float(self.angle_green_min - self.impact_angle))
            self.set_feedback(f"[伸展] 再打开 {need:.0f}°", "good")
        elif self.impact_angle <= self.angle_yellow_max:
            need = max(0.0, float(self.impact_angle - self.angle_green_max))
            self.set_feedback(f"[伸展] 稍直：回收 {need:.0f}°", "good")
        else:
            need = max(0.0, float(self.impact_angle - self.angle_green_max))
            self.set_feedback(f"[伸展] 过直：回收 {need:.0f}°", "good")

    def reset(self):
        self.angle_buffer.clear()
        self.current_angle = 0.0
        self.impact_angle = 0.0
        self.feedback_message = ""
        self.quality_status = "unknown"


# =============================================================================
# Module 2: Knee Load Monitor (下肢加载监控)
# =============================================================================

class KneeLoadMonitor(BaseMonitor):
    """
    Monitor lower-body loading via knee bend (approx).
    用膝角近似“下肢加载/下沉”。

    说明：
    - 该指标不追求绝对“标准角度”，只给出一个保守的“是否太直”的提示。
    - 评价以击球前/击球附近的膝角中位数为准，避免单帧抖动。
    """

    def __init__(
        self,
        *,
        good_max_angle: float = 150.0,
        ok_max_angle: float = 160.0,
        is_right_handed: bool = True,
    ):
        super().__init__(is_right_handed)
        self.good_max_angle = float(good_max_angle)
        self.ok_max_angle = float(ok_max_angle)

        self.l_hip = KEYPOINT_NAMES["left_hip"]
        self.r_hip = KEYPOINT_NAMES["right_hip"]
        self.l_knee = KEYPOINT_NAMES["left_knee"]
        self.r_knee = KEYPOINT_NAMES["right_knee"]
        self.l_ankle = KEYPOINT_NAMES["left_ankle"]
        self.r_ankle = KEYPOINT_NAMES["right_ankle"]

        self._buf: deque[float] = deque(maxlen=15)
        self.current_angle: float = 0.0
        self.impact_angle: float = 0.0

    def _angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Optional[float]:
        v1 = np.asarray(p1, dtype=np.float32) - np.asarray(p2, dtype=np.float32)
        v2 = np.asarray(p3, dtype=np.float32) - np.asarray(p2, dtype=np.float32)
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 < 1e-6 or n2 < 1e-6:
            return None
        c = float(np.clip(float(np.dot(v1, v2)) / (n1 * n2), -1.0, 1.0))
        return float(np.degrees(np.arccos(c)))

    def _min_knee_angle(self, keypoints: np.ndarray, confidence: np.ndarray) -> Optional[float]:
        min_conf = 0.3

        angles: list[float] = []
        if (
            confidence[self.l_hip] >= min_conf
            and confidence[self.l_knee] >= min_conf
            and confidence[self.l_ankle] >= min_conf
        ):
            a = self._angle(keypoints[self.l_hip], keypoints[self.l_knee], keypoints[self.l_ankle])
            if a is not None:
                angles.append(float(a))

        if (
            confidence[self.r_hip] >= min_conf
            and confidence[self.r_knee] >= min_conf
            and confidence[self.r_ankle] >= min_conf
        ):
            a = self._angle(keypoints[self.r_hip], keypoints[self.r_knee], keypoints[self.r_ankle])
            if a is not None:
                angles.append(float(a))

        if not angles:
            return None
        # Smaller angle => more bend. Use the more-bent leg as the load proxy.
        return float(min(angles))

    def update(
        self,
        keypoints: np.ndarray,
        confidence: np.ndarray,
        is_forehand: bool,
        **kwargs,
    ) -> Dict[str, Any]:
        ang = self._min_knee_angle(keypoints, confidence)
        if ang is not None and 60.0 <= float(ang) <= 180.0:
            self.current_angle = float(ang)
            self._buf.append(float(ang))

        is_impact = bool(kwargs.get("is_impact", False))
        if is_impact and len(self._buf) >= 3:
            recent = list(self._buf)[-5:]
            self.impact_angle = float(np.median(np.asarray(recent, dtype=np.float32)))
            self._evaluate()

        return {
            "knee_angle": self.current_angle,
            "impact_knee_angle": self.impact_angle,
            "status": self.quality_status,
            "message": self.feedback_message,
        }

    def _evaluate(self) -> None:
        if self.impact_angle <= self.good_max_angle:
            self.set_feedback("[下肢] 好!", "good")
            return

        need = max(0.0, float(self.impact_angle - self.good_max_angle))
        if self.impact_angle <= self.ok_max_angle:
            self.set_feedback(f"[下肢] 再下沉 {need:.0f}°", "good")
        else:
            self.set_feedback(f"[下肢] 下沉不足：再下沉 {need:.0f}°", "warning")

    def reset(self) -> None:
        self._buf.clear()
        self.current_angle = 0.0
        self.impact_angle = 0.0
        self.feedback_message = ""
        self.quality_status = "unknown"


# =============================================================================
# Module 3: Linearization Monitor (随挥路径监测)
# =============================================================================

class LinearizationMonitor(BaseMonitor):
    """
    Monitor follow-through path direction.
    检测随挥路径是否向前送出（而非向上刷）

    改进逻辑：
    1. 击球后前3帧为"驱动窗口"(Drive Window)，检测初始穿透
    2. 如果驱动窗口内 ΔY/ΔX > 0.3，直接判定为"向上捞了"
    3. 只有驱动窗口足够平，才算"向前送"
    4. 后续随挥高度不影响判定（保护击球初期的水平位移）
    """

    def __init__(
        self,
        drive_slope_threshold: float = 0.3,  # 驱动窗口斜率阈值
        drive_window_frames: int = 3,  # 驱动窗口帧数
        is_right_handed: bool = True
    ):
        super().__init__(is_right_handed)
        self.drive_slope_threshold = drive_slope_threshold
        self.drive_window_frames = drive_window_frames

        # Post-impact tracking
        self.tracking_active = False
        self.post_impact_positions: List[np.ndarray] = []
        self.current_slope = 0.0

    def update(
        self,
        keypoints: np.ndarray,
        confidence: np.ndarray,
        is_forehand: bool,
        frame_idx: int = 0,
        **kwargs
    ) -> Dict[str, Any]:
        is_impact = bool(kwargs.get("is_impact", False))
        impact_keypoints = kwargs.get("impact_keypoints")
        impact_confidence = kwargs.get("impact_confidence")

        # Seed at the *estimated* impact frame when available, since the impact
        # detector triggers one frame after the speed peak.
        if (
            is_impact
            and impact_keypoints is not None
            and impact_confidence is not None
            and impact_confidence[self.wrist_idx] >= 0.3
        ):
            wrist_pos = impact_keypoints[self.wrist_idx].copy()
        else:
            if confidence[self.wrist_idx] < 0.3:
                return {
                    "slope": self.current_slope,
                    "status": self.quality_status,
                    "message": self.feedback_message
                }
            wrist_pos = keypoints[self.wrist_idx].copy()

        if is_impact:
            # Start tracking post-impact path
            self.tracking_active = True
            self.post_impact_positions = [wrist_pos]
        elif self.tracking_active:
            self.post_impact_positions.append(wrist_pos)

            # Evaluate after drive window frames collected
            if len(self.post_impact_positions) >= self.drive_window_frames + 1:
                self._evaluate()
                self.tracking_active = False
                self.post_impact_positions.clear()

        return {
            "slope": self.current_slope,
            "status": self.quality_status,
            "message": self.feedback_message
        }

    def _evaluate(self):
        if len(self.post_impact_positions) < self.drive_window_frames + 1:
            return

        # Drive Window: T0 (impact) to T3 (3 frames after)
        # Index 0 = T0 (impact frame)
        # Index 1,2,3 = T1,T2,T3 (drive window)
        t0 = self.post_impact_positions[0]
        t3 = self.post_impact_positions[self.drive_window_frames]

        dx = t3[0] - t0[0]  # Horizontal displacement
        dy = t3[1] - t0[1]  # Vertical displacement (positive = down in image coords)

        # Calculate drive window slope
        # In image coords: y increases downward
        # Upward movement = negative dy
        # We want to detect upward scooping: -dy / |dx|

        abs_dx = abs(dx)
        if abs_dx > 3:  # Minimum horizontal movement
            # If dy < 0, wrist moved up (bad for drive)
            if dy < 0:
                self.current_slope = -dy / abs_dx  # Positive value = upward slope
            else:
                self.current_slope = 0  # Downward or flat is fine
        else:
            # Very little horizontal movement - check if went up
            if dy < -5:
                self.current_slope = 1.0  # Went up without forward movement
            else:
                self.current_slope = 0

        # Evaluate based on drive window only
        if self.current_slope > self.drive_slope_threshold:
            self.set_feedback(f"[路径] 向上捞了", "warning")
        else:
            self.set_feedback("[路径] 向前送，好!", "good")

    def reset(self):
        self.tracking_active = False
        self.post_impact_positions.clear()
        self.current_slope = 0.0
        self.feedback_message = ""
        self.quality_status = "unknown"


# =============================================================================
# Module 3: Spacing Monitor (击球空间监测)
# =============================================================================

class SpacingMonitor(BaseMonitor):
    """
    Monitor hitting space - distance from wrist to body.
    检测击球时是否有足够的击球空间

    改进：
    1. 使用前10帧的平均肩宽作为静态基准
    2. 置信度检查：wrist.conf > 0.5
    3. 数据丢失时保持上一帧数据，不输出0
    4. Impact触发需要手腕在身体1.5倍肩宽范围内

    参数：
    - 肩宽基准: 41cm
    - 最小击球空间: 45.5cm (70% 臂长 或 1.2x 肩宽)
    """

    def __init__(
        self,
        shoulder_width_cm: float = SHOULDER_WIDTH_REF_CM,  # 41cm
        min_spacing_cm: float = MIN_SPACE_THRESHOLD_CM,    # 45.5cm
        impact_range_ratio: float = 1.5,  # 击球范围（肩宽倍数）
        is_right_handed: bool = True
    ):
        super().__init__(is_right_handed)
        self.shoulder_width_cm = shoulder_width_cm
        self.min_spacing_cm = min_spacing_cm
        self.impact_range_ratio = impact_range_ratio

        # Static baseline from first 10 frames
        self.calibration_frames: List[float] = []
        self.static_pixels_per_cm: Optional[float] = None
        self.calibration_complete = False

        # Current metrics with hold logic
        self.current_spacing_cm = 0.0
        self.last_valid_spacing_cm = 30.0  # Default reasonable value

    def _calibrate(self, shoulder_width_pixels: float):
        """Collect first 10 frames to establish static baseline."""
        if self.calibration_complete:
            return

        if shoulder_width_pixels > 20:  # Valid measurement
            self.calibration_frames.append(shoulder_width_pixels)

        if len(self.calibration_frames) >= 10:
            avg_shoulder_pixels = np.mean(self.calibration_frames)
            self.static_pixels_per_cm = avg_shoulder_pixels / self.shoulder_width_cm
            self.calibration_complete = True

    def calculate_spacing(
        self,
        keypoints: np.ndarray,
        confidence: np.ndarray
    ) -> Optional[float]:
        """Calculate horizontal wrist distance from body center in cm."""
        l_shoulder = KEYPOINT_NAMES["left_shoulder"]
        r_shoulder = KEYPOINT_NAMES["right_shoulder"]
        l_hip = KEYPOINT_NAMES["left_hip"]
        r_hip = KEYPOINT_NAMES["right_hip"]

        # Strict confidence check for wrist
        if confidence[self.wrist_idx] < 0.5:
            return None  # Will use last valid value

        min_conf = 0.3
        if (confidence[l_shoulder] < min_conf or
            confidence[r_shoulder] < min_conf or
            confidence[l_hip] < min_conf or
            confidence[r_hip] < min_conf):
            return None

        # Calculate current shoulder width
        shoulder_width_pixels = np.linalg.norm(
            keypoints[r_shoulder] - keypoints[l_shoulder]
        )

        # Calibration phase
        self._calibrate(shoulder_width_pixels)

        # Use static baseline if available, otherwise use current (with minimum)
        if self.static_pixels_per_cm is not None:
            pixels_per_cm = self.static_pixels_per_cm
        elif shoulder_width_pixels > 20:
            pixels_per_cm = shoulder_width_pixels / self.shoulder_width_cm
        else:
            return None  # Can't calculate

        # Body center x (average of shoulders and hips)
        body_center_x = (keypoints[l_shoulder][0] + keypoints[r_shoulder][0] +
                         keypoints[l_hip][0] + keypoints[r_hip][0]) / 4

        # Horizontal distance from wrist to body center
        wrist_x = keypoints[self.wrist_idx][0]
        horizontal_dist_pixels = abs(wrist_x - body_center_x)

        # Convert to cm
        spacing_cm = horizontal_dist_pixels / pixels_per_cm

        # Sanity check - spacing should be reasonable (0-100cm)
        if spacing_cm < 0 or spacing_cm > 100:
            return None

        return spacing_cm

    def _is_in_impact_zone(
        self,
        keypoints: np.ndarray,
        confidence: np.ndarray
    ) -> bool:
        """Check if wrist is in valid impact zone (within 1.5x shoulder width of body)."""
        l_shoulder = KEYPOINT_NAMES["left_shoulder"]
        r_shoulder = KEYPOINT_NAMES["right_shoulder"]

        if confidence[self.wrist_idx] < 0.5:
            return False

        # Use static baseline for impact zone calculation
        if self.static_pixels_per_cm is None:
            return True  # Allow during calibration

        shoulder_width_pixels = self.static_pixels_per_cm * self.shoulder_width_cm
        impact_range_pixels = shoulder_width_pixels * self.impact_range_ratio

        # Body center
        body_center_x = (keypoints[l_shoulder][0] + keypoints[r_shoulder][0]) / 2
        wrist_x = keypoints[self.wrist_idx][0]

        distance_from_body = abs(wrist_x - body_center_x)

        return distance_from_body <= impact_range_pixels

    def update(
        self,
        keypoints: np.ndarray,
        confidence: np.ndarray,
        is_forehand: bool,
        **kwargs
    ) -> Dict[str, Any]:
        # Calculate current spacing
        spacing = self.calculate_spacing(keypoints, confidence)

        if spacing is not None:
            self.current_spacing_cm = spacing
            self.last_valid_spacing_cm = spacing
        else:
            # Hold last valid value instead of showing 0
            self.current_spacing_cm = self.last_valid_spacing_cm

        # Evaluate on external impact signal (provided by manager).
        is_impact = bool(kwargs.get("is_impact", False))
        if is_impact:
            # Prefer evaluating spacing at the estimated impact pose (usually previous frame),
            # because the impact detector triggers one frame after the speed peak.
            impact_kps = kwargs.get("impact_keypoints")
            impact_conf = kwargs.get("impact_confidence")
            if impact_kps is not None and impact_conf is not None:
                spacing_i = self.calculate_spacing(impact_kps, impact_conf)
                if spacing_i is not None and self._is_in_impact_zone(impact_kps, impact_conf):
                    self._evaluate(spacing_i)
            elif spacing is not None and self._is_in_impact_zone(keypoints, confidence):
                self._evaluate(spacing)

        return {
            "spacing_cm": self.current_spacing_cm,
            "status": self.quality_status,
            "message": self.feedback_message
        }

    def _evaluate(self, spacing_cm: float):
        if spacing_cm >= self.min_spacing_cm:
            self.set_feedback("[空间] 好!", "good")
        else:
            need = max(0.0, float(self.min_spacing_cm - spacing_cm))
            self.set_feedback(f"[空间] 太近：再增加 {need:.0f}cm 距离", "warning")

    def reset(self):
        self.calibration_frames.clear()
        self.static_pixels_per_cm = None
        self.calibration_complete = False
        self.current_spacing_cm = 0.0
        self.last_valid_spacing_cm = 30.0
        self.feedback_message = ""
        self.quality_status = "unknown"


# =============================================================================
# Module 4: X-Factor Monitor (转体幅度监测)
# =============================================================================

class XFactorMonitor(BaseMonitor):
    """
    Unit Turn / X-Factor Monitor (behind-view preferred).
    转体幅度 + 转体时机（更适合背后视角）

    2D 关键点很难直接恢复真实的 3D "hip-shoulder separation"。
    这里采用一个更稳健的近似：
      - 使用肩宽的“缩短”（透视压缩）来估计躯干 yaw 旋转角度：
            turn_deg ≈ arccos(shoulder_width_px / max_shoulder_width_px)
      - 取击球前一段窗口内的最大 turn_deg 作为“转体最深点”
      - 同时检查“转体最深点”是否发生得足够早（避免临击球才转）

    注意：该近似在侧面视角会严重失真（肩宽长期被压缩），因此需要由上层
    manager 做视角 gating（侧面不算、背面才算）。
    """

    def __init__(
        self,
        xfactor_threshold: float = 35.0,  # 最小转体角度（近似）
        xfactor_excellent: float = 55.0,  # 优秀转体角度（近似）
        unit_turn_early_s: float = 0.25,  # 转体最深点距离击球至少多少秒（经验值）
        is_right_handed: bool = True
    ):
        super().__init__(is_right_handed)
        self.xfactor_threshold = xfactor_threshold
        self.xfactor_excellent = xfactor_excellent
        self.unit_turn_early_s = float(unit_turn_early_s)

        # Track shoulder widths over swing (store frame_idx for timing).
        self.xfactor_buffer: deque = deque(maxlen=60)  # (frame_idx, turn_deg)
        self._max_shoulder_width_px: float = 0.0
        self.current_xfactor = 0.0
        self.max_xfactor = 0.0
        self.max_xfactor_frame_idx: Optional[int] = None
        self.time_to_impact_s: float = 0.0

    def _turn_deg_from_width_ratio(self, ratio: float) -> float:
        # ratio in (0,1]; 1 => facing camera => 0 deg, smaller => more turn.
        r = float(np.clip(float(ratio), 1e-3, 1.0))
        return float(np.degrees(np.arccos(r)))

    def calculate_xfactor(
        self,
        keypoints: np.ndarray,
        confidence: np.ndarray
    ) -> Optional[float]:
        """Estimate unit-turn angle from shoulder width compression."""
        l_shoulder = KEYPOINT_NAMES["left_shoulder"]
        r_shoulder = KEYPOINT_NAMES["right_shoulder"]

        min_conf = 0.35
        if confidence[l_shoulder] < min_conf or confidence[r_shoulder] < min_conf:
            return None

        sw_px = float(np.linalg.norm(np.asarray(keypoints[r_shoulder]) - np.asarray(keypoints[l_shoulder])))
        if sw_px < 1e-6:
            return None

        # Update running max within this swing window.
        self._max_shoulder_width_px = max(float(self._max_shoulder_width_px), sw_px)
        if self._max_shoulder_width_px < 1e-6:
            return None

        ratio = float(sw_px) / float(self._max_shoulder_width_px)
        return self._turn_deg_from_width_ratio(ratio)

    def update(
        self,
        keypoints: np.ndarray,
        confidence: np.ndarray,
        is_forehand: bool,
        frame_idx: int = 0,
        **kwargs
    ) -> Dict[str, Any]:
        # Calculate current X-Factor
        xfactor = self.calculate_xfactor(keypoints, confidence)
        if xfactor is not None:
            self.current_xfactor = xfactor
            self.xfactor_buffer.append((int(frame_idx), float(xfactor)))

        # Evaluate on external impact signal (provided by manager).
        is_impact = bool(kwargs.get("is_impact", False))
        if is_impact and len(self.xfactor_buffer) >= 5:
            fps = float(kwargs.get("fps", 30.0)) if kwargs.get("fps") else 30.0
            impact_frame_idx = kwargs.get("impact_frame_idx")
            finalize_idx = int(impact_frame_idx) if impact_frame_idx is not None else int(frame_idx)
            self._finalize_on_impact(frame_idx=finalize_idx, fps=fps)
            self.xfactor_buffer.clear()
            self._max_shoulder_width_px = 0.0

        return {
            "xfactor": self.current_xfactor,
            "max_xfactor": self.max_xfactor,
            "time_to_impact_s": self.time_to_impact_s,
            "status": self.quality_status,
            "message": self.feedback_message
        }

    def _finalize_on_impact(self, frame_idx: int, fps: float) -> None:
        """Compute max xfactor + timing, then set feedback."""
        if not self.xfactor_buffer:
            return
        # Pick the frame with the maximum X-Factor in the buffer.
        max_frame, max_val = max(self.xfactor_buffer, key=lambda t: t[1])
        self.max_xfactor = float(max_val)
        self.max_xfactor_frame_idx = int(max_frame)

        fps = float(fps) if fps and fps > 0 else 30.0
        self.time_to_impact_s = max(0.0, float(frame_idx - max_frame) / fps)

        # Depth first.
        if self.max_xfactor < self.xfactor_threshold:
            need = max(0.0, float(self.xfactor_threshold - self.max_xfactor))
            self.set_feedback(f"[转体] 再多转体 {need:.0f}°", "warning")
            return

        # Timing (unit turn happens too close to contact -> often causes late contact/arming).
        if self.time_to_impact_s < self.unit_turn_early_s:
            need_s = max(0.0, float(self.unit_turn_early_s - self.time_to_impact_s))
            self.set_feedback(f"[转体] 转体要更早：提前 {need_s:.2f}s", "warning")
            return

        # Good / excellent.
        if self.max_xfactor >= self.xfactor_excellent:
            self.set_feedback("[转体] 优秀且提前!", "good")
        else:
            self.set_feedback("[转体] 好且提前", "good")

    def reset(self):
        self.xfactor_buffer.clear()
        self._max_shoulder_width_px = 0.0
        self.current_xfactor = 0.0
        self.max_xfactor = 0.0
        self.max_xfactor_frame_idx = None
        self.time_to_impact_s = 0.0
        self.feedback_message = ""
        self.quality_status = "unknown"


# =============================================================================
# Module 5: Shoulder Alignment Monitor (肩线位置检查) - NEW
# =============================================================================

class ShoulderAlignmentMonitor(BaseMonitor):
    """
    Monitor shoulder alignment at impact.
    检测击球瞬间肩膀位置，确保击球点靠前，身体完全打开

    判定：
    - 持拍手肩膀在前 (R_Shoulder.x > L_Shoulder.x for right-hander) = 好
    - 持拍手肩膀在后 = 身体未打开
    """

    def __init__(self, is_right_handed: bool = True):
        super().__init__(is_right_handed)

        # Current state
        self.shoulder_front = False

    def update(
        self,
        keypoints: np.ndarray,
        confidence: np.ndarray,
        is_forehand: bool,
        **kwargs
    ) -> Dict[str, Any]:
        l_shoulder = KEYPOINT_NAMES["left_shoulder"]
        r_shoulder = KEYPOINT_NAMES["right_shoulder"]
        is_impact = bool(kwargs.get("is_impact", False))
        forward_sign = float(kwargs.get("forward_sign", 1.0))
        if is_impact:
            impact_kps = kwargs.get("impact_keypoints")
            impact_conf = kwargs.get("impact_confidence")
            if (
                impact_kps is not None
                and impact_conf is not None
                and impact_conf[l_shoulder] >= 0.5
                and impact_conf[r_shoulder] >= 0.5
            ):
                self._evaluate(impact_kps, l_shoulder, r_shoulder, forward_sign=forward_sign)
            elif confidence[l_shoulder] >= 0.5 and confidence[r_shoulder] >= 0.5:
                self._evaluate(keypoints, l_shoulder, r_shoulder, forward_sign=forward_sign)

        return {
            "shoulder_front": self.shoulder_front,
            "status": self.quality_status,
            "message": self.feedback_message
        }

    def _evaluate(
        self, keypoints: np.ndarray, l_shoulder: int, r_shoulder: int, *, forward_sign: float
    ):
        """Check if the dominant shoulder is "in front" along the inferred forward axis.

        We assume side-view-ish footage where the forward (net) direction mostly
        maps to image X. `forward_sign` indicates which X direction is "forward".
        """
        forward_sign = float(forward_sign) if abs(float(forward_sign)) > 1e-6 else 1.0

        dom = r_shoulder if self.is_right_handed else l_shoulder
        other = l_shoulder if self.is_right_handed else r_shoulder

        dom_x = float(keypoints[dom][0])
        other_x = float(keypoints[other][0])

        # Multiply by forward_sign so a single comparison works for both directions.
        self.shoulder_front = (dom_x * forward_sign) > (other_x * forward_sign)

        if self.shoulder_front:
            self.set_feedback("[肩线] 击球点靠前!", "good")
        else:
            self.set_feedback("[肩线] 身体未打开", "warning")

    def reset(self):
        self.shoulder_front = False
        self.feedback_message = ""
        self.quality_status = "unknown"


# =============================================================================
# Module 6: Wrist Laid-back Monitor (手腕锁定检测) - NEW
# =============================================================================

class WristLaidbackMonitor(BaseMonitor):
    """
    Monitor wrist laid-back position during backswing.
    检测引拍时手腕是否正确锁定（laid-back）

    判定1：小臂垂直夹角 (Forearm Verticality)
    - 计算 R_Wrist 和 R_Elbow 连线与水平线的夹角
    - 如果夹角 α > 20° (手比肘高或平)，判定为 "小臂过载 - 肌肉紧绷!" 🔴

    判定2：手腕预设检查 (Wrist Preset)
    - 如果手部高于肘部，手腕无法实现 135° 的 Laid-back
    - 显示 "物理死锁 - 无法产生离心力!" 🔴

    正确的 Laid-back：
    - 肘部抬起，手腕在肘部下方
    - 小臂与水平线夹角为负（手腕低于肘部）
    """

    def __init__(
        self,
        forearm_angle_threshold: float = 20.0,  # 小臂与水平线夹角阈值（度）
        elbow_wrist_height_threshold: float = 10.0,  # 肘腕高度差阈值（像素）
        is_right_handed: bool = True
    ):
        super().__init__(is_right_handed)
        self.forearm_angle_threshold = forearm_angle_threshold
        self.elbow_wrist_height_threshold = elbow_wrist_height_threshold

        # Track backswing phase
        self.in_backswing = False
        self.backswing_checked = False
        self.wrist_laid_back = True
        self.prev_wrist_x = None

    def calculate_forearm_angle(
        self,
        keypoints: np.ndarray,
        confidence: np.ndarray
    ) -> Optional[float]:
        """
        Calculate forearm angle relative to horizontal.
        Positive angle = wrist higher than elbow (bad)
        Negative angle = wrist lower than elbow (good)
        """
        if (confidence[self.elbow_idx] < 0.5 or
            confidence[self.wrist_idx] < 0.5):
            return None

        elbow = keypoints[self.elbow_idx]
        wrist = keypoints[self.wrist_idx]

        dx = wrist[0] - elbow[0]
        dy = wrist[1] - elbow[1]

        # In image coords, Y increases downward
        # If wrist.y < elbow.y, wrist is higher (bad), dy is negative
        # We want positive angle when wrist is higher, so negate dy
        angle = np.degrees(np.arctan2(-dy, abs(dx)))

        return angle

    def check_laid_back(
        self,
        keypoints: np.ndarray,
        confidence: np.ndarray
    ) -> Tuple[bool, str]:
        """
        Check if wrist is properly laid back.

        Returns:
            (is_laid_back, error_message)
        """
        if (confidence[self.elbow_idx] < 0.5 or
            confidence[self.wrist_idx] < 0.5):
            return True, ""  # Can't determine, assume OK

        elbow_y = keypoints[self.elbow_idx][1]
        wrist_y = keypoints[self.wrist_idx][1]

        # 判定2：手腕预设检查 - 手部高于肘部
        # In pixel coords, smaller Y = higher position
        if wrist_y < elbow_y - self.elbow_wrist_height_threshold:
            # Wrist too high relative to elbow: hard to keep a stable laid-back structure.
            return False, "[手腕] 手高于肘：手再低一点 / 肘再抬一点（更容易稳定 laid-back）"

        # 判定1：小臂垂直夹角
        forearm_angle = self.calculate_forearm_angle(keypoints, confidence)
        if forearm_angle is not None and forearm_angle > self.forearm_angle_threshold:
            return False, "[手腕] 小臂偏高：放松手臂，让拍头自然下坠（别硬抬）"

        # 正确状态：手腕在肘部下方或水平
        return True, ""

    def update(
        self,
        keypoints: np.ndarray,
        confidence: np.ndarray,
        is_forehand: bool,
        is_impact: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        if confidence[self.wrist_idx] < 0.5:
            return {
                "wrist_laid_back": self.wrist_laid_back,
                "status": self.quality_status,
                "message": self.feedback_message
            }

        wrist_x = keypoints[self.wrist_idx][0]
        forward_sign = float(kwargs.get("forward_sign", 1.0))
        if abs(forward_sign) < 1e-6:
            forward_sign = 1.0

        # Detect backswing phase (wrist moving backward)
        if self.prev_wrist_x is not None:
            # Generic: backswing means wrist moving opposite the forward direction.
            dx = float(wrist_x - self.prev_wrist_x)
            wrist_moving_back = (dx * forward_sign) < -3.0

            if wrist_moving_back and not self.backswing_checked:
                self.in_backswing = True

        self.prev_wrist_x = wrist_x

        # Check laid-back during backswing
        if self.in_backswing and not self.backswing_checked:
            is_laid_back, error_msg = self.check_laid_back(keypoints, confidence)

            if not is_laid_back:
                self.wrist_laid_back = False
                self.set_feedback(error_msg, "warning")
                self.backswing_checked = True
            else:
                self.wrist_laid_back = True

        # Reset after impact for next stroke
        if is_impact:
            self.in_backswing = False
            self.backswing_checked = False

        return {
            "wrist_laid_back": self.wrist_laid_back,
            "status": self.quality_status,
            "message": self.feedback_message
        }

    def reset(self):
        self.in_backswing = False
        self.backswing_checked = False
        self.wrist_laid_back = True
        self.prev_wrist_x = None
        self.feedback_message = ""
        self.quality_status = "unknown"


# =============================================================================
# Module 7: Balance Monitor (平衡/头部稳定) - NEW (tennis_coach "Off Balance")
# =============================================================================

class BalanceMonitor(BaseMonitor):
    """
    Monitor balance via head (nose) stability after impact.

    tennis_coach skill "Off Balance" heuristics:
    - Head falling over (large lean angle at impact)
    - Head swaying a lot right after contact (poor stability)
    """

    def __init__(
        self,
        window_frames: int = 8,
        lean_angle_threshold_deg: float = 20.0,
        head_sway_threshold_sw: float = 0.25,  # shoulder-widths
        is_right_handed: bool = True,
    ):
        super().__init__(is_right_handed)
        self.nose_idx = KEYPOINT_NAMES["nose"]
        self.l_hip = KEYPOINT_NAMES["left_hip"]
        self.r_hip = KEYPOINT_NAMES["right_hip"]
        self.l_shoulder = KEYPOINT_NAMES["left_shoulder"]
        self.r_shoulder = KEYPOINT_NAMES["right_shoulder"]

        self.window_frames = int(window_frames)
        self.lean_angle_threshold_deg = float(lean_angle_threshold_deg)
        self.head_sway_threshold_sw = float(head_sway_threshold_sw)

        self._tracking = False
        self._frames = 0
        self._nose0: Optional[np.ndarray] = None
        self._hip0: Optional[np.ndarray] = None
        self._shoulder_width_px: Optional[float] = None
        self._max_head_sway_sw: float = 0.0

    def signal_impact(self, keypoints: np.ndarray, confidence: np.ndarray) -> None:
        """Start a short post-impact tracking window seeded at impact."""
        min_conf = 0.5
        if (
            confidence[self.nose_idx] < min_conf
            or confidence[self.l_hip] < min_conf
            or confidence[self.r_hip] < min_conf
        ):
            return

        self._tracking = True
        self._frames = 0
        self._nose0 = np.asarray(keypoints[self.nose_idx], dtype=np.float32).copy()
        hip_center = (np.asarray(keypoints[self.l_hip], dtype=np.float32) + np.asarray(keypoints[self.r_hip], dtype=np.float32)) / 2.0
        self._hip0 = hip_center.copy()

        if confidence[self.l_shoulder] >= 0.35 and confidence[self.r_shoulder] >= 0.35:
            self._shoulder_width_px = float(
                np.linalg.norm(
                    np.asarray(keypoints[self.r_shoulder], dtype=np.float32)
                    - np.asarray(keypoints[self.l_shoulder], dtype=np.float32)
                )
            )
        else:
            self._shoulder_width_px = None

        self._max_head_sway_sw = 0.0

    def update_frame(self, keypoints: np.ndarray, confidence: np.ndarray) -> Dict[str, Any]:
        """Update tracking; returns current/frozen evaluation."""
        if not self._tracking or self._nose0 is None or self._hip0 is None:
            return {
                "lean_angle_deg": 0.0,
                "head_sway_sw": self._max_head_sway_sw,
                "status": self.quality_status,
                "message": self.feedback_message,
            }

        min_conf = 0.5
        if confidence[self.nose_idx] < min_conf:
            return {
                "lean_angle_deg": 0.0,
                "head_sway_sw": self._max_head_sway_sw,
                "status": self.quality_status,
                "message": self.feedback_message,
            }

        nose = np.asarray(keypoints[self.nose_idx], dtype=np.float32)

        # Head sway (horizontal) relative to impact frame, normalized by shoulder width.
        dx = float(abs(nose[0] - self._nose0[0]))
        if self._shoulder_width_px and self._shoulder_width_px > 1e-6:
            sway_sw = dx / self._shoulder_width_px
        else:
            sway_sw = 0.0
        self._max_head_sway_sw = max(self._max_head_sway_sw, sway_sw)

        self._frames += 1
        if self._frames < self.window_frames:
            return {
                "lean_angle_deg": 0.0,
                "head_sway_sw": self._max_head_sway_sw,
                "status": self.quality_status,
                "message": self.feedback_message,
            }

        # Evaluate at the end of the window, using the initial impact-frame geometry.
        vec = self._nose0 - self._hip0
        lean_angle = float(np.degrees(np.arctan2(abs(vec[0]), abs(vec[1]) + 1e-6)))

        if lean_angle > self.lean_angle_threshold_deg or self._max_head_sway_sw > self.head_sway_threshold_sw:
            self.set_feedback("[平衡] 头部不稳（击球后晃/歪）", "warning")
        else:
            self.set_feedback("[平衡] 稳 ✓", "good")

        self._tracking = False
        return {
            "lean_angle_deg": lean_angle,
            "head_sway_sw": self._max_head_sway_sw,
            "status": self.quality_status,
            "message": self.feedback_message,
        }

    def update(
        self,
        keypoints: np.ndarray,
        confidence: np.ndarray,
        is_forehand: bool,
        **kwargs,
    ) -> Dict[str, Any]:
        """BaseMonitor-compatible update wrapper."""
        is_impact = bool(kwargs.get("is_impact", False))
        impact_keypoints = kwargs.get("impact_keypoints")
        impact_confidence = kwargs.get("impact_confidence")
        if is_impact:
            if impact_keypoints is not None and impact_confidence is not None:
                self.signal_impact(impact_keypoints, impact_confidence)
            else:
                self.signal_impact(keypoints, confidence)
        return self.update_frame(keypoints, confidence)

    def reset(self) -> None:
        self._tracking = False
        self._frames = 0
        self._nose0 = None
        self._hip0 = None
        self._shoulder_width_px = None
        self._max_head_sway_sw = 0.0
        self.feedback_message = ""
        self.quality_status = "unknown"


# =============================================================================
# Kinetic Chain Manager (统一管理器)
# =============================================================================

class KineticChainManager:
    """
    Unified manager for all kinetic chain monitors.
    统一管理所有动力链监控模块 (v2.0 - Double Bend Edition)

    重要改进：使用共享的 ImpactDetector，确保所有模块同时评估
    """

    def __init__(
        self,
        is_right_handed: bool = True,
        fps: float = 30.0,
        *,
        camera_view: str = CAMERA_VIEW_UNKNOWN,
        profile: Optional[StrokeProfile] = None,
        impact_events_by_frame: Optional[Dict[int, ImpactEvent]] = None,
    ):
        """Initialize all monitors.

        Args:
            is_right_handed: Player handedness (affects dominant-side keypoints).
            fps: Video FPS (used for impact detection normalization).
        """
        self.is_right_handed = is_right_handed
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.camera_view = str(camera_view or CAMERA_VIEW_UNKNOWN).lower()
        if self.camera_view not in (CAMERA_VIEW_SIDE, CAMERA_VIEW_BACK, CAMERA_VIEW_UNKNOWN):
            self.camera_view = CAMERA_VIEW_UNKNOWN
        self.profile = profile or FOREHAND_PROFILE
        # Optional external impact schedule (e.g., audio+pose two-pass impacts).
        # When provided, it becomes the single source of truth for `is_impact`.
        self._impact_events_by_frame = impact_events_by_frame

        # Shared impact detector - single source of truth for all checkpoint evaluations
        self.shared_impact_detector = WristSpeedImpactDetector(
            fps=self.fps,
            is_right_handed=is_right_handed,
            cooldown_frames=25,
        )

        # Big 3 Checkpoints (tennis_coach skill priority order)
        p = self.profile
        self.big3_contact_point = Big3ContactPointMonitor(
            is_right_handed=is_right_handed,
            good_min_cm=p.contact_good_min_cm,
            good_max_cm=p.contact_good_max_cm,
            ok_min_cm=p.contact_ok_min_cm,
            good_threshold_px=p.contact_good_threshold_px,
            ok_threshold_px=p.contact_ok_threshold_px,
        )
        self.big3_weight_transfer = Big3WeightTransferMonitor(
            is_right_handed=is_right_handed,
            good_threshold_px=p.weight_transfer_good_px,
            ok_threshold_px=p.weight_transfer_ok_px,
            good_threshold_cm=p.weight_transfer_good_cm,
            ok_threshold_cm=p.weight_transfer_ok_cm,
        )
        cz_window_frames = int(round(max(0.10, float(p.contact_zone_window_s)) * self.fps))
        # Keep within a reasonable range to avoid capturing recovery steps.
        cz_window_frames = int(min(18, max(4, cz_window_frames)))
        self.big3_contact_zone = Big3ContactZoneMonitor(
            is_right_handed=is_right_handed,
            window_size=cz_window_frames,
            good_forward_cm=p.contact_zone_good_forward_cm,
            good_upward_cm=p.contact_zone_good_upward_cm,
            good_forward_px=p.contact_zone_good_forward_px,
            good_upward_px=p.contact_zone_good_upward_px,
        )

        self._last_big3 = {
            "contact_point": None,
            "weight_transfer": None,
            "contact_zone": None,
        }

        # Keep previous-frame snapshots so we can align Big3 evaluation to the
        # *estimated* impact frame (speed peak occurs one frame before detector triggers).
        self._prev_keypoints: Optional[np.ndarray] = None
        self._prev_confidence: Optional[np.ndarray] = None
        self._prev_frame_idx: Optional[int] = None

        # Initialize all monitors with tuned parameters
        self.extension_monitor = ExtensionMonitor(
            angle_green_min=120.0,   # 🟢 绿区下限
            angle_green_max=145.0,   # 🟢 绿区上限
            angle_yellow_min=115.0,  # 🟡 黄区下限 (below = 🔴)
            angle_yellow_max=155.0,  # 🟡 黄区上限
            is_right_handed=is_right_handed
        )
        self.knee_load_monitor = KneeLoadMonitor(is_right_handed=is_right_handed)
        self.spacing_monitor = SpacingMonitor(
            shoulder_width_cm=SHOULDER_WIDTH_REF_CM,  # 41cm
            min_spacing_cm=MIN_SPACE_THRESHOLD_CM,    # 45.5cm
            is_right_handed=is_right_handed
        )
        self.xfactor_monitor = XFactorMonitor(
            xfactor_threshold=35.0,   # 转体 >= 35° 为好（近似）
            xfactor_excellent=55.0,   # 转体 >= 55° 为优秀（近似）
            unit_turn_early_s=p.unit_turn_early_s,
            is_right_handed=is_right_handed
        )
        self.wrist_monitor = WristLaidbackMonitor(
            forearm_angle_threshold=20.0,        # 小臂与水平线夹角阈值（度）
            elbow_wrist_height_threshold=10.0,   # 肘腕高度差阈值（像素）
            is_right_handed=is_right_handed
        )
        self.balance_monitor = BalanceMonitor(is_right_handed=is_right_handed)

        # Forward direction estimation (side-view ambiguity).
        self._dom_wrist_idx = (
            KEYPOINT_NAMES["right_wrist"] if is_right_handed else KEYPOINT_NAMES["left_wrist"]
        )
        self._prev_dom_wrist_x: Optional[float] = None
        self._dom_wrist_dx_hist: deque[float] = deque(maxlen=7)
        # Latch forward sign briefly after impact so post-impact windows stay consistent.
        self._active_forward_sign: Optional[float] = None
        self._active_forward_ttl: int = 0
        self._active_forward_axis_unit: Optional[Tuple[float, float]] = None

    def update(
        self,
        keypoints: np.ndarray,
        confidence: np.ndarray,
        is_forehand: bool,
        frame_idx: int = 0
    ) -> Dict[str, Dict[str, Any]]:
        """
        Update all monitors with synchronized impact detection.

        Returns:
            Dictionary with results from all monitors
        """
        # ---------------------------------------------------------------------
        # 1) Forward direction inference (side-view ambiguity)
        # ---------------------------------------------------------------------
        if confidence[self._dom_wrist_idx] >= 0.35:
            wx = float(keypoints[self._dom_wrist_idx][0])
            if self._prev_dom_wrist_x is not None:
                self._dom_wrist_dx_hist.append(wx - self._prev_dom_wrist_x)
            self._prev_dom_wrist_x = wx

        inferred_forward_sign = 1.0
        if len(self._dom_wrist_dx_hist) >= 3:
            dx_med = float(np.median(np.asarray(self._dom_wrist_dx_hist, dtype=np.float32)))
            if abs(dx_med) > 1e-3:
                inferred_forward_sign = 1.0 if dx_med > 0 else -1.0

        # ---------------------------------------------------------------------
        # 2) Shared impact detection (single source of truth)
        # ---------------------------------------------------------------------
        pose_event, speed = self.shared_impact_detector.update(frame_idx, keypoints, confidence)

        # If a schedule is provided, use it and ignore pose-triggered impacts.
        if self._impact_events_by_frame is not None:
            impact_event = self._impact_events_by_frame.get(int(frame_idx))
            is_impact = impact_event is not None
            if impact_event is not None:
                # Keep speed metric consistent with the scheduled impact.
                speed = float(impact_event.peak_speed_px_s)
        else:
            impact_event = pose_event
            is_impact = impact_event is not None

        impact_frame_idx: Optional[int] = None
        impact_kps: Optional[np.ndarray] = None
        impact_conf: Optional[np.ndarray] = None
        if impact_event is not None:
            impact_frame_idx = int(impact_event.impact_frame_idx)
            if (
                self._prev_keypoints is not None
                and self._prev_confidence is not None
                and self._prev_frame_idx == impact_frame_idx
            ):
                impact_kps = self._prev_keypoints
                impact_conf = self._prev_confidence
            else:
                impact_kps = keypoints
                impact_conf = confidence

        # Prefer the impact-event velocity direction to set forward_sign.
        forward_sign = inferred_forward_sign
        forward_axis_unit: Optional[Tuple[float, float]] = None
        if impact_event is not None:
            dx = float(impact_event.peak_velocity_unit[0])
            if abs(dx) > 1e-3:
                forward_sign = 1.0 if dx > 0 else -1.0
            # Latch for post-impact windows.
            self._active_forward_sign = forward_sign
            self._active_forward_ttl = 20
            forward_axis_unit = impact_event.peak_velocity_unit
            self._active_forward_axis_unit = forward_axis_unit
        elif self._active_forward_ttl > 0 and self._active_forward_sign is not None:
            forward_sign = float(self._active_forward_sign)
            forward_axis_unit = self._active_forward_axis_unit
            self._active_forward_ttl -= 1
        else:
            self._active_forward_sign = None
            self._active_forward_ttl = 0
            self._active_forward_axis_unit = None

        view = self.camera_view
        is_side_view = view == CAMERA_VIEW_SIDE
        is_back_view = view == CAMERA_VIEW_BACK

        # ---------------------------------------------------------------------
        # 3) Big3 monitors (freeze at impact)
        # ---------------------------------------------------------------------
        cp_unknown = Big3ContactPointStatus(
            status=Big3Status.UNKNOWN,
            delta_px=0.0,
            delta_sw=None,
            delta_cm=None,
            message="",
        )
        cz_unknown = Big3ContactZoneStatus(
            status=Big3Status.UNKNOWN,
            forward_px=0.0,
            forward_sw=None,
            forward_cm=None,
            upward_px=0.0,
            upward_sw=None,
            upward_cm=None,
            message="",
        )

        # Contact point is only meaningful in side view (depth is not observable from behind).
        if is_side_view:
            cp_status = self.big3_contact_point.update(
                keypoints,
                confidence,
                frame_idx=int(frame_idx),
                is_impact=False,
                forward_sign=forward_sign,
                forward_axis_unit=forward_axis_unit,
                camera_view=view,
                fps=self.fps,
            )
        else:
            cp_status = cp_unknown

        wt_status = self.big3_weight_transfer.update(
            keypoints,
            confidence,
            frame_idx=int(frame_idx),
            is_impact=is_impact,
            impact_frame_idx=impact_frame_idx,
            impact_keypoints=impact_kps,
            impact_confidence=impact_conf,
            forward_sign=forward_sign,
            fps=self.fps,
            camera_view=view,
        )

        if is_impact and impact_kps is not None and impact_conf is not None:
            # Freeze CP at the estimated impact frame.
            if is_side_view:
                bounce_idx = int(getattr(impact_event, "bounce_frame_idx")) if (impact_event and getattr(impact_event, "bounce_frame_idx", None) is not None) else None
                self._last_big3["contact_point"] = self.big3_contact_point.update(
                    impact_kps,
                    impact_conf,
                    frame_idx=int(impact_frame_idx) if impact_frame_idx is not None else int(frame_idx),
                    is_impact=True,
                    impact_frame_idx=int(impact_frame_idx) if impact_frame_idx is not None else int(frame_idx),
                    bounce_frame_idx=bounce_idx,
                    forward_sign=forward_sign,
                    forward_axis_unit=forward_axis_unit,
                    camera_view=view,
                    fps=self.fps,
                )
            else:
                self._last_big3["contact_point"] = cp_unknown
            # Weight transfer evaluates on impact and returns a frozen status.
            self._last_big3["weight_transfer"] = wt_status

            # Contact zone (penetration + brush) is only reliable in side view.
            if is_side_view:
                # Start contact-zone tracking at the estimated impact frame and seed it.
                if impact_frame_idx is not None:
                    self.big3_contact_zone.signal_impact(int(impact_frame_idx))
                    self.big3_contact_zone.update(
                        impact_kps,
                        impact_conf,
                        forward_sign=forward_sign,
                        forward_axis_unit=forward_axis_unit,
                        camera_view=view,
                    )
                else:
                    self.big3_contact_zone.signal_impact(int(frame_idx))
                    self.big3_contact_zone.update(
                        keypoints,
                        confidence,
                        forward_sign=forward_sign,
                        forward_axis_unit=forward_axis_unit,
                        camera_view=view,
                    )

                # Reset contact-zone snapshot for this new impact.
                self._last_big3["contact_zone"] = None
            else:
                self._last_big3["contact_zone"] = cz_unknown

        if is_side_view:
            cz_status = self.big3_contact_zone.update(
                keypoints,
                confidence,
                forward_sign=forward_sign,
                forward_axis_unit=forward_axis_unit,
                camera_view=view,
            )
            if (
                self._last_big3["contact_zone"] is None
                and cz_status is not None
                and hasattr(cz_status, "status")
                and cz_status.status != Big3Status.UNKNOWN
            ):
                self._last_big3["contact_zone"] = cz_status
        else:
            cz_status = cz_unknown
            self._last_big3["contact_zone"] = cz_unknown

        # ---------------------------------------------------------------------
        # 4) Secondary monitors (use shared impact + aligned impact pose)
        # ---------------------------------------------------------------------
        common_kwargs = {
            "is_impact": is_impact,
            "impact_event": impact_event,
            "impact_frame_idx": impact_frame_idx,
            "impact_keypoints": impact_kps,
            "impact_confidence": impact_conf,
            "forward_sign": forward_sign,
            "fps": self.fps,
        }

        # View-gated secondary monitors.
        na = {"status": "unknown", "message": ""}

        # Swing speed is intentionally disabled in reports and monitoring output.
        # Keep the field for backward compatibility with downstream consumers.
        swing_speed_kmh: Optional[float] = None

        results = {
            "big3": {
                "contact_point": self._last_big3["contact_point"] or cp_status,
                "weight_transfer": self._last_big3["weight_transfer"] or wt_status,
                "contact_zone": self._last_big3["contact_zone"] or cz_status,
                "is_impact": is_impact,
                # Estimated contact frame (aligned to the speed peak). Useful for reporting.
                "impact_frame_idx": int(impact_frame_idx) if impact_frame_idx is not None else None,
                "wrist_speed_px_s": speed,
                "wrist_speed_sw_s": float(impact_event.peak_speed_sw_s) if (impact_event and impact_event.peak_speed_sw_s is not None) else None,
                "swing_speed_kmh": swing_speed_kmh,
            },
            "extension": self.extension_monitor.update(keypoints, confidence, is_forehand, **common_kwargs),
            "knee_load": self.knee_load_monitor.update(keypoints, confidence, is_forehand, **common_kwargs),
            # Back-view only: spacing/crowding is most meaningful from behind.
            "spacing": self.spacing_monitor.update(keypoints, confidence, is_forehand, **common_kwargs)
            if is_back_view
            else dict(na),
            # Back-view only: unit turn estimate uses shoulder-width compression.
            "xfactor": self.xfactor_monitor.update(
                keypoints, confidence, is_forehand, frame_idx=int(frame_idx), **common_kwargs
            )
            if is_back_view
            else dict(na),
            # Side-view only: wrist laid-back / structure is easiest to judge from the side.
            "wrist": self.wrist_monitor.update(
                keypoints, confidence, is_forehand, is_impact=is_impact, **{"forward_sign": forward_sign}
            )
            if is_side_view
            else dict(na),
            "balance": self.balance_monitor.update(keypoints, confidence, is_forehand, **common_kwargs),
        }

        # Check for Double Bend success (复合判定) - only on impact
        if is_impact:
            ext_angle = results["extension"].get("impact_angle", 0)
            spacing = results["spacing"].get("spacing_cm", 0)

            if (120 <= ext_angle <= 145 and spacing >= MIN_SPACE_THRESHOLD_CM):
                self.extension_monitor.set_feedback("[伸展] Double Bend 通透!", "good")
                results["extension"]["message"] = self.extension_monitor.feedback_message

        # Save previous-frame snapshots for next update call.
        self._prev_keypoints = np.asarray(keypoints).copy()
        self._prev_confidence = np.asarray(confidence).copy()
        self._prev_frame_idx = int(frame_idx)

        return results

    def get_active_feedback(self) -> List[Tuple[str, str, str]]:
        """
        Get all active feedback messages.

        Returns:
            List of (module_name, message, status) tuples
        """
        feedbacks = []

        # Big 3 first (skill priority order), then secondary checkpoints.
        monitors = [
            ("contact_point", self._last_big3.get("contact_point")),
            ("weight_transfer", self._last_big3.get("weight_transfer")),
            ("contact_zone", self._last_big3.get("contact_zone")),
            ("extension", self.extension_monitor),
            ("knee_load", self.knee_load_monitor),
            ("spacing", self.spacing_monitor),
            ("xfactor", self.xfactor_monitor),
            ("wrist", self.wrist_monitor),
            ("balance", self.balance_monitor),
        ]

        for name, monitor in monitors:
            if monitor is None:
                continue
            # Big3 monitor statuses are dataclasses with `.message` and `.status`.
            if hasattr(monitor, "message") and hasattr(monitor, "status"):
                msg = getattr(monitor, "message", "") or ""
                st = getattr(monitor, "status", Big3Status.UNKNOWN)
                if msg and st != Big3Status.UNKNOWN:
                    status_str = "good" if st == Big3Status.GOOD else "warning"
                    feedbacks.append((name, msg, status_str))
                continue

            # Secondary monitors (BaseMonitor) keep feedback in the object.
            if getattr(monitor, "feedback_message", ""):
                feedbacks.append((name, monitor.feedback_message, monitor.quality_status))

        return feedbacks

    def get_arm_highlight_status(self) -> Tuple[bool, str]:
        """
        Get whether arm should be highlighted and with what status.

        Returns:
            (should_highlight, status) where status is 'good' or 'warning'
        """
        # Priority: extension > spacing
        if self.extension_monitor.quality_status in ("good", "warning"):
            return True, self.extension_monitor.quality_status
        if self.spacing_monitor.quality_status in ("good", "warning"):
            return True, self.spacing_monitor.quality_status
        return False, "unknown"

    def reset(self):
        """Reset all monitors and shared state."""
        self.shared_impact_detector.reset()
        self.big3_contact_point.reset()
        self.big3_weight_transfer.reset()
        self.big3_contact_zone.reset()
        self._last_big3 = {"contact_point": None, "weight_transfer": None, "contact_zone": None}
        self._prev_keypoints = None
        self._prev_confidence = None
        self._prev_frame_idx = None
        self.extension_monitor.reset()
        self.knee_load_monitor.reset()
        self.spacing_monitor.reset()
        self.xfactor_monitor.reset()
        self.wrist_monitor.reset()
        self.balance_monitor.reset()
        self._prev_dom_wrist_x = None
        self._dom_wrist_dx_hist.clear()
        self._active_forward_sign = None
        self._active_forward_ttl = 0
