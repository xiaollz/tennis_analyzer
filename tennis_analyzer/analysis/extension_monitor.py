"""Extension monitor for forehand stroke quality analysis.

Deprecated: this file is kept for backwards compatibility with earlier
experiments. Prefer `tennis_analyzer.analysis.kinetic_chain.ExtensionMonitor`.
"""

import numpy as np
from typing import Optional, Tuple, List
from collections import deque
from enum import Enum

from ..config.keypoints import KEYPOINT_NAMES


class ExtensionQuality(Enum):
    """Extension quality status."""
    UNKNOWN = "unknown"
    GOOD = "good"        # θ >= 150°
    TIGHT = "tight"      # θ < 150°


class ExtensionMonitor:
    """
    Monitor arm extension during forehand impact.

    Detects "impact frame" and checks if arm angle stays above threshold
    during impact and early follow-through (5 frames window).
    """

    def __init__(
        self,
        angle_threshold: float = 150.0,
        check_window: int = 5,
        velocity_threshold: float = 50.0,
        is_right_handed: bool = True
    ):
        """
        Initialize extension monitor.

        Args:
            angle_threshold: Minimum angle for "good extension" (degrees)
            check_window: Number of frames to check after impact
            velocity_threshold: Velocity change threshold to detect impact
            is_right_handed: Whether player is right-handed
        """
        self.angle_threshold = angle_threshold
        self.check_window = check_window
        self.velocity_threshold = velocity_threshold
        self.is_right_handed = is_right_handed

        # Keypoint indices
        if is_right_handed:
            self.shoulder_idx = KEYPOINT_NAMES["right_shoulder"]
            self.elbow_idx = KEYPOINT_NAMES["right_elbow"]
            self.wrist_idx = KEYPOINT_NAMES["right_wrist"]
        else:
            self.shoulder_idx = KEYPOINT_NAMES["left_shoulder"]
            self.elbow_idx = KEYPOINT_NAMES["left_elbow"]
            self.wrist_idx = KEYPOINT_NAMES["left_wrist"]

        # State tracking
        self.prev_wrist_pos: Optional[np.ndarray] = None
        self.prev_velocity: Optional[np.ndarray] = None
        self.velocity_history: deque = deque(maxlen=10)

        # Impact detection
        self.impact_detected = False
        self.frames_since_impact = 0
        self.impact_angles: List[float] = []

        # Result
        self.current_quality = ExtensionQuality.UNKNOWN
        self.current_angle = 0.0
        self.feedback_message = ""
        self.feedback_frames_remaining = 0

    def calculate_arm_angle(
        self,
        keypoints: np.ndarray,
        confidence: np.ndarray
    ) -> Optional[float]:
        """
        Calculate arm angle at elbow (shoulder-elbow-wrist).

        Args:
            keypoints: Keypoint coordinates
            confidence: Confidence scores

        Returns:
            Angle in degrees, or None if keypoints not confident
        """
        min_conf = 0.3
        if (confidence[self.shoulder_idx] < min_conf or
            confidence[self.elbow_idx] < min_conf or
            confidence[self.wrist_idx] < min_conf):
            return None

        shoulder = keypoints[self.shoulder_idx]
        elbow = keypoints[self.elbow_idx]
        wrist = keypoints[self.wrist_idx]

        # Vectors from elbow
        ba = shoulder - elbow  # elbow to shoulder
        bc = wrist - elbow     # elbow to wrist

        # Calculate angle using dot product
        dot = np.dot(ba, bc)
        mag_ba = np.linalg.norm(ba)
        mag_bc = np.linalg.norm(bc)

        if mag_ba < 1e-6 or mag_bc < 1e-6:
            return None

        cos_angle = dot / (mag_ba * mag_bc)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))

        return angle

    def detect_impact(
        self,
        keypoints: np.ndarray,
        confidence: np.ndarray
    ) -> bool:
        """
        Detect impact frame based on wrist velocity change.

        Impact is detected when:
        - Velocity magnitude peaks (deceleration after acceleration)
        - Or velocity direction changes significantly

        Args:
            keypoints: Keypoint coordinates
            confidence: Confidence scores

        Returns:
            True if impact detected this frame
        """
        if confidence[self.wrist_idx] < 0.3:
            return False

        wrist_pos = keypoints[self.wrist_idx]

        if self.prev_wrist_pos is None:
            self.prev_wrist_pos = wrist_pos.copy()
            return False

        # Calculate velocity
        velocity = wrist_pos - self.prev_wrist_pos
        speed = np.linalg.norm(velocity)
        self.velocity_history.append(speed)
        self.prev_wrist_pos = wrist_pos.copy()

        # Need enough history
        if len(self.velocity_history) < 5:
            self.prev_velocity = velocity.copy()
            return False

        # Detect velocity peak (was accelerating, now decelerating)
        speeds = list(self.velocity_history)
        if len(speeds) >= 3:
            # Check if we just passed a peak
            if speeds[-2] > speeds[-1] and speeds[-2] > speeds[-3]:
                if speeds[-2] > self.velocity_threshold:
                    self.prev_velocity = velocity.copy()
                    return True

        # Detect direction change
        if self.prev_velocity is not None:
            prev_speed = np.linalg.norm(self.prev_velocity)
            if prev_speed > 1e-6 and speed > 1e-6:
                cos_angle = np.dot(velocity, self.prev_velocity) / (speed * prev_speed)
                if cos_angle < 0.5:  # Direction changed > 60 degrees
                    if speed > self.velocity_threshold * 0.5:
                        self.prev_velocity = velocity.copy()
                        return True

        self.prev_velocity = velocity.copy()
        return False

    def update(
        self,
        keypoints: np.ndarray,
        confidence: np.ndarray,
        is_forehand: bool
    ) -> Tuple[ExtensionQuality, float, str]:
        """
        Update monitor with new frame.

        Args:
            keypoints: Keypoint coordinates
            confidence: Confidence scores
            is_forehand: Whether currently in forehand stroke

        Returns:
            Tuple of (quality, angle, feedback_message)
        """
        # Calculate current arm angle
        angle = self.calculate_arm_angle(keypoints, confidence)
        if angle is not None:
            self.current_angle = angle

        # Decrease feedback display counter
        if self.feedback_frames_remaining > 0:
            self.feedback_frames_remaining -= 1

        # Only monitor during forehand
        if not is_forehand:
            self.impact_detected = False
            self.frames_since_impact = 0
            self.impact_angles.clear()
            if self.feedback_frames_remaining <= 0:
                self.current_quality = ExtensionQuality.UNKNOWN
                self.feedback_message = ""
            return self.current_quality, self.current_angle, self.feedback_message

        # Detect impact
        if not self.impact_detected:
            if self.detect_impact(keypoints, confidence):
                self.impact_detected = True
                self.frames_since_impact = 0
                self.impact_angles.clear()

        # During check window after impact
        if self.impact_detected:
            if angle is not None:
                self.impact_angles.append(angle)

            self.frames_since_impact += 1

            # End of check window - evaluate
            if self.frames_since_impact >= self.check_window:
                self._evaluate_extension()
                self.impact_detected = False

        return self.current_quality, self.current_angle, self.feedback_message

    def _evaluate_extension(self):
        """Evaluate extension quality based on collected angles."""
        if not self.impact_angles:
            return

        # Use minimum angle during impact window
        min_angle = min(self.impact_angles)
        avg_angle = sum(self.impact_angles) / len(self.impact_angles)

        if min_angle >= self.angle_threshold:
            self.current_quality = ExtensionQuality.GOOD
            self.feedback_message = "Great Extension! 通透感十足!"
        else:
            self.current_quality = ExtensionQuality.TIGHT
            self.feedback_message = f"Space Too Tight! 手肘收缩! ({min_angle:.0f}°<{self.angle_threshold:.0f}°)"

        # Show feedback for 60 frames (~2 seconds)
        self.feedback_frames_remaining = 60

    def reset(self):
        """Reset monitor state."""
        self.prev_wrist_pos = None
        self.prev_velocity = None
        self.velocity_history.clear()
        self.impact_detected = False
        self.frames_since_impact = 0
        self.impact_angles.clear()
        self.current_quality = ExtensionQuality.UNKNOWN
        self.current_angle = 0.0
        self.feedback_message = ""
        self.feedback_frames_remaining = 0
