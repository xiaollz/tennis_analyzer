"""Action classifier for tennis strokes."""

import numpy as np
from typing import Optional, Tuple, List
from enum import Enum

from ..config.keypoints import KEYPOINT_NAMES


class StrokeType(Enum):
    """Tennis stroke types."""
    UNKNOWN = "unknown"
    FOREHAND = "forehand"
    BACKHAND = "backhand"
    SERVE = "serve"
    VOLLEY = "volley"
    READY = "ready"  # Ready position


class ActionClassifier:
    """
    Rule-based classifier for tennis actions.

    Uses keypoint positions and angles to determine stroke type.
    Assumes right-handed player by default.

    Includes state machine logic to prevent false transitions
    (e.g., forehand follow-through misclassified as backhand).
    """

    def __init__(self, is_right_handed: bool = True):
        """
        Initialize action classifier.

        Args:
            is_right_handed: Whether player is right-handed
        """
        self.is_right_handed = is_right_handed
        # Dominant side indices
        if is_right_handed:
            self.dominant_shoulder = KEYPOINT_NAMES["right_shoulder"]
            self.dominant_elbow = KEYPOINT_NAMES["right_elbow"]
            self.dominant_wrist = KEYPOINT_NAMES["right_wrist"]
            self.non_dominant_shoulder = KEYPOINT_NAMES["left_shoulder"]
            self.non_dominant_wrist = KEYPOINT_NAMES["left_wrist"]
        else:
            self.dominant_shoulder = KEYPOINT_NAMES["left_shoulder"]
            self.dominant_elbow = KEYPOINT_NAMES["left_elbow"]
            self.dominant_wrist = KEYPOINT_NAMES["left_wrist"]
            self.non_dominant_shoulder = KEYPOINT_NAMES["right_shoulder"]
            self.non_dominant_wrist = KEYPOINT_NAMES["right_wrist"]

        # History for smoothing
        self.history: List[StrokeType] = []
        self.history_size = 5

        # State machine for stroke transitions
        self.current_stroke = StrokeType.UNKNOWN
        self.stroke_frames = 0  # Frames in current stroke
        self.min_stroke_frames = 10  # Minimum frames before allowing transition

        # Wrist velocity tracking for direction detection
        self.prev_wrist_pos: Optional[np.ndarray] = None
        self.wrist_velocity_history: List[np.ndarray] = []
        self.velocity_history_size = 5

    def classify(
        self,
        keypoints: np.ndarray,
        confidence: np.ndarray,
        min_confidence: float = 0.3
    ) -> Tuple[StrokeType, float]:
        """
        Classify the current pose into a stroke type.

        Args:
            keypoints: (17, 2) array of x, y coordinates
            confidence: (17,) array of confidence scores
            min_confidence: Minimum confidence for keypoints

        Returns:
            Tuple of (StrokeType, confidence_score)
        """
        # Check if we have enough keypoints
        required_points = [
            self.dominant_shoulder, self.dominant_elbow, self.dominant_wrist,
            KEYPOINT_NAMES["left_hip"], KEYPOINT_NAMES["right_hip"],
            KEYPOINT_NAMES["nose"]
        ]

        for idx in required_points:
            if confidence[idx] < min_confidence:
                return StrokeType.UNKNOWN, 0.0

        # Get key positions
        dom_wrist = keypoints[self.dominant_wrist]
        dom_shoulder = keypoints[self.dominant_shoulder]
        dom_elbow = keypoints[self.dominant_elbow]
        non_dom_shoulder = keypoints[self.non_dominant_shoulder]
        nose = keypoints[KEYPOINT_NAMES["nose"]]

        # Update wrist velocity tracking
        wrist_direction = self._update_wrist_velocity(dom_wrist)

        # Calculate body center (midpoint of shoulders)
        body_center_x = (dom_shoulder[0] + non_dom_shoulder[0]) / 2

        # Calculate relative positions
        wrist_above_head = dom_wrist[1] < nose[1]  # y increases downward
        wrist_height_ratio = (nose[1] - dom_wrist[1]) / (nose[1] + 1e-6)

        # Wrist position relative to body center
        wrist_side = "dominant" if self._is_on_dominant_side(dom_wrist[0], body_center_x) else "non_dominant"

        # Arm extension (elbow angle approximation)
        arm_extended = self._is_arm_extended(dom_shoulder, dom_elbow, dom_wrist)

        # Classify based on rules
        raw_stroke, conf = self._apply_rules(
            wrist_above_head=wrist_above_head,
            wrist_height_ratio=wrist_height_ratio,
            wrist_side=wrist_side,
            arm_extended=arm_extended,
            dom_wrist=dom_wrist,
            dom_shoulder=dom_shoulder,
            body_center_x=body_center_x,
        )

        # Apply state machine to prevent false transitions
        stroke_type = self._apply_state_machine(raw_stroke, wrist_direction)

        # Smooth with history
        self.history.append(stroke_type)
        if len(self.history) > self.history_size:
            self.history.pop(0)

        smoothed_type = self._smooth_classification()

        return smoothed_type, conf

    def _update_wrist_velocity(self, wrist_pos: np.ndarray) -> Optional[str]:
        """
        Track wrist velocity and return movement direction.

        Returns:
            'forward' (toward non-dominant side), 'backward' (toward dominant side), or None
        """
        if self.prev_wrist_pos is None:
            self.prev_wrist_pos = wrist_pos.copy()
            return None

        velocity = wrist_pos - self.prev_wrist_pos
        self.wrist_velocity_history.append(velocity)
        if len(self.wrist_velocity_history) > self.velocity_history_size:
            self.wrist_velocity_history.pop(0)

        self.prev_wrist_pos = wrist_pos.copy()

        # Calculate average velocity direction
        if len(self.wrist_velocity_history) < 3:
            return None

        avg_velocity = np.mean(self.wrist_velocity_history, axis=0)
        horizontal_speed = abs(avg_velocity[0])

        if horizontal_speed < 5:  # Too slow to determine direction
            return None

        # For right-hander: negative x = forward (toward left/non-dominant side)
        # For left-hander: positive x = forward (toward right/non-dominant side)
        if self.is_right_handed:
            return "forward" if avg_velocity[0] < 0 else "backward"
        else:
            return "forward" if avg_velocity[0] > 0 else "backward"

    def _apply_state_machine(self, raw_stroke: StrokeType, wrist_direction: Optional[str]) -> StrokeType:
        """
        Apply state machine logic to prevent false stroke transitions.

        Key rule: Forehand follow-through should not be misclassified as backhand.
        """
        self.stroke_frames += 1

        # If same stroke, just continue
        if raw_stroke == self.current_stroke:
            return raw_stroke

        # Transition from FOREHAND to BACKHAND needs extra validation
        if self.current_stroke == StrokeType.FOREHAND and raw_stroke == StrokeType.BACKHAND:
            # Don't allow quick transition - likely follow-through
            if self.stroke_frames < self.min_stroke_frames:
                return StrokeType.FOREHAND

            # Check wrist direction: if still moving forward, it's follow-through not backhand
            if wrist_direction == "forward":
                return StrokeType.FOREHAND

            # If wrist is moving backward (toward dominant side), could be backhand prep
            # But require more frames to confirm
            if self.stroke_frames < self.min_stroke_frames * 2:
                return StrokeType.FOREHAND

        # Transition from BACKHAND to FOREHAND needs similar validation
        if self.current_stroke == StrokeType.BACKHAND and raw_stroke == StrokeType.FOREHAND:
            if self.stroke_frames < self.min_stroke_frames:
                return StrokeType.BACKHAND

            if wrist_direction == "forward":
                return StrokeType.BACKHAND

            if self.stroke_frames < self.min_stroke_frames * 2:
                return StrokeType.BACKHAND

        # Allow transition to READY/UNKNOWN more easily (recovery phase)
        if raw_stroke in (StrokeType.READY, StrokeType.UNKNOWN):
            if self.stroke_frames >= self.min_stroke_frames // 2:
                self.current_stroke = raw_stroke
                self.stroke_frames = 0
                return raw_stroke
            return self.current_stroke

        # Allow transition
        self.current_stroke = raw_stroke
        self.stroke_frames = 0
        return raw_stroke

    def _is_on_dominant_side(self, x: float, center_x: float) -> bool:
        """Check if x position is on dominant side."""
        if self.is_right_handed:
            return x > center_x
        return x < center_x

    def _is_arm_extended(
        self,
        shoulder: np.ndarray,
        elbow: np.ndarray,
        wrist: np.ndarray
    ) -> bool:
        """Check if arm is relatively extended."""
        # Calculate angle at elbow
        v1 = shoulder - elbow
        v2 = wrist - elbow

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

        return angle > 140  # Extended if angle > 140 degrees

    def _apply_rules(
        self,
        wrist_above_head: bool,
        wrist_height_ratio: float,
        wrist_side: str,
        arm_extended: bool,
        dom_wrist: np.ndarray,
        dom_shoulder: np.ndarray,
        body_center_x: float,
    ) -> Tuple[StrokeType, float]:
        """Apply classification rules."""

        # SERVE: Wrist significantly above head
        if wrist_above_head and wrist_height_ratio > 0.3:
            return StrokeType.SERVE, 0.85

        # Calculate horizontal distance from body center
        wrist_distance = abs(dom_wrist[0] - body_center_x)
        shoulder_width = abs(dom_shoulder[0] - body_center_x) * 2
        relative_distance = wrist_distance / (shoulder_width + 1e-6)

        # FOREHAND: Wrist on dominant side, arm moving/extended
        if wrist_side == "dominant" and relative_distance > 0.8:
            if arm_extended:
                return StrokeType.FOREHAND, 0.80
            return StrokeType.FOREHAND, 0.65

        # BACKHAND: Wrist on non-dominant side
        if wrist_side == "non_dominant" and relative_distance > 0.6:
            if arm_extended:
                return StrokeType.BACKHAND, 0.80
            return StrokeType.BACKHAND, 0.65

        # VOLLEY: Wrist in front, not too high, compact position
        # This is harder to detect without temporal info
        if not wrist_above_head and not arm_extended and relative_distance < 0.5:
            # Could be ready position or volley
            return StrokeType.READY, 0.50

        return StrokeType.UNKNOWN, 0.3

    def _smooth_classification(self) -> StrokeType:
        """Smooth classification using history."""
        if not self.history:
            return StrokeType.UNKNOWN

        # Count occurrences
        counts = {}
        for stroke in self.history:
            counts[stroke] = counts.get(stroke, 0) + 1

        # Return most common
        return max(counts, key=counts.get)

    def reset_history(self):
        """Reset classification history and state machine."""
        self.history.clear()
        self.current_stroke = StrokeType.UNKNOWN
        self.stroke_frames = 0
        self.prev_wrist_pos = None
        self.wrist_velocity_history.clear()


class StrokePhase(Enum):
    """Phases of a tennis stroke."""
    PREPARATION = "preparation"  # 引拍
    FORWARD_SWING = "forward_swing"  # 挥拍
    CONTACT = "contact"  # 击球
    FOLLOW_THROUGH = "follow_through"  # 随挥
    RECOVERY = "recovery"  # 还原


class PhaseDetector:
    """
    Detect the phase of a tennis stroke.

    Uses velocity and position changes to determine stroke phase.
    """

    def __init__(self):
        """Initialize phase detector."""
        self.prev_keypoints: Optional[np.ndarray] = None
        self.prev_phase = StrokePhase.RECOVERY
        self.phase_history: List[StrokePhase] = []

    def detect(
        self,
        keypoints: np.ndarray,
        confidence: np.ndarray,
        stroke_type: StrokeType,
        fps: float = 30.0
    ) -> StrokePhase:
        """
        Detect current stroke phase.

        Args:
            keypoints: Current keypoints
            confidence: Confidence scores
            stroke_type: Current stroke type
            fps: Video frame rate

        Returns:
            Current stroke phase
        """
        if self.prev_keypoints is None:
            self.prev_keypoints = keypoints.copy()
            return StrokePhase.RECOVERY

        # Calculate wrist velocity
        wrist_idx = KEYPOINT_NAMES["right_wrist"]  # Assume right-handed

        if confidence[wrist_idx] < 0.3:
            return self.prev_phase

        wrist_velocity = (keypoints[wrist_idx] - self.prev_keypoints[wrist_idx]) * fps
        speed = np.linalg.norm(wrist_velocity)

        # Direction (positive x = forward for right-hander forehand)
        direction = wrist_velocity[0]

        # Simple phase detection based on speed and direction
        phase = self._classify_phase(speed, direction, stroke_type)

        self.prev_keypoints = keypoints.copy()
        self.prev_phase = phase

        return phase

    def _classify_phase(
        self,
        speed: float,
        direction: float,
        stroke_type: StrokeType
    ) -> StrokePhase:
        """Classify phase based on motion."""

        # Thresholds (pixels per second, need tuning)
        slow_threshold = 100
        fast_threshold = 500

        if speed < slow_threshold:
            return StrokePhase.RECOVERY

        if stroke_type == StrokeType.FOREHAND:
            if direction < 0:  # Moving backward (preparation)
                return StrokePhase.PREPARATION
            elif speed > fast_threshold:
                return StrokePhase.FORWARD_SWING
            else:
                return StrokePhase.FOLLOW_THROUGH

        elif stroke_type == StrokeType.BACKHAND:
            if direction > 0:  # Moving backward for backhand
                return StrokePhase.PREPARATION
            elif speed > fast_threshold:
                return StrokePhase.FORWARD_SWING
            else:
                return StrokePhase.FOLLOW_THROUGH

        elif stroke_type == StrokeType.SERVE:
            if direction < 0:  # Racket going up/back
                return StrokePhase.PREPARATION
            elif speed > fast_threshold:
                return StrokePhase.FORWARD_SWING
            else:
                return StrokePhase.FOLLOW_THROUGH

        return StrokePhase.RECOVERY

    def reset(self):
        """Reset detector state."""
        self.prev_keypoints = None
        self.prev_phase = StrokePhase.RECOVERY
        self.phase_history.clear()
