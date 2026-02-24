"""Base class for all tennis stroke monitors."""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from abc import ABC, abstractmethod
from collections import deque

from ..config.keypoints import KEYPOINT_NAMES


class BaseMonitor(ABC):
    """Abstract base class for stroke quality monitors."""

    def __init__(self, is_right_handed: bool = True):
        """
        Initialize base monitor.

        Args:
            is_right_handed: Whether player is right-handed
        """
        self.is_right_handed = is_right_handed

        # Common keypoint indices
        if is_right_handed:
            self.shoulder_idx = KEYPOINT_NAMES["right_shoulder"]
            self.elbow_idx = KEYPOINT_NAMES["right_elbow"]
            self.wrist_idx = KEYPOINT_NAMES["right_wrist"]
            self.hip_idx = KEYPOINT_NAMES["right_hip"]
            self.other_shoulder_idx = KEYPOINT_NAMES["left_shoulder"]
            self.other_hip_idx = KEYPOINT_NAMES["left_hip"]
        else:
            self.shoulder_idx = KEYPOINT_NAMES["left_shoulder"]
            self.elbow_idx = KEYPOINT_NAMES["left_elbow"]
            self.wrist_idx = KEYPOINT_NAMES["left_wrist"]
            self.hip_idx = KEYPOINT_NAMES["left_hip"]
            self.other_shoulder_idx = KEYPOINT_NAMES["right_shoulder"]
            self.other_hip_idx = KEYPOINT_NAMES["right_hip"]

        # Feedback state - persists until replaced
        self.feedback_message = ""
        self.quality_status = "unknown"  # "good", "warning", "unknown"

    def get_shoulder_width(
        self,
        keypoints: np.ndarray,
        confidence: np.ndarray,
        min_conf: float = 0.3
    ) -> Optional[float]:
        """Get shoulder width for normalization."""
        l_shoulder = KEYPOINT_NAMES["left_shoulder"]
        r_shoulder = KEYPOINT_NAMES["right_shoulder"]

        if confidence[l_shoulder] < min_conf or confidence[r_shoulder] < min_conf:
            return None

        return np.linalg.norm(keypoints[l_shoulder] - keypoints[r_shoulder])

    def normalize_distance(
        self,
        distance: float,
        keypoints: np.ndarray,
        confidence: np.ndarray
    ) -> Optional[float]:
        """Normalize distance by shoulder width."""
        shoulder_width = self.get_shoulder_width(keypoints, confidence)
        if shoulder_width is None or shoulder_width < 1e-6:
            return None
        return distance / shoulder_width

    def set_feedback(self, message: str, status: str):
        """Set feedback message (persists until replaced)."""
        self.feedback_message = message
        self.quality_status = status

    @abstractmethod
    def update(
        self,
        keypoints: np.ndarray,
        confidence: np.ndarray,
        is_forehand: bool,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Update monitor with new frame.

        Args:
            keypoints: Keypoint coordinates
            confidence: Confidence scores
            is_forehand: Whether currently in forehand stroke

        Returns:
            Dictionary with monitor results
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset monitor state."""
        pass
