"""Biomechanics analysis for tennis poses."""

import numpy as np
from typing import Dict, Optional, Tuple

from ..config.keypoints import KEYPOINT_NAMES


class BiomechanicsAnalyzer:
    """Calculate biomechanical metrics from pose keypoints."""

    def __init__(self):
        """Initialize biomechanics analyzer."""
        pass

    def analyze(self, keypoints: np.ndarray, confidence: np.ndarray) -> Dict[str, float]:
        """
        Analyze pose and return all metrics.

        Args:
            keypoints: (17, 2) array of x, y coordinates
            confidence: (17,) array of confidence scores

        Returns:
            Dictionary of metric names to values
        """
        metrics = {}

        # Knee angles
        left_knee = self.calculate_knee_angle(keypoints, confidence, "left")
        right_knee = self.calculate_knee_angle(keypoints, confidence, "right")
        if left_knee is not None:
            metrics["L Knee"] = left_knee
        if right_knee is not None:
            metrics["R Knee"] = right_knee

        # Elbow angles
        left_elbow = self.calculate_elbow_angle(keypoints, confidence, "left")
        right_elbow = self.calculate_elbow_angle(keypoints, confidence, "right")
        if left_elbow is not None:
            metrics["L Elbow"] = left_elbow
        if right_elbow is not None:
            metrics["R Elbow"] = right_elbow

        # Hip-shoulder separation (X-factor)
        x_factor = self.calculate_hip_shoulder_separation(keypoints, confidence)
        if x_factor is not None:
            metrics["X-Factor"] = x_factor

        # Shoulder rotation
        shoulder_angle = self.calculate_shoulder_rotation(keypoints, confidence)
        if shoulder_angle is not None:
            metrics["Shoulder"] = shoulder_angle

        return metrics

    def calculate_angle(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        p3: np.ndarray
    ) -> float:
        """
        Calculate angle at p2 formed by p1-p2-p3.

        Args:
            p1, p2, p3: Points as (x, y) arrays

        Returns:
            Angle in degrees
        """
        v1 = p1 - p2
        v2 = p3 - p2

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)

        return np.degrees(angle)

    def calculate_knee_angle(
        self,
        keypoints: np.ndarray,
        confidence: np.ndarray,
        side: str = "left"
    ) -> Optional[float]:
        """Calculate knee bend angle."""
        if side == "left":
            hip_idx = KEYPOINT_NAMES["left_hip"]
            knee_idx = KEYPOINT_NAMES["left_knee"]
            ankle_idx = KEYPOINT_NAMES["left_ankle"]
        else:
            hip_idx = KEYPOINT_NAMES["right_hip"]
            knee_idx = KEYPOINT_NAMES["right_knee"]
            ankle_idx = KEYPOINT_NAMES["right_ankle"]

        # Check confidence
        min_conf = 0.3
        if (confidence[hip_idx] < min_conf or
            confidence[knee_idx] < min_conf or
            confidence[ankle_idx] < min_conf):
            return None

        return self.calculate_angle(
            keypoints[hip_idx],
            keypoints[knee_idx],
            keypoints[ankle_idx]
        )

    def calculate_elbow_angle(
        self,
        keypoints: np.ndarray,
        confidence: np.ndarray,
        side: str = "left"
    ) -> Optional[float]:
        """Calculate elbow bend angle."""
        if side == "left":
            shoulder_idx = KEYPOINT_NAMES["left_shoulder"]
            elbow_idx = KEYPOINT_NAMES["left_elbow"]
            wrist_idx = KEYPOINT_NAMES["left_wrist"]
        else:
            shoulder_idx = KEYPOINT_NAMES["right_shoulder"]
            elbow_idx = KEYPOINT_NAMES["right_elbow"]
            wrist_idx = KEYPOINT_NAMES["right_wrist"]

        min_conf = 0.3
        if (confidence[shoulder_idx] < min_conf or
            confidence[elbow_idx] < min_conf or
            confidence[wrist_idx] < min_conf):
            return None

        return self.calculate_angle(
            keypoints[shoulder_idx],
            keypoints[elbow_idx],
            keypoints[wrist_idx]
        )

    def calculate_hip_shoulder_separation(
        self,
        keypoints: np.ndarray,
        confidence: np.ndarray
    ) -> Optional[float]:
        """
        Calculate hip-shoulder separation angle (X-factor).

        This measures the rotation difference between hip line and shoulder line.
        Important for power generation in tennis strokes.
        """
        l_shoulder = KEYPOINT_NAMES["left_shoulder"]
        r_shoulder = KEYPOINT_NAMES["right_shoulder"]
        l_hip = KEYPOINT_NAMES["left_hip"]
        r_hip = KEYPOINT_NAMES["right_hip"]

        min_conf = 0.3
        if (confidence[l_shoulder] < min_conf or
            confidence[r_shoulder] < min_conf or
            confidence[l_hip] < min_conf or
            confidence[r_hip] < min_conf):
            return None

        # Calculate shoulder line angle
        shoulder_vec = keypoints[r_shoulder] - keypoints[l_shoulder]
        shoulder_angle = np.arctan2(shoulder_vec[1], shoulder_vec[0])

        # Calculate hip line angle
        hip_vec = keypoints[r_hip] - keypoints[l_hip]
        hip_angle = np.arctan2(hip_vec[1], hip_vec[0])

        # Separation angle
        separation = np.abs(np.degrees(shoulder_angle - hip_angle))
        if separation > 180:
            separation = 360 - separation

        return separation

    def calculate_shoulder_rotation(
        self,
        keypoints: np.ndarray,
        confidence: np.ndarray
    ) -> Optional[float]:
        """
        Calculate shoulder rotation angle relative to horizontal.

        Useful for measuring unit turn in forehand/backhand.
        """
        l_shoulder = KEYPOINT_NAMES["left_shoulder"]
        r_shoulder = KEYPOINT_NAMES["right_shoulder"]

        min_conf = 0.3
        if (confidence[l_shoulder] < min_conf or
            confidence[r_shoulder] < min_conf):
            return None

        shoulder_vec = keypoints[r_shoulder] - keypoints[l_shoulder]
        angle = np.arctan2(shoulder_vec[1], shoulder_vec[0])

        return np.degrees(angle)

    def get_body_center(
        self,
        keypoints: np.ndarray,
        confidence: np.ndarray
    ) -> Optional[Tuple[float, float]]:
        """Calculate approximate body center (midpoint of hips)."""
        l_hip = KEYPOINT_NAMES["left_hip"]
        r_hip = KEYPOINT_NAMES["right_hip"]

        if confidence[l_hip] < 0.3 or confidence[r_hip] < 0.3:
            return None

        center = (keypoints[l_hip] + keypoints[r_hip]) / 2
        return tuple(center)
