"""Skeleton drawing utilities."""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from collections import deque

from ..config.keypoints import (
    SKELETON_CONNECTIONS,
    CONNECTION_COLORS,
    KEYPOINT_TO_PART,
    KEYPOINT_COLORS,
    FACE_KEYPOINTS,
    KEYPOINT_NAMES,
)


class WristTracker:
    """Track wrist position and draw trajectory trail."""

    def __init__(self, max_trail_length: int = 60, trail_color: Tuple[int, int, int] = (0, 255, 255), min_distance: int = 8):
        """
        Initialize wrist tracker.

        Args:
            max_trail_length: Maximum number of points in trail
            trail_color: Color for trail (BGR) - default yellow
            min_distance: Minimum pixel distance between trail points
        """
        self.max_trail_length = max_trail_length
        self.trail_color = trail_color
        self.min_distance = min_distance
        self.trail: deque = deque(maxlen=max_trail_length)
        self.is_tracking = False

    def update(self, keypoints: np.ndarray, confidence: np.ndarray, is_stroke: bool, is_right_handed: bool = True):
        """
        Update wrist trail.

        Args:
            keypoints: Keypoint coordinates
            confidence: Confidence scores
            is_stroke: Whether currently in a stroke (forehand/backhand)
            is_right_handed: Whether player is right-handed
        """
        # Get dominant wrist index
        wrist_idx = KEYPOINT_NAMES["right_wrist"] if is_right_handed else KEYPOINT_NAMES["left_wrist"]

        if confidence[wrist_idx] > 0.3:
            wrist_pos = tuple(keypoints[wrist_idx].astype(int))

            if is_stroke or self.is_tracking:
                # Only add point if far enough from last point (avoid overlap)
                if len(self.trail) == 0:
                    self.trail.append(wrist_pos)
                else:
                    last_pos = self.trail[-1]
                    dist = np.sqrt((wrist_pos[0] - last_pos[0])**2 + (wrist_pos[1] - last_pos[1])**2)
                    if dist >= self.min_distance:
                        self.trail.append(wrist_pos)

                if is_stroke:
                    self.is_tracking = True

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw wrist trail on frame - tiny dots for clean visualization.

        Args:
            frame: BGR image

        Returns:
            Frame with trail drawn
        """
        if len(self.trail) < 2:
            return frame

        frame = frame.copy()

        # Draw tiny trail dots (radius 1 = smallest visible)
        for pos in self.trail:
            cv2.circle(frame, pos, 1, self.trail_color, -1, cv2.LINE_AA)

        return frame

    def clear(self):
        """Clear the trail."""
        self.trail.clear()
        self.is_tracking = False


class SkeletonDrawer:
    """Draw pose skeleton on frames."""

    def __init__(
        self,
        line_thickness: int = 1,
        point_radius: int = 3,
        confidence_threshold: float = 0.5,
        draw_face: bool = False,
    ):
        """
        Initialize skeleton drawer.

        Args:
            line_thickness: Thickness of skeleton lines
            point_radius: Radius of keypoint circles
            confidence_threshold: Minimum confidence to draw keypoint
            draw_face: Whether to draw face keypoints
        """
        self.line_thickness = line_thickness
        self.point_radius = point_radius
        self.confidence_threshold = confidence_threshold
        self.draw_face = draw_face

    def draw(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        confidence: np.ndarray,
        color_override: Optional[Tuple[int, int, int]] = None,
    ) -> np.ndarray:
        """
        Draw skeleton on frame.

        Args:
            frame: BGR image
            keypoints: (17, 2) array of x, y coordinates
            confidence: (17,) array of confidence scores
            color_override: Optional single color for all elements

        Returns:
            Frame with skeleton drawn
        """
        frame = frame.copy()

        # Draw connections first (so points are on top)
        self._draw_connections(frame, keypoints, confidence, color_override)

        # Draw keypoints
        self._draw_keypoints(frame, keypoints, confidence, color_override)

        return frame

    def _draw_connections(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        confidence: np.ndarray,
        color_override: Optional[Tuple[int, int, int]] = None,
    ):
        """Draw skeleton connections."""
        for idx, (start_idx, end_idx) in enumerate(SKELETON_CONNECTIONS):
            if (confidence[start_idx] < self.confidence_threshold or
                confidence[end_idx] < self.confidence_threshold):
                continue

            start_point = tuple(keypoints[start_idx].astype(int))
            end_point = tuple(keypoints[end_idx].astype(int))

            # Skip if points are at origin (not detected)
            if start_point == (0, 0) or end_point == (0, 0):
                continue

            color = color_override if color_override else CONNECTION_COLORS[idx]
            cv2.line(frame, start_point, end_point, color, self.line_thickness, cv2.LINE_AA)

    def _draw_keypoints(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        confidence: np.ndarray,
        color_override: Optional[Tuple[int, int, int]] = None,
    ):
        """Draw keypoint circles."""
        for idx, (point, conf) in enumerate(zip(keypoints, confidence)):
            # Skip face keypoints unless explicitly enabled
            if not self.draw_face and idx in FACE_KEYPOINTS:
                continue

            if conf < self.confidence_threshold:
                continue

            center = tuple(point.astype(int))
            if center == (0, 0):
                continue

            part = KEYPOINT_TO_PART.get(idx, "torso")
            color = color_override if color_override else KEYPOINT_COLORS[part]

            # Draw small filled circle
            cv2.circle(frame, center, self.point_radius, color, -1, cv2.LINE_AA)

    def draw_multiple(
        self,
        frame: np.ndarray,
        persons: List[dict],
    ) -> np.ndarray:
        """
        Draw skeletons for multiple persons.

        Args:
            frame: BGR image
            persons: List of person dicts with 'keypoints' and 'confidence'

        Returns:
            Frame with all skeletons drawn
        """
        for person in persons:
            frame = self.draw(
                frame,
                person["keypoints"],
                person["confidence"],
            )
        return frame

    def draw_arm_highlight(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        confidence: np.ndarray,
        color: Tuple[int, int, int],
        is_right: bool = True,
        thickness: int = 3,
    ) -> np.ndarray:
        """
        Draw highlighted arm (shoulder-elbow-wrist) with specific color.

        Args:
            frame: BGR image
            keypoints: Keypoint coordinates
            confidence: Confidence scores
            color: Color for arm highlight (BGR)
            is_right: Whether to highlight right arm
            thickness: Line thickness for highlight

        Returns:
            Frame with highlighted arm
        """
        frame = frame.copy()

        if is_right:
            shoulder_idx = KEYPOINT_NAMES["right_shoulder"]
            elbow_idx = KEYPOINT_NAMES["right_elbow"]
            wrist_idx = KEYPOINT_NAMES["right_wrist"]
        else:
            shoulder_idx = KEYPOINT_NAMES["left_shoulder"]
            elbow_idx = KEYPOINT_NAMES["left_elbow"]
            wrist_idx = KEYPOINT_NAMES["left_wrist"]

        min_conf = 0.3
        if (confidence[shoulder_idx] < min_conf or
            confidence[elbow_idx] < min_conf or
            confidence[wrist_idx] < min_conf):
            return frame

        shoulder = tuple(keypoints[shoulder_idx].astype(int))
        elbow = tuple(keypoints[elbow_idx].astype(int))
        wrist = tuple(keypoints[wrist_idx].astype(int))

        # Draw arm lines with highlight color
        cv2.line(frame, shoulder, elbow, color, thickness, cv2.LINE_AA)
        cv2.line(frame, elbow, wrist, color, thickness, cv2.LINE_AA)

        # Draw points
        cv2.circle(frame, shoulder, thickness + 1, color, -1, cv2.LINE_AA)
        cv2.circle(frame, elbow, thickness + 1, color, -1, cv2.LINE_AA)
        cv2.circle(frame, wrist, thickness + 1, color, -1, cv2.LINE_AA)

        return frame
