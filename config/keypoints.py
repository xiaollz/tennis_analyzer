"""COCO 17-keypoint definitions and skeleton topology for YOLO Pose.

This is the single source of truth for keypoint indices, names, skeleton
connections, and drawing colours used throughout the project.
"""

from typing import Dict, List, Tuple

# ── COCO 17 keypoints ────────────────────────────────────────────────
COCO_KEYPOINTS: Dict[int, str] = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle",
}

# Reverse mapping: name -> index
KEYPOINT_NAMES: Dict[str, int] = {v: k for k, v in COCO_KEYPOINTS.items()}

NUM_KEYPOINTS = 17

# ── Skeleton connections (body only, no face) ────────────────────────
SKELETON_CONNECTIONS: List[Tuple[int, int]] = [
    # Upper body
    (5, 6),            # shoulders
    (5, 7), (7, 9),   # left arm
    (6, 8), (8, 10),  # right arm
    # Torso
    (5, 11), (6, 12), (11, 12),
    # Lower body
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
]

# Face keypoint indices (to skip in drawing)
FACE_KEYPOINTS = {0, 1, 2, 3, 4}

# ── Colours (BGR for OpenCV) ─────────────────────────────────────────
KEYPOINT_COLORS: Dict[str, Tuple[int, int, int]] = {
    "face":      (255, 200, 150),
    "left_arm":  (0, 255, 0),
    "right_arm": (0, 165, 255),
    "torso":     (255, 255, 0),
    "left_leg":  (255, 0, 255),
    "right_leg": (0, 255, 255),
}

KEYPOINT_TO_PART: Dict[int, str] = {
    0: "face", 1: "face", 2: "face", 3: "face", 4: "face",
    5: "torso", 6: "torso",
    7: "left_arm", 9: "left_arm",
    8: "right_arm", 10: "right_arm",
    11: "torso", 12: "torso",
    13: "left_leg", 15: "left_leg",
    14: "right_leg", 16: "right_leg",
}

CONNECTION_COLORS: List[Tuple[int, int, int]] = [
    (255, 255, 0),                   # shoulders
    (0, 255, 0), (0, 255, 0),       # left arm
    (0, 165, 255), (0, 165, 255),   # right arm
    (255, 255, 0), (255, 255, 0), (255, 255, 0),  # torso
    (255, 0, 255), (255, 0, 255),   # left leg
    (0, 255, 255), (0, 255, 255),   # right leg
]

# ── Convenience groupings for analysis ───────────────────────────────
UPPER_BODY_INDICES = [5, 6, 7, 8, 9, 10]
LOWER_BODY_INDICES = [11, 12, 13, 14, 15, 16]
TORSO_INDICES = [5, 6, 11, 12]
