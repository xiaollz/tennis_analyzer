"""COCO keypoint definitions for pose estimation."""

# COCO 17 keypoints
COCO_KEYPOINTS = {
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

# Reverse mapping
KEYPOINT_NAMES = {v: k for k, v in COCO_KEYPOINTS.items()}

# Skeleton connections for drawing (NO FACE - only body)
SKELETON_CONNECTIONS = [
    # Upper body
    (5, 6),   # shoulders
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

# Colors for different body parts (BGR format for OpenCV)
KEYPOINT_COLORS = {
    "face": (255, 200, 150),      # light blue (not used)
    "left_arm": (0, 255, 0),      # green
    "right_arm": (0, 165, 255),   # orange
    "torso": (255, 255, 0),       # cyan
    "left_leg": (255, 0, 255),    # magenta
    "right_leg": (0, 255, 255),   # yellow
}

# Map keypoint index to body part for coloring
KEYPOINT_TO_PART = {
    0: "face", 1: "face", 2: "face", 3: "face", 4: "face",
    5: "torso", 6: "torso",
    7: "left_arm", 9: "left_arm",
    8: "right_arm", 10: "right_arm",
    11: "torso", 12: "torso",
    13: "left_leg", 15: "left_leg",
    14: "right_leg", 16: "right_leg",
}

# Connection colors (no face connections now)
CONNECTION_COLORS = [
    # Shoulders (0)
    (255, 255, 0),
    # Left arm (1-2)
    (0, 255, 0), (0, 255, 0),
    # Right arm (3-4)
    (0, 165, 255), (0, 165, 255),
    # Torso (5-7)
    (255, 255, 0), (255, 255, 0), (255, 255, 0),
    # Left leg (8-9)
    (255, 0, 255), (255, 0, 255),
    # Right leg (10-11)
    (0, 255, 255), (0, 255, 255),
]
