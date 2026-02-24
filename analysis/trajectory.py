"""Joint trajectory management, smoothing, and kinematic derivation.

Each ``JointTrajectory`` stores the time-series of a single keypoint and
provides methods for computing velocity, acceleration, and angular velocity.
``TrajectoryStore`` is a convenience container that manages trajectories for
all 17 COCO keypoints simultaneously.
"""

from __future__ import annotations

import numpy as np
from collections import deque
from typing import Optional, Dict, List, Tuple

from config.keypoints import COCO_KEYPOINTS, KEYPOINT_NAMES, NUM_KEYPOINTS


class JointTrajectory:
    """Time-series data for a single keypoint.

    Stores raw and smoothed (x, y) positions plus the confidence at each
    frame, and lazily computes velocity / acceleration when requested.
    """

    # Leg keypoints need less smoothing (fast movement in tennis)
    _LEG_INDICES = {11, 12, 13, 14, 15, 16}

    def __init__(
        self,
        joint_index: int,
        fps: float = 30.0,
        smoothing_factor: float = 0.4,
        max_history: int = 600,
    ):
        self.joint_index = joint_index
        self.joint_name = COCO_KEYPOINTS.get(joint_index, f"kp_{joint_index}")
        self.fps = fps
        self.dt = 1.0 / max(fps, 1.0)

        # Adaptive smoothing: less for legs
        if joint_index in self._LEG_INDICES:
            self.alpha = max(0.1, smoothing_factor - 0.25)
        else:
            self.alpha = smoothing_factor

        # Storage
        self.max_history = max_history
        self.raw_positions: List[np.ndarray] = []        # (x, y) per frame
        self.smoothed_positions: List[np.ndarray] = []
        self.confidences: List[float] = []
        self.frame_indices: List[int] = []

        self._prev_smooth: Optional[np.ndarray] = None

    # ── recording ─────────────────────────────────────────────────────
    def add(self, x: float, y: float, confidence: float, frame_idx: int):
        """Record a new observation."""
        pos = np.array([x, y], dtype=np.float64)
        self.raw_positions.append(pos)
        self.confidences.append(float(confidence))
        self.frame_indices.append(int(frame_idx))

        # EMA smoothing
        if self._prev_smooth is None or confidence < 0.3:
            smooth = pos.copy()
        else:
            smooth = self.alpha * self._prev_smooth + (1.0 - self.alpha) * pos
        self._prev_smooth = smooth.copy()
        self.smoothed_positions.append(smooth)

        # Trim if over limit
        if len(self.raw_positions) > self.max_history:
            self.raw_positions.pop(0)
            self.smoothed_positions.pop(0)
            self.confidences.pop(0)
            self.frame_indices.pop(0)

    @property
    def length(self) -> int:
        return len(self.raw_positions)

    # ── derived quantities ────────────────────────────────────────────
    def get_positions(self, smoothed: bool = True) -> np.ndarray:
        """Return (N, 2) array of positions."""
        data = self.smoothed_positions if smoothed else self.raw_positions
        if not data:
            return np.empty((0, 2), dtype=np.float64)
        return np.array(data, dtype=np.float64)

    def get_velocities(self, smoothed: bool = True) -> np.ndarray:
        """Return (N-1, 2) array of velocity vectors (pixels/s)."""
        pos = self.get_positions(smoothed)
        if len(pos) < 2:
            return np.empty((0, 2), dtype=np.float64)
        return np.diff(pos, axis=0) / self.dt

    def get_speeds(self, smoothed: bool = True) -> np.ndarray:
        """Return (N-1,) array of scalar speeds (pixels/s)."""
        vel = self.get_velocities(smoothed)
        if len(vel) == 0:
            return np.empty(0, dtype=np.float64)
        return np.linalg.norm(vel, axis=1)

    def get_accelerations(self, smoothed: bool = True) -> np.ndarray:
        """Return (N-2, 2) array of acceleration vectors (pixels/s²)."""
        vel = self.get_velocities(smoothed)
        if len(vel) < 2:
            return np.empty((0, 2), dtype=np.float64)
        return np.diff(vel, axis=0) / self.dt

    def peak_speed_frame(self, smoothed: bool = True) -> Optional[Tuple[int, float]]:
        """Return (frame_index, speed) of the maximum speed, or None."""
        speeds = self.get_speeds(smoothed)
        if len(speeds) == 0:
            return None
        idx = int(np.argmax(speeds))
        # Speed[i] corresponds to the interval between frame[i] and frame[i+1]
        # We attribute it to frame[i+1]
        if idx + 1 < len(self.frame_indices):
            return self.frame_indices[idx + 1], float(speeds[idx])
        return self.frame_indices[-1], float(speeds[idx])

    def position_at_frame(self, frame_idx: int, smoothed: bool = True) -> Optional[np.ndarray]:
        """Return the position at a specific frame index, or None."""
        try:
            i = self.frame_indices.index(frame_idx)
        except ValueError:
            return None
        data = self.smoothed_positions if smoothed else self.raw_positions
        return data[i].copy()

    def reset(self):
        self.raw_positions.clear()
        self.smoothed_positions.clear()
        self.confidences.clear()
        self.frame_indices.clear()
        self._prev_smooth = None


class TrajectoryStore:
    """Container managing ``JointTrajectory`` for all 17 COCO keypoints."""

    def __init__(self, fps: float = 30.0, smoothing_factor: float = 0.4):
        self.fps = fps
        self.trajectories: Dict[int, JointTrajectory] = {
            idx: JointTrajectory(idx, fps=fps, smoothing_factor=smoothing_factor)
            for idx in range(NUM_KEYPOINTS)
        }

    def update(self, keypoints: np.ndarray, confidence: np.ndarray, frame_idx: int):
        """Record one frame of keypoint data for all joints.

        Parameters
        ----------
        keypoints : (17, 2) array
        confidence : (17,) array
        frame_idx : int
        """
        for idx in range(min(len(keypoints), NUM_KEYPOINTS)):
            self.trajectories[idx].add(
                float(keypoints[idx, 0]),
                float(keypoints[idx, 1]),
                float(confidence[idx]),
                frame_idx,
            )

    def get(self, joint: int | str) -> JointTrajectory:
        """Retrieve trajectory by index or name."""
        if isinstance(joint, str):
            joint = KEYPOINT_NAMES[joint]
        return self.trajectories[joint]

    def reset(self):
        for t in self.trajectories.values():
            t.reset()
