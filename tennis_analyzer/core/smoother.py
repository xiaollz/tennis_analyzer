"""Temporal smoothing for keypoints and metrics."""

import numpy as np
from typing import Dict, Optional, List
from collections import deque


class KeypointSmoother:
    """
    Smooth keypoints over time using exponential moving average.

    Reduces jitter/noise in pose detection.
    Uses adaptive smoothing - less smoothing for fast-moving body parts (legs).
    """

    # Leg keypoint indices (need less smoothing due to fast movement)
    LEG_KEYPOINTS = {11, 12, 13, 14, 15, 16}  # hips, knees, ankles

    def __init__(self, smoothing_factor: float = 0.5, window_size: int = 5):
        """
        Initialize keypoint smoother.

        Args:
            smoothing_factor: EMA factor (0-1). Higher = more smoothing, more lag.
            window_size: Number of frames to consider for smoothing.
        """
        self.smoothing_factor = smoothing_factor
        # Use much less smoothing for legs (they move fast in tennis)
        self.leg_smoothing_factor = max(0.1, smoothing_factor - 0.3)
        self.window_size = window_size
        self.history: deque = deque(maxlen=window_size)
        self.prev_keypoints: Optional[np.ndarray] = None

    def smooth(
        self,
        keypoints: np.ndarray,
        confidence: np.ndarray,
        min_confidence: float = 0.3
    ) -> np.ndarray:
        """
        Apply temporal smoothing to keypoints.

        Args:
            keypoints: (17, 2) array of x, y coordinates
            confidence: (17,) array of confidence scores
            min_confidence: Minimum confidence to include in smoothing

        Returns:
            Smoothed keypoints
        """
        if self.prev_keypoints is None:
            self.prev_keypoints = keypoints.copy()
            self.history.append(keypoints.copy())
            return keypoints

        smoothed = np.zeros_like(keypoints)

        for i in range(len(keypoints)):
            if confidence[i] >= min_confidence:
                # Use less smoothing for legs (faster movement)
                if i in self.LEG_KEYPOINTS:
                    factor = self.leg_smoothing_factor
                else:
                    factor = self.smoothing_factor

                # EMA smoothing
                smoothed[i] = (
                    factor * self.prev_keypoints[i] +
                    (1 - factor) * keypoints[i]
                )
            else:
                # Keep previous position if low confidence
                smoothed[i] = self.prev_keypoints[i]

        self.prev_keypoints = smoothed.copy()
        self.history.append(smoothed.copy())

        return smoothed

    def smooth_moving_average(
        self,
        keypoints: np.ndarray,
        confidence: np.ndarray,
        min_confidence: float = 0.3
    ) -> np.ndarray:
        """
        Apply moving average smoothing (alternative method).

        Args:
            keypoints: Current keypoints
            confidence: Confidence scores
            min_confidence: Minimum confidence threshold

        Returns:
            Smoothed keypoints
        """
        self.history.append(keypoints.copy())

        if len(self.history) < 2:
            return keypoints

        # Stack history and compute weighted average
        history_array = np.array(self.history)

        # More recent frames get higher weight
        weights = np.linspace(0.5, 1.0, len(self.history))
        weights = weights / weights.sum()

        smoothed = np.zeros_like(keypoints)
        for i in range(len(keypoints)):
            if confidence[i] >= min_confidence:
                smoothed[i] = np.average(history_array[:, i], axis=0, weights=weights)
            else:
                smoothed[i] = keypoints[i]

        return smoothed

    def reset(self):
        """Reset smoother state."""
        self.history.clear()
        self.prev_keypoints = None


class MetricsSmoother:
    """Smooth metrics over time to reduce flickering."""

    def __init__(self, smoothing_factor: float = 0.7, window_size: int = 10):
        """
        Initialize metrics smoother.

        Args:
            smoothing_factor: EMA factor for smoothing
            window_size: Window size for moving average
        """
        self.smoothing_factor = smoothing_factor
        self.window_size = window_size
        self.history: Dict[str, deque] = {}
        self.prev_metrics: Dict[str, float] = {}

    def smooth(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Smooth metrics using EMA.

        Args:
            metrics: Dictionary of metric name to value

        Returns:
            Smoothed metrics
        """
        smoothed = {}

        for name, value in metrics.items():
            if name not in self.prev_metrics:
                self.prev_metrics[name] = value
                smoothed[name] = value
            else:
                # EMA smoothing
                smoothed[name] = (
                    self.smoothing_factor * self.prev_metrics[name] +
                    (1 - self.smoothing_factor) * value
                )
                self.prev_metrics[name] = smoothed[name]

        return smoothed

    def smooth_moving_average(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Smooth metrics using moving average.

        Args:
            metrics: Dictionary of metric name to value

        Returns:
            Smoothed metrics
        """
        smoothed = {}

        for name, value in metrics.items():
            if name not in self.history:
                self.history[name] = deque(maxlen=self.window_size)

            self.history[name].append(value)

            # Compute weighted moving average
            values = list(self.history[name])
            if len(values) > 1:
                weights = np.linspace(0.5, 1.0, len(values))
                smoothed[name] = np.average(values, weights=weights)
            else:
                smoothed[name] = value

        return smoothed

    def reset(self):
        """Reset smoother state."""
        self.history.clear()
        self.prev_metrics.clear()


class ActionSmoother:
    """Smooth action classification to reduce flickering."""

    def __init__(self, window_size: int = 15, threshold: float = 0.6):
        """
        Initialize action smoother.

        Args:
            window_size: Number of frames to consider
            threshold: Minimum ratio to confirm action change
        """
        self.window_size = window_size
        self.threshold = threshold
        self.history: deque = deque(maxlen=window_size)
        self.current_action = None

    def smooth(self, action) -> any:
        """
        Smooth action classification.

        Args:
            action: Current detected action

        Returns:
            Smoothed action (most stable)
        """
        self.history.append(action)

        if len(self.history) < 3:
            return action

        # Count occurrences
        counts = {}
        for a in self.history:
            counts[a] = counts.get(a, 0) + 1

        # Find most common
        most_common = max(counts, key=counts.get)
        ratio = counts[most_common] / len(self.history)

        # Only change if confident enough
        if ratio >= self.threshold:
            self.current_action = most_common
        elif self.current_action is None:
            self.current_action = most_common

        return self.current_action

    def reset(self):
        """Reset smoother state."""
        self.history.clear()
        self.current_action = None
