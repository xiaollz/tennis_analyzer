from __future__ import annotations

import numpy as np

from tennis_analyzer.analysis.impact import WristSpeedImpactDetector
from tennis_analyzer.config.keypoints import KEYPOINT_NAMES


def _empty_pose() -> tuple[np.ndarray, np.ndarray]:
    keypoints = np.zeros((17, 2), dtype=np.float32)
    confidence = np.zeros((17,), dtype=np.float32)
    return keypoints, confidence


def test_wrist_speed_peak_indexing_and_trigger_frame():
    det = WristSpeedImpactDetector(fps=30.0, is_right_handed=True, cooldown_frames=10)

    wrist = KEYPOINT_NAMES["right_wrist"]
    ls = KEYPOINT_NAMES["left_shoulder"]
    rs = KEYPOINT_NAMES["right_shoulder"]

    # Provide a stable shoulder width so normalized speeds are available.
    def frame(wrist_x: float) -> tuple[np.ndarray, np.ndarray]:
        kps, conf = _empty_pose()
        kps[ls] = np.array([0.0, 0.0], dtype=np.float32)
        kps[rs] = np.array([100.0, 0.0], dtype=np.float32)
        conf[ls] = 1.0
        conf[rs] = 1.0
        kps[wrist] = np.array([wrist_x, 0.0], dtype=np.float32)
        conf[wrist] = 1.0
        return kps, conf

    # Wrist positions (px): 0 -> 10 -> 30 -> 35 produces a clear speed peak at frame 2,
    # detected when processing frame 3.
    poses = [frame(x) for x in (0.0, 10.0, 30.0, 35.0)]

    events = []
    for idx, (kps, conf) in enumerate(poses):
        event, _speed = det.update(idx, kps, conf)
        events.append(event)

    assert events[0] is None
    assert events[1] is None
    assert events[2] is None
    assert events[3] is not None
    assert events[3].impact_frame_idx == 2
    assert events[3].trigger_frame_idx == 3
    assert events[3].peak_speed_sw_s is not None


def test_wrist_speed_cooldown_blocks_then_allows_next_peak():
    det = WristSpeedImpactDetector(fps=30.0, is_right_handed=True, cooldown_frames=3)
    wrist = KEYPOINT_NAMES["right_wrist"]
    ls = KEYPOINT_NAMES["left_shoulder"]
    rs = KEYPOINT_NAMES["right_shoulder"]

    def frame(wrist_x: float) -> tuple[np.ndarray, np.ndarray]:
        kps, conf = _empty_pose()
        kps[ls] = np.array([0.0, 0.0], dtype=np.float32)
        kps[rs] = np.array([100.0, 0.0], dtype=np.float32)
        conf[ls] = 1.0
        conf[rs] = 1.0
        kps[wrist] = np.array([wrist_x, 0.0], dtype=np.float32)
        conf[wrist] = 1.0
        return kps, conf

    # Two peaks: one detected at trigger frame 3 (impact=2), another at trigger frame 7 (impact=6).
    xs = (0.0, 10.0, 30.0, 35.0, 40.0, 50.0, 80.0, 85.0)
    impacts = []
    for idx, x in enumerate(xs):
        kps, conf = frame(x)
        event, _speed = det.update(idx, kps, conf)
        if event is not None:
            impacts.append((event.trigger_frame_idx, event.impact_frame_idx))

    assert impacts == [(3, 2), (7, 6)]
