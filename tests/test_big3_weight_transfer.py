from __future__ import annotations

import numpy as np

from tennis_analyzer.analysis.big3_monitors import Status, WeightTransferMonitor
from tennis_analyzer.config.keypoints import KEYPOINT_NAMES


def _pose(
    *,
    ankle_y: float,
    hip_x: float,
    wrist_x: float = 0.0,
    conf_ankle: float = 1.0,
    conf_hip: float = 1.0,
    conf_wrist: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    kps = np.zeros((17, 2), dtype=np.float32)
    conf = np.zeros((17,), dtype=np.float32)

    ra = KEYPOINT_NAMES["right_ankle"]
    rh = KEYPOINT_NAMES["right_hip"]
    rw = KEYPOINT_NAMES["right_wrist"]

    kps[ra] = np.array([0.0, ankle_y], dtype=np.float32)
    kps[rh] = np.array([hip_x, 0.0], dtype=np.float32)
    kps[rw] = np.array([wrist_x, 0.0], dtype=np.float32)

    conf[ra] = conf_ankle
    conf[rh] = conf_hip
    conf[rw] = conf_wrist

    return kps, conf


def test_weight_transfer_uses_preimpact_baseline_and_impact_pose():
    wt = WeightTransferMonitor(
        is_right_handed=True,
        baseline_window_frames=20,
        baseline_exclude_tail_frames=2,
        min_baseline_samples=6,
        good_threshold_px=30.0,
        ok_threshold_px=10.0,
    )

    # Baseline: frames 0..19, ankle_y=100, hip_x=0
    for t in range(20):
        kps, conf = _pose(ankle_y=100.0, hip_x=0.0)
        st = wt.update(kps, conf, frame_idx=t, is_impact=False)
        assert st.status == Status.UNKNOWN

    # Impact pose at frame 20 (ankle lifted by 10px, hip moved forward by 10px).
    impact_kps, impact_conf = _pose(ankle_y=90.0, hip_x=10.0)
    # Current frame (trigger frame) can be anything; evaluation should use impact_*.
    cur_kps, cur_conf = _pose(ankle_y=95.0, hip_x=5.0)

    st = wt.update(
        cur_kps,
        cur_conf,
        frame_idx=21,
        is_impact=True,
        impact_frame_idx=20,
        impact_keypoints=impact_kps,
        impact_confidence=impact_conf,
        forward_sign=1.0,
    )

    # ankle_lift_px=10, hip_forward_px=10 => score_px=15 => OK (>=10, <30)
    assert st.status == Status.OK
    assert st.ankle_lift_px == 10.0
    assert st.hip_forward_px == 10.0
    assert st.score_px == 15.0

    # After impact, the monitor should keep returning the frozen result (until next impact/reset).
    st2 = wt.update(cur_kps, cur_conf, frame_idx=22, is_impact=False)
    assert st2.status == Status.OK
    assert st2.score_px == 15.0
