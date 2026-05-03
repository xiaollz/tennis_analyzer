"""Tests for evaluation/hsa_detector.py.

覆盖 5 种 HSA 闭合失败模式 + 1 种健康模式 + 边界情况。
"""

from __future__ import annotations

import numpy as np
import pytest

from config.keypoints import KEYPOINT_NAMES
from evaluation.hsa_detector import (
    HSAMetrics,
    classify_closure_pattern,
    compute_health_score,
    compute_hsa_trajectory,
    cross_body_finish_distance_2d,
    detect_hsa,
    hsa_angle_2d,
)


# =============================================================================
# 工具：合成单帧 / 时间序列 keypoints
# =============================================================================

def _build_frame(
    angle_deg: float,
    elbow_len: float = 80.0,
    wrist_extra: float = 60.0,
    l_sho: tuple = (100.0, 100.0),
    r_sho: tuple = (200.0, 100.0),
) -> tuple:
    """构造一帧 COCO 17 keypoints + confidence。

    angle_deg: 大臂相对躯干横线的角度。
    """
    kpts = np.zeros((17, 2), dtype=np.float64)
    conf = np.ones(17, dtype=np.float64)

    kpts[KEYPOINT_NAMES["left_shoulder"]] = list(l_sho)
    kpts[KEYPOINT_NAMES["right_shoulder"]] = list(r_sho)
    kpts[KEYPOINT_NAMES["left_hip"]] = [l_sho[0], l_sho[1] + 200]
    kpts[KEYPOINT_NAMES["right_hip"]] = [r_sho[0], r_sho[1] + 200]

    # 大臂方向：从右肩出发，相对躯干横线方向(→左肩)外展 angle_deg
    rad = np.deg2rad(180.0 - angle_deg)
    elb = (r_sho[0] + elbow_len * np.cos(rad), r_sho[1] + elbow_len * np.sin(rad))
    wri = (r_sho[0] + (elbow_len + wrist_extra) * np.cos(rad),
           r_sho[1] + (elbow_len + wrist_extra) * np.sin(rad))
    kpts[KEYPOINT_NAMES["right_elbow"]] = list(elb)
    kpts[KEYPOINT_NAMES["right_wrist"]] = list(wri)

    return kpts, conf


def _build_sequence(angle_trajectory: list, fps: float = 60.0) -> tuple:
    """根据 HSA 角度轨迹构造时间序列。"""
    T = len(angle_trajectory)
    kpts_seq = np.zeros((T, 17, 2), dtype=np.float64)
    conf_seq = np.ones((T, 17), dtype=np.float64)
    for t, angle in enumerate(angle_trajectory):
        k, c = _build_frame(angle)
        kpts_seq[t] = k
        conf_seq[t] = c
    return kpts_seq, conf_seq


# =============================================================================
# 单帧几何
# =============================================================================

def test_hsa_angle_unit_turn_typical_value():
    """引拍顶点 HSA 应在 90-100° 区间。"""
    k, c = _build_frame(95.0)
    angle = hsa_angle_2d(k, c)
    assert angle == pytest.approx(95.0, abs=1.0)


def test_hsa_angle_contact_typical_value():
    """接触瞬间 HSA 应在 50-70°。"""
    k, c = _build_frame(60.0)
    angle = hsa_angle_2d(k, c)
    assert angle == pytest.approx(60.0, abs=1.0)


def test_hsa_angle_returns_none_when_keypoint_low_confidence():
    """关键点置信度不足时返回 None。"""
    k, c = _build_frame(60.0)
    c[KEYPOINT_NAMES["right_elbow"]] = 0.05
    assert hsa_angle_2d(k, c) is None


def test_cross_body_finish_positive_when_wrist_crosses():
    """随挥末端腕越过左肩 → 距离 > 0。"""
    # 角度 30°：大臂跨过身体中线
    k, c = _build_frame(20.0, elbow_len=100, wrist_extra=120)
    dist = cross_body_finish_distance_2d(k, c)
    assert dist is not None and dist > 0


def test_cross_body_finish_negative_when_wrist_outside():
    """随挥未跨胸 → 距离 < 0。"""
    k, c = _build_frame(95.0)
    dist = cross_body_finish_distance_2d(k, c)
    assert dist is not None and dist < 0


# =============================================================================
# 时间序列
# =============================================================================

def test_hsa_trajectory_decreasing_during_swing():
    """模拟健康挥拍：角度从 95° 单调下降到 35°。"""
    angles = list(np.linspace(95.0, 35.0, 60))
    kpts_seq, conf_seq = _build_sequence(angles)
    traj = compute_hsa_trajectory(kpts_seq, conf_seq)
    assert traj[0] == pytest.approx(95.0, abs=1.0)
    assert traj[-1] == pytest.approx(35.0, abs=1.0)
    # 总体单调递减
    assert np.all(np.diff(traj[~np.isnan(traj)]) < 0)


# =============================================================================
# 5 种闭合模式分类
# =============================================================================

def test_pattern_healthy():
    """健康挥拍：30-45° 闭合，接触角在区间内。"""
    angles = list(np.linspace(95.0, 45.0, 30)) + list(np.linspace(45.0, 30.0, 30))
    kpts, conf = _build_sequence(angles)
    result = detect_hsa(kpts, conf, fps=60.0, phase_frames={
        "unit_turn_peak": 0,
        "forward_swing_start": 15,
        "contact": 29,
        "follow_through": 59,
    })
    assert result.hsa_closure_pattern == "healthy"
    assert result.hsa_health_score >= 80


def test_pattern_no_closure():
    """no_closure：接触瞬间角度仍 > 85°。"""
    angles = [95.0] * 60
    kpts, conf = _build_sequence(angles)
    result = detect_hsa(kpts, conf, fps=60.0, phase_frames={
        "unit_turn_peak": 0,
        "contact": 30,
        "follow_through": 59,
    })
    assert result.hsa_closure_pattern == "no_closure"
    assert result.hsa_health_score < 50


def test_pattern_static():
    """static：总闭合 < 10°。"""
    angles = list(np.linspace(75.0, 70.0, 60))  # 5° 闭合
    kpts, conf = _build_sequence(angles)
    result = detect_hsa(kpts, conf, fps=60.0, phase_frames={
        "unit_turn_peak": 0,
        "contact": 30,
        "follow_through": 59,
    })
    assert result.hsa_closure_pattern == "static"


def test_pattern_early_closure():
    """early_closure：接触瞬间角度 < 45°。"""
    angles = list(np.linspace(95.0, 30.0, 30)) + [30.0] * 30
    kpts, conf = _build_sequence(angles)
    result = detect_hsa(kpts, conf, fps=60.0, phase_frames={
        "unit_turn_peak": 0,
        "contact": 35,
        "follow_through": 59,
    })
    assert result.hsa_closure_pattern == "early_closure"


def test_pattern_late_closure():
    """late_closure：接触后闭合 > 接触前闭合 × 1.5。"""
    # 接触前: 95° → 80° (15° 闭合); 接触后: 80° → 30° (50° 闭合)
    pre_contact = list(np.linspace(95.0, 80.0, 30))
    post_contact = list(np.linspace(80.0, 30.0, 30))
    angles = pre_contact + post_contact
    kpts, conf = _build_sequence(angles)
    result = detect_hsa(kpts, conf, fps=60.0, phase_frames={
        "unit_turn_peak": 0,
        "contact": 29,
        "follow_through": 59,
    })
    assert result.hsa_closure_pattern == "late_closure"


def test_pattern_insufficient_cross_body():
    """随挥末端腕未跨过非持拍肩。"""
    # 健康闭合幅度 + 接触角度，但末端 wrist 仍在右侧
    # 通过限制 elbow_len 和 wrist_extra 让 wrist 不跨身
    T = 60
    kpts_seq = np.zeros((T, 17, 2), dtype=np.float64)
    conf_seq = np.ones((T, 17), dtype=np.float64)
    for t in range(T):
        progress = t / (T - 1)
        angle = 95.0 - 35.0 * progress  # 95 → 60
        k, c = _build_frame(angle, elbow_len=50, wrist_extra=20)  # 短手臂，不能跨胸
        kpts_seq[t] = k
        conf_seq[t] = c
    result = detect_hsa(kpts_seq, conf_seq, fps=60.0, phase_frames={
        "unit_turn_peak": 0,
        "contact": 30,
        "follow_through": 59,
    })
    assert result.cross_body_finish is False
    # 即使健康闭合，cross_body 缺失也分类为 insufficient_cross_body
    assert result.hsa_closure_pattern == "insufficient_cross_body"


# =============================================================================
# 健康分边界
# =============================================================================

def test_health_score_full_for_perfect_swing():
    angles = list(np.linspace(95.0, 35.0, 60))
    kpts, conf = _build_sequence(angles)
    result = detect_hsa(kpts, conf, fps=60.0, phase_frames={
        "unit_turn_peak": 0,
        "forward_swing_start": 15,
        "contact": 45,
        "follow_through": 59,
    })
    assert result.hsa_health_score == pytest.approx(100.0, abs=1.0)


def test_health_score_clipped_to_zero():
    """全坏的 swing：分数被夹到 0。"""
    angles = [95.0] * 60  # 完全无闭合
    kpts, conf = _build_sequence(angles)
    result = detect_hsa(kpts, conf, fps=60.0, phase_frames={
        "unit_turn_peak": 0,
        "contact": 30,
        "follow_through": 59,
    })
    assert 0.0 <= result.hsa_health_score < 50.0


# =============================================================================
# 边界 / 缺失数据
# =============================================================================

def test_uncertain_when_phase_missing():
    """关键 phase 缺失时返回 uncertain。"""
    metrics = HSAMetrics()  # 全空
    pattern, evidence = classify_closure_pattern(metrics)
    assert pattern == "uncertain"


def test_empty_sequence_returns_empty_metrics():
    kpts = np.zeros((0, 17, 2), dtype=np.float64)
    conf = np.zeros((0, 17), dtype=np.float64)
    result = detect_hsa(kpts, conf, fps=60.0, phase_frames={
        "unit_turn_peak": 0, "contact": 0, "follow_through": 0,
    })
    assert result.hsa_closure_pattern == "uncertain"
    assert result.hsa_angle_at_contact is None


def test_closure_velocity_negative_during_healthy_swing():
    """健康挥拍中接触前 100ms 平均闭合速度应为负值（角度递减）。"""
    angles = list(np.linspace(95.0, 35.0, 60))
    kpts, conf = _build_sequence(angles, fps=60.0)
    result = detect_hsa(kpts, conf, fps=60.0, phase_frames={
        "unit_turn_peak": 0,
        "contact": 45,
        "follow_through": 59,
    })
    assert result.hsa_closure_rate_deg_per_s is not None
    assert result.hsa_closure_rate_deg_per_s < 0  # 闭合 = 角度减小 = 负速度
