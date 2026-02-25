"""Tennis Analyzer v3 — 综合测试套件。

覆盖所有核心模块：配置（8阶段 + 单反）、运动学计算（v3 新增函数）、
轨迹管理、击球检测、KPI 评分（20 正手 + 17 单反）、评估引擎、
可视化和报告生成。
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np

from config.framework_config import DEFAULT_CONFIG, FrameworkConfig, ScoringWeights
from config.keypoints import COCO_KEYPOINTS, KEYPOINT_NAMES, SKELETON_CONNECTIONS, NUM_KEYPOINTS
from analysis.trajectory import JointTrajectory, TrajectoryStore
from analysis.kinematic_calculator import (
    joint_angle, elbow_angle, knee_angle, min_knee_angle,
    shoulder_hip_angle, spine_angle_from_vertical,
    torso_height_px, shoulder_width_px,
    wrist_forward_normalised, nose_position,
    shoulder_center, hip_center,
    # v3 new
    elbow_behind_torso_normalised,
    wrist_below_elbow_distance,
    hip_center_vertical_position,
    compute_vertical_acceleration,
    elbow_to_torso_distance,
    forearm_angle,
    compute_angular_velocity,
    wrist_lateral_displacement,
    compute_wiper_sweep_angle,
    hip_line_angle,
    shoulder_line_angle,
    compute_rotation_speed,
    compute_peak_rotation_speed,
    hip_shoulder_separation_timing,
)
from evaluation.event_detector import (
    WristSpeedDetector, ImpactEvent, SwingEvent, SwingPhaseEstimator,
)
from evaluation.kpi import (
    ShoulderRotationKPI, KneeBendKPI, SpineAngleKPI,
    ElbowBackPositionKPI, RacketDropKPI,
    GroundForceProxyKPI, HipRotationSpeedKPI,
    HipShoulderSeparationKPI, HipShoulderTimingKPI,
    ElbowTuckKPI, HandPathLinearityKPI,
    ContactPointKPI, ElbowAngleAtContactKPI,
    BodyFreezeKPI, HeadStabilityAtContactKPI, SIRProxyKPI,
    ForwardExtensionKPI, WiperSweepKPI,
    OverallHeadStabilityKPI, SpineConsistencyKPI,
    ALL_KPIS, KPIResult, _linear_score, _rating_from_score,
)
from evaluation.forehand_evaluator import (
    ForehandEvaluator, MultiSwingReport, SwingEvaluation, PhaseScore,
)
from report.visualizer import SkeletonDrawer, TrajectoryDrawer, ChartGenerator, JOINT_CN
from report.report_generator import ReportGenerator


# ── Fixtures ─────────────────────────────────────────────────────────

def _make_standing_pose():
    """创建一个合成的站立姿势关键点数组 (17, 2)。"""
    kp = np.zeros((17, 2), dtype=np.float32)
    conf = np.ones(17, dtype=np.float32)
    kp[0] = [200, 50]    # nose
    kp[1] = [195, 45]    # left_eye
    kp[2] = [205, 45]    # right_eye
    kp[3] = [190, 50]    # left_ear
    kp[4] = [210, 50]    # right_ear
    kp[5] = [170, 100]   # left_shoulder
    kp[6] = [230, 100]   # right_shoulder
    kp[7] = [150, 150]   # left_elbow
    kp[8] = [250, 150]   # right_elbow
    kp[9] = [140, 200]   # left_wrist
    kp[10] = [260, 200]  # right_wrist
    kp[11] = [180, 200]  # left_hip
    kp[12] = [220, 200]  # right_hip
    kp[13] = [175, 280]  # left_knee
    kp[14] = [225, 280]  # right_knee
    kp[15] = [170, 360]  # left_ankle
    kp[16] = [230, 360]  # right_ankle
    return kp, conf


def _make_swing_sequence(n_frames=60, fps=30.0):
    """创建一个合成的正手挥拍序列。"""
    kp_series = []
    conf_series = []
    base_kp, base_conf = _make_standing_pose()
    for i in range(n_frames):
        kp = base_kp.copy()
        t = i / n_frames
        if t < 0.3:
            kp[10, 0] = 260 - t * 100
        elif t < 0.6:
            swing_t = (t - 0.3) / 0.3
            kp[10, 0] = 230 + swing_t * 150
            kp[10, 1] = 200 - swing_t * 30
        else:
            ft_t = (t - 0.6) / 0.4
            kp[10, 0] = 380 - ft_t * 50
            kp[10, 1] = 170 - ft_t * 60
        if t < 0.3:
            kp[5, 0] = 170 + t * 30
            kp[6, 0] = 230 - t * 30
        elif t < 0.6:
            swing_t = (t - 0.3) / 0.3
            kp[5, 0] = 179 - swing_t * 20
            kp[6, 0] = 221 + swing_t * 20
        kp_series.append(kp)
        conf_series.append(base_conf.copy())
    return kp_series, conf_series


# =====================================================================
# Config tests
# =====================================================================

class TestConfig:
    def test_keypoint_count(self):
        assert len(COCO_KEYPOINTS) == 17

    def test_keypoint_names_reverse(self):
        for idx, name in COCO_KEYPOINTS.items():
            assert KEYPOINT_NAMES[name] == idx

    def test_skeleton_connections(self):
        assert len(SKELETON_CONNECTIONS) == 12
        for s, e in SKELETON_CONNECTIONS:
            assert 0 <= s < 17
            assert 0 <= e < 17

    def test_default_config_v3_8_phases(self):
        """v3: 8 阶段权重配置。"""
        cfg = DEFAULT_CONFIG
        weights = cfg.scoring.as_dict()
        assert len(weights) == 8
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        # 检查所有 8 个阶段都存在
        expected_phases = ["unit_turn", "slot_prep", "leg_drive", "torso_pull",
                           "lag_drive", "contact", "wiper", "balance"]
        for phase in expected_phases:
            assert phase in weights, f"Missing phase: {phase}"

    def test_scoring_weights_v3(self):
        w = ScoringWeights()
        assert w.unit_turn == 0.10
        assert w.slot_prep == 0.10
        assert w.leg_drive == 0.15
        assert w.torso_pull == 0.15
        assert w.lag_drive == 0.10
        assert w.contact == 0.20
        assert w.wiper == 0.10
        assert w.balance == 0.10


# =====================================================================
# Kinematic calculator tests
# =====================================================================

class TestKinematicCalculator:
    def test_joint_angle_straight(self):
        kp = np.zeros((17, 2), dtype=np.float32)
        conf = np.ones(17, dtype=np.float32)
        kp[5] = [0, 0]; kp[7] = [100, 0]; kp[9] = [200, 0]
        angle = joint_angle(kp, conf, 5, 7, 9)
        assert angle is not None
        assert abs(angle - 180.0) < 1.0

    def test_joint_angle_right_angle(self):
        kp = np.zeros((17, 2), dtype=np.float32)
        conf = np.ones(17, dtype=np.float32)
        kp[5] = [0, 0]; kp[7] = [100, 0]; kp[9] = [100, 100]
        angle = joint_angle(kp, conf, 5, 7, 9)
        assert angle is not None
        assert abs(angle - 90.0) < 1.0

    def test_elbow_angle(self):
        kp, conf = _make_standing_pose()
        angle = elbow_angle(kp, conf, right=True)
        assert angle is not None
        assert 0 < angle <= 180

    def test_knee_angle(self):
        kp, conf = _make_standing_pose()
        angle = knee_angle(kp, conf, right=True)
        assert angle is not None
        assert 0 < angle <= 180

    def test_min_knee_angle(self):
        kp, conf = _make_standing_pose()
        assert min_knee_angle(kp, conf) is not None

    def test_shoulder_hip_angle(self):
        kp, conf = _make_standing_pose()
        angle = shoulder_hip_angle(kp, conf)
        assert angle is not None
        assert angle < 30

    def test_spine_angle(self):
        kp, conf = _make_standing_pose()
        angle = spine_angle_from_vertical(kp, conf)
        assert angle is not None
        assert angle < 30

    def test_torso_height(self):
        kp, conf = _make_standing_pose()
        assert torso_height_px(kp, conf) is not None

    def test_shoulder_width(self):
        kp, conf = _make_standing_pose()
        assert shoulder_width_px(kp, conf) is not None

    def test_wrist_forward_normalised(self):
        kp, conf = _make_standing_pose()
        val = wrist_forward_normalised(kp, conf, is_right_handed=True, forward_sign=1.0)
        assert val is not None

    def test_low_confidence_returns_none(self):
        kp, conf = _make_standing_pose()
        conf[:] = 0.1
        assert elbow_angle(kp, conf) is None
        assert shoulder_hip_angle(kp, conf) is None

    def test_nose_position(self):
        kp, conf = _make_standing_pose()
        pos = nose_position(kp, conf)
        assert pos is not None and len(pos) == 2

    # v3 new functions
    def test_elbow_behind_torso_normalised(self):
        kp, conf = _make_standing_pose()
        val = elbow_behind_torso_normalised(kp, conf, is_right_handed=True, forward_sign=1.0)
        assert val is not None

    def test_wrist_below_elbow_distance(self):
        kp, conf = _make_standing_pose()
        val = wrist_below_elbow_distance(kp, conf, right=True)
        assert val is not None

    def test_hip_center_vertical_position(self):
        kp, conf = _make_standing_pose()
        val = hip_center_vertical_position(kp, conf)
        assert val is not None

    def test_elbow_to_torso_distance(self):
        kp, conf = _make_standing_pose()
        val = elbow_to_torso_distance(kp, conf, right=True)
        assert val is not None

    def test_forearm_angle(self):
        kp, conf = _make_standing_pose()
        val = forearm_angle(kp, conf, right=True)
        assert val is not None

    def test_hip_line_angle(self):
        kp, conf = _make_standing_pose()
        val = hip_line_angle(kp, conf)
        assert val is not None

    def test_shoulder_line_angle(self):
        kp, conf = _make_standing_pose()
        val = shoulder_line_angle(kp, conf)
        assert val is not None


# =====================================================================
# Trajectory tests
# =====================================================================

class TestTrajectory:
    def test_joint_trajectory_basic(self):
        traj = JointTrajectory(10, fps=30.0)
        for i in range(10):
            traj.add(float(i * 10), float(i * 5), 0.9, i)
        assert traj.length == 10
        assert traj.get_positions().shape == (10, 2)

    def test_velocity_computation(self):
        traj = JointTrajectory(10, fps=30.0)
        for i in range(10):
            traj.add(float(i * 30), 0.0, 0.9, i)
        vel = traj.get_velocities()
        assert vel.shape == (9, 2)
        assert vel[0, 0] > 0

    def test_speed_computation(self):
        traj = JointTrajectory(10, fps=30.0)
        for i in range(10):
            traj.add(float(i * 30), 0.0, 0.9, i)
        speeds = traj.get_speeds()
        assert len(speeds) == 9
        assert all(s > 0 for s in speeds)

    def test_peak_speed_frame(self):
        traj = JointTrajectory(10, fps=30.0)
        positions = [0, 5, 10, 15, 50, 90, 95, 98, 99, 100]
        for i, x in enumerate(positions):
            traj.add(float(x), 0.0, 0.9, i)
        peak = traj.peak_speed_frame()
        assert peak is not None and peak[1] > 0

    def test_trajectory_store(self):
        store = TrajectoryStore(fps=30.0)
        kp = np.random.rand(17, 2).astype(np.float32) * 100
        conf = np.ones(17, dtype=np.float32)
        store.update(kp, conf, 0)
        store.update(kp + 10, conf, 1)
        assert store.get("right_wrist").length == 2


# =====================================================================
# Impact detector tests
# =====================================================================

class TestImpactDetector:
    def test_no_impact_on_static(self):
        det = WristSpeedDetector(fps=30.0)
        kp, conf = _make_standing_pose()
        for i in range(30):
            event, speed = det.update(i, kp, conf)
            assert event is None

    def test_impact_on_speed_peak(self):
        det = WristSpeedDetector(fps=30.0, is_right_handed=True)
        kp, conf = _make_standing_pose()
        wrist_idx = KEYPOINT_NAMES["right_wrist"]
        positions = [0, 5, 10, 15, 20, 30, 50, 100, 180, 250, 280, 290, 295, 298, 299, 300]
        detected = []
        for i, x in enumerate(positions):
            kp_frame = kp.copy()
            kp_frame[wrist_idx, 0] = float(x)
            event, speed = det.update(i, kp_frame, conf)
            if event is not None:
                detected.append(event)
        assert len(detected) >= 1

    def test_swing_phase_estimator(self):
        est = SwingPhaseEstimator(fps=30.0)
        speeds = np.array([10.0] * 10 + [50.0, 100.0, 200.0, 100.0, 50.0] + [10.0] * 10)
        frames = list(range(len(speeds)))
        event = est.estimate_phases(12, speeds, frames)
        assert event.impact_frame == 12
        assert event.prep_start_frame is not None
        assert event.followthrough_end_frame is not None


# =====================================================================
# KPI tests (v3 — 20 KPIs, 8 phases)
# =====================================================================

class TestKPIs:
    def test_linear_score_larger_is_better(self):
        assert _linear_score(100, 45, 70, 90) == 100.0
        assert _linear_score(90, 45, 70, 90) == 100.0
        assert _linear_score(70, 45, 70, 90) == 70.0
        assert _linear_score(45, 45, 70, 90) == 20.0

    def test_linear_score_smaller_is_better(self):
        assert _linear_score(5, 25, 15, 5) == 100.0
        assert _linear_score(15, 25, 15, 5) == 70.0

    def test_rating_from_score(self):
        assert _rating_from_score(90) == "优秀"
        assert _rating_from_score(70) == "良好"
        assert _rating_from_score(50) == "一般"
        assert _rating_from_score(30) == "较差"

    # Phase 1: Unit Turn
    def test_shoulder_rotation_kpi(self):
        kpi = ShoulderRotationKPI()
        result = kpi.evaluate(shoulder_rotation_values=[85.0, 90.0, 75.0])
        assert isinstance(result, KPIResult)
        assert result.raw_value == 90.0
        assert result.score > 70
        assert result.phase == "unit_turn"

    def test_knee_bend_kpi(self):
        kpi = KneeBendKPI()
        result = kpi.evaluate(knee_angle_values=[130.0, 125.0, 140.0])
        assert result.raw_value == 125.0
        assert result.score > 50
        assert result.phase == "unit_turn"

    def test_spine_angle_kpi(self):
        kpi = SpineAngleKPI()
        result = kpi.evaluate(spine_angle_values=[10.0, 12.0, 8.0])
        assert result.score > 70
        assert result.phase == "unit_turn"

    # Phase 2: Slot Prep
    def test_elbow_back_position_kpi(self):
        kpi = ElbowBackPositionKPI()
        result = kpi.evaluate(elbow_behind_norm_values=[0.3, 0.4, 0.5])
        assert result.phase == "slot_prep"
        assert 0 <= result.score <= 100

    def test_racket_drop_kpi(self):
        kpi = RacketDropKPI()
        result = kpi.evaluate(wrist_below_elbow_values=[30.0, 40.0, 50.0])
        assert result.phase == "slot_prep"
        assert 0 <= result.score <= 100

    # Phase 3: Leg Drive
    def test_ground_force_proxy_kpi(self):
        kpi = GroundForceProxyKPI()
        result = kpi.evaluate(vertical_accel_values=[500.0, 600.0, 700.0])
        assert result.phase == "leg_drive"
        assert 0 <= result.score <= 100

    def test_hip_rotation_speed_kpi(self):
        kpi = HipRotationSpeedKPI()
        result = kpi.evaluate(hip_rotation_speed_values=[200.0, 300.0, 400.0])
        assert result.phase == "leg_drive"
        assert 0 <= result.score <= 100

    # Phase 4: Torso Pull
    def test_hip_shoulder_separation_kpi(self):
        kpi = HipShoulderSeparationKPI()
        result = kpi.evaluate(hip_shoulder_sep_values=[30.0, 35.0, 40.0])
        assert result.phase == "torso_pull"
        assert 0 <= result.score <= 100

    def test_hip_shoulder_timing_kpi(self):
        kpi = HipShoulderTimingKPI()
        result = kpi.evaluate(hip_shoulder_timing_ms=50.0)
        assert result.phase == "torso_pull"
        assert 0 <= result.score <= 100

    # Phase 5: Lag & Elbow Drive
    def test_elbow_tuck_kpi(self):
        kpi = ElbowTuckKPI()
        result = kpi.evaluate(elbow_torso_distance_values=[20.0, 25.0, 30.0])
        assert result.phase == "lag_drive"
        assert 0 <= result.score <= 100

    def test_hand_path_linearity_kpi(self):
        kpi = HandPathLinearityKPI()
        result = kpi.evaluate(hand_path_r2=0.85)
        assert result.phase == "lag_drive"
        assert result.score > 50

    # Phase 6: Contact & SIR
    def test_contact_point_kpi(self):
        kpi = ContactPointKPI()
        result = kpi.evaluate(contact_forward_norm=0.5)
        assert result.phase == "contact"
        assert result.score > 70

    def test_elbow_angle_straight_arm(self):
        kpi = ElbowAngleAtContactKPI()
        result = kpi.evaluate(elbow_angle_at_contact=170.0)
        assert result.phase == "contact"
        assert result.score >= 85
        assert "直臂" in result.feedback

    def test_elbow_angle_double_bend(self):
        kpi = ElbowAngleAtContactKPI()
        result = kpi.evaluate(elbow_angle_at_contact=135.0)
        assert result.score >= 85
        assert "双弯" in result.feedback

    def test_body_freeze_kpi(self):
        kpi = BodyFreezeKPI()
        result = kpi.evaluate(decel_magnitude=800.0)
        assert result.phase == "contact"
        assert 0 <= result.score <= 100

    def test_head_stability_at_contact_kpi(self):
        kpi = HeadStabilityAtContactKPI()
        result = kpi.evaluate(head_displacement_norm=0.02)
        assert result.phase == "contact"
        assert result.score > 50

    def test_sir_proxy_kpi(self):
        kpi = SIRProxyKPI()
        result = kpi.evaluate(sir_angular_velocity=300.0)
        assert result.phase == "contact"
        assert 0 <= result.score <= 100

    # Phase 7: Wiper
    def test_forward_extension_kpi(self):
        kpi = ForwardExtensionKPI()
        result = kpi.evaluate(extension_ratio=0.6)
        assert result.phase == "wiper"
        assert result.score > 50

    def test_wiper_sweep_kpi(self):
        kpi = WiperSweepKPI()
        result = kpi.evaluate(wiper_sweep_angle=100.0)
        assert result.phase == "wiper"
        assert 0 <= result.score <= 100

    # Phase 8: Balance
    def test_overall_head_stability_kpi(self):
        kpi = OverallHeadStabilityKPI()
        result = kpi.evaluate(head_std_norm=0.01)
        assert result.phase == "balance"
        assert result.score > 50

    def test_spine_consistency_kpi(self):
        kpi = SpineConsistencyKPI()
        result = kpi.evaluate(spine_angle_std=3.0)
        assert result.phase == "balance"
        assert result.score > 50


    def test_all_kpis_registered_v3(self):
        """v3: 20 个正手 KPI。"""
        assert len(ALL_KPIS) == 20

    def test_all_kpis_phases_are_v3(self):
        """v3: 所有 KPI 的 phase 应该属于 8 阶段。"""
        valid_phases = {"unit_turn", "slot_prep", "leg_drive", "torso_pull",
                        "lag_drive", "contact", "wiper", "balance"}
        for kpi_cls in ALL_KPIS:
            kpi = kpi_cls()
            assert kpi.phase in valid_phases, f"{kpi.kpi_id} has invalid phase: {kpi.phase}"

    def test_kpi_na_on_empty_data(self):
        kpi = ShoulderRotationKPI()
        result = kpi.evaluate(shoulder_rotation_values=[])
        assert result.rating == "无数据"


# =====================================================================
# Evaluator integration test (v3)
# =====================================================================

class TestForehandEvaluator:
    def test_evaluate_synthetic_swing(self):
        kp_series, conf_series = _make_swing_sequence(n_frames=60, fps=30.0)
        evaluator = ForehandEvaluator(fps=30.0, is_right_handed=True)
        report = evaluator.evaluate(kp_series, conf_series, list(range(60)))
        assert isinstance(report, MultiSwingReport)
        assert 0 <= report.average_score <= 100
        assert len(report.swing_evaluations) >= 1
        # v3: 20 KPIs
        for ev in report.swing_evaluations:
            assert len(ev.kpi_results) == 20
            valid_phases = {"unit_turn", "slot_prep", "leg_drive", "torso_pull",
                            "lag_drive", "contact", "wiper", "balance"}
            for kpi in ev.kpi_results:
                assert kpi.phase in valid_phases, f"KPI {kpi.kpi_id} has invalid phase: {kpi.phase}"

    def test_evaluate_static_pose(self):
        kp, conf = _make_standing_pose()
        kp_series = [kp] * 30
        conf_series = [conf] * 30
        evaluator = ForehandEvaluator(fps=30.0)
        report = evaluator.evaluate(kp_series, conf_series, list(range(30)))
        assert isinstance(report, MultiSwingReport)
        assert len(report.swing_evaluations) >= 1

    def test_evaluate_multi_impact(self):
        kp_series, conf_series = _make_swing_sequence(n_frames=120, fps=30.0)
        evaluator = ForehandEvaluator(fps=30.0)
        impacts = [
            ImpactEvent(30, 31, 500.0, 5.0, (1.0, 0.0), True),
            ImpactEvent(80, 81, 600.0, 6.0, (1.0, 0.0), False),
        ]
        report = evaluator.evaluate_multi(kp_series, conf_series, list(range(120)), impacts)
        assert report.total_swings == 2
        assert len(report.swing_evaluations) == 2
        assert report.average_score > 0

    def test_v3_phase_scores_8_phases(self):
        """v3: 评估结果应包含 8 个阶段的评分。"""
        kp_series, conf_series = _make_swing_sequence(n_frames=60, fps=30.0)
        evaluator = ForehandEvaluator(fps=30.0, is_right_handed=True)
        report = evaluator.evaluate(kp_series, conf_series, list(range(60)))
        for ev in report.swing_evaluations:
            expected_phases = {"unit_turn", "slot_prep", "leg_drive", "torso_pull",
                               "lag_drive", "contact", "wiper", "balance"}
            assert set(ev.phase_scores.keys()) == expected_phases


# =====================================================================
# Visualizer tests
# =====================================================================

class TestVisualizer:
    def test_skeleton_drawer(self):
        drawer = SkeletonDrawer()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        kp, conf = _make_standing_pose()
        result = drawer.draw(frame, kp, conf)
        assert result.shape == frame.shape
        assert result.sum() > 0

    def test_trajectory_drawer_with_fade(self):
        drawer = TrajectoryDrawer(joint="right_wrist", max_trail=30, fade=True)
        kp, conf = _make_standing_pose()
        for i in range(40):
            kp_mod = kp.copy()
            kp_mod[10, 0] += i * 10
            drawer.update(kp_mod, conf, frame_idx=i)
        assert len(drawer.trail) <= 31
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = drawer.draw(frame)
        assert result.sum() > 0

    def test_trajectory_drawer_clear(self):
        drawer = TrajectoryDrawer(joint="right_wrist")
        kp, conf = _make_standing_pose()
        drawer.update(kp, conf, frame_idx=0)
        drawer.clear()
        assert len(drawer.trail) == 0

    def test_joint_cn_mapping(self):
        assert JOINT_CN["right_wrist"] == "右手腕"
        assert JOINT_CN["nose"] == "鼻子"

    def test_chart_generator_radar_v3(self, tmp_path):
        """v3: 8 阶段雷达图。"""
        path = str(tmp_path / "radar.png")
        result = ChartGenerator.radar_chart(
            {"unit_turn": 80, "slot_prep": 75, "leg_drive": 70,
             "torso_pull": 85, "lag_drive": 65, "contact": 90,
             "wiper": 60, "balance": 75},
            path,
        )
        assert result != ""
        assert Path(path).exists()

    def test_chart_generator_kpi_bar(self, tmp_path):
        kpis = [
            KPIResult("UT1.1", "肩部旋转", "unit_turn", 85.0, "度", 80, "良好", "良好"),
            KPIResult("UT1.2", "膝盖弯曲", "unit_turn", 130.0, "度", 70, "良好", "尚可"),
        ]
        path = str(tmp_path / "kpi_bar.png")
        result = ChartGenerator.kpi_bar_chart(kpis, path)
        assert result != ""

    def test_multi_swing_summary_chart(self, tmp_path):
        swing_scores = [(0, 75.0), (1, 82.0), (2, 68.0)]
        path = str(tmp_path / "multi.png")
        result = ChartGenerator.multi_swing_summary_chart(swing_scores, path)
        assert result != ""

    def test_phase_labels_include_v3(self):
        """v3: PHASE_LABELS 包含 8 阶段。"""
        expected = ["unit_turn", "slot_prep", "leg_drive", "torso_pull",
                    "lag_drive", "contact", "wiper", "balance"]
        for phase in expected:
            assert phase in ChartGenerator.PHASE_LABELS
            assert phase in ChartGenerator.PHASE_COLORS


# =====================================================================
# Report generator tests (v3)
# =====================================================================

class TestReportGenerator:
    def test_generate_single_swing_report_v3(self, tmp_path):
        gen = ReportGenerator(output_dir=str(tmp_path))
        kpi_results = [
            KPIResult("UT1.1", "肩部旋转", "unit_turn", 85.0, "度", 80, "良好", "良好的肩部旋转。"),
        ]
        phase_scores = {"unit_turn": PhaseScore("unit_turn", 80.0, kpi_results)}
        swing_eval = SwingEvaluation(
            swing_index=0,
            swing_event=SwingEvent(swing_index=0, impact_frame=30),
            phase_scores=phase_scores,
            overall_score=75.0,
            kpi_results=kpi_results,
            arm_style="直臂型",
        )
        report = MultiSwingReport(
            swing_evaluations=[swing_eval],
            average_score=75.0,
            best_swing_index=0,
            worst_swing_index=0,
            impact_frames=[30],
            total_swings=1,
        )
        path = gen.generate(report, video_name="test")
        assert Path(path).exists()
        content = Path(path).read_text(encoding="utf-8")
        assert "现代正手技术分析报告" in content
        assert "综合评分" in content
        assert "肩部旋转" in content
        assert "8 阶段" in content or "8阶段" in content

    def test_generate_multi_swing_report_v3(self, tmp_path):
        gen = ReportGenerator(output_dir=str(tmp_path))
        kpi1 = [KPIResult("UT1.1", "肩部旋转", "unit_turn", 85.0, "度", 80, "良好", "良好")]
        kpi2 = [KPIResult("UT1.1", "肩部旋转", "unit_turn", 70.0, "度", 60, "一般", "一般")]
        ev1 = SwingEvaluation(0, SwingEvent(0, impact_frame=30),
                              {"unit_turn": PhaseScore("unit_turn", 80.0, kpi1)},
                              80.0, kpi1, arm_style="直臂型")
        ev2 = SwingEvaluation(1, SwingEvent(1, impact_frame=80),
                              {"unit_turn": PhaseScore("unit_turn", 60.0, kpi2)},
                              60.0, kpi2, arm_style="双弯型")
        report = MultiSwingReport([ev1, ev2], 70.0, 0, 1, [30, 80], 2)
        path = gen.generate(report, video_name="multi_test")
        content = Path(path).read_text(encoding="utf-8")
        assert "第 1 次击球" in content
        assert "第 2 次击球" in content
        assert "检测到击球次数" in content

    def test_forehand_phase_order_v3(self):
        """v3: 报告生成器的阶段顺序应为 8 阶段。"""
        assert len(ReportGenerator.FOREHAND_PHASE_ORDER) == 8
        assert ReportGenerator.FOREHAND_PHASE_ORDER[0] == "unit_turn"
        assert ReportGenerator.FOREHAND_PHASE_ORDER[-1] == "balance"

    def test_training_prescription_drills_v3(self):
        """v3: 训练处方应包含 8 阶段的训练方法。"""
        gen = ReportGenerator()
        # 检查 drill IDs 覆盖了 8 个阶段
        drill_phases = set()
        for drill_id in gen.FOREHAND_DRILLS:
            if drill_id.startswith("UT") or drill_id.startswith("P1"):
                drill_phases.add("unit_turn")
            elif drill_id.startswith("S2"):
                drill_phases.add("slot_prep")
            elif drill_id.startswith("L3"):
                drill_phases.add("leg_drive")
            elif drill_id.startswith("T4"):
                drill_phases.add("torso_pull")
            elif drill_id.startswith("D5"):
                drill_phases.add("lag_drive")
            elif drill_id.startswith("C6"):
                drill_phases.add("contact")
            elif drill_id.startswith("W7"):
                drill_phases.add("wiper")
            elif drill_id.startswith("B8"):
                drill_phases.add("balance")
        assert len(drill_phases) >= 7  # at least 7 of 8 phases covered


# =====================================================================
# Backhand Config tests
# =====================================================================

class TestBackhandConfig:
    def test_backhand_config_loads(self):
        from config.backhand_config import DEFAULT_BACKHAND_CONFIG
        assert DEFAULT_BACKHAND_CONFIG is not None
        sw = DEFAULT_BACKHAND_CONFIG.scoring
        total = sw.preparation + sw.backswing + sw.kinetic_chain + sw.contact + sw.extension + sw.balance
        assert abs(total - 1.0) < 0.01

    def test_backhand_config_phases(self):
        from config.backhand_config import DEFAULT_BACKHAND_CONFIG
        assert hasattr(DEFAULT_BACKHAND_CONFIG, "preparation")
        assert hasattr(DEFAULT_BACKHAND_CONFIG, "backswing")
        assert hasattr(DEFAULT_BACKHAND_CONFIG, "kinetic_chain")
        assert hasattr(DEFAULT_BACKHAND_CONFIG, "contact")
        assert hasattr(DEFAULT_BACKHAND_CONFIG, "extension")
        assert hasattr(DEFAULT_BACKHAND_CONFIG, "balance")


# =====================================================================
# Backhand KPI tests
# =====================================================================

class TestBackhandKPI:
    def test_ohb_shoulder_rotation(self):
        from evaluation.backhand_kpi import OHB_ShoulderRotationKPI
        from config.backhand_config import DEFAULT_BACKHAND_CONFIG
        kpi = OHB_ShoulderRotationKPI(DEFAULT_BACKHAND_CONFIG)
        result = kpi.evaluate(shoulder_rotation_values=[50.0, 60.0, 70.0])
        assert result.kpi_id == "BP1.1"
        assert 0 <= result.score <= 100
        assert result.rating != "无数据"

    def test_ohb_knee_bend(self):
        from evaluation.backhand_kpi import OHB_KneeBendKPI
        from config.backhand_config import DEFAULT_BACKHAND_CONFIG
        kpi = OHB_KneeBendKPI(DEFAULT_BACKHAND_CONFIG)
        result = kpi.evaluate(knee_angle_values=[130.0, 125.0, 135.0])
        assert result.kpi_id == "BP1.2"
        assert 0 <= result.score <= 100

    def test_ohb_contact_arm_extension(self):
        from evaluation.backhand_kpi import OHB_ArmExtensionKPI
        from config.backhand_config import DEFAULT_BACKHAND_CONFIG
        kpi = OHB_ArmExtensionKPI(DEFAULT_BACKHAND_CONFIG)
        result = kpi.evaluate(elbow_angle_at_contact=165.0)
        assert result.kpi_id == "BP4.2"
        assert result.score > 70

    def test_ohb_none_returns_no_data(self):
        from evaluation.backhand_kpi import OHB_ShoulderRotationKPI
        from config.backhand_config import DEFAULT_BACKHAND_CONFIG
        kpi = OHB_ShoulderRotationKPI(DEFAULT_BACKHAND_CONFIG)
        result = kpi.evaluate(shoulder_rotation_values=[])
        assert result.rating == "无数据"
        assert result.score == 0.0

    def test_ohb_head_stability(self):
        from evaluation.backhand_kpi import OHB_HeadStabilityAtContactKPI
        from config.backhand_config import DEFAULT_BACKHAND_CONFIG
        kpi = OHB_HeadStabilityAtContactKPI(DEFAULT_BACKHAND_CONFIG)
        result = kpi.evaluate(head_displacement_norm=0.02)
        assert result.kpi_id == "BP4.5"
        assert result.score > 50


# =====================================================================
# Backhand Evaluator tests
# =====================================================================

class TestBackhandEvaluator:
    def test_evaluate_synthetic_swing(self):
        from evaluation.backhand_evaluator import BackhandEvaluator
        from config.backhand_config import DEFAULT_BACKHAND_CONFIG
        kp_series, conf_series = _make_swing_sequence(n_frames=60, fps=30.0)
        evaluator = BackhandEvaluator(fps=30.0, is_right_handed=True, cfg=DEFAULT_BACKHAND_CONFIG)
        report = evaluator.evaluate_multi(kp_series, conf_series, list(range(60)), [])
        assert isinstance(report, MultiSwingReport)
        assert 0 <= report.average_score <= 100
        assert len(report.swing_evaluations) >= 1

    def test_evaluate_multi_impact(self):
        from evaluation.backhand_evaluator import BackhandEvaluator
        from config.backhand_config import DEFAULT_BACKHAND_CONFIG
        kp_series, conf_series = _make_swing_sequence(n_frames=120, fps=30.0)
        impacts = [
            ImpactEvent(30, 31, 500.0, 5.0, (1.0, 0.0), True),
            ImpactEvent(80, 81, 600.0, 6.0, (1.0, 0.0), False),
        ]
        evaluator = BackhandEvaluator(fps=30.0, is_right_handed=True, cfg=DEFAULT_BACKHAND_CONFIG)
        report = evaluator.evaluate_multi(kp_series, conf_series, list(range(120)), impacts)
        assert report.total_swings == 2
        assert len(report.swing_evaluations) == 2

    def test_backhand_phases_are_ohb(self):
        from evaluation.backhand_evaluator import BackhandEvaluator
        from config.backhand_config import DEFAULT_BACKHAND_CONFIG
        kp_series, conf_series = _make_swing_sequence(n_frames=60, fps=30.0)
        evaluator = BackhandEvaluator(fps=30.0, is_right_handed=True, cfg=DEFAULT_BACKHAND_CONFIG)
        report = evaluator.evaluate_multi(kp_series, conf_series, list(range(60)), [])
        for ev in report.swing_evaluations:
            for phase_key in ev.phase_scores:
                assert phase_key.startswith("ohb_"), f"Phase {phase_key} should start with ohb_"


# =====================================================================
# Stroke Classifier tests
# =====================================================================

class TestStrokeClassifier:
    def test_classify_forehand(self):
        from evaluation.stroke_classifier import StrokeClassifier, StrokeType
        classifier = StrokeClassifier(is_right_handed=True)
        kp, conf = _make_standing_pose()
        kp_series = []
        conf_series = []
        for i in range(30):
            kp_mod = kp.copy()
            kp_mod[10, 0] = 300 - i * 8  # right wrist moves left (forehand for right-handed)
            kp_series.append(kp_mod)
            conf_series.append(conf.copy())
        result = classifier.classify_swing(kp_series, conf_series, list(range(30)), 15)
        assert result.stroke_type in (StrokeType.FOREHAND, StrokeType.UNKNOWN)

    def test_classify_all_swings(self):
        from evaluation.stroke_classifier import StrokeClassifier
        classifier = StrokeClassifier(is_right_handed=True)
        kp, conf = _make_standing_pose()
        kp_series = [kp.copy() for _ in range(60)]
        conf_series = [conf.copy() for _ in range(60)]
        results = classifier.classify_all_swings(kp_series, conf_series, list(range(60)), [15, 45])
        assert len(results) == 2


# =====================================================================
# Report generator — backhand
# =====================================================================

class TestBackhandReportGenerator:
    def test_generate_backhand_report(self, tmp_path):
        gen = ReportGenerator(output_dir=str(tmp_path))
        kpi_results = [
            KPIResult("BP1.1", "侧身转体", "ohb_preparation", 50.0, "度", 80, "良好", "良好的侧身转体。"),
        ]
        phase_scores = {"ohb_preparation": PhaseScore("ohb_preparation", 80.0, kpi_results)}
        swing_eval = SwingEvaluation(
            swing_index=0,
            swing_event=SwingEvent(swing_index=0, impact_frame=30),
            phase_scores=phase_scores,
            overall_score=75.0,
            kpi_results=kpi_results,
            arm_style=None,
        )
        report = MultiSwingReport(
            swing_evaluations=[swing_eval],
            average_score=75.0,
            best_swing_index=0,
            worst_swing_index=0,
            impact_frames=[30],
            total_swings=1,
        )
        path = gen.generate(report, video_name="test_bh", stroke_type="one_handed_backhand")
        assert Path(path).exists()
        content = Path(path).read_text(encoding="utf-8")
        assert "单手反拍" in content
        assert "综合评分" in content
        assert "侧身转体" in content

    def test_radar_chart_with_ohb_phases(self, tmp_path):
        path = str(tmp_path / "radar_ohb.png")
        result = ChartGenerator.radar_chart(
            {
                "ohb_preparation": 80,
                "ohb_backswing": 70,
                "ohb_kinetic_chain": 75,
                "ohb_contact": 90,
                "ohb_extension": 60,
                "ohb_balance": 85,
            },
            path,
        )
        assert result != ""
        assert Path(path).exists()
