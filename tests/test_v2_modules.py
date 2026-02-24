"""Tennis Analyzer v2 — 综合测试套件。

覆盖所有核心模块：配置、运动学计算、轨迹管理、击球检测、KPI 评分、
评估引擎、可视化和报告生成。
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np

from config.framework_config import DEFAULT_CONFIG, FrameworkConfig
from config.keypoints import COCO_KEYPOINTS, KEYPOINT_NAMES, SKELETON_CONNECTIONS, NUM_KEYPOINTS
from analysis.trajectory import JointTrajectory, TrajectoryStore
from analysis.kinematic_calculator import (
    joint_angle, elbow_angle, knee_angle, min_knee_angle,
    shoulder_hip_angle, spine_angle_from_vertical,
    torso_height_px, shoulder_width_px,
    wrist_forward_normalised, nose_position,
    shoulder_center, hip_center,
)
from evaluation.event_detector import (
    WristSpeedDetector, ImpactEvent, SwingEvent, SwingPhaseEstimator,
)
from evaluation.kpi import (
    ShoulderRotationKPI, KneeBendKPI, SpineAngleKPI,
    KineticChainSequenceKPI, HipShoulderSeparationKPI,
    HandPathLinearityKPI, ContactPointKPI, ElbowAngleAtContactKPI,
    BodyFreezeKPI, HeadStabilityAtContactKPI,
    ForwardExtensionKPI, FollowThroughPathKPI,
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

    def test_default_config(self):
        cfg = DEFAULT_CONFIG
        assert cfg.scoring.preparation == 0.15
        assert cfg.scoring.contact == 0.25
        weights = cfg.scoring.as_dict()
        assert abs(sum(weights.values()) - 1.0) < 1e-6


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
# KPI tests
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

    def test_shoulder_rotation_kpi(self):
        kpi = ShoulderRotationKPI()
        result = kpi.evaluate(shoulder_rotation_values=[85.0, 90.0, 75.0])
        assert isinstance(result, KPIResult)
        assert result.raw_value == 90.0
        assert result.score > 70

    def test_knee_bend_kpi(self):
        kpi = KneeBendKPI()
        result = kpi.evaluate(knee_angle_values=[130.0, 125.0, 140.0])
        assert result.raw_value == 125.0
        assert result.score > 50

    def test_spine_angle_kpi(self):
        kpi = SpineAngleKPI()
        result = kpi.evaluate(spine_angle_values=[10.0, 12.0, 8.0])
        assert result.score > 70

    def test_kinetic_chain_correct_order(self):
        kpi = KineticChainSequenceKPI()
        result = kpi.evaluate(
            hip_peak_frame=10, shoulder_peak_frame=13,
            elbow_peak_frame=16, wrist_peak_frame=19, fps=30.0,
        )
        assert result.score >= 85

    def test_kinetic_chain_wrong_order(self):
        kpi = KineticChainSequenceKPI()
        result = kpi.evaluate(
            hip_peak_frame=19, shoulder_peak_frame=16,
            elbow_peak_frame=13, wrist_peak_frame=10, fps=30.0,
        )
        assert result.score < 50

    def test_contact_point_kpi(self):
        kpi = ContactPointKPI()
        result = kpi.evaluate(contact_forward_norm=0.5)
        assert result.score > 70

    def test_elbow_angle_straight_arm(self):
        kpi = ElbowAngleAtContactKPI()
        result = kpi.evaluate(elbow_angle_at_contact=170.0)
        assert result.score >= 85
        assert "直臂" in result.feedback

    def test_elbow_angle_double_bend(self):
        kpi = ElbowAngleAtContactKPI()
        result = kpi.evaluate(elbow_angle_at_contact=135.0)
        assert result.score >= 85
        assert "双弯" in result.feedback

    def test_all_kpis_registered(self):
        assert len(ALL_KPIS) == 14

    def test_kpi_na_on_empty_data(self):
        kpi = ShoulderRotationKPI()
        result = kpi.evaluate(shoulder_rotation_values=[])
        assert result.rating == "无数据"


# =====================================================================
# Evaluator integration test
# =====================================================================

class TestForehandEvaluator:
    def test_evaluate_synthetic_swing(self):
        kp_series, conf_series = _make_swing_sequence(n_frames=60, fps=30.0)
        evaluator = ForehandEvaluator(fps=30.0, is_right_handed=True)
        report = evaluator.evaluate(kp_series, conf_series, list(range(60)))
        assert isinstance(report, MultiSwingReport)
        assert 0 <= report.average_score <= 100
        assert len(report.swing_evaluations) >= 1
        for ev in report.swing_evaluations:
            assert len(ev.kpi_results) == 14
            for kpi in ev.kpi_results:
                assert kpi.phase in ["preparation", "loading", "kinetic_chain", "contact", "extension", "balance"]

    def test_evaluate_static_pose(self):
        kp, conf = _make_standing_pose()
        kp_series = [kp] * 30
        conf_series = [conf] * 30
        evaluator = ForehandEvaluator(fps=30.0)
        report = evaluator.evaluate(kp_series, conf_series, list(range(30)))
        assert isinstance(report, MultiSwingReport)
        # 即使没有检测到击球，也应该有一个默认评估
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
        assert len(drawer.trail) <= 31  # deque maxlen + 1 tolerance
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

    def test_chart_generator_radar(self, tmp_path):
        path = str(tmp_path / "radar.png")
        result = ChartGenerator.radar_chart(
            {"preparation": 80, "kinetic_chain": 70, "contact": 90, "extension": 60, "balance": 75},
            path,
        )
        assert result != ""
        assert Path(path).exists()

    def test_chart_generator_kpi_bar(self, tmp_path):
        kpis = [
            KPIResult("P1.1", "肩部旋转", "preparation", 85.0, "度", 80, "良好", "良好"),
            KPIResult("P1.4", "膝盖弯曲", "preparation", 130.0, "度", 70, "良好", "尚可"),
        ]
        path = str(tmp_path / "kpi_bar.png")
        result = ChartGenerator.kpi_bar_chart(kpis, path)
        assert result != ""

    def test_multi_swing_summary_chart(self, tmp_path):
        swing_scores = [(0, 75.0), (1, 82.0), (2, 68.0)]
        path = str(tmp_path / "multi.png")
        result = ChartGenerator.multi_swing_summary_chart(swing_scores, path)
        assert result != ""


# =====================================================================
# Report generator tests
# =====================================================================

class TestReportGenerator:
    def test_generate_single_swing_report(self, tmp_path):
        gen = ReportGenerator(output_dir=str(tmp_path))
        kpi_results = [
            KPIResult("P1.1", "肩部旋转", "preparation", 85.0, "度", 80, "良好", "良好的肩部旋转。"),
        ]
        phase_scores = {"preparation": PhaseScore("preparation", 80.0, kpi_results)}
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

    def test_generate_multi_swing_report(self, tmp_path):
        gen = ReportGenerator(output_dir=str(tmp_path))
        kpi1 = [KPIResult("P1.1", "肩部旋转", "preparation", 85.0, "度", 80, "良好", "良好")]
        kpi2 = [KPIResult("P1.1", "肩部旋转", "preparation", 70.0, "度", 60, "一般", "一般")]
        ev1 = SwingEvaluation(0, SwingEvent(0, impact_frame=30),
                              {"preparation": PhaseScore("preparation", 80.0, kpi1)},
                              80.0, kpi1, arm_style="直臂型")
        ev2 = SwingEvaluation(1, SwingEvent(1, impact_frame=80),
                              {"preparation": PhaseScore("preparation", 60.0, kpi2)},
                              60.0, kpi2, arm_style="双弯型")
        report = MultiSwingReport([ev1, ev2], 70.0, 0, 1, [30, 80], 2)
        path = gen.generate(report, video_name="multi_test")
        content = Path(path).read_text(encoding="utf-8")
        assert "第 1 次击球" in content
        assert "第 2 次击球" in content
        assert "检测到击球次数" in content
