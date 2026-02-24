"""Unit tests for Tennis Analyzer v2 modules.

Tests the core logic without requiring YOLO model or video files.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from config.keypoints import COCO_KEYPOINTS, KEYPOINT_NAMES, SKELETON_CONNECTIONS
from config.framework_config import DEFAULT_CONFIG
from analysis.trajectory import JointTrajectory, TrajectoryStore
from analysis.kinematic_calculator import (
    joint_angle, elbow_angle, knee_angle, min_knee_angle,
    shoulder_hip_angle, spine_angle_from_vertical,
    torso_height_px, shoulder_width_px,
    wrist_forward_normalised, nose_position,
    shoulder_center, hip_center,
)
from evaluation.event_detector import ImpactDetector, SwingPhaseEstimator, ImpactEvent
from evaluation.kpi import (
    ShoulderRotationKPI, KneeBendKPI, SpineAngleKPI,
    KineticChainSequenceKPI, HipShoulderSeparationKPI,
    HandPathLinearityKPI, ContactPointKPI, ElbowAngleAtContactKPI,
    BodyFreezeKPI, HeadStabilityAtContactKPI,
    ForwardExtensionKPI, FollowThroughPathKPI,
    OverallHeadStabilityKPI, SpineConsistencyKPI,
    ALL_KPIS, KPIResult, _linear_score, _rating_from_score,
)
from evaluation.forehand_evaluator import ForehandEvaluator, EvaluationReport
from report.visualizer import SkeletonDrawer, TrajectoryDrawer, ChartGenerator
from report.report_generator import ReportGenerator


# ── Fixtures ─────────────────────────────────────────────────────────

def _make_standing_pose():
    """Create a synthetic standing person keypoints array (17, 2)."""
    kp = np.zeros((17, 2), dtype=np.float32)
    conf = np.ones(17, dtype=np.float32)

    # Approximate standing pose (image coords: y increases downward)
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
    """Create a synthetic forehand swing sequence."""
    kp_series = []
    conf_series = []
    base_kp, base_conf = _make_standing_pose()

    for i in range(n_frames):
        kp = base_kp.copy()
        t = i / n_frames

        # Simulate wrist moving forward during swing
        # Phase 0-0.3: preparation (wrist moves back)
        # Phase 0.3-0.6: forward swing (wrist accelerates forward)
        # Phase 0.6-1.0: follow-through (wrist decelerates)
        if t < 0.3:
            kp[10, 0] = 260 - t * 100  # wrist moves back
        elif t < 0.6:
            swing_t = (t - 0.3) / 0.3
            kp[10, 0] = 230 + swing_t * 150  # wrist moves forward fast
            kp[10, 1] = 200 - swing_t * 30   # slight upward
        else:
            ft_t = (t - 0.6) / 0.4
            kp[10, 0] = 380 - ft_t * 50   # wrist decelerates
            kp[10, 1] = 170 - ft_t * 60   # wrist goes up (follow-through)

        # Simulate shoulder rotation
        if t < 0.3:
            # Shoulders rotate (left shoulder moves forward)
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
        kp = np.array([[0, 0]] * 17, dtype=np.float32)
        conf = np.ones(17, dtype=np.float32)
        kp[5] = [0, 0]    # shoulder
        kp[7] = [100, 0]  # elbow
        kp[9] = [200, 0]  # wrist
        angle = joint_angle(kp, conf, 5, 7, 9)
        assert angle is not None
        assert abs(angle - 180.0) < 1.0

    def test_joint_angle_right_angle(self):
        kp = np.array([[0, 0]] * 17, dtype=np.float32)
        conf = np.ones(17, dtype=np.float32)
        kp[5] = [0, 0]
        kp[7] = [100, 0]
        kp[9] = [100, 100]
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
        angle = min_knee_angle(kp, conf)
        assert angle is not None

    def test_shoulder_hip_angle(self):
        kp, conf = _make_standing_pose()
        angle = shoulder_hip_angle(kp, conf)
        assert angle is not None
        # Standing pose: shoulders and hips roughly parallel
        assert angle < 30

    def test_spine_angle(self):
        kp, conf = _make_standing_pose()
        angle = spine_angle_from_vertical(kp, conf)
        assert angle is not None
        # Standing pose should be roughly upright
        assert angle < 30

    def test_torso_height(self):
        kp, conf = _make_standing_pose()
        th = torso_height_px(kp, conf)
        assert th is not None
        assert th > 0

    def test_shoulder_width(self):
        kp, conf = _make_standing_pose()
        sw = shoulder_width_px(kp, conf)
        assert sw is not None
        assert sw > 0

    def test_wrist_forward_normalised(self):
        kp, conf = _make_standing_pose()
        val = wrist_forward_normalised(kp, conf, is_right_handed=True, forward_sign=1.0)
        assert val is not None

    def test_low_confidence_returns_none(self):
        kp, conf = _make_standing_pose()
        conf[:] = 0.1  # all low confidence
        assert elbow_angle(kp, conf) is None
        assert shoulder_hip_angle(kp, conf) is None

    def test_nose_position(self):
        kp, conf = _make_standing_pose()
        pos = nose_position(kp, conf)
        assert pos is not None
        assert len(pos) == 2


# =====================================================================
# Trajectory tests
# =====================================================================

class TestTrajectory:
    def test_joint_trajectory_basic(self):
        traj = JointTrajectory(10, fps=30.0)
        for i in range(10):
            traj.add(float(i * 10), float(i * 5), 0.9, i)
        assert traj.length == 10
        pos = traj.get_positions()
        assert pos.shape == (10, 2)

    def test_velocity_computation(self):
        traj = JointTrajectory(10, fps=30.0)
        for i in range(10):
            traj.add(float(i * 30), 0.0, 0.9, i)  # 30 px/frame = 900 px/s at 30fps
        vel = traj.get_velocities()
        assert vel.shape == (9, 2)
        # Due to EMA smoothing, velocities will be attenuated
        # Just verify they are positive and in the right direction
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
        # Slow, then fast, then slow
        positions = [0, 5, 10, 15, 50, 90, 95, 98, 99, 100]
        for i, x in enumerate(positions):
            traj.add(float(x), 0.0, 0.9, i)
        peak = traj.peak_speed_frame()
        assert peak is not None
        assert peak[1] > 0

    def test_trajectory_store(self):
        store = TrajectoryStore(fps=30.0)
        kp = np.random.rand(17, 2).astype(np.float32) * 100
        conf = np.ones(17, dtype=np.float32)
        store.update(kp, conf, 0)
        store.update(kp + 10, conf, 1)
        traj = store.get("right_wrist")
        assert traj.length == 2


# =====================================================================
# Impact detector tests
# =====================================================================

class TestImpactDetector:
    def test_no_impact_on_static(self):
        det = ImpactDetector(fps=30.0)
        kp, conf = _make_standing_pose()
        for i in range(30):
            event, speed = det.update(i, kp, conf)
            assert event is None

    def test_impact_on_speed_peak(self):
        det = ImpactDetector(fps=30.0)
        kp, conf = _make_standing_pose()
        wrist_idx = KEYPOINT_NAMES["right_wrist"]

        # Simulate: slow → fast → slow (speed peak)
        positions = [0, 5, 10, 15, 20, 30, 50, 100, 180, 250, 280, 290, 295, 298, 299, 300]
        detected = []
        for i, x in enumerate(positions):
            kp_frame = kp.copy()
            kp_frame[wrist_idx, 0] = float(x)
            event, speed = det.update(i, kp_frame, conf)
            if event is not None:
                detected.append(event)

        # Should detect at least one impact
        assert len(detected) >= 1


# =====================================================================
# KPI tests
# =====================================================================

class TestKPIs:
    def test_linear_score_larger_is_better(self):
        # shoulder rotation: larger is better
        assert _linear_score(100, 45, 70, 90) == 100.0
        assert _linear_score(90, 45, 70, 90) == 100.0
        assert _linear_score(70, 45, 70, 90) == 70.0
        assert _linear_score(45, 45, 70, 90) == 20.0

    def test_linear_score_smaller_is_better(self):
        # spine lean: smaller is better
        assert _linear_score(5, 25, 15, 5) == 100.0
        assert _linear_score(15, 25, 15, 5) == 70.0

    def test_rating_from_score(self):
        assert _rating_from_score(90) == "excellent"
        assert _rating_from_score(75) == "good"
        assert _rating_from_score(50) == "fair"
        assert _rating_from_score(20) == "poor"

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
            elbow_peak_frame=16, wrist_peak_frame=19, fps=30.0
        )
        assert result.score >= 85

    def test_kinetic_chain_wrong_order(self):
        kpi = KineticChainSequenceKPI()
        result = kpi.evaluate(
            hip_peak_frame=19, shoulder_peak_frame=16,
            elbow_peak_frame=13, wrist_peak_frame=10, fps=30.0
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
        assert "straight" in result.details.get("style", "").lower()

    def test_elbow_angle_double_bend(self):
        kpi = ElbowAngleAtContactKPI()
        result = kpi.evaluate(elbow_angle_at_contact=135.0)
        assert result.score >= 85
        assert "double" in result.details.get("style", "").lower()

    def test_all_kpis_registered(self):
        assert len(ALL_KPIS) == 14

    def test_kpi_na_on_empty_data(self):
        kpi = ShoulderRotationKPI()
        result = kpi.evaluate(shoulder_rotation_values=[])
        assert result.rating == "n/a"


# =====================================================================
# Evaluator integration test
# =====================================================================

class TestForehandEvaluator:
    def test_evaluate_synthetic_swing(self):
        kp_series, conf_series = _make_swing_sequence(n_frames=60, fps=30.0)
        evaluator = ForehandEvaluator(fps=30.0, is_right_handed=True)
        report = evaluator.evaluate(kp_series, conf_series)

        assert isinstance(report, EvaluationReport)
        assert 0 <= report.overall_score <= 100
        assert len(report.kpi_results) == 14

        # Check that all phases are present
        for kpi in report.kpi_results:
            assert kpi.phase in ["preparation", "kinetic_chain", "contact", "extension", "balance"]

    def test_evaluate_static_pose(self):
        """Static pose should still produce a valid report (no impact detected)."""
        kp, conf = _make_standing_pose()
        kp_series = [kp] * 30
        conf_series = [conf] * 30
        evaluator = ForehandEvaluator(fps=30.0)
        report = evaluator.evaluate(kp_series, conf_series)
        assert isinstance(report, EvaluationReport)
        assert report.swing_event.impact_frame is None


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
        # Should have drawn something (not all black)
        assert result.sum() > 0

    def test_trajectory_drawer(self):
        drawer = TrajectoryDrawer(joint="right_wrist")
        kp, conf = _make_standing_pose()
        for i in range(10):
            kp_mod = kp.copy()
            kp_mod[10, 0] += i * 10
            drawer.update(kp_mod, conf)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = drawer.draw(frame)
        assert result.sum() > 0

    def test_chart_generator_radar(self):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = ChartGenerator.radar_chart(
                {"preparation": 80, "kinetic_chain": 70, "contact": 90, "extension": 60, "balance": 75},
                f.name,
            )
            assert path != ""

    def test_chart_generator_kpi_bar(self):
        import tempfile
        results = [
            KPIResult("P1.1", "Test", "preparation", 80.0, "deg", 80, "good", "Good."),
            KPIResult("C4.1", "Test2", "contact", 60.0, "deg", 60, "fair", "Fair."),
        ]
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = ChartGenerator.kpi_bar_chart(results, f.name)
            assert path != ""


# =====================================================================
# Report generator test
# =====================================================================

class TestReportGenerator:
    def test_generate_report(self):
        kp_series, conf_series = _make_swing_sequence()
        evaluator = ForehandEvaluator(fps=30.0)
        report = evaluator.evaluate(kp_series, conf_series)

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = ReportGenerator(output_dir=tmpdir)
            path = gen.generate(report, video_name="test")
            assert Path(path).exists()
            content = Path(path).read_text()
            assert "Modern Forehand" in content
            assert "Overall Score" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
