"""Modern Forehand Evaluator — 编排层。

``ForehandEvaluator`` 负责：
    1. 接收完整的逐帧关键点时间序列。
    2. 使用 ``HybridImpactDetector`` 检测 **所有** 击球事件。
    3. 对每次击球独立分段（准备 → 击球 → 随挥）。
    4. 对每次击球独立计算所有生物力学指标。
    5. 对每次击球独立评分。
    6. 汇总为 ``MultiSwingReport``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

import numpy as np

from config.framework_config import FrameworkConfig, DEFAULT_CONFIG
from config.keypoints import KEYPOINT_NAMES
from analysis.trajectory import TrajectoryStore
from analysis.kinematic_calculator import (
    shoulder_hip_angle,
    min_knee_angle,
    spine_angle_from_vertical,
    elbow_angle,
    wrist_forward_normalised,
    nose_position,
    torso_height_px,
    shoulder_rotation_signed,
    hip_center,
    shoulder_center,
)
from evaluation.event_detector import (
    HybridImpactDetector,
    SwingPhaseEstimator,
    SwingEvent,
    ImpactEvent,
)
from evaluation.kpi import (
    KPIResult,
    ShoulderRotationKPI,
    KneeBendKPI,
    SpineAngleKPI,
    KineticChainSequenceKPI,
    HipShoulderSeparationKPI,
    HandPathLinearityKPI,
    ContactPointKPI,
    ElbowAngleAtContactKPI,
    BodyFreezeKPI,
    HeadStabilityAtContactKPI,
    ForwardExtensionKPI,
    FollowThroughPathKPI,
    OverallHeadStabilityKPI,
    SpineConsistencyKPI,
)


# ── 数据容器 ────────────────────────────────────────────────────────

@dataclass
class PhaseScore:
    phase: str
    score: float
    kpis: List[KPIResult]


@dataclass
class SwingEvaluation:
    """单次击球的完整评估结果。"""
    swing_index: int
    swing_event: SwingEvent
    phase_scores: Dict[str, PhaseScore]
    overall_score: float
    kpi_results: List[KPIResult]
    forward_sign: float = 1.0
    arm_style: str = "未知"
    raw_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiSwingReport:
    """多次击球的综合报告。"""
    swing_evaluations: List[SwingEvaluation]
    average_score: float
    best_swing_index: int
    worst_swing_index: int
    impact_frames: List[int]
    total_swings: int


class ForehandEvaluator:
    """评估正手挥拍 — 支持视频中多次击球独立评分。"""

    def __init__(
        self,
        fps: float = 30.0,
        is_right_handed: bool = True,
        cfg: FrameworkConfig = DEFAULT_CONFIG,
    ):
        self.fps = fps
        self.is_right_handed = is_right_handed
        self.cfg = cfg

        self.wrist_key = "right_wrist" if is_right_handed else "left_wrist"
        self.elbow_key = "right_elbow" if is_right_handed else "left_elbow"
        self.shoulder_key = "right_shoulder" if is_right_handed else "left_shoulder"

    # ── 公开 API ────────────────────────────────────────────────────

    def evaluate_multi(
        self,
        keypoints_series: List[np.ndarray],
        confidence_series: List[np.ndarray],
        frame_indices: List[int],
        impact_events: List[ImpactEvent],
    ) -> MultiSwingReport:
        """对多次击球进行独立评估。

        Parameters
        ----------
        keypoints_series : list of (17, 2) arrays
        confidence_series : list of (17,) arrays
        frame_indices : list of int
        impact_events : list of ImpactEvent (已排序)

        Returns
        -------
        MultiSwingReport
        """
        n_frames = len(keypoints_series)

        # 构建 TrajectoryStore
        store = TrajectoryStore(fps=self.fps)
        for kp, conf, fidx in zip(keypoints_series, confidence_series, frame_indices):
            store.update(kp, conf, fidx)

        # 获取手腕速度序列
        wrist_traj = store.get(self.wrist_key)
        wrist_speeds = wrist_traj.get_speeds(smoothed=True)
        speed_frame_indices = wrist_traj.frame_indices[1:] if len(wrist_traj.frame_indices) > 1 else []

        # 如果没有检测到击球，尝试作为单次评估
        if not impact_events:
            eval_result = self._evaluate_no_impact(
                keypoints_series, confidence_series, frame_indices, store
            )
            return MultiSwingReport(
                swing_evaluations=[eval_result],
                average_score=eval_result.overall_score,
                best_swing_index=0,
                worst_swing_index=0,
                impact_frames=[],
                total_swings=0,
            )

        # 对每次击球独立评估
        evaluations: List[SwingEvaluation] = []
        phase_estimator = SwingPhaseEstimator(fps=self.fps)

        for i, impact_event in enumerate(impact_events):
            # 确定前后击球帧，用于限制阶段边界
            prev_impact = impact_events[i - 1].impact_frame_idx if i > 0 else None
            next_impact = impact_events[i + 1].impact_frame_idx if i < len(impact_events) - 1 else None

            # 估计挥拍阶段
            swing_event = phase_estimator.estimate_phases(
                impact_frame=impact_event.impact_frame_idx,
                wrist_speeds=wrist_speeds,
                frame_indices=speed_frame_indices,
                prev_impact_frame=prev_impact,
                next_impact_frame=next_impact,
            )
            swing_event.swing_index = i
            swing_event.impact_event = impact_event

            # 推断前进方向
            forward_sign = 1.0
            if abs(impact_event.peak_velocity_unit[0]) > 0.1:
                forward_sign = 1.0 if impact_event.peak_velocity_unit[0] > 0 else -1.0

            # 计算原始指标
            raw = self._compute_raw_metrics(
                keypoints_series, confidence_series, frame_indices,
                store, swing_event, forward_sign,
            )

            # 评估所有 KPI
            kpi_results = self._evaluate_kpis(raw, store, frame_indices)

            # 汇总阶段评分
            phase_scores = self._aggregate_phases(kpi_results)
            overall_score = self._compute_overall_score(phase_scores)

            # 判断手臂风格
            elbow_kpi = next((k for k in kpi_results if k.kpi_id == "C4.2"), None)
            arm_style = elbow_kpi.details.get("style", "未知") if elbow_kpi and elbow_kpi.details else "未知"

            evaluations.append(SwingEvaluation(
                swing_index=i,
                swing_event=swing_event,
                phase_scores=phase_scores,
                overall_score=overall_score,
                kpi_results=kpi_results,
                forward_sign=forward_sign,
                arm_style=arm_style,
                raw_metrics=raw,
            ))

        # 汇总
        scores = [e.overall_score for e in evaluations]
        avg_score = float(np.mean(scores)) if scores else 0.0
        best_idx = int(np.argmax(scores)) if scores else 0
        worst_idx = int(np.argmin(scores)) if scores else 0
        impact_frames = [e.impact_frame_idx for e in impact_events]

        return MultiSwingReport(
            swing_evaluations=evaluations,
            average_score=avg_score,
            best_swing_index=best_idx,
            worst_swing_index=worst_idx,
            impact_frames=impact_frames,
            total_swings=len(evaluations),
        )

    # ── 单次击球评估（向后兼容）────────────────────────────────────

    def evaluate(
        self,
        keypoints_series: List[np.ndarray],
        confidence_series: List[np.ndarray],
        frame_indices: Optional[List[int]] = None,
        impact_events: Optional[List[ImpactEvent]] = None,
    ) -> MultiSwingReport:
        """向后兼容接口，自动检测或使用提供的击球事件。"""
        if frame_indices is None:
            frame_indices = list(range(len(keypoints_series)))

        if impact_events is None:
            impact_events = []

        return self.evaluate_multi(
            keypoints_series, confidence_series, frame_indices, impact_events,
        )

    # ── 内部：原始指标计算 ──────────────────────────────────────────

    def _compute_raw_metrics(
        self,
        kp_series, conf_series, frame_indices,
        store: TrajectoryStore,
        swing: SwingEvent,
        forward_sign: float,
    ) -> Dict[str, Any]:
        """计算单次击球所需的所有生物力学原始指标。"""
        raw: Dict[str, Any] = {}
        impact_frame = swing.impact_frame
        prep_start = swing.prep_start_frame or frame_indices[0]
        ft_end = swing.followthrough_end_frame or frame_indices[-1]

        # 帧索引 → 数组位置映射
        f2p = {f: i for i, f in enumerate(frame_indices)}
        impact_pos = f2p.get(impact_frame, len(frame_indices) // 2)
        prep_pos = f2p.get(prep_start, 0)
        ft_pos = f2p.get(ft_end, len(frame_indices) - 1)

        # ── 准备阶段指标 ────────────────────────────────────────────
        shoulder_rots = []
        knee_angles = []
        spine_angles = []
        for i in range(prep_pos, min(impact_pos + 1, len(kp_series))):
            kp, conf = kp_series[i], conf_series[i]
            sr = shoulder_hip_angle(kp, conf)
            if sr is not None:
                shoulder_rots.append(sr)
            ka = min_knee_angle(kp, conf)
            if ka is not None:
                knee_angles.append(ka)
            sa = spine_angle_from_vertical(kp, conf)
            if sa is not None:
                spine_angles.append(sa)

        raw["shoulder_rotation_values"] = shoulder_rots
        raw["knee_angle_values"] = knee_angles
        raw["spine_angle_values"] = spine_angles

        # ── 动力链指标 ──────────────────────────────────────────────
        # 限定在本次挥拍范围内查找峰值速度帧
        hip_traj = store.get("right_hip" if self.is_right_handed else "left_hip")
        shoulder_traj = store.get(self.shoulder_key)
        elbow_traj = store.get(self.elbow_key)
        wrist_traj = store.get(self.wrist_key)

        raw["hip_peak_frame"] = self._peak_in_range(hip_traj, prep_pos, ft_pos, frame_indices)
        raw["shoulder_peak_frame"] = self._peak_in_range(shoulder_traj, prep_pos, ft_pos, frame_indices)
        raw["elbow_peak_frame"] = self._peak_in_range(elbow_traj, prep_pos, ft_pos, frame_indices)
        raw["wrist_peak_frame"] = self._peak_in_range(wrist_traj, prep_pos, ft_pos, frame_indices)

        # 髋肩分离角
        hip_shoulder_seps = []
        search_start = max(0, impact_pos - int(0.3 * self.fps))
        for i in range(search_start, min(impact_pos + 1, len(kp_series))):
            kp, conf = kp_series[i], conf_series[i]
            angle = shoulder_hip_angle(kp, conf)
            if angle is not None:
                hip_shoulder_seps.append(angle)
        raw["hip_shoulder_sep_values"] = hip_shoulder_seps

        # 手部路径线性度（击球前后 ±5 帧）
        contact_zone_half = max(3, int(0.05 * self.fps))
        wrist_positions_cz = []
        wrist_idx = KEYPOINT_NAMES[self.wrist_key]
        for i in range(max(0, impact_pos - contact_zone_half),
                       min(len(kp_series), impact_pos + contact_zone_half + 1)):
            kp, conf = kp_series[i], conf_series[i]
            if conf[wrist_idx] >= 0.3:
                wrist_positions_cz.append(kp[wrist_idx].copy())
        raw["wrist_positions_contact_zone"] = np.array(wrist_positions_cz) if wrist_positions_cz else None

        # ── 击球点指标 ──────────────────────────────────────────────
        if impact_pos < len(kp_series):
            kp_impact = kp_series[impact_pos]
            conf_impact = conf_series[impact_pos]

            raw["contact_forward_norm"] = wrist_forward_normalised(
                kp_impact, conf_impact, self.is_right_handed, forward_sign
            )
            raw["elbow_angle_at_contact"] = elbow_angle(
                kp_impact, conf_impact, right=self.is_right_handed
            )

            # 身体刹车：击球时躯干角速度
            if 0 < impact_pos < len(kp_series) - 1:
                sr_before = shoulder_hip_angle(kp_series[impact_pos - 1], conf_series[impact_pos - 1])
                sr_after = shoulder_hip_angle(
                    kp_series[min(impact_pos + 1, len(kp_series) - 1)],
                    conf_series[min(impact_pos + 1, len(conf_series) - 1)]
                )
                if sr_before is not None and sr_after is not None:
                    dt = 2.0 / self.fps
                    raw["torso_angular_velocity_at_contact"] = abs(sr_after - sr_before) / dt
                else:
                    raw["torso_angular_velocity_at_contact"] = None
            else:
                raw["torso_angular_velocity_at_contact"] = None

            # 击球点附近头部稳定性（±5 帧）
            head_window = 5
            nose_positions = []
            for i in range(max(0, impact_pos - head_window),
                           min(len(kp_series), impact_pos + head_window + 1)):
                np_pos = nose_position(kp_series[i], conf_series[i])
                if np_pos is not None:
                    nose_positions.append(np_pos)
            if len(nose_positions) >= 3:
                nose_arr = np.array(nose_positions)
                displacement = float(np.std(nose_arr[:, 1]))
                th = torso_height_px(kp_impact, conf_impact)
                if th and th > 1e-6:
                    raw["head_displacement_norm"] = displacement / th
                else:
                    raw["head_displacement_norm"] = None
            else:
                raw["head_displacement_norm"] = None
        else:
            raw["contact_forward_norm"] = None
            raw["elbow_angle_at_contact"] = None
            raw["torso_angular_velocity_at_contact"] = None
            raw["head_displacement_norm"] = None

        # ── 延伸指标 ────────────────────────────────────────────────
        post_contact_frames = int(self.cfg.extension.post_contact_window_s * self.fps)
        if impact_pos < len(kp_series):
            wrist_at_contact = (
                kp_series[impact_pos][wrist_idx].copy()
                if conf_series[impact_pos][wrist_idx] >= 0.3 else None
            )

            if wrist_at_contact is not None:
                max_forward = 0.0
                max_upward = 0.0
                for i in range(impact_pos + 1,
                               min(len(kp_series), impact_pos + post_contact_frames + 1)):
                    if conf_series[i][wrist_idx] >= 0.3:
                        delta = kp_series[i][wrist_idx] - wrist_at_contact
                        forward_dist = delta[0] * forward_sign
                        upward_dist = -delta[1]
                        max_forward = max(max_forward, forward_dist)
                        max_upward = max(max_upward, upward_dist)

                th = torso_height_px(kp_series[impact_pos], conf_series[impact_pos])
                if th and th > 1e-6:
                    raw["forward_extension_norm"] = max_forward / th
                    if max_forward > 1e-6:
                        raw["upward_forward_ratio"] = max_upward / max_forward
                    else:
                        raw["upward_forward_ratio"] = None
                else:
                    raw["forward_extension_norm"] = None
                    raw["upward_forward_ratio"] = None
            else:
                raw["forward_extension_norm"] = None
                raw["upward_forward_ratio"] = None
        else:
            raw["forward_extension_norm"] = None
            raw["upward_forward_ratio"] = None

        # ── 平衡指标（整个挥拍范围）────────────────────────────────
        nose_y_values = []
        spine_all = []
        for i in range(prep_pos, min(ft_pos + 1, len(kp_series))):
            np_pos = nose_position(kp_series[i], conf_series[i])
            if np_pos is not None:
                nose_y_values.append(np_pos[1])
            sa = spine_angle_from_vertical(kp_series[i], conf_series[i])
            if sa is not None:
                spine_all.append(sa)

        if len(nose_y_values) >= 3:
            th_ref = (
                torso_height_px(kp_series[impact_pos], conf_series[impact_pos])
                if impact_pos < len(kp_series) else None
            )
            if th_ref and th_ref > 1e-6:
                raw["head_y_std_norm"] = float(np.std(nose_y_values)) / th_ref
            else:
                raw["head_y_std_norm"] = None
        else:
            raw["head_y_std_norm"] = None

        raw["spine_angle_std"] = float(np.std(spine_all)) if len(spine_all) >= 3 else None
        raw["fps"] = self.fps
        return raw

    def _peak_in_range(
        self,
        traj,
        start_pos: int,
        end_pos: int,
        frame_indices: List[int],
    ) -> Optional[int]:
        """在指定帧范围内查找轨迹的峰值速度帧。"""
        speeds = traj.get_speeds(smoothed=True)
        if len(speeds) == 0:
            return None

        start_frame = frame_indices[start_pos] if start_pos < len(frame_indices) else 0
        end_frame = frame_indices[end_pos] if end_pos < len(frame_indices) else frame_indices[-1]

        best_speed = -1.0
        best_frame = None
        traj_frames = traj.frame_indices
        for i, speed in enumerate(speeds):
            # speed[i] 对应 frame_indices[i+1]
            if i + 1 < len(traj_frames):
                f = traj_frames[i + 1]
                if start_frame <= f <= end_frame and speed > best_speed:
                    best_speed = speed
                    best_frame = f
        return best_frame

    # ── 内部：KPI 评估 ─────────────────────────────────────────────

    def _evaluate_kpis(self, raw: Dict[str, Any], store: TrajectoryStore, frame_indices: List[int]) -> List[KPIResult]:
        """实例化并评估所有 KPI。"""
        results = []

        # 阶段 1：准备
        results.append(ShoulderRotationKPI(self.cfg).evaluate(
            shoulder_rotation_values=raw.get("shoulder_rotation_values", [])))
        results.append(KneeBendKPI(self.cfg).evaluate(
            knee_angle_values=raw.get("knee_angle_values", [])))
        results.append(SpineAngleKPI(self.cfg).evaluate(
            spine_angle_values=raw.get("spine_angle_values", [])))

        # 阶段 3：动力链
        results.append(KineticChainSequenceKPI(self.cfg).evaluate(
            hip_peak_frame=raw.get("hip_peak_frame"),
            shoulder_peak_frame=raw.get("shoulder_peak_frame"),
            elbow_peak_frame=raw.get("elbow_peak_frame"),
            wrist_peak_frame=raw.get("wrist_peak_frame"),
            fps=self.fps))
        results.append(HipShoulderSeparationKPI(self.cfg).evaluate(
            hip_shoulder_sep_values=raw.get("hip_shoulder_sep_values", [])))
        results.append(HandPathLinearityKPI(self.cfg).evaluate(
            wrist_positions_contact_zone=raw.get("wrist_positions_contact_zone")))

        # 阶段 4：击球
        results.append(ContactPointKPI(self.cfg).evaluate(
            contact_forward_norm=raw.get("contact_forward_norm")))
        results.append(ElbowAngleAtContactKPI(self.cfg).evaluate(
            elbow_angle_at_contact=raw.get("elbow_angle_at_contact")))
        results.append(BodyFreezeKPI(self.cfg).evaluate(
            torso_angular_velocity_at_contact=raw.get("torso_angular_velocity_at_contact")))
        results.append(HeadStabilityAtContactKPI(self.cfg).evaluate(
            head_displacement_norm=raw.get("head_displacement_norm")))

        # 阶段 5：延伸
        results.append(ForwardExtensionKPI(self.cfg).evaluate(
            forward_extension_norm=raw.get("forward_extension_norm")))
        results.append(FollowThroughPathKPI(self.cfg).evaluate(
            upward_forward_ratio=raw.get("upward_forward_ratio")))

        # 阶段 6：平衡
        results.append(OverallHeadStabilityKPI(self.cfg).evaluate(
            head_y_std_norm=raw.get("head_y_std_norm")))
        results.append(SpineConsistencyKPI(self.cfg).evaluate(
            spine_angle_std=raw.get("spine_angle_std")))

        return results

    # ── 内部：汇总 ─────────────────────────────────────────────────

    def _aggregate_phases(self, kpi_results: List[KPIResult]) -> Dict[str, PhaseScore]:
        """按阶段分组并计算阶段评分。"""
        phase_map: Dict[str, List[KPIResult]] = {}
        for kpi in kpi_results:
            phase_map.setdefault(kpi.phase, []).append(kpi)

        phase_scores = {}
        for phase, kpis in phase_map.items():
            valid_scores = [k.score for k in kpis if k.rating != "n/a"]
            avg = float(np.mean(valid_scores)) if valid_scores else 0.0
            phase_scores[phase] = PhaseScore(phase=phase, score=avg, kpis=kpis)

        return phase_scores

    def _compute_overall_score(self, phase_scores: Dict[str, PhaseScore]) -> float:
        """加权平均计算总分。"""
        weights = self.cfg.scoring.as_dict()
        total_weight = 0.0
        weighted_sum = 0.0
        for phase, weight in weights.items():
            if phase in phase_scores:
                weighted_sum += phase_scores[phase].score * weight
                total_weight += weight
        return weighted_sum / max(total_weight, 1e-6)

    # ── 无击球回退 ─────────────────────────────────────────────────

    def _evaluate_no_impact(self, kp_series, conf_series, frame_indices, store) -> SwingEvaluation:
        """无法检测到击球时的回退评估。"""
        swing = SwingEvent(
            swing_index=0,
            prep_start_frame=frame_indices[0] if frame_indices else None,
            impact_frame=None,
            followthrough_end_frame=frame_indices[-1] if frame_indices else None,
        )

        shoulder_rots, knee_angles, spine_angles = [], [], []
        for kp, conf in zip(kp_series, conf_series):
            sr = shoulder_hip_angle(kp, conf)
            if sr is not None:
                shoulder_rots.append(sr)
            ka = min_knee_angle(kp, conf)
            if ka is not None:
                knee_angles.append(ka)
            sa = spine_angle_from_vertical(kp, conf)
            if sa is not None:
                spine_angles.append(sa)

        raw = {
            "shoulder_rotation_values": shoulder_rots,
            "knee_angle_values": knee_angles,
            "spine_angle_values": spine_angles,
        }

        kpi_results = [
            ShoulderRotationKPI(self.cfg).evaluate(shoulder_rotation_values=shoulder_rots),
            KneeBendKPI(self.cfg).evaluate(knee_angle_values=knee_angles),
            SpineAngleKPI(self.cfg).evaluate(spine_angle_values=spine_angles),
        ]

        na_kpis = [
            KineticChainSequenceKPI, HipShoulderSeparationKPI, HandPathLinearityKPI,
            ContactPointKPI, ElbowAngleAtContactKPI, BodyFreezeKPI,
            HeadStabilityAtContactKPI, ForwardExtensionKPI, FollowThroughPathKPI,
            OverallHeadStabilityKPI, SpineConsistencyKPI,
        ]
        for kpi_cls in na_kpis:
            kpi = kpi_cls(self.cfg)
            kpi_results.append(KPIResult(
                kpi.kpi_id, kpi.name, kpi.phase, None, kpi.unit, 0, "n/a",
                "未检测到击球 — 无法评估此指标。"
            ))

        phase_scores = self._aggregate_phases(kpi_results)
        overall = self._compute_overall_score(phase_scores)

        return SwingEvaluation(
            swing_index=0,
            swing_event=swing,
            phase_scores=phase_scores,
            overall_score=overall,
            kpi_results=kpi_results,
            raw_metrics=raw,
        )
