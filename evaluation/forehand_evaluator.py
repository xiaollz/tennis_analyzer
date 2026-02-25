"""Modern Forehand Evaluator v3 — 8阶段编排层。

``ForehandEvaluator`` 负责：
    1. 接收完整的逐帧关键点时间序列。
    2. 使用 ``HybridImpactDetector`` 检测 **所有** 击球事件。
    3. 对每次击球独立分段（准备 → 击球 → 随挥）。
    4. 对每次击球独立计算所有生物力学原始指标（含 v3 新增指标）。
    5. 对每次击球独立评估 21 个 KPI。
    6. 汇总为 ``MultiSwingReport``。

v3 升级：
    - 6阶段 → 8阶段（新增 Slot准备、蹬转启动、滞后驱动、雨刷随挥）
    - 14 KPI → 21 KPI（新增 SIR代理、肘部后撤、拍头下垂、蹬地力量等）
    - 新增训练处方系统
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

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
    shoulder_width_px,
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
    HybridImpactDetector,
    SwingPhaseEstimator,
    SwingEvent,
    ImpactEvent,
)
from evaluation.kpi import (
    KPIResult,
    # Phase 1: Unit Turn
    ShoulderRotationKPI,
    KneeBendKPI,
    SpineAngleKPI,
    # Phase 2: Slot Preparation
    ElbowBackPositionKPI,
    RacketDropKPI,
    # Phase 3: Leg Drive
    GroundForceProxyKPI,
    HipRotationSpeedKPI,
    # Phase 4: Torso Pull
    HipShoulderSeparationKPI,
    HipShoulderTimingKPI,
    # Phase 5: Lag & Elbow Drive
    ElbowTuckKPI,
    HandPathLinearityKPI,
    # Phase 6: Contact & SIR
    ContactPointKPI,
    ElbowAngleAtContactKPI,
    BodyFreezeKPI,
    HeadStabilityAtContactKPI,
    SIRProxyKPI,
    # Phase 7: Wiper Follow-Through
    ForwardExtensionKPI,
    WiperSweepKPI,
    # Phase 8: Balance
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
    """评估正手挥拍 — v3 8阶段模型，支持视频中多次击球独立评分。"""

    # 8 阶段名称（用于报告和 UI）
    PHASE_ORDER = [
        "unit_turn", "slot_prep", "leg_drive", "torso_pull",
        "lag_drive", "contact", "wiper", "balance",
    ]

    PHASE_NAMES_CN = {
        "unit_turn": "一体化转体",
        "slot_prep": "槽位准备",
        "leg_drive": "蹬转与髋部启动",
        "torso_pull": "躯干与肩部牵引",
        "lag_drive": "滞后与肘部驱动",
        "contact": "击球与肩内旋",
        "wiper": "雨刷式随挥",
        "balance": "减速与平衡",
    }

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
        self.hip_key = "right_hip" if is_right_handed else "left_hip"

    # ── 公开 API ────────────────────────────────────────────────────

    def evaluate_multi(
        self,
        keypoints_series: List[np.ndarray],
        confidence_series: List[np.ndarray],
        frame_indices: List[int],
        impact_events: List[ImpactEvent],
    ) -> MultiSwingReport:
        """对多次击球进行独立评估。"""
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
            prev_impact = impact_events[i - 1].impact_frame_idx if i > 0 else None
            next_impact = impact_events[i + 1].impact_frame_idx if i < len(impact_events) - 1 else None

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

            # 计算原始指标（v3 扩展）
            raw = self._compute_raw_metrics(
                keypoints_series, confidence_series, frame_indices,
                store, swing_event, forward_sign,
            )

            # 评估所有 21 个 KPI
            kpi_results = self._evaluate_kpis(raw)

            # 汇总阶段评分
            phase_scores = self._aggregate_phases(kpi_results)
            overall_score = self._compute_overall_score(phase_scores)

            # 判断手臂风格
            elbow_kpi = next((k for k in kpi_results if k.kpi_id == "C6.2"), None)
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

    def evaluate(
        self,
        keypoints_series: List[np.ndarray],
        confidence_series: List[np.ndarray],
        frame_indices: Optional[List[int]] = None,
        impact_events: Optional[List[ImpactEvent]] = None,
    ) -> MultiSwingReport:
        """向后兼容接口。"""
        if frame_indices is None:
            frame_indices = list(range(len(keypoints_series)))
        if impact_events is None:
            impact_events = []
        return self.evaluate_multi(
            keypoints_series, confidence_series, frame_indices, impact_events,
        )

    # ── 内部：原始指标计算（v3 扩展）──────────────────────────────────

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

        # 前挥起始位置（大约在击球前 0.15 秒）
        forward_swing_start = max(prep_pos, impact_pos - int(0.15 * self.fps))

        wrist_idx = KEYPOINT_NAMES[self.wrist_key]
        elbow_idx = KEYPOINT_NAMES[self.elbow_key]

        # ================================================================
        # 阶段 1：一体化转体 (Unit Turn)
        # ================================================================
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

        # ================================================================
        # 阶段 2：槽位准备 (Slot Preparation)
        # ================================================================
        # 在引拍阶段（准备到前挥起始）测量肘部后撤和拍头下垂
        slot_phase_end = min(forward_swing_start + 1, len(kp_series))
        elbow_behind_values = []
        wrist_below_elbow_values = []
        for i in range(prep_pos, slot_phase_end):
            kp, conf = kp_series[i], conf_series[i]
            eb = elbow_behind_torso_normalised(kp, conf, self.is_right_handed, forward_sign)
            if eb is not None:
                elbow_behind_values.append(eb)
            wbe = wrist_below_elbow_distance(kp, conf, self.is_right_handed)
            if wbe is not None:
                wrist_below_elbow_values.append(wbe)

        raw["elbow_behind_values"] = elbow_behind_values
        raw["wrist_below_elbow_values"] = wrist_below_elbow_values

        # ================================================================
        # 阶段 3：蹬转与髋部启动 (Leg Drive)
        # ================================================================
        # 髋部垂直位置序列（用于计算蹬地力量代理）
        hip_y_positions = []
        hip_angles_series = []
        shoulder_angles_series = []
        leg_drive_start = max(prep_pos, impact_pos - int(0.3 * self.fps))
        leg_drive_frame_indices = []

        for i in range(leg_drive_start, min(impact_pos + 1, len(kp_series))):
            kp, conf = kp_series[i], conf_series[i]
            hy = hip_center_vertical_position(kp, conf)
            if hy is not None:
                hip_y_positions.append(hy)
            ha = hip_line_angle(kp, conf)
            hip_angles_series.append(ha)
            sa = shoulder_line_angle(kp, conf)
            shoulder_angles_series.append(sa)
            if i < len(frame_indices):
                leg_drive_frame_indices.append(frame_indices[i])

        # 蹬地力量代理
        raw["peak_hip_acceleration"] = compute_vertical_acceleration(hip_y_positions, self.fps)

        # 髋部旋转速度
        raw["peak_hip_rotation_speed"] = compute_peak_rotation_speed(hip_angles_series, self.fps)

        # ================================================================
        # 阶段 4：躯干与肩部牵引 (Torso Pull)
        # ================================================================
        # 髋肩分离角
        hip_shoulder_seps = []
        search_start = max(0, impact_pos - int(0.3 * self.fps))
        for i in range(search_start, min(impact_pos + 1, len(kp_series))):
            kp, conf = kp_series[i], conf_series[i]
            angle = shoulder_hip_angle(kp, conf)
            if angle is not None:
                hip_shoulder_seps.append(angle)
        raw["hip_shoulder_sep_values"] = hip_shoulder_seps

        # 髋肩旋转时序差
        raw["hip_shoulder_timing_delay"] = hip_shoulder_separation_timing(
            hip_angles_series, shoulder_angles_series, self.fps, leg_drive_frame_indices
        )

        # ================================================================
        # 阶段 5：滞后与肘部驱动 (Lag & Elbow Drive)
        # ================================================================
        # 肘部收紧距离（前挥阶段）
        elbow_tuck_values = []
        for i in range(forward_swing_start, min(impact_pos + 1, len(kp_series))):
            kp, conf = kp_series[i], conf_series[i]
            et = elbow_to_torso_distance(kp, conf, self.is_right_handed)
            if et is not None:
                elbow_tuck_values.append(et)
        raw["min_elbow_tuck"] = min(elbow_tuck_values) if elbow_tuck_values else None

        # 手部路径线性度（击球前后 ±5 帧）
        contact_zone_half = max(3, int(0.05 * self.fps))
        wrist_positions_cz = []
        for i in range(max(0, impact_pos - contact_zone_half),
                       min(len(kp_series), impact_pos + contact_zone_half + 1)):
            kp, conf = kp_series[i], conf_series[i]
            if conf[wrist_idx] >= 0.3:
                wrist_positions_cz.append(kp[wrist_idx].copy())
        raw["wrist_positions_contact_zone"] = np.array(wrist_positions_cz) if wrist_positions_cz else None

        # ================================================================
        # 阶段 6：击球与肩内旋 (Contact & SIR)
        # ================================================================
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

            # SIR 代理：击球前后前臂角速度
            sir_window = max(3, int(0.1 * self.fps))
            forearm_angles = []
            for i in range(max(0, impact_pos - sir_window),
                           min(len(kp_series), impact_pos + sir_window + 1)):
                fa = forearm_angle(kp_series[i], conf_series[i], self.is_right_handed)
                forearm_angles.append(fa if fa is not None else 0.0)
            raw["forearm_angular_velocity"] = compute_angular_velocity(forearm_angles, self.fps)

        else:
            raw["contact_forward_norm"] = None
            raw["elbow_angle_at_contact"] = None
            raw["torso_angular_velocity_at_contact"] = None
            raw["head_displacement_norm"] = None
            raw["forearm_angular_velocity"] = None

        # ================================================================
        # 阶段 7：雨刷式随挥 (Wiper Follow-Through)
        # ================================================================
        post_contact_frames = int(self.cfg.wiper.post_contact_window_s * self.fps)
        if impact_pos < len(kp_series):
            wrist_at_contact = (
                kp_series[impact_pos][wrist_idx].copy()
                if conf_series[impact_pos][wrist_idx] >= 0.3 else None
            )

            if wrist_at_contact is not None:
                max_forward = 0.0
                for i in range(impact_pos + 1,
                               min(len(kp_series), impact_pos + post_contact_frames + 1)):
                    if conf_series[i][wrist_idx] >= 0.3:
                        delta = kp_series[i][wrist_idx] - wrist_at_contact
                        forward_dist = delta[0] * forward_sign
                        max_forward = max(max_forward, forward_dist)

                th = torso_height_px(kp_series[impact_pos], conf_series[impact_pos])
                if th and th > 1e-6:
                    raw["forward_extension_norm"] = max_forward / th
                else:
                    raw["forward_extension_norm"] = None
            else:
                raw["forward_extension_norm"] = None

            # 雨刷扫过角度
            wiper_wrist_positions = []
            wiper_shoulder_centers = []
            for i in range(impact_pos,
                           min(len(kp_series), impact_pos + post_contact_frames + 1)):
                kp, conf = kp_series[i], conf_series[i]
                if conf[wrist_idx] >= 0.3:
                    wiper_wrist_positions.append(kp[wrist_idx].astype(np.float64).copy())
                    sc = shoulder_center(kp, conf)
                    if sc is not None:
                        wiper_shoulder_centers.append(sc)
                    else:
                        wiper_wrist_positions.pop()

            raw["wiper_sweep_angle"] = compute_wiper_sweep_angle(
                wiper_wrist_positions, wiper_shoulder_centers
            )
        else:
            raw["forward_extension_norm"] = None
            raw["wiper_sweep_angle"] = None

        # ================================================================
        # 阶段 8：减速与平衡 (Balance)
        # ================================================================
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

    # ── 内部：KPI 评估（v3: 21 个 KPI）─────────────────────────────────

    def _evaluate_kpis(self, raw: Dict[str, Any]) -> List[KPIResult]:
        """实例化并评估所有 21 个 KPI。"""
        results = []

        # ── 阶段 1：一体化转体 ──
        results.append(ShoulderRotationKPI(self.cfg).evaluate(
            shoulder_rotation_values=raw.get("shoulder_rotation_values", [])))
        results.append(KneeBendKPI(self.cfg).evaluate(
            knee_angle_values=raw.get("knee_angle_values", [])))
        results.append(SpineAngleKPI(self.cfg).evaluate(
            spine_angle_values=raw.get("spine_angle_values", [])))

        # ── 阶段 2：槽位准备 ──
        results.append(ElbowBackPositionKPI(self.cfg).evaluate(
            elbow_behind_values=raw.get("elbow_behind_values", [])))
        results.append(RacketDropKPI(self.cfg).evaluate(
            wrist_below_elbow_values=raw.get("wrist_below_elbow_values", [])))

        # ── 阶段 3：蹬转与髋部启动 ──
        results.append(GroundForceProxyKPI(self.cfg).evaluate(
            peak_hip_acceleration=raw.get("peak_hip_acceleration")))
        results.append(HipRotationSpeedKPI(self.cfg).evaluate(
            peak_hip_rotation_speed=raw.get("peak_hip_rotation_speed")))

        # ── 阶段 4：躯干与肩部牵引 ──
        results.append(HipShoulderSeparationKPI(self.cfg).evaluate(
            hip_shoulder_sep_values=raw.get("hip_shoulder_sep_values", [])))
        results.append(HipShoulderTimingKPI(self.cfg).evaluate(
            hip_shoulder_timing_delay=raw.get("hip_shoulder_timing_delay")))

        # ── 阶段 5：滞后与肘部驱动 ──
        results.append(ElbowTuckKPI(self.cfg).evaluate(
            min_elbow_tuck=raw.get("min_elbow_tuck")))
        results.append(HandPathLinearityKPI(self.cfg).evaluate(
            wrist_positions_contact_zone=raw.get("wrist_positions_contact_zone")))

        # ── 阶段 6：击球与肩内旋 ──
        results.append(ContactPointKPI(self.cfg).evaluate(
            contact_forward_norm=raw.get("contact_forward_norm")))
        results.append(ElbowAngleAtContactKPI(self.cfg).evaluate(
            elbow_angle_at_contact=raw.get("elbow_angle_at_contact")))
        results.append(BodyFreezeKPI(self.cfg).evaluate(
            torso_angular_velocity_at_contact=raw.get("torso_angular_velocity_at_contact")))
        results.append(HeadStabilityAtContactKPI(self.cfg).evaluate(
            head_displacement_norm=raw.get("head_displacement_norm")))
        results.append(SIRProxyKPI(self.cfg).evaluate(
            forearm_angular_velocity=raw.get("forearm_angular_velocity")))

        # ── 阶段 7：雨刷式随挥 ──
        results.append(ForwardExtensionKPI(self.cfg).evaluate(
            forward_extension_norm=raw.get("forward_extension_norm")))
        results.append(WiperSweepKPI(self.cfg).evaluate(
            wiper_sweep_angle=raw.get("wiper_sweep_angle")))

        # ── 阶段 8：减速与平衡 ──
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
            valid_scores = [k.score for k in kpis if k.rating != "无数据"]
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

        # 其余 KPI 标记为无数据
        na_kpi_classes = [
            ElbowBackPositionKPI, RacketDropKPI,
            GroundForceProxyKPI, HipRotationSpeedKPI,
            HipShoulderSeparationKPI, HipShoulderTimingKPI,
            ElbowTuckKPI, HandPathLinearityKPI,
            ContactPointKPI, ElbowAngleAtContactKPI, BodyFreezeKPI,
            HeadStabilityAtContactKPI, SIRProxyKPI,
            ForwardExtensionKPI, WiperSweepKPI,
            OverallHeadStabilityKPI, SpineConsistencyKPI,
        ]
        for kpi_cls in na_kpi_classes:
            kpi = kpi_cls(self.cfg)
            kpi_results.append(KPIResult(
                kpi.kpi_id, kpi.name, kpi.phase, None, kpi.unit, 0, "无数据",
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
