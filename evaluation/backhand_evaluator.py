"""One-Handed Backhand Evaluator — 编排层。

``BackhandEvaluator`` 负责：
    1. 接收完整的逐帧关键点时间序列。
    2. 对每次击球独立分段（准备 → 击球 → 随挥）。
    3. 对每次击球独立计算单反特有的生物力学指标。
    4. 对每次击球独立评分。
    5. 汇总为 ``MultiSwingReport``。

与 ForehandEvaluator 的关键差异：
    - 持拍手/非持拍手定义相反（右手持拍者的反手用左手）
    - 新增：非持拍手辅助引拍评估、L形杠杆评估、ATA收拍评估、保持侧身评估
    - 手臂伸展评估替代了肘部角度（直臂/双弯）评估
    - 非持拍手反向平衡评估
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

import numpy as np

from config.backhand_config import BackhandConfig, DEFAULT_BACKHAND_CONFIG
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
    shoulder_width_px,
    shoulder_rotation_signed,
    hip_center,
    shoulder_center,
)
from evaluation.event_detector import (
    SwingPhaseEstimator,
    SwingEvent,
    ImpactEvent,
)
from evaluation.kpi import KPIResult
from evaluation.backhand_kpi import (
    OHB_ShoulderRotationKPI,
    OHB_KneeBendKPI,
    OHB_NonDomHandPrepKPI,
    OHB_SpineAngleKPI,
    OHB_LLeverageKPI,
    OHB_KineticChainSequenceKPI,
    OHB_HipShoulderSeparationKPI,
    OHB_InsideOutPathKPI,
    OHB_ContactPointKPI,
    OHB_ArmExtensionKPI,
    OHB_BodyFreezeKPI,
    OHB_NonDomHandBalanceKPI,
    OHB_HeadStabilityAtContactKPI,
    OHB_ATA_KPI,
    OHB_StaySidewaysKPI,
    OHB_OverallHeadStabilityKPI,
    OHB_SpineConsistencyKPI,
)

# 复用正手的数据容器
from evaluation.forehand_evaluator import (
    PhaseScore,
    SwingEvaluation,
    MultiSwingReport,
)


class BackhandEvaluator:
    """评估单反挥拍 — 支持视频中多次击球独立评分。"""

    def __init__(
        self,
        fps: float = 30.0,
        is_right_handed: bool = True,
        cfg: BackhandConfig = DEFAULT_BACKHAND_CONFIG,
    ):
        self.fps = fps
        self.is_right_handed = is_right_handed
        self.cfg = cfg

        # 单反：持拍手在非惯用侧
        # 右手持拍者的反手 → 持拍手仍然是右手，但击球在左侧
        self.dom_wrist_key = "right_wrist" if is_right_handed else "left_wrist"
        self.dom_elbow_key = "right_elbow" if is_right_handed else "left_elbow"
        self.dom_shoulder_key = "right_shoulder" if is_right_handed else "left_shoulder"

        # 非持拍手
        self.non_dom_wrist_key = "left_wrist" if is_right_handed else "right_wrist"
        self.non_dom_elbow_key = "left_elbow" if is_right_handed else "right_elbow"
        self.non_dom_shoulder_key = "left_shoulder" if is_right_handed else "right_shoulder"

    # ── 公开 API ────────────────────────────────────────────────────

    def evaluate_multi(
        self,
        keypoints_series: List[np.ndarray],
        confidence_series: List[np.ndarray],
        frame_indices: List[int],
        impact_events: List[ImpactEvent],
    ) -> MultiSwingReport:
        """对多次击球进行独立评估。"""
        n_frames = len(keypoints_series)

        # 构建 TrajectoryStore
        store = TrajectoryStore(fps=self.fps)
        for kp, conf, fidx in zip(keypoints_series, confidence_series, frame_indices):
            store.update(kp, conf, fidx)

        # 获取持拍手腕速度序列
        wrist_traj = store.get(self.dom_wrist_key)
        wrist_speeds = wrist_traj.get_speeds(smoothed=True)
        speed_frame_indices = wrist_traj.frame_indices[1:] if len(wrist_traj.frame_indices) > 1 else []

        # 如果没有检测到击球，回退
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

            # 单反的前进方向（与正手相反）
            forward_sign = 1.0
            if abs(impact_event.peak_velocity_unit[0]) > 0.1:
                forward_sign = 1.0 if impact_event.peak_velocity_unit[0] > 0 else -1.0

            # 计算原始指标
            raw = self._compute_raw_metrics(
                keypoints_series, confidence_series, frame_indices,
                store, swing_event, forward_sign,
            )

            # 评估所有 KPI
            kpi_results = self._evaluate_kpis(raw)

            # 汇总阶段评分
            phase_scores = self._aggregate_phases(kpi_results)
            overall_score = self._compute_overall_score(phase_scores)

            evaluations.append(SwingEvaluation(
                swing_index=i,
                swing_event=swing_event,
                phase_scores=phase_scores,
                overall_score=overall_score,
                kpi_results=kpi_results,
                forward_sign=forward_sign,
                arm_style="单反",
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

    # ── 内部：原始指标计算 ──────────────────────────────────────────

    def _compute_raw_metrics(
        self,
        kp_series, conf_series, frame_indices,
        store: TrajectoryStore,
        swing: SwingEvent,
        forward_sign: float,
    ) -> Dict[str, Any]:
        """计算单次单反击球所需的所有生物力学原始指标。"""
        raw: Dict[str, Any] = {}
        impact_frame = swing.impact_frame
        prep_start = swing.prep_start_frame or frame_indices[0]
        ft_end = swing.followthrough_end_frame or frame_indices[-1]

        # 帧索引 → 数组位置映射
        f2p = {f: i for i, f in enumerate(frame_indices)}
        impact_pos = f2p.get(impact_frame, len(frame_indices) // 2)
        prep_pos = f2p.get(prep_start, 0)
        ft_pos = f2p.get(ft_end, len(frame_indices) - 1)

        dom_wrist_idx = KEYPOINT_NAMES[self.dom_wrist_key]
        non_dom_wrist_idx = KEYPOINT_NAMES[self.non_dom_wrist_key]
        dom_elbow_idx = KEYPOINT_NAMES[self.dom_elbow_key]
        dom_shoulder_idx = KEYPOINT_NAMES[self.dom_shoulder_key]

        # ── 阶段 1：准备阶段指标 ──────────────────────────────────────
        shoulder_rots = []
        knee_angles = []
        spine_angles = []
        non_dom_distances_prep = []

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

            # 非持拍手距离（准备阶段前半段，两手应在一起）
            if i < prep_pos + (impact_pos - prep_pos) // 2:
                sw = shoulder_width_px(kp, conf)
                if (sw and sw > 1e-6 and
                    conf[dom_wrist_idx] >= 0.3 and conf[non_dom_wrist_idx] >= 0.3):
                    dist = float(np.linalg.norm(
                        kp[dom_wrist_idx].astype(np.float64) - kp[non_dom_wrist_idx].astype(np.float64)
                    ))
                    non_dom_distances_prep.append(dist / sw)

        raw["shoulder_rotation_values"] = shoulder_rots
        raw["knee_angle_values"] = knee_angles
        raw["spine_angle_values"] = spine_angles
        raw["non_dom_hand_distance_prep"] = (
            float(np.mean(non_dom_distances_prep)) if non_dom_distances_prep else None
        )

        # ── 阶段 2：引拍阶段指标 ──────────────────────────────────────
        # 引拍阶段的肘部角度（准备阶段后半段到击球前）
        backswing_start = prep_pos + (impact_pos - prep_pos) // 3
        backswing_end = max(backswing_start + 1, impact_pos - int(0.05 * self.fps))
        backswing_elbow_angles = []
        for i in range(backswing_start, min(backswing_end, len(kp_series))):
            ea = elbow_angle(kp_series[i], conf_series[i], right=self.is_right_handed)
            if ea is not None:
                backswing_elbow_angles.append(ea)

        raw["backswing_elbow_angle"] = (
            float(np.mean(backswing_elbow_angles)) if backswing_elbow_angles else None
        )

        # ── 阶段 3：动力链指标 ────────────────────────────────────────
        hip_key = "right_hip" if self.is_right_handed else "left_hip"
        hip_traj = store.get(hip_key)
        shoulder_traj = store.get(self.dom_shoulder_key)
        elbow_traj = store.get(self.dom_elbow_key)
        wrist_traj = store.get(self.dom_wrist_key)

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

        # Inside-Out 路径线性度
        contact_zone_half = max(3, int(0.05 * self.fps))
        wrist_positions_cz = []
        for i in range(max(0, impact_pos - contact_zone_half),
                       min(len(kp_series), impact_pos + contact_zone_half + 1)):
            kp, conf = kp_series[i], conf_series[i]
            if conf[dom_wrist_idx] >= 0.3:
                wrist_positions_cz.append(kp[dom_wrist_idx].copy())
        raw["wrist_positions_contact_zone"] = np.array(wrist_positions_cz) if wrist_positions_cz else None

        # ── 阶段 4：击球点指标 ────────────────────────────────────────
        if impact_pos < len(kp_series):
            kp_impact = kp_series[impact_pos]
            conf_impact = conf_series[impact_pos]

            # 击球点位置
            raw["contact_forward_norm"] = wrist_forward_normalised(
                kp_impact, conf_impact, self.is_right_handed, forward_sign
            )

            # 手臂伸展（肘部角度）
            raw["elbow_angle_at_contact"] = elbow_angle(
                kp_impact, conf_impact, right=self.is_right_handed
            )

            # 身体刹车
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

            # 非持拍手反向平衡（击球时两手距离）
            sw = shoulder_width_px(kp_impact, conf_impact)
            if (sw and sw > 1e-6 and
                conf_impact[dom_wrist_idx] >= 0.3 and conf_impact[non_dom_wrist_idx] >= 0.3):
                dist = float(np.linalg.norm(
                    kp_impact[dom_wrist_idx].astype(np.float64) - kp_impact[non_dom_wrist_idx].astype(np.float64)
                ))
                raw["non_dom_hand_spread_contact"] = dist / sw
            else:
                raw["non_dom_hand_spread_contact"] = None

            # 头部稳定性
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
            raw["non_dom_hand_spread_contact"] = None
            raw["head_displacement_norm"] = None

        # ── 阶段 5：延伸指标 ──────────────────────────────────────────
        post_contact_frames = int(self.cfg.extension.post_contact_window_s * self.fps)
        if impact_pos < len(kp_series):
            kp_impact = kp_series[impact_pos]
            conf_impact = conf_series[impact_pos]

            # ATA 收拍高度：随挥后手腕最高点 vs 肩部
            dom_shoulder_y_at_contact = None
            if conf_impact[dom_shoulder_idx] >= 0.3:
                dom_shoulder_y_at_contact = float(kp_impact[dom_shoulder_idx][1])

            max_wrist_above_shoulder = None
            if dom_shoulder_y_at_contact is not None:
                th = torso_height_px(kp_impact, conf_impact)
                if th and th > 1e-6:
                    best_above = -999.0
                    for i in range(impact_pos + 1,
                                   min(len(kp_series), impact_pos + post_contact_frames + 1)):
                        if conf_series[i][dom_wrist_idx] >= 0.3:
                            wrist_y = float(kp_series[i][dom_wrist_idx][1])
                            # 在图像坐标中 y 向下，所以 shoulder_y - wrist_y > 0 表示手腕在肩上方
                            above = (dom_shoulder_y_at_contact - wrist_y) / th
                            best_above = max(best_above, above)
                    if best_above > -999.0:
                        max_wrist_above_shoulder = best_above

            raw["ata_wrist_above_shoulder"] = max_wrist_above_shoulder

            # 保持侧身：击球后肩部旋转变化
            sr_at_contact = shoulder_hip_angle(kp_impact, conf_impact)
            post_shoulder_rots = []
            for i in range(impact_pos + 1,
                           min(len(kp_series), impact_pos + post_contact_frames + 1)):
                sr = shoulder_hip_angle(kp_series[i], conf_series[i])
                if sr is not None:
                    post_shoulder_rots.append(sr)

            if sr_at_contact is not None and post_shoulder_rots:
                # 最大变化
                max_change = max(abs(sr - sr_at_contact) for sr in post_shoulder_rots)
                raw["post_contact_shoulder_change"] = max_change
            else:
                raw["post_contact_shoulder_change"] = None
        else:
            raw["ata_wrist_above_shoulder"] = None
            raw["post_contact_shoulder_change"] = None

        # ── 阶段 6：平衡指标 ──────────────────────────────────────────
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

    def _peak_in_range(self, traj, start_pos, end_pos, frame_indices) -> Optional[int]:
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
            if i + 1 < len(traj_frames):
                f = traj_frames[i + 1]
                if start_frame <= f <= end_frame and speed > best_speed:
                    best_speed = speed
                    best_frame = f
        return best_frame

    # ── 内部：KPI 评估 ─────────────────────────────────────────────

    def _evaluate_kpis(self, raw: Dict[str, Any]) -> List[KPIResult]:
        """实例化并评估所有单反 KPI。"""
        results = []

        # 阶段 1：准备
        results.append(OHB_ShoulderRotationKPI(self.cfg).evaluate(
            shoulder_rotation_values=raw.get("shoulder_rotation_values", [])))
        results.append(OHB_KneeBendKPI(self.cfg).evaluate(
            knee_angle_values=raw.get("knee_angle_values", [])))
        results.append(OHB_NonDomHandPrepKPI(self.cfg).evaluate(
            non_dom_hand_distance_prep=raw.get("non_dom_hand_distance_prep")))
        results.append(OHB_SpineAngleKPI(self.cfg).evaluate(
            spine_angle_values=raw.get("spine_angle_values", [])))

        # 阶段 2：引拍
        results.append(OHB_LLeverageKPI(self.cfg).evaluate(
            backswing_elbow_angle=raw.get("backswing_elbow_angle")))

        # 阶段 3：动力链
        results.append(OHB_KineticChainSequenceKPI(self.cfg).evaluate(
            hip_peak_frame=raw.get("hip_peak_frame"),
            shoulder_peak_frame=raw.get("shoulder_peak_frame"),
            elbow_peak_frame=raw.get("elbow_peak_frame"),
            wrist_peak_frame=raw.get("wrist_peak_frame"),
            fps=self.fps))
        results.append(OHB_HipShoulderSeparationKPI(self.cfg).evaluate(
            hip_shoulder_sep_values=raw.get("hip_shoulder_sep_values", [])))
        results.append(OHB_InsideOutPathKPI(self.cfg).evaluate(
            wrist_positions_contact_zone=raw.get("wrist_positions_contact_zone")))

        # 阶段 4：击球
        results.append(OHB_ContactPointKPI(self.cfg).evaluate(
            contact_forward_norm=raw.get("contact_forward_norm")))
        results.append(OHB_ArmExtensionKPI(self.cfg).evaluate(
            elbow_angle_at_contact=raw.get("elbow_angle_at_contact")))
        results.append(OHB_BodyFreezeKPI(self.cfg).evaluate(
            torso_angular_velocity_at_contact=raw.get("torso_angular_velocity_at_contact")))
        results.append(OHB_NonDomHandBalanceKPI(self.cfg).evaluate(
            non_dom_hand_spread_contact=raw.get("non_dom_hand_spread_contact")))
        results.append(OHB_HeadStabilityAtContactKPI(self.cfg).evaluate(
            head_displacement_norm=raw.get("head_displacement_norm")))

        # 阶段 5：延伸
        results.append(OHB_ATA_KPI(self.cfg).evaluate(
            ata_wrist_above_shoulder=raw.get("ata_wrist_above_shoulder")))
        results.append(OHB_StaySidewaysKPI(self.cfg).evaluate(
            post_contact_shoulder_change=raw.get("post_contact_shoulder_change")))

        # 阶段 6：平衡
        results.append(OHB_OverallHeadStabilityKPI(self.cfg).evaluate(
            head_y_std_norm=raw.get("head_y_std_norm")))
        results.append(OHB_SpineConsistencyKPI(self.cfg).evaluate(
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
            OHB_ShoulderRotationKPI(self.cfg).evaluate(shoulder_rotation_values=shoulder_rots),
            OHB_KneeBendKPI(self.cfg).evaluate(knee_angle_values=knee_angles),
            OHB_SpineAngleKPI(self.cfg).evaluate(spine_angle_values=spine_angles),
        ]

        # 其余 KPI 标记为无数据
        na_kpis = [
            OHB_NonDomHandPrepKPI, OHB_LLeverageKPI,
            OHB_KineticChainSequenceKPI, OHB_HipShoulderSeparationKPI, OHB_InsideOutPathKPI,
            OHB_ContactPointKPI, OHB_ArmExtensionKPI, OHB_BodyFreezeKPI,
            OHB_NonDomHandBalanceKPI, OHB_HeadStabilityAtContactKPI,
            OHB_ATA_KPI, OHB_StaySidewaysKPI,
            OHB_OverallHeadStabilityKPI, OHB_SpineConsistencyKPI,
        ]
        for kpi_cls in na_kpis:
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
