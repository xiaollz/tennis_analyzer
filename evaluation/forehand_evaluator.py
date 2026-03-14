"""Fault-tolerant forehand evaluator.

This evaluator follows the principle order from
``The Fault Tolerant Forehand``:
    - simple unit turn and preparation,
    - hip-led chain,
    - front contact with workable arm structure,
    - out / through hand path,
    - head and trunk stability.

Metrics that require racket tracking are intentionally omitted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from config.framework_config import DEFAULT_CONFIG, FrameworkConfig
from config.keypoints import KEYPOINT_NAMES
from analysis.trajectory import TrajectoryStore
from analysis.kinematic_calculator import (
    elbow_angle,
    hip_center,
    hip_line_angle,
    hip_shoulder_separation_timing,
    min_knee_angle,
    nose_position,
    shoulder_center,
    shoulder_hip_angle,
    shoulder_line_angle,
    spine_angle_from_vertical,
    torso_height_px,
    wrist_forward_normalised,
)
from evaluation.event_detector import ImpactEvent, SwingEvent, SwingPhaseEstimator
from evaluation.kpi import (
    ALL_KPIS,
    KPIResult,
    ContactPointKPI,
    ContactSpacingKPI,
    ElbowAngleAtContactKPI,
    ForwardExtensionKPI,
    HandPathLinearityKPI,
    HeadStabilityAtContactKPI,
    HipShoulderSeparationKPI,
    HipShoulderTimingKPI,
    KneeBendKPI,
    OverallHeadStabilityKPI,
    ShoulderRotationKPI,
    SpineAngleKPI,
    SpineConsistencyKPI,
    OutsideExtensionKPI,
)


@dataclass
class PhaseScore:
    phase: str
    score: float
    kpis: List[KPIResult]


@dataclass
class SwingEvaluation:
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
    swing_evaluations: List[SwingEvaluation]
    average_score: float
    best_swing_index: int
    worst_swing_index: int
    impact_frames: List[int]
    total_swings: int


class ForehandEvaluator:
    """Principle-driven forehand evaluator."""

    PHASE_ORDER = ["unit_turn", "chain", "contact", "through", "stability"]
    PHASE_NAMES_CN = {
        "unit_turn": "转开、备手与下肢准备",
        "chain": "转髋带动与解旋顺序",
        "contact": "前方接触与手臂结构",
        "through": "向外、向前穿过",
        "stability": "头部与躯干稳定",
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

    def evaluate_multi(
        self,
        keypoints_series: List[np.ndarray],
        confidence_series: List[np.ndarray],
        frame_indices: List[int],
        impact_events: List[ImpactEvent],
    ) -> MultiSwingReport:
        store = TrajectoryStore(fps=self.fps)
        for kp, conf, fidx in zip(keypoints_series, confidence_series, frame_indices):
            store.update(kp, conf, fidx)

        wrist_traj = store.get(self.wrist_key)
        wrist_speeds = wrist_traj.get_speeds(smoothed=True)
        speed_frame_indices = wrist_traj.frame_indices[1:] if len(wrist_traj.frame_indices) > 1 else []

        if not impact_events:
            evaluation = self._evaluate_no_impact(keypoints_series, confidence_series, frame_indices)
            return MultiSwingReport(
                swing_evaluations=[evaluation],
                average_score=evaluation.overall_score,
                best_swing_index=0,
                worst_swing_index=0,
                impact_frames=[],
                total_swings=0,
            )

        evaluations: List[SwingEvaluation] = []
        phase_estimator = SwingPhaseEstimator(fps=self.fps)

        for index, impact_event in enumerate(impact_events):
            prev_impact = impact_events[index - 1].impact_frame_idx if index > 0 else None
            next_impact = impact_events[index + 1].impact_frame_idx if index < len(impact_events) - 1 else None
            swing_event = phase_estimator.estimate_phases(
                impact_frame=impact_event.impact_frame_idx,
                wrist_speeds=wrist_speeds,
                frame_indices=speed_frame_indices,
                prev_impact_frame=prev_impact,
                next_impact_frame=next_impact,
            )
            swing_event.swing_index = index
            swing_event.impact_event = impact_event

            forward_sign = 1.0
            if abs(impact_event.peak_velocity_unit[0]) > 0.1:
                forward_sign = 1.0 if impact_event.peak_velocity_unit[0] > 0 else -1.0
            forward_axis = self._estimate_forward_axis(impact_event)

            raw = self._compute_raw_metrics(
                keypoints_series,
                confidence_series,
                frame_indices,
                swing_event,
                forward_sign,
                forward_axis,
            )
            kpi_results = self._evaluate_kpis(raw)
            phase_scores = self._aggregate_phases(kpi_results)
            overall_score = self._compute_overall_score(phase_scores)

            elbow_kpi = next((k for k in kpi_results if k.kpi_id == "C3.3"), None)
            arm_style = elbow_kpi.details.get("style", "未知") if elbow_kpi else "未知"

            evaluations.append(
                SwingEvaluation(
                    swing_index=index,
                    swing_event=swing_event,
                    phase_scores=phase_scores,
                    overall_score=overall_score,
                    kpi_results=kpi_results,
                    forward_sign=forward_sign,
                    arm_style=arm_style,
                    raw_metrics=raw,
                )
            )

        scores = [evaluation.overall_score for evaluation in evaluations]
        avg_score = float(np.mean(scores)) if scores else 0.0
        best_idx = int(np.argmax(scores)) if scores else 0
        worst_idx = int(np.argmin(scores)) if scores else 0
        impact_frames = [event.impact_frame_idx for event in impact_events]
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
        if frame_indices is None:
            frame_indices = list(range(len(keypoints_series)))
        return self.evaluate_multi(
            keypoints_series,
            confidence_series,
            frame_indices,
            impact_events or [],
        )

    @staticmethod
    def _estimate_forward_axis(impact_event: ImpactEvent) -> Optional[np.ndarray]:
        vx, vy = impact_event.peak_velocity_unit
        axis = np.array([float(vx), float(vy)], dtype=np.float64)
        norm = float(np.linalg.norm(axis))
        if norm < 1e-6:
            return None
        return axis / norm

    @staticmethod
    def _relative_rotation_series(angles: List[Optional[float]]) -> List[float]:
        valid = [float(angle) for angle in angles if angle is not None]
        if len(valid) < 2:
            return []
        arr = np.asarray(valid, dtype=np.float64)
        doubled = np.deg2rad(arr * 2.0)
        unwrapped = np.rad2deg(np.unwrap(doubled)) / 2.0
        base = float(unwrapped[0])
        return np.clip(np.abs(unwrapped - base), 0.0, 120.0).tolist()

    @staticmethod
    def _normalised_rms_displacement(points: List[np.ndarray], scale: Optional[float]) -> Optional[float]:
        if len(points) < 3 or scale is None or scale <= 1e-6:
            return None
        arr = np.asarray(points, dtype=np.float64)
        centered = arr - np.mean(arr, axis=0, keepdims=True)
        rms = float(np.sqrt(np.mean(np.sum(centered ** 2, axis=1))))
        return rms / scale

    @staticmethod
    def _point_to_line_distance(point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> Optional[float]:
        line_vec = np.asarray(line_end, dtype=np.float64) - np.asarray(line_start, dtype=np.float64)
        norm = float(np.linalg.norm(line_vec))
        if norm < 1e-6:
            return None
        point_vec = np.asarray(point, dtype=np.float64) - np.asarray(line_start, dtype=np.float64)
        area_twice = abs(float(np.cross(line_vec, point_vec)))
        return area_twice / norm

    @staticmethod
    def _relative_head_point(keypoints: np.ndarray, confidence: np.ndarray) -> Optional[np.ndarray]:
        nose = nose_position(keypoints, confidence)
        sc = shoulder_center(keypoints, confidence)
        hc = hip_center(keypoints, confidence)
        if nose is None or sc is None or hc is None:
            return None
        torso_anchor = 0.5 * (sc + hc)
        return nose - torso_anchor

    def _dominant_side_axis(self, keypoints: np.ndarray, confidence: np.ndarray) -> Optional[np.ndarray]:
        shoulder_idx = KEYPOINT_NAMES[self.shoulder_key]
        if float(confidence[shoulder_idx]) < 0.3:
            return None
        sc = shoulder_center(keypoints, confidence)
        if sc is None:
            return None
        axis = keypoints[shoulder_idx].astype(np.float64) - sc
        norm = float(np.linalg.norm(axis))
        if norm < 1e-6:
            return None
        return axis / norm

    def _compute_raw_metrics(
        self,
        kp_series: List[np.ndarray],
        conf_series: List[np.ndarray],
        frame_indices: List[int],
        swing: SwingEvent,
        forward_sign: float,
        forward_axis_hint: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        raw: Dict[str, Any] = {}
        if not frame_indices:
            return raw

        impact_frame = swing.impact_frame if swing.impact_frame is not None else frame_indices[len(frame_indices) // 2]
        prep_start = swing.prep_start_frame if swing.prep_start_frame is not None else frame_indices[0]
        ft_end = swing.followthrough_end_frame if swing.followthrough_end_frame is not None else frame_indices[-1]

        frame_to_pos = {frame: idx for idx, frame in enumerate(frame_indices)}
        impact_pos = frame_to_pos.get(impact_frame, len(frame_indices) // 2)
        prep_pos = frame_to_pos.get(prep_start, 0)
        ft_pos = frame_to_pos.get(ft_end, len(frame_indices) - 1)
        forward_swing_start = max(prep_pos, impact_pos - int(0.15 * self.fps))

        wrist_idx = KEYPOINT_NAMES[self.wrist_key]
        shoulder_angles: List[Optional[float]] = []
        knee_angles: List[float] = []
        spine_angles: List[float] = []

        for idx in range(prep_pos, min(impact_pos + 1, len(kp_series))):
            kp = kp_series[idx]
            conf = conf_series[idx]
            shoulder_angles.append(shoulder_line_angle(kp, conf))
            knee_angle = min_knee_angle(kp, conf)
            if knee_angle is not None:
                knee_angles.append(float(knee_angle))
            spine_angle = spine_angle_from_vertical(kp, conf)
            if spine_angle is not None:
                spine_angles.append(float(spine_angle))

        raw["shoulder_rotation_values"] = self._relative_rotation_series(shoulder_angles)
        raw["knee_angle_values"] = knee_angles
        raw["spine_angle_values"] = spine_angles

        forward_axis = None
        wrist_positions_fs = []
        for idx in range(forward_swing_start, min(impact_pos + 1, len(kp_series))):
            if float(conf_series[idx][wrist_idx]) >= 0.35:
                wrist_positions_fs.append(kp_series[idx][wrist_idx].astype(np.float64).copy())
        if len(wrist_positions_fs) >= 2:
            disp = wrist_positions_fs[-1] - wrist_positions_fs[0]
            norm = float(np.linalg.norm(disp))
            if norm > 3.0:
                forward_axis = disp / norm
        if forward_axis is None and forward_axis_hint is not None:
            axis = np.asarray(forward_axis_hint, dtype=np.float64).reshape(-1)[:2]
            norm = float(np.linalg.norm(axis))
            if norm > 1e-6:
                forward_axis = axis / norm
        raw["forward_axis"] = forward_axis.tolist() if forward_axis is not None else None

        chain_start = max(prep_pos, impact_pos - int(0.30 * self.fps))
        hip_angles_series: List[Optional[float]] = []
        shoulder_chain_series: List[Optional[float]] = []
        chain_frame_indices: List[int] = []
        hip_shoulder_seps: List[float] = []
        for idx in range(chain_start, min(impact_pos + 1, len(kp_series))):
            kp = kp_series[idx]
            conf = conf_series[idx]
            separation = shoulder_hip_angle(kp, conf)
            if separation is not None:
                hip_shoulder_seps.append(float(separation))
            hip_angles_series.append(hip_line_angle(kp, conf))
            shoulder_chain_series.append(shoulder_line_angle(kp, conf))
            chain_frame_indices.append(frame_indices[idx])
        raw["hip_shoulder_sep_values"] = hip_shoulder_seps
        raw["hip_shoulder_timing_s"] = hip_shoulder_separation_timing(
            hip_angles_series,
            shoulder_chain_series,
            self.fps,
            chain_frame_indices,
        )

        raw["wrist_positions_contact_zone"] = None
        raw["contact_forward_norm"] = None
        raw["contact_spacing_norm"] = None
        raw["elbow_angle_at_contact"] = None
        raw["head_displacement_norm"] = None
        raw["forward_extension_norm"] = None
        raw["outside_extension_norm"] = None

        if impact_pos < len(kp_series):
            kp_impact = kp_series[impact_pos]
            conf_impact = conf_series[impact_pos]
            th = torso_height_px(kp_impact, conf_impact)
            side_axis = self._dominant_side_axis(kp_impact, conf_impact)
            raw["side_axis"] = side_axis.tolist() if side_axis is not None else None

            contact_half_window = max(3, int(0.05 * self.fps))
            contact_forward_values: List[float] = []
            contact_spacing_values: List[float] = []
            elbow_angles_contact: List[float] = []
            for idx in range(max(0, impact_pos - contact_half_window), min(len(kp_series), impact_pos + contact_half_window + 1)):
                kp_window = kp_series[idx]
                conf_window = conf_series[idx]
                forward_val = wrist_forward_normalised(
                    kp_window,
                    conf_window,
                    is_right_handed=self.is_right_handed,
                    forward_sign=forward_sign,
                    forward_axis=forward_axis,
                )
                if forward_val is not None:
                    contact_forward_values.append(float(forward_val))
                elbow_val = elbow_angle(kp_window, conf_window, right=self.is_right_handed)
                if elbow_val is not None:
                    elbow_angles_contact.append(float(elbow_val))
                th_window = torso_height_px(kp_window, conf_window)
                if th_window is not None and th_window > 1e-6 and float(conf_window[wrist_idx]) >= 0.3:
                    sc = shoulder_center(kp_window, conf_window)
                    hc = hip_center(kp_window, conf_window)
                    if sc is not None and hc is not None:
                        wrist = kp_window[wrist_idx].astype(np.float64)
                        torso_clearance = self._point_to_line_distance(wrist, sc, hc)
                        if torso_clearance is not None:
                            contact_spacing_values.append(float(torso_clearance / th_window))

            raw["contact_forward_norm"] = float(np.median(contact_forward_values)) if contact_forward_values else None
            raw["contact_spacing_norm"] = float(np.median(contact_spacing_values)) if contact_spacing_values else None
            raw["elbow_angle_at_contact"] = float(np.median(elbow_angles_contact)) if elbow_angles_contact else None

            wrist_positions_contact_zone: List[np.ndarray] = []
            for idx in range(max(0, impact_pos - contact_half_window), min(len(kp_series), impact_pos + contact_half_window + 1)):
                if float(conf_series[idx][wrist_idx]) >= 0.3:
                    wrist_positions_contact_zone.append(kp_series[idx][wrist_idx].astype(np.float64).copy())
            if wrist_positions_contact_zone:
                raw["wrist_positions_contact_zone"] = np.asarray(wrist_positions_contact_zone, dtype=np.float64)

            head_points_contact: List[np.ndarray] = []
            for idx in range(max(0, impact_pos - 5), min(len(kp_series), impact_pos + 6)):
                head_point = self._relative_head_point(kp_series[idx], conf_series[idx])
                if head_point is not None:
                    head_points_contact.append(head_point)
            raw["head_displacement_norm"] = self._normalised_rms_displacement(head_points_contact, th)

            post_contact_frames = max(6, int(self.cfg.through.post_contact_window_s * self.fps))
            if float(conf_impact[wrist_idx]) >= 0.3 and th is not None and th > 1e-6:
                wrist_at_contact = kp_impact[wrist_idx].astype(np.float64).copy()
                max_forward = 0.0
                max_outside = 0.0
                valid_post_frames = 0
                for idx in range(impact_pos + 1, min(len(kp_series), impact_pos + post_contact_frames + 1)):
                    if float(conf_series[idx][wrist_idx]) < 0.3:
                        continue
                    valid_post_frames += 1
                    delta = kp_series[idx][wrist_idx].astype(np.float64) - wrist_at_contact
                    if forward_axis is not None:
                        max_forward = max(max_forward, float(np.dot(delta, forward_axis)))
                    else:
                        max_forward = max(max_forward, float(delta[0]) * forward_sign)
                    if side_axis is not None:
                        max_outside = max(max_outside, float(np.dot(delta, side_axis)))
                if valid_post_frames >= 3:
                    raw["forward_extension_norm"] = max_forward / th
                    raw["outside_extension_norm"] = max_outside / th if side_axis is not None else None
                raw["post_contact_valid_frames"] = valid_post_frames

        overall_head_points: List[np.ndarray] = []
        all_spine_angles: List[float] = []
        scale_frame_pos = impact_pos if impact_pos < len(kp_series) else prep_pos
        scale = torso_height_px(kp_series[scale_frame_pos], conf_series[scale_frame_pos]) if kp_series else None
        for idx in range(prep_pos, min(ft_pos + 1, len(kp_series))):
            head_point = self._relative_head_point(kp_series[idx], conf_series[idx])
            if head_point is not None:
                overall_head_points.append(head_point)
            spine_angle = spine_angle_from_vertical(kp_series[idx], conf_series[idx])
            if spine_angle is not None:
                all_spine_angles.append(float(spine_angle))

        raw["overall_head_displacement_norm"] = self._normalised_rms_displacement(overall_head_points, scale)
        raw["spine_angle_std"] = float(np.std(all_spine_angles)) if len(all_spine_angles) >= 3 else None
        raw["fps"] = self.fps
        return raw

    def _evaluate_kpis(self, raw: Dict[str, Any]) -> List[KPIResult]:
        return [
            ShoulderRotationKPI(self.cfg).evaluate(shoulder_rotation_values=raw.get("shoulder_rotation_values")),
            KneeBendKPI(self.cfg).evaluate(knee_angle_values=raw.get("knee_angle_values")),
            SpineAngleKPI(self.cfg).evaluate(spine_angle_values=raw.get("spine_angle_values")),
            HipShoulderSeparationKPI(self.cfg).evaluate(hip_shoulder_sep_values=raw.get("hip_shoulder_sep_values")),
            HipShoulderTimingKPI(self.cfg).evaluate(hip_shoulder_timing_s=raw.get("hip_shoulder_timing_s")),
            ContactPointKPI(self.cfg).evaluate(contact_forward_norm=raw.get("contact_forward_norm")),
            ContactSpacingKPI(self.cfg).evaluate(contact_spacing_norm=raw.get("contact_spacing_norm")),
            ElbowAngleAtContactKPI(self.cfg).evaluate(elbow_angle_at_contact=raw.get("elbow_angle_at_contact")),
            HandPathLinearityKPI(self.cfg).evaluate(wrist_positions_contact_zone=raw.get("wrist_positions_contact_zone")),
            ForwardExtensionKPI(self.cfg).evaluate(forward_extension_norm=raw.get("forward_extension_norm")),
            OutsideExtensionKPI(self.cfg).evaluate(outside_extension_norm=raw.get("outside_extension_norm")),
            HeadStabilityAtContactKPI(self.cfg).evaluate(head_displacement_norm=raw.get("head_displacement_norm")),
            OverallHeadStabilityKPI(self.cfg).evaluate(overall_head_displacement_norm=raw.get("overall_head_displacement_norm")),
            SpineConsistencyKPI(self.cfg).evaluate(spine_angle_std=raw.get("spine_angle_std")),
        ]

    def _aggregate_phases(self, kpi_results: List[KPIResult]) -> Dict[str, PhaseScore]:
        phase_map: Dict[str, List[KPIResult]] = {}
        for kpi in kpi_results:
            phase_map.setdefault(kpi.phase, []).append(kpi)

        phase_scores: Dict[str, PhaseScore] = {}
        for phase, kpis in phase_map.items():
            valid_scores = [kpi.score for kpi in kpis if kpi.rating not in ("无数据", "n/a")]
            avg_score = float(np.mean(valid_scores)) if valid_scores else 0.0
            phase_scores[phase] = PhaseScore(phase=phase, score=avg_score, kpis=kpis)
        return phase_scores

    def _compute_overall_score(self, phase_scores: Dict[str, PhaseScore]) -> float:
        weights = self.cfg.scoring.as_dict()
        total_weight = 0.0
        weighted_sum = 0.0
        for phase, weight in weights.items():
            if phase not in phase_scores:
                continue
            weighted_sum += phase_scores[phase].score * weight
            total_weight += weight
        return weighted_sum / max(total_weight, 1e-6)

    def _evaluate_no_impact(
        self,
        keypoints_series: List[np.ndarray],
        confidence_series: List[np.ndarray],
        frame_indices: List[int],
    ) -> SwingEvaluation:
        swing = SwingEvent(
            swing_index=0,
            prep_start_frame=frame_indices[0] if frame_indices else None,
            impact_frame=None,
            followthrough_end_frame=frame_indices[-1] if frame_indices else None,
        )

        shoulder_angles: List[Optional[float]] = []
        knee_angles: List[float] = []
        spine_angles: List[float] = []
        for kp, conf in zip(keypoints_series, confidence_series):
            shoulder_angles.append(shoulder_line_angle(kp, conf))
            knee_angle = min_knee_angle(kp, conf)
            if knee_angle is not None:
                knee_angles.append(float(knee_angle))
            spine_angle = spine_angle_from_vertical(kp, conf)
            if spine_angle is not None:
                spine_angles.append(float(spine_angle))

        raw = {
            "shoulder_rotation_values": self._relative_rotation_series(shoulder_angles),
            "knee_angle_values": knee_angles,
            "spine_angle_values": spine_angles,
        }
        kpi_results = self._evaluate_kpis(raw)
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
