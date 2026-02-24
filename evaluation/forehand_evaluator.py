"""Modern Forehand Evaluator — the orchestration layer.

``ForehandEvaluator`` is the "conductor" that:
    1. Receives the full per-frame keypoint time-series.
    2. Builds ``TrajectoryStore`` and detects impact events.
    3. Segments the swing into phases (preparation → contact → follow-through).
    4. Computes all raw biomechanical metrics for each phase.
    5. Feeds them into the KPI scorers.
    6. Aggregates results into a structured ``EvaluationReport``.
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
from evaluation.event_detector import ImpactDetector, SwingPhaseEstimator, SwingEvent
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


# ── Result containers ────────────────────────────────────────────────

@dataclass
class PhaseScore:
    phase: str
    score: float
    kpis: List[KPIResult]


@dataclass
class EvaluationReport:
    """Complete evaluation output for one swing."""
    swing_event: SwingEvent
    phase_scores: Dict[str, PhaseScore]
    overall_score: float
    kpi_results: List[KPIResult]
    forward_sign: float = 1.0
    arm_style: str = "unknown"
    raw_metrics: Dict[str, Any] = field(default_factory=dict)


class ForehandEvaluator:
    """Evaluate a single forehand swing from keypoint time-series data."""

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

    # ── Public API ────────────────────────────────────────────────────

    def evaluate(
        self,
        keypoints_series: List[np.ndarray],
        confidence_series: List[np.ndarray],
        frame_indices: Optional[List[int]] = None,
    ) -> EvaluationReport:
        """Run the full evaluation pipeline.

        Parameters
        ----------
        keypoints_series : list of (17, 2) arrays, one per frame.
        confidence_series : list of (17,) arrays, one per frame.
        frame_indices : optional list of frame indices (defaults to 0..N-1).

        Returns
        -------
        EvaluationReport
        """
        n_frames = len(keypoints_series)
        if frame_indices is None:
            frame_indices = list(range(n_frames))

        # 1. Build trajectory store
        store = TrajectoryStore(fps=self.fps)
        for kp, conf, fidx in zip(keypoints_series, confidence_series, frame_indices):
            store.update(kp, conf, fidx)

        # 2. Detect impact
        detector = ImpactDetector(fps=self.fps, is_right_handed=self.is_right_handed)
        for kp, conf, fidx in zip(keypoints_series, confidence_series, frame_indices):
            detector.update(fidx, kp, conf)

        if not detector.events:
            # No impact detected — evaluate what we can
            return self._evaluate_no_impact(keypoints_series, confidence_series, frame_indices, store)

        # Use the first (or strongest) impact
        impact_event = max(detector.events, key=lambda e: e.peak_speed_px_s)
        impact_frame = impact_event.impact_frame_idx

        # 3. Estimate swing phases
        wrist_traj = store.get(self.wrist_key)
        wrist_speeds = wrist_traj.get_speeds(smoothed=True)
        phase_estimator = SwingPhaseEstimator(fps=self.fps)
        swing_event = phase_estimator.estimate_phases(
            impact_frame, wrist_speeds, wrist_traj.frame_indices[1:]  # speeds are N-1
        )
        swing_event.impact_event = impact_event

        # 4. Infer forward direction from impact velocity
        forward_sign = 1.0
        if abs(impact_event.peak_velocity_unit[0]) > 0.1:
            forward_sign = 1.0 if impact_event.peak_velocity_unit[0] > 0 else -1.0

        # 5. Compute raw metrics per phase
        raw = self._compute_raw_metrics(
            keypoints_series, confidence_series, frame_indices,
            store, swing_event, forward_sign,
        )

        # 6. Evaluate all KPIs
        kpi_results = self._evaluate_kpis(raw, store, frame_indices)

        # 7. Aggregate into phase scores and overall score
        phase_scores = self._aggregate_phases(kpi_results)
        overall_score = self._compute_overall_score(phase_scores)

        # Determine arm style
        elbow_kpi = next((k for k in kpi_results if k.kpi_id == "C4.2"), None)
        arm_style = elbow_kpi.details.get("style", "unknown") if elbow_kpi and elbow_kpi.details else "unknown"

        return EvaluationReport(
            swing_event=swing_event,
            phase_scores=phase_scores,
            overall_score=overall_score,
            kpi_results=kpi_results,
            forward_sign=forward_sign,
            arm_style=arm_style,
            raw_metrics=raw,
        )

    # ── Internal: raw metric computation ─────────────────────────────

    def _compute_raw_metrics(
        self,
        kp_series, conf_series, frame_indices,
        store: TrajectoryStore,
        swing: SwingEvent,
        forward_sign: float,
    ) -> Dict[str, Any]:
        """Compute all raw biomechanical metrics needed by KPIs."""
        raw: Dict[str, Any] = {}
        impact_frame = swing.impact_frame
        prep_start = swing.prep_start_frame or frame_indices[0]
        ft_end = swing.followthrough_end_frame or frame_indices[-1]

        # Map frame index to array position
        f2p = {f: i for i, f in enumerate(frame_indices)}
        impact_pos = f2p.get(impact_frame, len(frame_indices) // 2)
        prep_pos = f2p.get(prep_start, 0)
        ft_pos = f2p.get(ft_end, len(frame_indices) - 1)

        # ── Preparation phase metrics ────────────────────────────────
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

        # ── Kinetic chain metrics ────────────────────────────────────
        # Peak speed frames for hip, shoulder, elbow, wrist
        hip_traj = store.get("right_hip" if self.is_right_handed else "left_hip")
        shoulder_traj = store.get(self.shoulder_key)
        elbow_traj = store.get(self.elbow_key)
        wrist_traj = store.get(self.wrist_key)

        raw["hip_peak_frame"] = hip_traj.peak_speed_frame()[0] if hip_traj.peak_speed_frame() else None
        raw["shoulder_peak_frame"] = shoulder_traj.peak_speed_frame()[0] if shoulder_traj.peak_speed_frame() else None
        raw["elbow_peak_frame"] = elbow_traj.peak_speed_frame()[0] if elbow_traj.peak_speed_frame() else None
        raw["wrist_peak_frame"] = wrist_traj.peak_speed_frame()[0] if wrist_traj.peak_speed_frame() else None

        # Hip-shoulder separation: compute the difference in rotation angles
        # during the forward swing (between prep and impact)
        hip_shoulder_seps = []
        for i in range(max(0, impact_pos - int(0.3 * self.fps)), impact_pos + 1):
            if i < len(kp_series):
                kp, conf = kp_series[i], conf_series[i]
                angle = shoulder_hip_angle(kp, conf)
                if angle is not None:
                    hip_shoulder_seps.append(angle)
        raw["hip_shoulder_sep_values"] = hip_shoulder_seps

        # Hand path linearity through contact zone (±5 frames around impact)
        contact_zone_half = max(3, int(0.05 * self.fps))
        wrist_positions_cz = []
        for i in range(max(0, impact_pos - contact_zone_half), min(len(kp_series), impact_pos + contact_zone_half + 1)):
            kp, conf = kp_series[i], conf_series[i]
            wrist_idx = KEYPOINT_NAMES[self.wrist_key]
            if conf[wrist_idx] >= 0.3:
                wrist_positions_cz.append(kp[wrist_idx].copy())
        raw["wrist_positions_contact_zone"] = np.array(wrist_positions_cz) if wrist_positions_cz else None

        # ── Contact metrics ──────────────────────────────────────────
        if impact_pos < len(kp_series):
            kp_impact = kp_series[impact_pos]
            conf_impact = conf_series[impact_pos]

            raw["contact_forward_norm"] = wrist_forward_normalised(
                kp_impact, conf_impact, self.is_right_handed, forward_sign
            )
            raw["elbow_angle_at_contact"] = elbow_angle(
                kp_impact, conf_impact, right=self.is_right_handed
            )

            # Body freeze: torso angular velocity at contact
            if impact_pos > 0 and impact_pos < len(kp_series) - 1:
                sr_before = shoulder_hip_angle(kp_series[impact_pos - 1], conf_series[impact_pos - 1])
                sr_after = shoulder_hip_angle(kp_series[min(impact_pos + 1, len(kp_series) - 1)],
                                              conf_series[min(impact_pos + 1, len(conf_series) - 1)])
                if sr_before is not None and sr_after is not None:
                    dt = 2.0 / self.fps
                    raw["torso_angular_velocity_at_contact"] = abs(sr_after - sr_before) / dt
                else:
                    raw["torso_angular_velocity_at_contact"] = None
            else:
                raw["torso_angular_velocity_at_contact"] = None

            # Head stability around contact (±5 frames)
            head_window = 5
            nose_positions = []
            for i in range(max(0, impact_pos - head_window), min(len(kp_series), impact_pos + head_window + 1)):
                np_pos = nose_position(kp_series[i], conf_series[i])
                if np_pos is not None:
                    nose_positions.append(np_pos)
            if len(nose_positions) >= 3:
                nose_arr = np.array(nose_positions)
                displacement = float(np.std(nose_arr[:, 1]))  # vertical std
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

        # ── Extension metrics ────────────────────────────────────────
        post_contact_frames = int(self.cfg.extension.post_contact_window_s * self.fps)
        if impact_pos < len(kp_series):
            wrist_idx = KEYPOINT_NAMES[self.wrist_key]
            wrist_at_contact = kp_series[impact_pos][wrist_idx].copy() if conf_series[impact_pos][wrist_idx] >= 0.3 else None

            if wrist_at_contact is not None:
                # Find the furthest forward point after contact
                max_forward = 0.0
                max_upward = 0.0
                for i in range(impact_pos + 1, min(len(kp_series), impact_pos + post_contact_frames + 1)):
                    if conf_series[i][wrist_idx] >= 0.3:
                        delta = kp_series[i][wrist_idx] - wrist_at_contact
                        forward_dist = delta[0] * forward_sign
                        upward_dist = -delta[1]  # image coords: y increases downward
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

        # ── Balance metrics (full swing) ─────────────────────────────
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
            th_ref = torso_height_px(kp_series[impact_pos], conf_series[impact_pos]) if impact_pos < len(kp_series) else None
            if th_ref and th_ref > 1e-6:
                raw["head_y_std_norm"] = float(np.std(nose_y_values)) / th_ref
            else:
                raw["head_y_std_norm"] = None
        else:
            raw["head_y_std_norm"] = None

        raw["spine_angle_std"] = float(np.std(spine_all)) if len(spine_all) >= 3 else None

        raw["fps"] = self.fps
        return raw

    # ── Internal: KPI evaluation ─────────────────────────────────────

    def _evaluate_kpis(self, raw: Dict[str, Any], store: TrajectoryStore, frame_indices: List[int]) -> List[KPIResult]:
        """Instantiate and evaluate all KPIs."""
        results = []

        # Phase 1: Preparation
        results.append(ShoulderRotationKPI(self.cfg).evaluate(
            shoulder_rotation_values=raw.get("shoulder_rotation_values", [])))
        results.append(KneeBendKPI(self.cfg).evaluate(
            knee_angle_values=raw.get("knee_angle_values", [])))
        results.append(SpineAngleKPI(self.cfg).evaluate(
            spine_angle_values=raw.get("spine_angle_values", [])))

        # Phase 3: Kinetic Chain
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

        # Phase 4: Contact
        results.append(ContactPointKPI(self.cfg).evaluate(
            contact_forward_norm=raw.get("contact_forward_norm")))
        results.append(ElbowAngleAtContactKPI(self.cfg).evaluate(
            elbow_angle_at_contact=raw.get("elbow_angle_at_contact")))
        results.append(BodyFreezeKPI(self.cfg).evaluate(
            torso_angular_velocity_at_contact=raw.get("torso_angular_velocity_at_contact")))
        results.append(HeadStabilityAtContactKPI(self.cfg).evaluate(
            head_displacement_norm=raw.get("head_displacement_norm")))

        # Phase 5: Extension
        results.append(ForwardExtensionKPI(self.cfg).evaluate(
            forward_extension_norm=raw.get("forward_extension_norm")))
        results.append(FollowThroughPathKPI(self.cfg).evaluate(
            upward_forward_ratio=raw.get("upward_forward_ratio")))

        # Phase 6: Balance
        results.append(OverallHeadStabilityKPI(self.cfg).evaluate(
            head_y_std_norm=raw.get("head_y_std_norm")))
        results.append(SpineConsistencyKPI(self.cfg).evaluate(
            spine_angle_std=raw.get("spine_angle_std")))

        return results

    # ── Internal: aggregation ────────────────────────────────────────

    def _aggregate_phases(self, kpi_results: List[KPIResult]) -> Dict[str, PhaseScore]:
        """Group KPIs by phase and compute phase-level scores."""
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
        """Weighted average of phase scores."""
        weights = self.cfg.scoring.as_dict()
        total_weight = 0.0
        weighted_sum = 0.0
        for phase, weight in weights.items():
            if phase in phase_scores:
                weighted_sum += phase_scores[phase].score * weight
                total_weight += weight
        return weighted_sum / max(total_weight, 1e-6)

    # ── Fallback for no-impact case ──────────────────────────────────

    def _evaluate_no_impact(self, kp_series, conf_series, frame_indices, store) -> EvaluationReport:
        """Evaluate what we can when no impact is detected."""
        swing = SwingEvent(
            prep_start_frame=frame_indices[0] if frame_indices else None,
            impact_frame=None,
            followthrough_end_frame=frame_indices[-1] if frame_indices else None,
        )

        # Compute preparation metrics over all frames
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

        # Add n/a results for other KPIs
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
                "No impact detected — cannot evaluate this metric."
            ))

        phase_scores = self._aggregate_phases(kpi_results)
        overall = self._compute_overall_score(phase_scores)

        return EvaluationReport(
            swing_event=swing,
            phase_scores=phase_scores,
            overall_score=overall,
            kpi_results=kpi_results,
            raw_metrics=raw,
        )
