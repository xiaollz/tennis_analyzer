"""Fault-tolerant forehand KPI definitions.

The KPI set in this module is intentionally conservative: it only scores
principles that can be estimated with reasonable stability from COCO-17 body
keypoints. True racket-face angle, racket/forearm angle, wrist relaxation, and
explicit racket-head lag are therefore not scored directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from config.framework_config import DEFAULT_CONFIG, FrameworkConfig


@dataclass
class KPIResult:
    kpi_id: str
    name: str
    phase: str
    raw_value: Optional[float]
    unit: str
    score: float
    rating: str
    feedback: str
    drill: str = ""
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


def _linear_score(value: float, poor: float, good: float, excellent: float) -> float:
    if excellent > poor:
        if value >= excellent:
            return 100.0
        if value <= poor:
            return max(0.0, 20.0 * value / max(poor, 1e-6))
        if value >= good:
            return 70.0 + 30.0 * (value - good) / max(excellent - good, 1e-6)
        return 20.0 + 50.0 * (value - poor) / max(good - poor, 1e-6)

    if value <= excellent:
        return 100.0
    if value >= poor:
        return max(0.0, 20.0 * (1.0 - value / max(poor, 1e-6)))
    if value <= good:
        return 70.0 + 30.0 * (good - value) / max(good - excellent, 1e-6)
    return 20.0 + 50.0 * (poor - value) / max(poor - good, 1e-6)


def _band_score(
    value: float,
    poor_min: float,
    good_min: float,
    excellent_min: float,
    excellent_max: float,
    good_max: float,
    poor_max: float,
) -> float:
    if excellent_min <= value <= excellent_max:
        return 100.0
    if good_min <= value < excellent_min:
        return 75.0 + 25.0 * (value - good_min) / max(excellent_min - good_min, 1e-6)
    if excellent_max < value <= good_max:
        return 75.0 + 25.0 * (good_max - value) / max(good_max - excellent_max, 1e-6)
    if poor_min <= value < good_min:
        return 20.0 + 55.0 * (value - poor_min) / max(good_min - poor_min, 1e-6)
    if good_max < value <= poor_max:
        return 20.0 + 55.0 * (poor_max - value) / max(poor_max - good_max, 1e-6)
    if value < poor_min:
        return max(0.0, 20.0 * value / max(poor_min, 1e-6))
    overflow = value - poor_max
    return max(0.0, 20.0 * (1.0 - overflow / max(poor_max, 1e-6)))


def _rating_from_score(score: float) -> str:
    if score >= 85:
        return "优秀"
    if score >= 65:
        return "良好"
    if score >= 40:
        return "一般"
    return "较差"


class BaseKPI(ABC):
    kpi_id: str = ""
    name: str = ""
    phase: str = ""
    unit: str = ""

    def __init__(self, cfg: FrameworkConfig = DEFAULT_CONFIG):
        self.cfg = cfg

    @abstractmethod
    def evaluate(self, **kwargs) -> KPIResult:
        ...


class ShoulderRotationKPI(BaseKPI):
    kpi_id = "UT1.1"
    name = "肩部转开幅度"
    phase = "unit_turn"
    unit = "度"

    def evaluate(self, *, shoulder_rotation_values: Optional[List[float]] = None, **kwargs) -> KPIResult:
        values = [float(v) for v in (shoulder_rotation_values or [])]
        if not values:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "无数据", "肩部转开数据不足，无法评估。")

        val = float(max(values))
        c = self.cfg.unit_turn
        score = _linear_score(val, c.shoulder_rotation_poor, c.shoulder_rotation_good, c.shoulder_rotation_excellent)
        rating = _rating_from_score(score)
        if val >= c.shoulder_rotation_excellent:
            fb = f"转开完整（{val:.0f}°），准备阶段有足够的上身储能。"
        elif val >= c.shoulder_rotation_good:
            fb = f"转开尚可（{val:.0f}°），再简洁完整一点会更稳。"
        else:
            fb = f"转开不足（{val:.0f}°），容易变成只用手把拍子带走。"
        return KPIResult(self.kpi_id, self.name, self.phase, val, self.unit, score, rating, fb)


class KneeBendKPI(BaseKPI):
    kpi_id = "UT1.2"
    name = "下肢承载"
    phase = "unit_turn"
    unit = "度"

    def evaluate(self, *, knee_angle_values: Optional[List[float]] = None, **kwargs) -> KPIResult:
        values = [float(v) for v in (knee_angle_values or [])]
        if not values:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "无数据", "膝部数据不足，无法评估。")

        val = float(min(values))
        c = self.cfg.unit_turn
        score = _linear_score(val, c.knee_bend_poor, c.knee_bend_good, c.knee_bend_excellent)
        rating = _rating_from_score(score)
        if val <= c.knee_bend_excellent:
            fb = f"下肢承载良好（最小膝角 {val:.0f}°），更容易用腿和躯干处理来球高度。"
        elif val <= c.knee_bend_good:
            fb = f"下肢参与尚可（最小膝角 {val:.0f}°）。"
        else:
            fb = f"腿部承载偏弱（最小膝角 {val:.0f}°），来球低或来不及时更容易用手补。"
        return KPIResult(self.kpi_id, self.name, self.phase, val, self.unit, score, rating, fb)


class SpineAngleKPI(BaseKPI):
    kpi_id = "UT1.3"
    name = "脊柱姿态"
    phase = "unit_turn"
    unit = "度"

    def evaluate(self, *, spine_angle_values: Optional[List[float]] = None, **kwargs) -> KPIResult:
        values = [float(v) for v in (spine_angle_values or [])]
        if not values:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "无数据", "脊柱姿态数据不足，无法评估。")

        val = float(np.mean(values))
        c = self.cfg.unit_turn
        score = _linear_score(val, c.spine_lean_warning, c.spine_lean_good_max, 4.0)
        rating = _rating_from_score(score)
        if val <= c.spine_lean_good_max:
            fb = f"脊柱姿态稳定（平均倾斜 {val:.1f}°），准备阶段结构干净。"
        else:
            fb = f"脊柱倾斜偏大（平均 {val:.1f}°），建议更多用腿调高度，不要先折腰或伸手找球。"
        return KPIResult(self.kpi_id, self.name, self.phase, val, self.unit, score, rating, fb)


class HipShoulderSeparationKPI(BaseKPI):
    kpi_id = "KC2.1"
    name = "髋肩分离"
    phase = "chain"
    unit = "度"

    def evaluate(self, *, hip_shoulder_sep_values: Optional[List[float]] = None, **kwargs) -> KPIResult:
        values = [float(v) for v in (hip_shoulder_sep_values or [])]
        if not values:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "无数据", "髋肩分离数据不足，无法评估。")

        val = float(max(values))
        c = self.cfg.chain
        score = _linear_score(val, c.hip_shoulder_separation_poor, c.hip_shoulder_separation_good, c.hip_shoulder_separation_excellent)
        rating = _rating_from_score(score)
        if val >= c.hip_shoulder_separation_excellent:
            fb = f"髋肩分离充分（{val:.0f}°），转髋带肩的弹性空间较好。"
        elif val >= c.hip_shoulder_separation_good:
            fb = f"髋肩分离尚可（{val:.0f}°）。"
        else:
            fb = f"髋肩分离不足（{val:.0f}°），容易变成髋肩一起转，前挥缺少层次。"
        return KPIResult(self.kpi_id, self.name, self.phase, val, self.unit, score, rating, fb)


class HipShoulderTimingKPI(BaseKPI):
    kpi_id = "KC2.2"
    name = "转髋领先顺序"
    phase = "chain"
    unit = "秒"

    def evaluate(self, *, hip_shoulder_timing_s: Optional[float] = None, **kwargs) -> KPIResult:
        if hip_shoulder_timing_s is None:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "无数据", "髋肩先后顺序数据不足，无法评估。")

        val = float(hip_shoulder_timing_s)
        c = self.cfg.chain
        score = _linear_score(val, c.hip_lead_timing_poor, c.hip_lead_timing_good, c.hip_lead_timing_excellent)
        rating = _rating_from_score(score)
        if val >= c.hip_lead_timing_good:
            fb = f"转髋先于上身（约 {val * 1000:.0f} ms），前挥顺序合理。"
        elif val >= 0.0:
            fb = f"髋部略领先肩部（约 {val * 1000:.0f} ms），顺序基本正确。"
        else:
            fb = f"肩部没有明显被髋部带动（约 {val * 1000:.0f} ms），前挥更像一起拉。"
        return KPIResult(self.kpi_id, self.name, self.phase, val, self.unit, score, rating, fb)


class ContactPointKPI(BaseKPI):
    kpi_id = "C3.1"
    name = "前方击球点"
    phase = "contact"
    unit = "归一化前向距离"

    def evaluate(self, *, contact_forward_norm: Optional[float] = None, **kwargs) -> KPIResult:
        if contact_forward_norm is None:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "无数据", "击球点前向距离不足，无法评估。")

        val = float(contact_forward_norm)
        c = self.cfg.contact
        score = _band_score(
            val,
            c.contact_forward_poor_min,
            c.contact_forward_good_min,
            c.contact_forward_excellent_min,
            c.contact_forward_excellent_max,
            c.contact_forward_good_max,
            c.contact_forward_good_max + 0.60,
        )
        rating = _rating_from_score(score)
        if c.contact_forward_excellent_min <= val <= c.contact_forward_excellent_max:
            fb = f"击球点在身体前方且余量充足（{val:.2f}）。"
        elif val < c.contact_forward_good_min:
            fb = f"击球点偏晚偏挤（{val:.2f}），书里的“始终尽量在前方接球”没有做到位。"
        else:
            fb = f"击球点略偏远（{val:.2f}），可以更自然地在身体前方完成接触。"
        return KPIResult(self.kpi_id, self.name, self.phase, val, self.unit, score, rating, fb)


class ContactSpacingKPI(BaseKPI):
    kpi_id = "C3.2"
    name = "击球间距"
    phase = "contact"
    unit = "归一化离躯干距离"

    def evaluate(self, *, contact_spacing_norm: Optional[float] = None, **kwargs) -> KPIResult:
        if contact_spacing_norm is None:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "无数据", "击球间距数据不足，无法评估。")

        val = float(contact_spacing_norm)
        c = self.cfg.contact
        score = _band_score(
            val,
            c.contact_spacing_poor_min,
            c.contact_spacing_good_min,
            c.contact_spacing_excellent_min,
            c.contact_spacing_excellent_max,
            c.contact_spacing_good_max,
            c.contact_spacing_poor_max,
        )
        rating = _rating_from_score(score)
        if c.contact_spacing_excellent_min <= val <= c.contact_spacing_excellent_max:
            fb = f"离躯干空间合理（{val:.2f}），没有被球挤住，也没有明显够球。"
        elif val < c.contact_spacing_good_min:
            fb = f"离躯干空间偏小（{val:.2f}），击球更容易贴身发力。"
        else:
            fb = f"离躯干空间偏大（{val:.2f}），容易把手臂伸得过远。"
        return KPIResult(self.kpi_id, self.name, self.phase, val, self.unit, score, rating, fb)


class ElbowAngleAtContactKPI(BaseKPI):
    kpi_id = "C3.3"
    name = "击球时手臂结构"
    phase = "contact"
    unit = "度"

    def evaluate(self, *, elbow_angle_at_contact: Optional[float] = None, **kwargs) -> KPIResult:
        if elbow_angle_at_contact is None:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "无数据", "肘部角度数据不足，无法评估。")

        angle = float(elbow_angle_at_contact)
        if angle < 60.0 or angle > 178.0:
            return KPIResult(
                self.kpi_id,
                self.name,
                self.phase,
                None,
                self.unit,
                0,
                "无数据",
                f"肘部接触角度疑似受遮挡或单帧抖动影响（原始值 {angle:.0f}°），本次不纳入评分。",
            )

        c = self.cfg.contact
        score = _band_score(
            angle,
            c.elbow_angle_poor_min,
            c.elbow_angle_good_min,
            c.elbow_angle_excellent_min,
            c.elbow_angle_excellent_max,
            c.elbow_angle_good_max,
            c.elbow_angle_poor_max,
        )
        rating = _rating_from_score(score)
        if angle >= 160.0:
            style = "较直臂型"
        elif angle <= 135.0:
            style = "双弯型"
        else:
            style = "中间型"

        if c.elbow_angle_good_min <= angle <= c.elbow_angle_good_max:
            fb = f"手臂结构处在可工作的区间（{angle:.0f}°，{style}）。重点是稳定，而不是刻意追求某一种职业外形。"
        elif angle < c.elbow_angle_good_min:
            fb = f"肘部过于收缩（{angle:.0f}°），击球更容易贴身发力。"
        else:
            fb = f"肘部接近锁死（{angle:.0f}°），对失位球和时机波动的容错会下降。"
        return KPIResult(self.kpi_id, self.name, self.phase, angle, self.unit, score, rating, fb, details={"style": style})


class HandPathLinearityKPI(BaseKPI):
    kpi_id = "T4.1"
    name = "击球区手路径"
    phase = "through"
    unit = "R²"

    def evaluate(self, *, wrist_positions_contact_zone: Optional[np.ndarray] = None, **kwargs) -> KPIResult:
        if wrist_positions_contact_zone is None:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "无数据", "击球区手路径数据不足，无法评估。")

        pts = np.asarray(wrist_positions_contact_zone, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] < 2:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "无数据", "击球区手路径数据不足，无法评估。")

        centered = pts[:, :2] - np.mean(pts[:, :2], axis=0, keepdims=True)
        _, svals, _ = np.linalg.svd(centered, full_matrices=False)
        denom = float(np.sum(svals ** 2))
        if denom < 1e-6:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "无数据", "击球区手路径变化太小，无法评估。")
        r_squared = float((svals[0] ** 2) / denom)

        c = self.cfg.through
        score = _linear_score(r_squared, c.hand_path_linearity_poor, c.hand_path_linearity_good, c.hand_path_linearity_excellent)
        rating = _rating_from_score(score)
        if r_squared >= c.hand_path_linearity_excellent:
            fb = f"击球区手路径非常干净（R²={r_squared:.2f}），更符合“向前穿过球”的原则。"
        elif r_squared >= c.hand_path_linearity_good:
            fb = f"击球区手路径基本稳定（R²={r_squared:.2f}）。"
        else:
            fb = f"击球区手路径偏散（R²={r_squared:.2f}），接触区不够可重复。"
        return KPIResult(self.kpi_id, self.name, self.phase, r_squared, self.unit, score, rating, fb)


class ForwardExtensionKPI(BaseKPI):
    kpi_id = "T4.2"
    name = "向前穿透"
    phase = "through"
    unit = "归一化距离"

    def evaluate(self, *, forward_extension_norm: Optional[float] = None, **kwargs) -> KPIResult:
        if forward_extension_norm is None:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "无数据", "向前穿透数据不足，无法评估。")

        val = float(forward_extension_norm)
        c = self.cfg.through
        score = _linear_score(val, c.forward_extension_poor, c.forward_extension_good, c.forward_extension_excellent)
        rating = _rating_from_score(score)
        if val >= c.forward_extension_excellent:
            fb = f"向前穿透充分（{val:.2f}），不是一碰到球就急着收。"
        elif val >= c.forward_extension_good:
            fb = f"向前穿透尚可（{val:.2f}）。"
        else:
            fb = f"向前穿透不足（{val:.2f}），更像只打到球，没有“through the ball”。"
        return KPIResult(self.kpi_id, self.name, self.phase, val, self.unit, score, rating, fb)


class OutsideExtensionKPI(BaseKPI):
    kpi_id = "T4.3"
    name = "向外送拍"
    phase = "through"
    unit = "归一化距离"

    def evaluate(self, *, outside_extension_norm: Optional[float] = None, **kwargs) -> KPIResult:
        if outside_extension_norm is None:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "无数据", "向外送拍数据不足，无法评估。")

        val = float(outside_extension_norm)
        c = self.cfg.through
        score = _linear_score(val, c.outside_extension_poor, c.outside_extension_good, c.outside_extension_excellent)
        rating = _rating_from_score(score)
        if val >= c.outside_extension_excellent:
            fb = f"向外送拍明显（{val:.2f}），更符合书里“out”的原则。"
        elif val >= c.outside_extension_good:
            fb = f"向外送拍尚可（{val:.2f}）。"
        else:
            fb = f"向外送拍不足（{val:.2f}），手更容易被身体卷回来。"
        return KPIResult(self.kpi_id, self.name, self.phase, val, self.unit, score, rating, fb)


class HeadStabilityAtContactKPI(BaseKPI):
    kpi_id = "S5.1"
    name = "击球时头部稳定"
    phase = "stability"
    unit = "归一化位移"

    def evaluate(self, *, head_displacement_norm: Optional[float] = None, **kwargs) -> KPIResult:
        if head_displacement_norm is None:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "无数据", "击球点附近头部稳定性不足，无法评估。")

        val = float(head_displacement_norm)
        c = self.cfg.stability
        score = _linear_score(val, c.head_contact_warning, c.head_contact_good, 0.015)
        rating = _rating_from_score(score)
        if val <= c.head_contact_good:
            fb = f"击球点附近头部相对躯干很稳定（{val:.3f}），更利于最后时刻的微调。"
        else:
            fb = f"击球点附近头部相对躯干漂移偏大（{val:.3f}），容易破坏接触一致性。"
        return KPIResult(self.kpi_id, self.name, self.phase, val, self.unit, score, rating, fb)


class OverallHeadStabilityKPI(BaseKPI):
    kpi_id = "S5.2"
    name = "全过程头部稳定"
    phase = "stability"
    unit = "归一化位移"

    def evaluate(self, *, overall_head_displacement_norm: Optional[float] = None, **kwargs) -> KPIResult:
        if overall_head_displacement_norm is None:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "无数据", "全过程头部稳定性不足，无法评估。")

        val = float(overall_head_displacement_norm)
        c = self.cfg.stability
        score = _linear_score(val, c.head_vertical_stability_warning, c.head_vertical_stability_good, 0.012)
        rating = _rating_from_score(score)
        if val <= c.head_vertical_stability_good:
            fb = f"全过程头部相对躯干控制良好（{val:.3f}）。"
        else:
            fb = f"挥拍全过程头部相对躯干摆动偏大（{val:.3f}），容错会下降。"
        return KPIResult(self.kpi_id, self.name, self.phase, val, self.unit, score, rating, fb)


class SpineConsistencyKPI(BaseKPI):
    kpi_id = "S5.3"
    name = "脊柱一致性"
    phase = "stability"
    unit = "度标准差"

    def evaluate(self, *, spine_angle_std: Optional[float] = None, **kwargs) -> KPIResult:
        if spine_angle_std is None:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "无数据", "脊柱一致性数据不足，无法评估。")

        val = float(spine_angle_std)
        c = self.cfg.stability
        score = _linear_score(val, c.spine_consistency_warning, c.spine_consistency_good, 2.0)
        rating = _rating_from_score(score)
        if val <= c.spine_consistency_good:
            fb = f"脊柱姿态一致（标准差 {val:.1f}°），动作更可重复。"
        else:
            fb = f"脊柱姿态波动偏大（标准差 {val:.1f}°），来球一难时更容易散。"
        return KPIResult(self.kpi_id, self.name, self.phase, val, self.unit, score, rating, fb)


# Reliable KPIs only — verified against VLM analysis for consistency.
# Removed: HipShoulderTimingKPI (unmeasurable at 30fps),
#          SpineAngleKPI (tilt can be good per FTT),
#          HandPathLinearityKPI (Out path isn't linear),
#          ContactSpacingKPI (contradictory scoring),
#          OverallHeadStabilityKPI (duplicate of S5.1),
#          HipShoulderSeparationKPI (2D projection too noisy).
ALL_KPIS = [
    ShoulderRotationKPI,       # UT1.1 — direction consistent with VLM
    KneeBendKPI,               # UT1.2 — reliable, matches VLM
    ElbowAngleAtContactKPI,    # C3.3 — reliable, matches VLM
    ContactPointKPI,           # C3.1 — moderate reliability
    # OutsideExtensionKPI removed: Out vector's main component is Z-axis
    # (depth), which is invisible in 2D side view. Use M7 swing_shape_label
    # (trajectory arc shape) as the Out vector proxy instead.
    ForwardExtensionKPI,       # T4.2 — Through vector, matches VLM
    HeadStabilityAtContactKPI, # S5.1 — reliable, matches VLM
    SpineConsistencyKPI,       # S5.3 — reliable
]
