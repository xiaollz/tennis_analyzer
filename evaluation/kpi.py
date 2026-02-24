"""关键绩效指标 (KPI) 定义 — Modern Forehand 评估。

每个 KPI 是一个独立的类：
    1. 接收相关数据（关键点、轨迹、帧索引）。
    2. 计算原始指标值。
    3. 使用 ``FrameworkConfig`` 中的阈值映射到 0-100 分。
    4. 返回 ``KPIResult``：分数、原始值、评级、中文反馈。

KPI 按挥拍阶段分组：
    阶段 1 – 准备：肩部旋转、膝盖弯曲、脊柱姿态
    阶段 3 – 动力链：动力链顺序、髋肩分离、手部路径线性度
    阶段 4 – 击球：击球点位置、肘部角度、身体刹车、头部稳定性
    阶段 5 – 延伸：前向延伸、随挥路径
    阶段 6 – 平衡：整体头部稳定性、脊柱一致性
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import numpy as np

from config.framework_config import FrameworkConfig, DEFAULT_CONFIG
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


# ── 结果容器 ────────────────────────────────────────────────────────

@dataclass
class KPIResult:
    """单个 KPI 的评估结果。"""
    kpi_id: str
    name: str
    phase: str
    raw_value: Optional[float]
    unit: str
    score: float          # 0-100
    rating: str           # "优秀", "良好", "一般", "较差", "无数据"
    feedback: str         # 中文教练反馈
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


def _linear_score(value: float, poor: float, good: float, excellent: float) -> float:
    """将值映射到 0-100 分（分段线性）。"""
    if excellent > poor:  # 越大越好
        if value >= excellent:
            return 100.0
        if value <= poor:
            return max(0.0, 20.0 * value / max(poor, 1e-6))
        if value >= good:
            return 70.0 + 30.0 * (value - good) / max(excellent - good, 1e-6)
        return 20.0 + 50.0 * (value - poor) / max(good - poor, 1e-6)
    else:  # 越小越好
        if value <= excellent:
            return 100.0
        if value >= poor:
            return max(0.0, 20.0 * (1.0 - value / max(poor, 1e-6)))
        if value <= good:
            return 70.0 + 30.0 * (good - value) / max(good - excellent, 1e-6)
        return 20.0 + 50.0 * (poor - value) / max(poor - good, 1e-6)


def _rating_from_score(score: float) -> str:
    if score >= 85:
        return "优秀"
    if score >= 65:
        return "良好"
    if score >= 40:
        return "一般"
    return "较差"


# ── 抽象基类 ────────────────────────────────────────────────────────

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


# =====================================================================
# 阶段 1：准备 & 转体
# =====================================================================

class ShoulderRotationKPI(BaseKPI):
    """P1.1 – 准备阶段最大肩部旋转（X-Factor）。"""
    kpi_id = "P1.1"
    name = "肩部旋转 (X-Factor)"
    phase = "preparation"
    unit = "度"

    def evaluate(self, *, shoulder_rotation_values: List[float], **kw) -> KPIResult:
        if not shoulder_rotation_values:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "无数据",
                             "肩部旋转数据不足，无法评估。")

        max_rot = float(max(shoulder_rotation_values))
        c = self.cfg.preparation
        score = _linear_score(max_rot, c.shoulder_rotation_poor, c.shoulder_rotation_good, c.shoulder_rotation_excellent)
        rating = _rating_from_score(score)

        if max_rot >= c.shoulder_rotation_excellent:
            fb = f"肩部转体出色（{max_rot:.0f}°），充分蓄力。背部几乎面向球网，这是现代正手的标志。"
        elif max_rot >= c.shoulder_rotation_good:
            fb = f"肩部转体良好（{max_rot:.0f}°）。尝试将肩膀再多转一些，目标是≥90°以获得最大蓄力。"
        else:
            fb = f"肩部转体不足（{max_rot:.0f}°）。需要加强整体转体（Unit Turn），肩膀相对于髋部应旋转≥90°。"

        return KPIResult(self.kpi_id, self.name, self.phase, max_rot, self.unit, score, rating, fb)


class KneeBendKPI(BaseKPI):
    """P1.4 – 准备阶段负重腿膝盖弯曲。"""
    kpi_id = "P1.4"
    name = "膝盖弯曲（蓄力）"
    phase = "preparation"
    unit = "度"

    def evaluate(self, *, knee_angle_values: List[float], **kw) -> KPIResult:
        if not knee_angle_values:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "无数据",
                             "膝盖弯曲数据不足，无法评估。")

        min_angle = float(min(knee_angle_values))
        c = self.cfg.preparation
        score = _linear_score(min_angle, c.knee_bend_poor, c.knee_bend_good, c.knee_bend_excellent)
        rating = _rating_from_score(score)

        if min_angle <= c.knee_bend_excellent:
            fb = f"膝盖弯曲充分（{min_angle:.0f}°），下肢蓄力出色。地面反作用力是力量的源泉。"
        elif min_angle <= c.knee_bend_good:
            fb = f"膝盖弯曲尚可（{min_angle:.0f}°）。尝试再蹲低一些，目标≤140°以获得更多地面力量。"
        else:
            fb = f"腿部过于僵直（{min_angle:.0f}°）。需要更多弯曲膝盖来蓄力，目标≤140°。"

        return KPIResult(self.kpi_id, self.name, self.phase, min_angle, self.unit, score, rating, fb)


class SpineAngleKPI(BaseKPI):
    """P1.3 – 脊柱姿态（偏离垂直角度）。"""
    kpi_id = "P1.3"
    name = "脊柱姿态"
    phase = "preparation"
    unit = "度"

    def evaluate(self, *, spine_angle_values: List[float], **kw) -> KPIResult:
        if not spine_angle_values:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "无数据",
                             "脊柱角度数据不足，无法评估。")

        avg_lean = float(np.mean(spine_angle_values))
        c = self.cfg.preparation
        score = _linear_score(avg_lean, c.spine_lean_warning, c.spine_lean_good_max, 5.0)
        rating = _rating_from_score(score)

        if avg_lean <= c.spine_lean_good_max:
            fb = f"脊柱姿态良好（平均倾斜{avg_lean:.1f}°），保持了直立的身体姿态。"
        else:
            fb = f"身体前倾过多（{avg_lean:.1f}°）。保持脊柱更直立，目标<{c.spine_lean_good_max:.0f}°。"

        return KPIResult(self.kpi_id, self.name, self.phase, avg_lean, self.unit, score, rating, fb)


# =====================================================================
# 阶段 3：动力链
# =====================================================================

class KineticChainSequenceKPI(BaseKPI):
    """KC3.1 – 动力链顺序：髋 → 肩 → 肘 → 腕。"""
    kpi_id = "KC3.1"
    name = "动力链顺序"
    phase = "kinetic_chain"
    unit = "顺序分"

    def evaluate(
        self,
        *,
        hip_peak_frame: Optional[int] = None,
        shoulder_peak_frame: Optional[int] = None,
        elbow_peak_frame: Optional[int] = None,
        wrist_peak_frame: Optional[int] = None,
        fps: float = 30.0,
        **kw,
    ) -> KPIResult:
        peaks = [
            ("髋", hip_peak_frame),
            ("肩", shoulder_peak_frame),
            ("肘", elbow_peak_frame),
            ("腕", wrist_peak_frame),
        ]
        available = [(name, f) for name, f in peaks if f is not None]
        if len(available) < 3:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "无数据",
                             "关节数据不足，无法评估动力链顺序。")

        correct_pairs = 0
        total_pairs = 0
        delays = []
        for i in range(len(available) - 1):
            total_pairs += 1
            delay_frames = available[i + 1][1] - available[i][1]
            delay_s = delay_frames / fps
            delays.append(delay_s)
            if delay_frames > 0:
                correct_pairs += 1

        seq_ratio = correct_pairs / max(total_pairs, 1)
        score = seq_ratio * 100.0
        rating = _rating_from_score(score)

        order_str = " → ".join(f"{n}(帧{f})" for n, f in available)
        if score >= 85:
            fb = f"动力链顺序正确：{order_str}。从近端到远端的依次发力模式良好。"
        elif score >= 50:
            fb = f"动力链部分正确：{order_str}。部分环节同时发力，需要加强髋部先行旋转。"
        else:
            fb = f"动力链顺序混乱：{order_str}。应遵循「腿→髋→躯干→手臂」的发力顺序。"

        return KPIResult(self.kpi_id, self.name, self.phase, score, self.unit, score, rating, fb,
                         details={"order": order_str, "delays_s": delays})


class HipShoulderSeparationKPI(BaseKPI):
    """KC3.2 – 前挥阶段最大髋肩分离角。"""
    kpi_id = "KC3.2"
    name = "髋肩分离角"
    phase = "kinetic_chain"
    unit = "度"

    def evaluate(self, *, hip_shoulder_sep_values: List[float], **kw) -> KPIResult:
        if not hip_shoulder_sep_values:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "无数据",
                             "髋肩分离数据不足，无法评估。")

        max_sep = float(max(hip_shoulder_sep_values))
        c = self.cfg.kinetic_chain
        score = _linear_score(max_sep, 10.0, c.hip_shoulder_separation_good, c.hip_shoulder_separation_excellent)
        rating = _rating_from_score(score)

        if max_sep >= c.hip_shoulder_separation_excellent:
            fb = f"髋肩分离角出色（{max_sep:.0f}°），X-Factor 拉伸充分。"
        elif max_sep >= c.hip_shoulder_separation_good:
            fb = f"髋肩分离角良好（{max_sep:.0f}°），髋部领先于肩部旋转。"
        else:
            fb = f"髋肩分离角不足（{max_sep:.0f}°）。让髋部先转，肩膀延迟跟随，制造「弹弓效应」。"

        return KPIResult(self.kpi_id, self.name, self.phase, max_sep, self.unit, score, rating, fb)


class HandPathLinearityKPI(BaseKPI):
    """KC3.4 – 击球区手部路径线性度。"""
    kpi_id = "KC3.4"
    name = "手部路径线性度"
    phase = "kinetic_chain"
    unit = "R²"

    def evaluate(self, *, wrist_positions_contact_zone: Optional[np.ndarray] = None, **kw) -> KPIResult:
        if wrist_positions_contact_zone is None or len(wrist_positions_contact_zone) < 3:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "无数据",
                             "击球区手部位置数据不足，无法评估。")

        pts = np.asarray(wrist_positions_contact_zone, dtype=np.float64)
        centroid = pts.mean(axis=0)
        centered = pts - centroid
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
        if s.sum() < 1e-6:
            r_squared = 1.0
        else:
            r_squared = float(s[0] ** 2 / (s ** 2).sum())

        c = self.cfg.kinetic_chain
        score = _linear_score(r_squared, 0.5, c.hand_path_linearity_good, c.hand_path_linearity_excellent)
        rating = _rating_from_score(score)

        if r_squared >= c.hand_path_linearity_excellent:
            fb = f"手部路径非常直线（R²={r_squared:.2f}），击球区挥拍路径干净利落。"
        elif r_squared >= c.hand_path_linearity_good:
            fb = f"手部路径较直（R²={r_squared:.2f}），有轻微弧度。"
        else:
            fb = f"手部路径弯曲过大（R²={r_squared:.2f}）。击球区应保持更直的挥拍路径，从内向外击球。"

        return KPIResult(self.kpi_id, self.name, self.phase, r_squared, self.unit, score, rating, fb)


# =====================================================================
# 阶段 4：击球
# =====================================================================

class ContactPointKPI(BaseKPI):
    """C4.1 – 击球点位置（手腕在髋部前方的距离）。"""
    kpi_id = "C4.1"
    name = "击球点位置"
    phase = "contact"
    unit = "躯干高度比"

    def evaluate(self, *, contact_forward_norm: Optional[float] = None, **kw) -> KPIResult:
        if contact_forward_norm is None:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "无数据",
                             "无法测量击球点位置。")

        c = self.cfg.contact
        val = contact_forward_norm
        if val < c.contact_forward_poor_min:
            score = max(0.0, 20.0 * val / max(c.contact_forward_poor_min, 1e-6))
        elif val < c.contact_forward_good_min:
            score = 20.0 + 50.0 * (val - c.contact_forward_poor_min) / max(c.contact_forward_good_min - c.contact_forward_poor_min, 1e-6)
        elif val <= c.contact_forward_good_max:
            score = 85.0
        else:
            overshoot = val - c.contact_forward_good_max
            score = max(40.0, 85.0 - overshoot * 100.0)

        score = float(np.clip(score, 0, 100))
        rating = _rating_from_score(score)

        if score >= 70:
            fb = f"击球点位置良好（{val:.2f}躯干高度），在身体前方充分击球。"
        elif val < c.contact_forward_good_min:
            fb = f"击球点过于靠近身体（{val:.2f}）。需要在更前方击球，手臂向前伸展。"
        else:
            fb = f"击球点过于靠前（{val:.2f}）。可能过度前伸，调整击球时机。"

        return KPIResult(self.kpi_id, self.name, self.phase, val, self.unit, score, rating, fb)


class ElbowAngleAtContactKPI(BaseKPI):
    """C4.2 – 击球时肘部角度。"""
    kpi_id = "C4.2"
    name = "击球时肘部角度"
    phase = "contact"
    unit = "度"

    def evaluate(self, *, elbow_angle_at_contact: Optional[float] = None, **kw) -> KPIResult:
        if elbow_angle_at_contact is None:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "无数据",
                             "无法测量击球时肘部角度。")

        c = self.cfg.contact
        angle = elbow_angle_at_contact

        in_straight = c.straight_arm_min <= angle <= c.straight_arm_max
        in_double_bend = c.double_bend_min <= angle <= c.double_bend_max

        if in_straight:
            score = 90.0
            style = "直臂型 (Gordon Type 3)"
            fb = f"直臂击球（{angle:.0f}°）— {style}风格。手臂充分伸展，符合现代正手标准。"
        elif in_double_bend:
            score = 90.0
            style = "双弯型"
            fb = f"双弯击球（{angle:.0f}°）— {style}风格。紧凑的手臂结构，控制力好。"
        elif c.double_bend_max < angle < c.straight_arm_min:
            score = 60.0
            style = "过渡型"
            fb = f"肘部角度{angle:.0f}°，介于双弯和直臂之间。建议选择一种风格并坚持练习。"
        elif angle < c.double_bend_min:
            score = 30.0
            style = "过度弯曲"
            fb = f"击球时手臂过度弯曲（{angle:.0f}°）。需要更多伸展，避免「夹臂」击球。"
        else:
            score = 70.0
            style = "过度伸展"
            fb = f"击球时肘部角度{angle:.0f}°，基本正常。"

        rating = _rating_from_score(score)
        return KPIResult(self.kpi_id, self.name, self.phase, angle, self.unit, score, rating, fb,
                         details={"style": style})


class BodyFreezeKPI(BaseKPI):
    """C4.3 – 击球时躯干角速度（应接近零 = 「刹车」）。"""
    kpi_id = "C4.3"
    name = "身体刹车"
    phase = "contact"
    unit = "度/秒"

    def evaluate(self, *, torso_angular_velocity_at_contact: Optional[float] = None, **kw) -> KPIResult:
        if torso_angular_velocity_at_contact is None:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "无数据",
                             "无法测量击球时躯干角速度。")

        c = self.cfg.contact
        val = abs(torso_angular_velocity_at_contact)
        score = _linear_score(val, c.body_freeze_warning, c.body_freeze_good_max, 10.0)
        rating = _rating_from_score(score)

        if val <= c.body_freeze_good_max:
            fb = f"击球时身体刹车良好（{val:.0f}°/s），胸部停止旋转提供了稳定的击球平台。"
        else:
            fb = f"击球时身体仍在旋转（{val:.0f}°/s）。尝试在击球瞬间「刹住」躯干，胸部面向目标后停止旋转。"

        return KPIResult(self.kpi_id, self.name, self.phase, val, self.unit, score, rating, fb)


class HeadStabilityAtContactKPI(BaseKPI):
    """C4.4 – 击球点附近头部稳定性。"""
    kpi_id = "C4.4"
    name = "击球时头部稳定性"
    phase = "contact"
    unit = "归一化位移"

    def evaluate(self, *, head_displacement_norm: Optional[float] = None, **kw) -> KPIResult:
        if head_displacement_norm is None:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "无数据",
                             "无法测量头部稳定性。")

        c = self.cfg.contact
        val = head_displacement_norm
        score = _linear_score(val, c.head_stability_warning, c.head_stability_good_max, 0.02)
        rating = _rating_from_score(score)

        if val <= c.head_stability_good_max:
            fb = f"头部稳定性出色（{val:.3f}），眼睛始终注视击球点。"
        else:
            fb = f"击球时头部移动过大（{val:.3f}）。保持眼睛注视击球点，头部应该是最后移动的部位。"

        return KPIResult(self.kpi_id, self.name, self.phase, val, self.unit, score, rating, fb)


# =====================================================================
# 阶段 5：延伸 & 随挥
# =====================================================================

class ForwardExtensionKPI(BaseKPI):
    """E5.1 – 击球后前向延伸距离。"""
    kpi_id = "E5.1"
    name = "前向延伸"
    phase = "extension"
    unit = "躯干高度比"

    def evaluate(self, *, forward_extension_norm: Optional[float] = None, **kw) -> KPIResult:
        if forward_extension_norm is None:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "无数据",
                             "无法测量前向延伸距离。")

        c = self.cfg.extension
        val = forward_extension_norm
        score = _linear_score(val, 0.1, c.forward_extension_good, c.forward_extension_excellent)
        rating = _rating_from_score(score)

        if val >= c.forward_extension_excellent:
            fb = f"前向延伸出色（{val:.2f}躯干高度），击球后充分穿透球体。"
        elif val >= c.forward_extension_good:
            fb = f"前向延伸良好（{val:.2f}）。继续向目标方向推送球拍。"
        else:
            fb = f"前向延伸不足（{val:.2f}）。击球后手臂应继续向目标方向延伸60-90厘米。"

        return KPIResult(self.kpi_id, self.name, self.phase, val, self.unit, score, rating, fb)


class FollowThroughPathKPI(BaseKPI):
    """E5.2 – 随挥路径：上升/前进比例。"""
    kpi_id = "E5.2"
    name = "随挥路径"
    phase = "extension"
    unit = "比值"

    def evaluate(self, *, upward_forward_ratio: Optional[float] = None, **kw) -> KPIResult:
        if upward_forward_ratio is None:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "无数据",
                             "无法测量随挥路径。")

        c = self.cfg.extension
        val = upward_forward_ratio
        score = _linear_score(val, c.followthrough_upward_forward_warning, c.followthrough_upward_forward_good_max, 0.5)
        rating = _rating_from_score(score)

        if val <= c.followthrough_upward_forward_good_max:
            fb = f"随挥路径平衡（上升/前进比={val:.2f}），先前伸再上升的模式正确。"
        else:
            fb = f"随挥上升过快（比值={val:.2f}）。击球后先向前延伸，再让球拍自然通过肩部旋转上升。"

        return KPIResult(self.kpi_id, self.name, self.phase, val, self.unit, score, rating, fb)


# =====================================================================
# 阶段 6：平衡
# =====================================================================

class OverallHeadStabilityKPI(BaseKPI):
    """B6.1 – 整个挥拍过程中头部垂直稳定性。"""
    kpi_id = "B6.1"
    name = "整体头部稳定性"
    phase = "balance"
    unit = "归一化标准差"

    def evaluate(self, *, head_y_std_norm: Optional[float] = None, **kw) -> KPIResult:
        if head_y_std_norm is None:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "无数据",
                             "无法测量整体头部稳定性。")

        c = self.cfg.balance
        val = head_y_std_norm
        score = _linear_score(val, c.head_vertical_stability_warning, c.head_vertical_stability_good, 0.01)
        rating = _rating_from_score(score)

        if val <= c.head_vertical_stability_good:
            fb = f"整个挥拍过程中头部稳定性出色（标准差={val:.3f}）。"
        else:
            fb = f"挥拍过程中头部上下跳动（标准差={val:.3f}）。保持头部高度一致，避免起伏。"

        return KPIResult(self.kpi_id, self.name, self.phase, val, self.unit, score, rating, fb)


class SpineConsistencyKPI(BaseKPI):
    """B6.2 – 挥拍过程中脊柱角度一致性。"""
    kpi_id = "B6.2"
    name = "脊柱一致性"
    phase = "balance"
    unit = "度标准差"

    def evaluate(self, *, spine_angle_std: Optional[float] = None, **kw) -> KPIResult:
        if spine_angle_std is None:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "无数据",
                             "无法测量脊柱一致性。")

        c = self.cfg.balance
        val = spine_angle_std
        score = _linear_score(val, c.spine_consistency_warning, c.spine_consistency_good, 2.0)
        rating = _rating_from_score(score)

        if val <= c.spine_consistency_good:
            fb = f"脊柱姿态一致（标准差={val:.1f}°），身体控制良好。"
        else:
            fb = f"脊柱角度变化过大（标准差={val:.1f}°）。保持躯干稳定，避免挥拍时弯腰或前倾。"

        return KPIResult(self.kpi_id, self.name, self.phase, val, self.unit, score, rating, fb)


# =====================================================================
# KPI 注册表
# =====================================================================

ALL_KPIS = [
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
]
