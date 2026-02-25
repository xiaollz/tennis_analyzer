"""关键绩效指标 (KPI) 定义 — One-Handed Backhand (单反) 评估。

每个 KPI 是一个独立的类，接收相关数据并返回 ``KPIResult``。

KPI 按挥拍阶段分组：
    阶段 1 – 准备：肩部旋转、膝盖弯曲、非持拍手辅助、脊柱姿态
    阶段 2 – 引拍：L形杠杆（肘部角度）
    阶段 3 – 动力链：动力链顺序、髋肩分离、Inside-Out路径
    阶段 4 – 击球：击球点位置、手臂伸展、身体刹车、非持拍手反向平衡、头部稳定性
    阶段 5 – 延伸：ATA收拍高度、保持侧身
    阶段 6 – 平衡：整体头部稳定性、脊柱一致性

Sources:
    - Tennis Doctor: L-for-Leverage, ATA (Air The Armpit), shoulder 180° motion
    - Feel Tennis: Double 45, forearm rotation, body brake
    - Tennisnerd 2: Inside-out path, stay sideways
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import numpy as np

from config.backhand_config import BackhandConfig, DEFAULT_BACKHAND_CONFIG
from evaluation.kpi import KPIResult, _linear_score, _rating_from_score


# ── 抽象基类 ────────────────────────────────────────────────────────

class BaseBackhandKPI(ABC):
    kpi_id: str = ""
    name: str = ""
    phase: str = ""
    unit: str = ""

    def __init__(self, cfg: BackhandConfig = DEFAULT_BACKHAND_CONFIG):
        self.cfg = cfg

    @abstractmethod
    def evaluate(self, **kwargs) -> KPIResult:
        ...


# =====================================================================
# 阶段 1：准备 & 转体
# =====================================================================

class OHB_ShoulderRotationKPI(BaseBackhandKPI):
    """P1.1 – 准备阶段最大肩部旋转。

    单反需要更充分的转体，理想情况下背部接近面向球网（>90°）。
    """
    kpi_id = "BP1.1"
    name = "肩部旋转 (转体)"
    phase = "ohb_preparation"
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
            fb = f"转体出色（{max_rot:.0f}°），背部几乎面向球网，蓄力充分。这是单反力量的基础。"
        elif max_rot >= c.shoulder_rotation_good:
            fb = f"转体良好（{max_rot:.0f}°）。尝试将肩膀再多转一些，目标是背部面向球网（≥90°）。"
        else:
            fb = f"转体不足（{max_rot:.0f}°）。单反需要比正手更充分的转体，整个上半身作为一个单元旋转（Unit Turn）。"

        return KPIResult(self.kpi_id, self.name, self.phase, max_rot, self.unit, score, rating, fb)


class OHB_KneeBendKPI(BaseBackhandKPI):
    """P1.2 – 准备阶段膝盖弯曲（"坐椅子"降低重心）。"""
    kpi_id = "BP1.2"
    name = "膝盖弯曲（蓄力）"
    phase = "ohb_preparation"
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
            fb = f"膝盖弯曲充分（{min_angle:.0f}°），像坐椅子一样降低重心，地面反作用力是力量的源泉。"
        elif min_angle <= c.knee_bend_good:
            fb = f"膝盖弯曲尚可（{min_angle:.0f}°）。尝试再蹲低一些，像坐在椅子上一样。"
        else:
            fb = f"腿部过于僵直（{min_angle:.0f}°）。单反需要充分弯曲膝盖来降低重心和蓄力。"

        return KPIResult(self.kpi_id, self.name, self.phase, min_angle, self.unit, score, rating, fb)


class OHB_NonDomHandPrepKPI(BaseBackhandKPI):
    """P1.3 – 非持拍手辅助引拍。

    准备阶段两只手应该都在球拍上（非持拍手辅助引拍和蓄力）。
    """
    kpi_id = "BP1.3"
    name = "非持拍手辅助"
    phase = "ohb_preparation"
    unit = "比例"

    def evaluate(self, *, non_dom_hand_distance_prep: Optional[float] = None, **kw) -> KPIResult:
        if non_dom_hand_distance_prep is None:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "无数据",
                             "无法测量非持拍手位置。")

        c = self.cfg.preparation
        val = non_dom_hand_distance_prep
        score = _linear_score(val, c.non_dom_hand_distance_warning, c.non_dom_hand_distance_good, 0.2)
        rating = _rating_from_score(score)

        if val <= c.non_dom_hand_distance_good:
            fb = f"非持拍手辅助引拍良好（距离={val:.2f}肩宽），双手在球拍上协同蓄力。"
        else:
            fb = f"非持拍手过早离开球拍（距离={val:.2f}肩宽）。准备阶段应保持双手在球拍上，非持拍手辅助引拍。"

        return KPIResult(self.kpi_id, self.name, self.phase, val, self.unit, score, rating, fb)


class OHB_SpineAngleKPI(BaseBackhandKPI):
    """P1.4 – 脊柱姿态。"""
    kpi_id = "BP1.4"
    name = "脊柱姿态"
    phase = "ohb_preparation"
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
            fb = f"身体前倾过多（{avg_lean:.1f}°）。保持脊柱更直立，避免弯腰击球。"

        return KPIResult(self.kpi_id, self.name, self.phase, avg_lean, self.unit, score, rating, fb)


# =====================================================================
# 阶段 2：引拍 & 落拍
# =====================================================================

class OHB_LLeverageKPI(BaseBackhandKPI):
    """P2.1 – L形杠杆（引拍时肘部角度）。

    单反引拍时手臂和球拍应形成 L 形，拍头保持在手腕上方。
    通过肘部角度近似评估（70-120° 为理想）。
    """
    kpi_id = "BP2.1"
    name = "L形杠杆"
    phase = "ohb_backswing"
    unit = "度"

    def evaluate(self, *, backswing_elbow_angle: Optional[float] = None, **kw) -> KPIResult:
        if backswing_elbow_angle is None:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "无数据",
                             "无法测量引拍时肘部角度。")

        c = self.cfg.backswing
        val = backswing_elbow_angle

        # 在理想范围内得高分
        if c.l_leverage_elbow_good_min <= val <= c.l_leverage_elbow_good_max:
            score = 90.0
            fb = f"L形杠杆良好（肘部角度{val:.0f}°），拍头保持在手腕上方，形成有效的力臂。"
        elif val < c.l_leverage_elbow_good_min:
            deviation = c.l_leverage_elbow_good_min - val
            score = max(20.0, 90.0 - deviation * 2.0)
            fb = f"手臂过度弯曲（{val:.0f}°）。引拍时保持手臂和球拍形成L形，拍头朝上。"
        else:
            deviation = val - c.l_leverage_elbow_good_max
            score = max(30.0, 90.0 - deviation * 1.5)
            fb = f"手臂过于伸直（{val:.0f}°）。引拍时手臂应适度弯曲形成L形，而非完全伸直。"

        score = float(np.clip(score, 0, 100))
        rating = _rating_from_score(score)
        return KPIResult(self.kpi_id, self.name, self.phase, val, self.unit, score, rating, fb)


# =====================================================================
# 阶段 3：动力链
# =====================================================================

class OHB_KineticChainSequenceKPI(BaseBackhandKPI):
    """P3.1 – 动力链顺序：脚→髋→肩→臂。"""
    kpi_id = "BP3.1"
    name = "动力链顺序"
    phase = "ohb_kinetic_chain"
    unit = "分"

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
            fb = f"动力链顺序正确：{order_str}。从地面到球拍的力量传递链完整。"
        elif score >= 50:
            fb = f"动力链部分正确：{order_str}。单反的力量来自腿部和躯干，确保髋部先行旋转。"
        else:
            fb = f"动力链顺序混乱：{order_str}。应遵循「脚→髋→肩→臂」的发力顺序。"

        return KPIResult(self.kpi_id, self.name, self.phase, score, self.unit, score, rating, fb,
                         details={"order": order_str, "delays_s": delays})


class OHB_HipShoulderSeparationKPI(BaseBackhandKPI):
    """P3.2 – 前挥阶段最大髋肩分离角。"""
    kpi_id = "BP3.2"
    name = "髋肩分离角"
    phase = "ohb_kinetic_chain"
    unit = "分"

    def evaluate(self, *, hip_shoulder_sep_values: List[float], **kw) -> KPIResult:
        if not hip_shoulder_sep_values:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "无数据",
                             "髋肩分离数据不足，无法评估。")

        max_sep = float(max(hip_shoulder_sep_values))
        c = self.cfg.kinetic_chain
        score = _linear_score(max_sep, 10.0, c.hip_shoulder_separation_good, c.hip_shoulder_separation_excellent)
        rating = _rating_from_score(score)

        if max_sep >= c.hip_shoulder_separation_excellent:
            fb = f"髋肩分离角出色（{max_sep:.0f}°），髋部领先肩部旋转，蓄力充分。"
        elif max_sep >= c.hip_shoulder_separation_good:
            fb = f"髋肩分离角良好（{max_sep:.0f}°）。"
        else:
            fb = f"髋肩分离角不足（{max_sep:.0f}°）。让髋部先转，肩膀延迟跟随。"

        return KPIResult(self.kpi_id, self.name, self.phase, max_sep, self.unit, score, rating, fb)


class OHB_InsideOutPathKPI(BaseBackhandKPI):
    """P3.3 – Inside-Out 挥拍路径线性度。

    单反应从靠近身体（Inside）向远离身体（Out）挥拍。
    """
    kpi_id = "BP3.3"
    name = "Inside-Out路径"
    phase = "ohb_kinetic_chain"
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
            fb = f"Inside-Out挥拍路径出色（R²={r_squared:.2f}），从身体内侧向外侧干净利落地挥出。"
        elif r_squared >= c.hand_path_linearity_good:
            fb = f"挥拍路径较好（R²={r_squared:.2f}），有轻微弧度。"
        else:
            fb = f"挥拍路径弯曲过大（R²={r_squared:.2f}）。单反应从身体内侧向外侧挥拍（Inside-Out）。"

        return KPIResult(self.kpi_id, self.name, self.phase, r_squared, self.unit, score, rating, fb)


# =====================================================================
# 阶段 4：击球点
# =====================================================================

class OHB_ContactPointKPI(BaseBackhandKPI):
    """P4.1 – 击球点位置。"""
    kpi_id = "BP4.1"
    name = "击球点位置"
    phase = "ohb_contact"
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
            fb = f"击球点过于靠近身体（{val:.2f}）。单反需要在身体前方击球，手臂向前伸展。"
        else:
            fb = f"击球点过于靠前（{val:.2f}）。可能过度前伸，调整击球时机。"

        return KPIResult(self.kpi_id, self.name, self.phase, val, self.unit, score, rating, fb)


class OHB_ArmExtensionKPI(BaseBackhandKPI):
    """P4.2 – 击球时手臂伸展程度。

    单反击球的关键：手臂必须从弯曲状态变为完全伸直。
    """
    kpi_id = "BP4.2"
    name = "手臂伸展"
    phase = "ohb_contact"
    unit = "度"

    def evaluate(self, *, elbow_angle_at_contact: Optional[float] = None, **kw) -> KPIResult:
        if elbow_angle_at_contact is None:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "无数据",
                             "无法测量击球时肘部角度。")

        c = self.cfg.contact
        angle = elbow_angle_at_contact

        if angle >= c.arm_extension_excellent_min:
            score = 95.0
            fb = f"手臂完全伸展（{angle:.0f}°），力量传递最大化。这是单反击球的关键。"
        elif angle >= c.arm_extension_good_min:
            score = 70.0 + 25.0 * (angle - c.arm_extension_good_min) / max(c.arm_extension_excellent_min - c.arm_extension_good_min, 1e-6)
            fb = f"手臂伸展良好（{angle:.0f}°）。尝试在击球瞬间完全伸直手臂，远离身体击球。"
        elif angle >= c.arm_extension_poor_min:
            score = 30.0 + 40.0 * (angle - c.arm_extension_poor_min) / max(c.arm_extension_good_min - c.arm_extension_poor_min, 1e-6)
            fb = f"手臂伸展不足（{angle:.0f}°）。击球时手臂应从弯曲变为完全伸直，这是单反力量的关键。"
        else:
            score = max(10.0, 30.0 * angle / max(c.arm_extension_poor_min, 1e-6))
            fb = f"手臂严重弯曲（{angle:.0f}°）。单反击球必须伸直手臂，否则无法产生足够力量。"

        score = float(np.clip(score, 0, 100))
        rating = _rating_from_score(score)
        return KPIResult(self.kpi_id, self.name, self.phase, angle, self.unit, score, rating, fb)


class OHB_BodyFreezeKPI(BaseBackhandKPI):
    """P4.3 – 击球时身体刹车。"""
    kpi_id = "BP4.3"
    name = "身体制动"
    phase = "ohb_contact"
    unit = "度"

    def evaluate(self, *, torso_angular_velocity_at_contact: Optional[float] = None, **kw) -> KPIResult:
        if torso_angular_velocity_at_contact is None:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "无数据",
                             "无法测量击球时躯干角速度。")

        c = self.cfg.contact
        val = abs(torso_angular_velocity_at_contact)
        score = _linear_score(val, c.body_freeze_warning, c.body_freeze_good_max, 10.0)
        rating = _rating_from_score(score)

        if val <= c.body_freeze_good_max:
            fb = f"击球时身体制动良好（{val:.0f}°/s）。身体安静，为手臂提供了稳定的击球平台。"
        else:
            fb = f"击球时身体仍在旋转（{val:.0f}°/s）。单反击球时身体必须「刹住」，非持拍手向后伸展帮助停止旋转。"

        return KPIResult(self.kpi_id, self.name, self.phase, val, self.unit, score, rating, fb)


class OHB_NonDomHandBalanceKPI(BaseBackhandKPI):
    """P4.4 – 非持拍手反向平衡。

    击球时非持拍手应向后伸展，两手距离应大。
    """
    kpi_id = "BP4.4"
    name = "非持拍手反向平衡"
    phase = "ohb_contact"
    unit = "比例"

    def evaluate(self, *, non_dom_hand_spread_contact: Optional[float] = None, **kw) -> KPIResult:
        if non_dom_hand_spread_contact is None:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "无数据",
                             "无法测量非持拍手位置。")

        c = self.cfg.contact
        val = non_dom_hand_spread_contact
        score = _linear_score(val, 0.5, c.non_dom_hand_spread_good, c.non_dom_hand_spread_excellent)
        rating = _rating_from_score(score)

        if val >= c.non_dom_hand_spread_excellent:
            fb = f"非持拍手反向平衡出色（距离={val:.2f}肩宽），像展翅一样向后伸展，完美平衡。"
        elif val >= c.non_dom_hand_spread_good:
            fb = f"非持拍手反向平衡良好（距离={val:.2f}肩宽）。"
        else:
            fb = f"非持拍手未充分向后伸展（距离={val:.2f}肩宽）。击球时非持拍手应主动向后伸展，帮助身体制动和保持平衡。"

        return KPIResult(self.kpi_id, self.name, self.phase, val, self.unit, score, rating, fb)


class OHB_HeadStabilityAtContactKPI(BaseBackhandKPI):
    """P4.5 – 击球时头部稳定性。"""
    kpi_id = "BP4.5"
    name = "击球时头部稳定性"
    phase = "ohb_contact"
    unit = "归一化"

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

class OHB_ATA_KPI(BaseBackhandKPI):
    """P5.1 – ATA (Air The Armpit) 收拍高度。

    单反随挥应高举，像自由女神像一样，亮出腋窝。
    通过测量随挥后手腕高于肩部的距离来评估。
    """
    kpi_id = "BP5.1"
    name = "ATA收拍高度"
    phase = "ohb_extension"
    unit = "高度比"

    def evaluate(self, *, ata_wrist_above_shoulder: Optional[float] = None, **kw) -> KPIResult:
        if ata_wrist_above_shoulder is None:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "无数据",
                             "无法测量ATA收拍高度。")

        c = self.cfg.extension
        val = ata_wrist_above_shoulder
        score = _linear_score(val, -0.1, c.ata_wrist_above_shoulder_good, c.ata_wrist_above_shoulder_excellent)
        rating = _rating_from_score(score)

        if val >= c.ata_wrist_above_shoulder_excellent:
            fb = f"ATA收拍出色（手腕高于肩部{val:.2f}躯干高度），像自由女神像一样高举。"
        elif val >= c.ata_wrist_above_shoulder_good:
            fb = f"收拍高度良好（{val:.2f}）。继续向上延伸，目标是「亮出腋窝」(Air The Armpit)。"
        else:
            fb = f"收拍过低（{val:.2f}）。单反随挥应高举过肩，想象自己是自由女神像。"

        return KPIResult(self.kpi_id, self.name, self.phase, val, self.unit, score, rating, fb)


class OHB_StaySidewaysKPI(BaseBackhandKPI):
    """P5.2 – 保持侧身。

    击球后身体应保持侧身，避免过早开胯。
    通过击球后肩部旋转变化来评估。
    """
    kpi_id = "BP5.2"
    name = "保持侧身"
    phase = "ohb_extension"
    unit = "度"

    def evaluate(self, *, post_contact_shoulder_change: Optional[float] = None, **kw) -> KPIResult:
        if post_contact_shoulder_change is None:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "无数据",
                             "无法测量击球后身体旋转变化。")

        c = self.cfg.extension
        val = abs(post_contact_shoulder_change)
        score = _linear_score(val, c.sidebody_rotation_change_warning, c.sidebody_rotation_change_good, 5.0)
        rating = _rating_from_score(score)

        if val <= c.sidebody_rotation_change_good:
            fb = f"击球后保持侧身出色（旋转变化{val:.0f}°），身体控制良好。"
        else:
            fb = f"击球后过早开身（旋转变化{val:.0f}°）。单反击球后应保持侧身，直到球离开拍面。"

        return KPIResult(self.kpi_id, self.name, self.phase, val, self.unit, score, rating, fb)


# =====================================================================
# 阶段 6：平衡
# =====================================================================

class OHB_OverallHeadStabilityKPI(BaseBackhandKPI):
    """P6.1 – 整体头部稳定性。"""
    kpi_id = "BP6.1"
    name = "整体头部稳定性"
    phase = "ohb_balance"
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
            fb = f"挥拍过程中头部上下跳动（标准差={val:.3f}）。保持头部高度一致。"

        return KPIResult(self.kpi_id, self.name, self.phase, val, self.unit, score, rating, fb)


class OHB_SpineConsistencyKPI(BaseBackhandKPI):
    """P6.2 – 脊柱一致性。"""
    kpi_id = "BP6.2"
    name = "脊柱一致性"
    phase = "ohb_balance"
    unit = "度"

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
            fb = f"脊柱角度变化过大（标准差={val:.1f}°）。保持躯干稳定。"

        return KPIResult(self.kpi_id, self.name, self.phase, val, self.unit, score, rating, fb)


# =====================================================================
# KPI 注册表
# =====================================================================

ALL_BACKHAND_KPIS = [
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
]
