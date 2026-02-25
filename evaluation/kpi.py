"""关键绩效指标 (KPI) 定义 — Modern Forehand 评估 v3（8阶段模型）。

每个 KPI 是一个独立的类：
    1. 接收相关数据（关键点、轨迹、帧索引）。
    2. 计算原始指标值。
    3. 使用 ``FrameworkConfig`` 中的阈值映射到 0-100 分。
    4. 返回 ``KPIResult``：分数、原始值、评级、中文反馈。

v3 8阶段模型：
    阶段 1 – 一体化转体 (unit_turn)：肩部旋转、膝盖弯曲、脊柱姿态
    阶段 2 – 槽位准备 (slot_prep)：肘部后撤、拍头下垂、肘部高度
    阶段 3 – 蹬转与髋部启动 (leg_drive)：地面力量代理、髋部旋转速度
    阶段 4 – 躯干与肩部牵引 (torso_pull)：髋肩分离角、髋肩时序
    阶段 5 – 滞后与肘部驱动 (lag_drive)：肘部收紧、手部路径线性度
    阶段 6 – 击球与肩内旋 (contact)：击球点、肘部角度、身体刹车、头部稳定性、SIR代理
    阶段 7 – 雨刷式随挥 (wiper)：前向延伸、雨刷扫过角度
    阶段 8 – 减速与平衡 (balance)：整体头部稳定性、脊柱一致性
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import numpy as np

from config.framework_config import FrameworkConfig, DEFAULT_CONFIG


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
    drill: str = ""       # 训练处方
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
# 阶段 1：一体化转体 (unit_turn)
# =====================================================================

class ShoulderRotationKPI(BaseKPI):
    """UT1.1 – 准备阶段最大肩部旋转（X-Factor）。"""
    kpi_id = "UT1.1"
    name = "肩部旋转 (X-Factor)"
    phase = "unit_turn"
    unit = "度"

    def evaluate(self, *, shoulder_rotation_values: List[float], **kw) -> KPIResult:
        if not shoulder_rotation_values:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "无数据",
                             "肩部旋转数据不足，无法评估。")

        max_rot = float(max(shoulder_rotation_values))
        c = self.cfg.unit_turn
        score = _linear_score(max_rot, c.shoulder_rotation_poor, c.shoulder_rotation_good, c.shoulder_rotation_excellent)
        rating = _rating_from_score(score)

        if max_rot >= c.shoulder_rotation_excellent:
            fb = f"肩部转体出色（{max_rot:.0f}°），背部几乎面向球网，这是现代正手的标志性蓄力姿态。"
            drill = ""
        elif max_rot >= c.shoulder_rotation_good:
            fb = f"肩部转体良好（{max_rot:.0f}°）。尝试将肩膀再多转一些，目标≥90°。"
            drill = "练习：面对墙壁站立，做Unit Turn时让非持拍手触碰墙壁，强化完整转体感觉。"
        else:
            fb = f"肩部转体不足（{max_rot:.0f}°）。Unit Turn 不够完整，肩膀相对于髋部应旋转≥90°。"
            drill = "练习：双手持拍在胸前，转体时让球拍指向身后的围栏。感受整个上半身作为一个整体旋转，而不只是手臂后拉。"

        return KPIResult(self.kpi_id, self.name, self.phase, max_rot, self.unit, score, rating, fb, drill)


class KneeBendKPI(BaseKPI):
    """UT1.2 – 准备阶段负重腿膝盖弯曲。"""
    kpi_id = "UT1.2"
    name = "膝盖弯曲（蓄力）"
    phase = "unit_turn"
    unit = "度"

    def evaluate(self, *, knee_angle_values: List[float], **kw) -> KPIResult:
        if not knee_angle_values:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "无数据",
                             "膝盖弯曲数据不足，无法评估。")

        min_angle = float(min(knee_angle_values))
        c = self.cfg.unit_turn
        score = _linear_score(min_angle, c.knee_bend_poor, c.knee_bend_good, c.knee_bend_excellent)
        rating = _rating_from_score(score)

        if min_angle <= c.knee_bend_excellent:
            fb = f"膝盖弯曲充分（{min_angle:.0f}°），下肢蓄力出色。地面反作用力是力量的源泉。"
            drill = ""
        elif min_angle <= c.knee_bend_good:
            fb = f"膝盖弯曲尚可（{min_angle:.0f}°）。尝试再蹲低一些，目标≤140°。"
            drill = "练习：在引拍时想象坐在一把高脚凳上，感受大腿肌肉的负重。"
        else:
            fb = f"腿部过于僵直（{min_angle:.0f}°）。需要更多弯曲膝盖来蓄力。"
            drill = "练习：做Split Step后立即下蹲转体，感受「坐下去」的感觉。每次击球前先蹲后打。"

        return KPIResult(self.kpi_id, self.name, self.phase, min_angle, self.unit, score, rating, fb, drill)


class SpineAngleKPI(BaseKPI):
    """UT1.3 – 脊柱姿态（偏离垂直角度）。"""
    kpi_id = "UT1.3"
    name = "脊柱姿态"
    phase = "unit_turn"
    unit = "度"

    def evaluate(self, *, spine_angle_values: List[float], **kw) -> KPIResult:
        if not spine_angle_values:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "无数据",
                             "脊柱角度数据不足，无法评估。")

        avg_lean = float(np.mean(spine_angle_values))
        c = self.cfg.unit_turn
        score = _linear_score(avg_lean, c.spine_lean_warning, c.spine_lean_good_max, 5.0)
        rating = _rating_from_score(score)

        if avg_lean <= c.spine_lean_good_max:
            fb = f"脊柱姿态良好（平均倾斜{avg_lean:.1f}°），保持了直立的身体姿态。"
            drill = ""
        else:
            fb = f"身体前倾过多（{avg_lean:.1f}°）。保持脊柱更直立，目标<{c.spine_lean_good_max:.0f}°。"
            drill = "练习：想象头顶有一根绳子向上拉你，转体时保持头部高度不变。"

        return KPIResult(self.kpi_id, self.name, self.phase, avg_lean, self.unit, score, rating, fb, drill)


# =====================================================================
# 阶段 2：槽位准备 (slot_prep)
# =====================================================================

class ElbowBackPositionKPI(BaseKPI):
    """SP2.1 – 肘部后撤深度（Rick Macci: "Elbow Back"）。"""
    kpi_id = "SP2.1"
    name = "肘部后撤 (Elbow Back)"
    phase = "slot_prep"
    unit = "肩宽比"

    def evaluate(self, *, elbow_behind_values: List[float], **kw) -> KPIResult:
        if not elbow_behind_values:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "无数据",
                             "肘部后撤数据不足，无法评估。")

        max_behind = float(max(elbow_behind_values))
        c = self.cfg.slot_prep
        score = _linear_score(max_behind, c.elbow_behind_poor, c.elbow_behind_good, c.elbow_behind_excellent)
        rating = _rating_from_score(score)

        if max_behind >= c.elbow_behind_excellent:
            fb = f"肘部后撤出色（{max_behind:.2f}肩宽），完美的Slot位置。Rick Macci强调的「肘部后拉」做得很好。"
            drill = ""
        elif max_behind >= c.elbow_behind_good:
            fb = f"肘部后撤良好（{max_behind:.2f}肩宽）。肘部已经在躯干后方，可以再深一些。"
            drill = "练习：转体后，用非持拍手轻推持拍手的肘部向后，感受肘部到达身体后方的位置。"
        else:
            fb = f"肘部后撤不足（{max_behind:.2f}肩宽）。肘部没有充分到达身体后方，影响了蓄力空间。"
            drill = "练习：Rick Macci的「肘部后拉」训练——转体时想象有人从后面拉你的肘部。肘部应该在肩膀后方，而不是身体侧面。"

        return KPIResult(self.kpi_id, self.name, self.phase, max_behind, self.unit, score, rating, fb, drill)


class RacketDropKPI(BaseKPI):
    """SP2.2 – 拍头下垂深度（"Pat the Dog" 放松度）。"""
    kpi_id = "SP2.2"
    name = "拍头下垂 (Racket Drop)"
    phase = "slot_prep"
    unit = "躯干高度比"

    def evaluate(self, *, wrist_below_elbow_values: List[float], **kw) -> KPIResult:
        if not wrist_below_elbow_values:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "无数据",
                             "拍头下垂数据不足，无法评估。")

        max_drop = float(max(wrist_below_elbow_values))
        c = self.cfg.slot_prep
        score = _linear_score(max_drop, c.racket_drop_poor, c.racket_drop_good, c.racket_drop_excellent)
        rating = _rating_from_score(score)

        if max_drop >= c.racket_drop_excellent:
            fb = f"拍头下垂出色（{max_drop:.2f}躯干高度），手腕完全放松，拍头自然下垂。这是产生拍头滞后的关键。"
            drill = ""
        elif max_drop >= c.racket_drop_good:
            fb = f"拍头下垂良好（{max_drop:.2f}）。手腕有一定放松度。"
            drill = "练习：在引拍最低点时，感受球拍重量完全挂在手腕上，像钟摆一样。"
        else:
            fb = f"拍头下垂不足（{max_drop:.2f}）。手腕过于僵硬，拍头没有充分下垂。需要更放松的手腕。"
            drill = "练习：「Pat the Dog」——想象你在拍一只站在你身侧的狗。拍头应该自然下垂到膝盖高度。握拍不要太紧，让重力帮你。"

        return KPIResult(self.kpi_id, self.name, self.phase, max_drop, self.unit, score, rating, fb, drill)


# =====================================================================
# 阶段 3：蹬转与髋部启动 (leg_drive)
# =====================================================================

class GroundForceProxyKPI(BaseKPI):
    """LD3.1 – 地面力量代理（髋部垂直加速度）。"""
    kpi_id = "LD3.1"
    name = "蹬地力量 (Ground Force)"
    phase = "leg_drive"
    unit = "px/s²"

    def evaluate(self, *, peak_hip_acceleration: Optional[float] = None, **kw) -> KPIResult:
        if peak_hip_acceleration is None:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "无数据",
                             "蹬地力量数据不足，无法评估。")

        val = peak_hip_acceleration
        c = self.cfg.leg_drive
        score = _linear_score(val, c.ground_force_poor, c.ground_force_good, c.ground_force_excellent)
        rating = _rating_from_score(score)

        if val >= c.ground_force_excellent:
            fb = f"蹬地力量出色（{val:.0f}），下肢爆发力强劲。力量从地面开始，通过腿部传递到躯干。"
            drill = ""
        elif val >= c.ground_force_good:
            fb = f"蹬地力量良好（{val:.0f}）。腿部有蹬地动作，可以更爆发。"
            drill = "练习：做分腿垫步后，用力蹬地同时转髋，感受从脚底到髋部的力量传递。"
        else:
            fb = f"蹬地力量不足（{val:.0f}）。腿部几乎没有参与发力，力量主要来自上半身。"
            drill = "练习：不拿球拍，只做蹬地转髋的动作。先蹲下，然后用力蹬地站起同时转髋。感受大腿和臀部的发力。这是所有力量的起点！"

        return KPIResult(self.kpi_id, self.name, self.phase, val, self.unit, score, rating, fb, drill)


class HipRotationSpeedKPI(BaseKPI):
    """LD3.2 – 髋部旋转速度。"""
    kpi_id = "LD3.2"
    name = "髋部旋转速度"
    phase = "leg_drive"
    unit = "度/秒"

    def evaluate(self, *, peak_hip_rotation_speed: Optional[float] = None, **kw) -> KPIResult:
        if peak_hip_rotation_speed is None:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "无数据",
                             "髋部旋转速度数据不足，无法评估。")

        val = peak_hip_rotation_speed
        c = self.cfg.leg_drive
        score = _linear_score(val, c.hip_rotation_speed_poor, c.hip_rotation_speed_good, c.hip_rotation_speed_excellent)
        rating = _rating_from_score(score)

        if val >= c.hip_rotation_speed_excellent:
            fb = f"髋部旋转速度出色（{val:.0f}°/s），髋部爆发性旋转是动力链的引擎。"
            drill = ""
        elif val >= c.hip_rotation_speed_good:
            fb = f"髋部旋转速度良好（{val:.0f}°/s）。髋部有主动旋转。"
            drill = "练习：做「髋部先行」训练——站立时只转髋不转肩，感受髋部独立旋转的能力。"
        else:
            fb = f"髋部旋转速度不足（{val:.0f}°/s）。髋部旋转太慢或没有主动旋转。"
            drill = "练习：双脚站稳，想象右脚脚跟踩灭一根烟（右手持拍），用力转髋。肩膀保持不动，只让髋部先转。这是制造「弹弓效应」的关键。"

        return KPIResult(self.kpi_id, self.name, self.phase, val, self.unit, score, rating, fb, drill)


# =====================================================================
# 阶段 4：躯干与肩部牵引 (torso_pull)
# =====================================================================

class HipShoulderSeparationKPI(BaseKPI):
    """TP4.1 – 前挥阶段最大髋肩分离角（X-Factor Stretch）。"""
    kpi_id = "TP4.1"
    name = "髋肩分离角 (X-Factor)"
    phase = "torso_pull"
    unit = "度"

    def evaluate(self, *, hip_shoulder_sep_values: List[float], **kw) -> KPIResult:
        if not hip_shoulder_sep_values:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "无数据",
                             "髋肩分离数据不足，无法评估。")

        max_sep = float(max(hip_shoulder_sep_values))
        c = self.cfg.torso_pull
        score = _linear_score(max_sep, c.hip_shoulder_separation_poor, c.hip_shoulder_separation_good, c.hip_shoulder_separation_excellent)
        rating = _rating_from_score(score)

        if max_sep >= c.hip_shoulder_separation_excellent:
            fb = f"髋肩分离角出色（{max_sep:.0f}°），X-Factor拉伸充分。髋部先转而肩膀延迟跟随，产生了强大的弹弓效应。"
            drill = ""
        elif max_sep >= c.hip_shoulder_separation_good:
            fb = f"髋肩分离角良好（{max_sep:.0f}°），髋部领先于肩部旋转。"
            drill = "练习：转体时用非持拍手按住持拍侧肩膀，只让髋部先转，感受躯干被「拧」的感觉。"
        else:
            fb = f"髋肩分离角不足（{max_sep:.0f}°）。髋部和肩膀几乎同时旋转，没有产生弹弓效应。"
            drill = "练习：背对球网站立，先转髋面向球网，保持肩膀不动2秒，然后释放肩膀。感受这个「分离」的感觉。这是职业选手力量的秘密。"

        return KPIResult(self.kpi_id, self.name, self.phase, max_sep, self.unit, score, rating, fb, drill)


class HipShoulderTimingKPI(BaseKPI):
    """TP4.2 – 髋肩旋转时序差（髋部应先于肩部达到峰值旋转速度）。"""
    kpi_id = "TP4.2"
    name = "髋肩旋转时序"
    phase = "torso_pull"
    unit = "秒"

    def evaluate(self, *, hip_shoulder_timing_delay: Optional[float] = None, **kw) -> KPIResult:
        if hip_shoulder_timing_delay is None:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "无数据",
                             "髋肩时序数据不足，无法评估。")

        val = hip_shoulder_timing_delay
        c = self.cfg.torso_pull
        score = _linear_score(val, c.timing_delay_poor, c.timing_delay_good, c.timing_delay_excellent)
        rating = _rating_from_score(score)

        if val >= c.timing_delay_excellent:
            fb = f"髋肩时序出色（延迟{val*1000:.0f}ms），髋部明显先于肩部达到峰值旋转。完美的近端到远端传递。"
            drill = ""
        elif val >= c.timing_delay_good:
            fb = f"髋肩时序良好（延迟{val*1000:.0f}ms），髋部先于肩部旋转。"
            drill = "练习：慢动作挥拍，刻意在髋部转完后停顿一拍再让肩膀跟上。"
        else:
            fb = f"髋肩时序不佳（延迟{val*1000:.0f}ms）。髋部和肩膀几乎同时旋转，甚至肩膀先转。"
            drill = "练习：用弹力带绑在腰间，固定在身后。转髋时对抗弹力带的阻力，强化髋部先行的意识。然后释放肩膀。"

        return KPIResult(self.kpi_id, self.name, self.phase, val, self.unit, score, rating, fb, drill)


# =====================================================================
# 阶段 5：滞后与肘部驱动 (lag_drive)
# =====================================================================

class ElbowTuckKPI(BaseKPI):
    """LE5.1 – 前挥时肘部收紧距离（肘部贴近身体）。"""
    kpi_id = "LE5.1"
    name = "肘部收紧 (Elbow Tuck)"
    phase = "lag_drive"
    unit = "躯干高度比"

    def evaluate(self, *, min_elbow_tuck: Optional[float] = None, **kw) -> KPIResult:
        if min_elbow_tuck is None:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "无数据",
                             "肘部收紧数据不足，无法评估。")

        val = min_elbow_tuck
        c = self.cfg.lag_drive
        # Smaller is better for tuck
        score = _linear_score(val, c.elbow_tuck_poor, c.elbow_tuck_good, c.elbow_tuck_excellent)
        rating = _rating_from_score(score)

        if val <= c.elbow_tuck_excellent:
            fb = f"肘部收紧出色（{val:.2f}），前挥时肘部紧贴身体驱动。这是Rick Macci强调的「肘部先行」。"
            drill = ""
        elif val <= c.elbow_tuck_good:
            fb = f"肘部收紧良好（{val:.2f}），肘部较贴近身体。"
            drill = "练习：前挥时想象肘部要「戳」前方的球，肘部先走，拍头滞后。"
        else:
            fb = f"肘部离身体太远（{val:.2f}）。前挥时手臂外展，失去了肘部驱动的效率。"
            drill = "练习：在腋下夹一个网球，做前挥动作。如果球掉了，说明肘部离身体太远。肘部应该贴着身体向前驱动，拍头自然滞后。"

        return KPIResult(self.kpi_id, self.name, self.phase, val, self.unit, score, rating, fb, drill)


class HandPathLinearityKPI(BaseKPI):
    """LE5.2 – 击球区手部路径线性度。"""
    kpi_id = "LE5.2"
    name = "手部路径线性度"
    phase = "lag_drive"
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

        c = self.cfg.lag_drive
        score = _linear_score(r_squared, 0.5, c.hand_path_linearity_good, c.hand_path_linearity_excellent)
        rating = _rating_from_score(score)

        if r_squared >= c.hand_path_linearity_excellent:
            fb = f"手部路径非常直线（R²={r_squared:.2f}），击球区挥拍路径干净利落。"
            drill = ""
        elif r_squared >= c.hand_path_linearity_good:
            fb = f"手部路径较直（R²={r_squared:.2f}），有轻微弧度。"
            drill = "练习：在击球区放一条直线标记，沿着直线做挥拍练习。"
        else:
            fb = f"手部路径弯曲过大（R²={r_squared:.2f}）。击球区应保持更直的挥拍路径。"
            drill = "练习：想象你在击球区推一扇门——手部应该沿直线向前推，而不是画弧线。从内向外击球。"

        return KPIResult(self.kpi_id, self.name, self.phase, r_squared, self.unit, score, rating, fb, drill)


# =====================================================================
# 阶段 6：击球与肩内旋 (contact)
# =====================================================================

class ContactPointKPI(BaseKPI):
    """C6.1 – 击球点位置（手腕在髋部前方的距离）。"""
    kpi_id = "C6.1"
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
            drill = ""
        elif val < c.contact_forward_good_min:
            fb = f"击球点过于靠近身体（{val:.2f}）。需要在更前方击球。"
            drill = "练习：在身体前方放一个标记物，练习在标记物位置击球。击球时手臂应该有一定伸展。"
        else:
            fb = f"击球点过于靠前（{val:.2f}）。可能过度前伸，调整击球时机。"
            drill = "练习：让球多弹一拍再击球，找到舒适的击球距离。"

        return KPIResult(self.kpi_id, self.name, self.phase, val, self.unit, score, rating, fb, drill)


class ElbowAngleAtContactKPI(BaseKPI):
    """C6.2 – 击球时肘部角度。"""
    kpi_id = "C6.2"
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
            fb = f"直臂击球（{angle:.0f}°）— {style}风格。手臂充分伸展，力臂最长，力量传递效率最高。"
            drill = ""
        elif in_double_bend:
            score = 90.0
            style = "双弯型"
            fb = f"双弯击球（{angle:.0f}°）— {style}风格。紧凑的手臂结构，控制力好。"
            drill = ""
        elif c.double_bend_max < angle < c.straight_arm_min:
            score = 60.0
            style = "过渡型"
            fb = f"肘部角度{angle:.0f}°，介于双弯和直臂之间。建议选择一种风格并坚持。"
            drill = "练习：如果想打直臂，击球时想象手臂要「推」向目标，让肘部完全伸直。"
        elif angle < c.double_bend_min:
            score = 30.0
            style = "过度弯曲"
            fb = f"击球时手臂过度弯曲（{angle:.0f}°）。需要更多伸展，避免「夹臂」击球。"
            drill = "练习：在击球点位置做静态伸展，感受手臂完全伸直的位置。"
        else:
            score = 70.0
            style = "过度伸展"
            fb = f"击球时肘部角度{angle:.0f}°，基本正常。"
            drill = ""

        rating = _rating_from_score(score)
        return KPIResult(self.kpi_id, self.name, self.phase, angle, self.unit, score, rating, fb, drill,
                         details={"style": style})


class BodyFreezeKPI(BaseKPI):
    """C6.3 – 击球时躯干角速度（应接近零 = 「刹车」）。"""
    kpi_id = "C6.3"
    name = "身体刹车 (Body Freeze)"
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
            fb = f"击球时身体刹车良好（{val:.0f}°/s），躯干停止旋转提供了稳定的击球平台。这是Tennis Doctor的四大不可妥协之一。"
            drill = ""
        else:
            fb = f"击球时身体仍在旋转（{val:.0f}°/s）。躯干应在击球瞬间「刹住」。"
            drill = "练习：击球时想象胸口有一个大灯照向目标，击球瞬间这个灯要「定住」不动。躯干停止旋转，让手臂和球拍像鞭梢一样甩出去。"

        return KPIResult(self.kpi_id, self.name, self.phase, val, self.unit, score, rating, fb, drill)


class HeadStabilityAtContactKPI(BaseKPI):
    """C6.4 – 击球点附近头部稳定性。"""
    kpi_id = "C6.4"
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
            fb = f"头部稳定性出色（{val:.3f}），眼睛始终注视击球点。费德勒的标志性动作。"
            drill = ""
        else:
            fb = f"击球时头部移动过大（{val:.3f}）。头部应该是最后移动的部位。"
            drill = "练习：击球后继续看击球点位置1秒钟，不要急着抬头看球的去向。像费德勒一样，让头部「留在那里」。"

        return KPIResult(self.kpi_id, self.name, self.phase, val, self.unit, score, rating, fb, drill)


class SIRProxyKPI(BaseKPI):
    """C6.5 – 肩内旋代理指标（击球前后前臂角速度）。

    高速的前臂旋前是肩内旋 (SIR) 的直接结果和可观测代理。
    Dr. Brian Gordon 指出 SIR 是正手力量的最大贡献者。
    """
    kpi_id = "C6.5"
    name = "肩内旋代理 (SIR Proxy)"
    phase = "contact"
    unit = "度/秒"

    def evaluate(self, *, forearm_angular_velocity: Optional[float] = None, **kw) -> KPIResult:
        if forearm_angular_velocity is None:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "无数据",
                             "前臂角速度数据不足，无法评估肩内旋。")

        val = forearm_angular_velocity
        c = self.cfg.contact
        score = _linear_score(val, c.sir_proxy_poor, c.sir_proxy_good, c.sir_proxy_excellent)
        rating = _rating_from_score(score)

        if val >= c.sir_proxy_excellent:
            fb = f"肩内旋指标出色（前臂角速度{val:.0f}°/s）。这说明你的肩部内旋非常充分，是力量的最大贡献者。"
            drill = ""
        elif val >= c.sir_proxy_good:
            fb = f"肩内旋指标良好（{val:.0f}°/s）。前臂有明显的旋前动作。"
            drill = "练习：做「翻门把手」的动作——击球时想象你在用力转动一个门把手。感受的不是手腕转，而是整个前臂从肩膀带动的旋转。"
        else:
            fb = f"肩内旋指标不足（{val:.0f}°/s）。前臂旋前速度低，说明肩部内旋不充分。"
            drill = "练习：Dr. Gordon的「胸肌发力」意象——想象你在用胸肌和背阔肌把上臂向内旋转，前臂只是被动跟随。不要主动转手腕，而是让肩膀的旋转自然传递到前臂。"

        return KPIResult(self.kpi_id, self.name, self.phase, val, self.unit, score, rating, fb, drill)


# =====================================================================
# 阶段 7：雨刷式随挥 (wiper)
# =====================================================================

class ForwardExtensionKPI(BaseKPI):
    """WW7.1 – 击球后前向延伸距离。"""
    kpi_id = "WW7.1"
    name = "前向延伸"
    phase = "wiper"
    unit = "躯干高度比"

    def evaluate(self, *, forward_extension_norm: Optional[float] = None, **kw) -> KPIResult:
        if forward_extension_norm is None:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "无数据",
                             "无法测量前向延伸距离。")

        c = self.cfg.wiper
        val = forward_extension_norm
        score = _linear_score(val, 0.1, c.forward_extension_good, c.forward_extension_excellent)
        rating = _rating_from_score(score)

        if val >= c.forward_extension_excellent:
            fb = f"前向延伸出色（{val:.2f}躯干高度），击球后充分穿透球体。"
            drill = ""
        elif val >= c.forward_extension_good:
            fb = f"前向延伸良好（{val:.2f}）。继续向目标方向推送。"
            drill = "练习：击球后想象你要把球拍递给对面的人，手臂向前伸展。"
        else:
            fb = f"前向延伸不足（{val:.2f}）。击球后手臂应继续向目标方向延伸。"
            drill = "练习：在击球后，让球拍头继续向前「推」60-90厘米。不要急着收拍，先延伸再雨刷。"

        return KPIResult(self.kpi_id, self.name, self.phase, val, self.unit, score, rating, fb, drill)


class WiperSweepKPI(BaseKPI):
    """WW7.2 – 雨刷式随挥扫过角度。"""
    kpi_id = "WW7.2"
    name = "雨刷式随挥"
    phase = "wiper"
    unit = "度"

    def evaluate(self, *, wiper_sweep_angle: Optional[float] = None, **kw) -> KPIResult:
        if wiper_sweep_angle is None:
            return KPIResult(self.kpi_id, self.name, self.phase, None, self.unit, 0, "无数据",
                             "雨刷随挥数据不足，无法评估。")

        val = wiper_sweep_angle
        c = self.cfg.wiper
        score = _linear_score(val, c.wiper_sweep_poor, c.wiper_sweep_good, c.wiper_sweep_excellent)
        rating = _rating_from_score(score)

        if val >= c.wiper_sweep_excellent:
            fb = f"雨刷式随挥出色（{val:.0f}°），球拍在身体前方完成了充分的横向扫过。这是产生上旋的关键。"
            drill = ""
        elif val >= c.wiper_sweep_good:
            fb = f"雨刷式随挥良好（{val:.0f}°），有明显的横向扫过动作。"
            drill = "练习：击球后让球拍像汽车雨刷一样从右向左扫过（右手持拍），收拍在身体左侧。"
        else:
            fb = f"雨刷式随挥不足（{val:.0f}°）。随挥太短或太直，缺少横向扫过。"
            drill = "练习：做「擦桌子」的动作——击球后想象你在用球拍面擦一张桌子，从右擦到左。这个横向的扫过动作会自然产生上旋。"

        return KPIResult(self.kpi_id, self.name, self.phase, val, self.unit, score, rating, fb, drill)


# =====================================================================
# 阶段 8：减速与平衡 (balance)
# =====================================================================

class OverallHeadStabilityKPI(BaseKPI):
    """B8.1 – 整个挥拍过程中头部垂直稳定性。"""
    kpi_id = "B8.1"
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
            fb = f"整个挥拍过程中头部稳定性出色（标准差={val:.3f}）。头部是身体的「陀螺仪」。"
            drill = ""
        else:
            fb = f"挥拍过程中头部上下跳动（标准差={val:.3f}）。保持头部高度一致。"
            drill = "练习：在头顶放一本书做挥拍练习（不用真放，想象就行）。如果书会掉，说明头部不够稳定。"

        return KPIResult(self.kpi_id, self.name, self.phase, val, self.unit, score, rating, fb, drill)


class SpineConsistencyKPI(BaseKPI):
    """B8.2 – 挥拍过程中脊柱角度一致性。"""
    kpi_id = "B8.2"
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
            drill = ""
        else:
            fb = f"脊柱角度变化过大（标准差={val:.1f}°）。保持躯干稳定。"
            drill = "练习：挥拍时保持「高个子」的感觉，不要弯腰或前倾。想象你的脊柱是一根不会弯的铁棍。"

        return KPIResult(self.kpi_id, self.name, self.phase, val, self.unit, score, rating, fb, drill)


# =====================================================================
# KPI 注册表 (v3: 21 个 KPI)
# =====================================================================

ALL_KPIS = [
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
]
