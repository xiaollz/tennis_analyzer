"""One-Handed Backhand (OHB) evaluation framework configuration.

All thresholds, weights, and scoring parameters for the single-handed
backhand are centralised here.

Sources:
    - Tennis Doctor (15 Years OHB Training, Biomechanics of OHB, Next Gen OHB)
    - Feel Tennis (6-Step OHB)
    - Tennisnerd 2 / ATP Coach Adri (Master the OHB)

Key differences from Forehand:
    - Shoulder rotation direction is reversed (back faces net)
    - Arm must extend from bent to fully straight at contact
    - Non-dominant hand plays a critical role (balance + stopping rotation)
    - Follow-through is ATA (Air The Armpit) — high finish
    - Body must stay sideways longer (no early hip opening)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict


# =====================================================================
# Phase 1: Preparation & Unit Turn (单反特有)
# =====================================================================

@dataclass(frozen=True)
class OHB_PreparationConfig:
    """单反准备阶段阈值。

    单反需要更充分的转体（背部面向球网），且非持拍手需要在球拍上辅助引拍。
    """
    # P1.1 肩部旋转（相对于髋部线）— 单反需要更大旋转
    shoulder_rotation_excellent: float = 100.0   # 度（背部几乎面向球网）
    shoulder_rotation_good: float = 80.0
    shoulder_rotation_poor: float = 55.0

    # P1.2 膝盖弯曲（"坐椅子"降低重心）
    knee_bend_excellent: float = 120.0   # 度（越小越弯）
    knee_bend_good: float = 140.0
    knee_bend_poor: float = 155.0

    # P1.3 非持拍手辅助 — 两只手腕距离应小（都在球拍上）
    non_dom_hand_distance_good: float = 0.6   # 归一化为肩宽
    non_dom_hand_distance_warning: float = 1.0

    # P1.4 脊柱姿态
    spine_lean_good_max: float = 15.0
    spine_lean_warning: float = 25.0


# =====================================================================
# Phase 2: Backswing & Slot
# =====================================================================

@dataclass(frozen=True)
class OHB_BackswingConfig:
    """单反引拍阶段阈值。

    关键：L-for-Leverage（拍头在手腕上方），击球臂U形靠近身体。
    """
    # P2.1 "L形杠杆" — 拍头应在手腕上方（通过肘部角度近似）
    # 引拍时肘部角度应在 70-120°（形成L形）
    l_leverage_elbow_good_min: float = 70.0
    l_leverage_elbow_good_max: float = 120.0

    # P2.3 击球臂U形 — 肘部靠近身体（肘-髋距离归一化为躯干高度）
    arm_u_shape_good_max: float = 0.5   # 肘部距离身体中线 < 50% 躯干高度
    arm_u_shape_warning: float = 0.8


# =====================================================================
# Phase 3: Forward Swing & Kinetic Chain
# =====================================================================

@dataclass(frozen=True)
class OHB_KineticChainConfig:
    """单反动力链阈值。"""
    # P3.1 动力链顺序：脚→髋→肩→臂→拍
    max_segment_delay_s: float = 0.20
    min_segment_delay_s: float = 0.01

    # P3.2 髋肩分离角
    hip_shoulder_separation_good: float = 25.0
    hip_shoulder_separation_excellent: float = 40.0

    # P3.3 Inside-Out 挥拍路径线性度
    hand_path_linearity_good: float = 0.80
    hand_path_linearity_excellent: float = 0.92


# =====================================================================
# Phase 4: Contact Point (单反特有)
# =====================================================================

@dataclass(frozen=True)
class OHB_ContactConfig:
    """单反击球点阈值。

    关键差异：
    - 手臂必须从弯曲变为完全伸直（165-180°）
    - 非持拍手必须向后伸展（反向平衡）
    - 身体刹车更为关键
    """
    # P4.1 击球点位置（手腕在髋部前方，归一化为躯干高度）
    contact_forward_good_min: float = 0.25
    contact_forward_good_max: float = 0.75
    contact_forward_poor_min: float = 0.05

    # P4.2 手臂完全伸展 — 单反击球时手臂应接近完全伸直
    arm_extension_excellent_min: float = 160.0   # 度
    arm_extension_good_min: float = 145.0
    arm_extension_poor_min: float = 120.0

    # P4.3 身体刹车（躯干角速度，度/秒）
    body_freeze_good_max: float = 50.0    # 单反需要更严格的刹车
    body_freeze_warning: float = 100.0

    # P4.4 非持拍手反向平衡 — 两手距离应大（非持拍手向后伸展）
    non_dom_hand_spread_good: float = 1.5   # 归一化为肩宽
    non_dom_hand_spread_excellent: float = 2.0

    # P4.5 头部稳定性
    head_stability_good_max: float = 0.05
    head_stability_warning: float = 0.10


# =====================================================================
# Phase 5: Extension & Follow-Through (ATA)
# =====================================================================

@dataclass(frozen=True)
class OHB_ExtensionConfig:
    """单反延伸和随挥阈值。

    关键：ATA (Air The Armpit) — 高收拍，像自由女神像。
    """
    # P5.1 ATA — 随挥后手腕应高于肩部（归一化为躯干高度）
    ata_wrist_above_shoulder_good: float = 0.1    # 手腕至少高于肩部 10% 躯干高度
    ata_wrist_above_shoulder_excellent: float = 0.3

    # P5.2 保持侧身 — 击球后肩部旋转应仍然保持侧身
    # 击球后 0.2 秒内肩部旋转变化应小
    sidebody_rotation_change_good: float = 15.0   # 度
    sidebody_rotation_change_warning: float = 30.0

    # 分析窗口
    post_contact_window_s: float = 0.35


# =====================================================================
# Phase 6: Balance & Recovery
# =====================================================================

@dataclass(frozen=True)
class OHB_BalanceConfig:
    """单反平衡阈值。"""
    head_vertical_stability_good: float = 0.03
    head_vertical_stability_warning: float = 0.06
    spine_consistency_good: float = 5.0
    spine_consistency_warning: float = 10.0


# =====================================================================
# Scoring Weights (单反权重分配)
# =====================================================================

@dataclass(frozen=True)
class OHB_ScoringWeights:
    """单反评分权重。

    击球点权重最高（25%），准备和动力链各 20%。
    """
    preparation: float = 0.20
    backswing: float = 0.10
    kinetic_chain: float = 0.20
    contact: float = 0.25
    extension: float = 0.15
    balance: float = 0.10

    def as_dict(self) -> Dict[str, float]:
        return {
            "ohb_preparation": self.preparation,
            "ohb_backswing": self.backswing,
            "ohb_kinetic_chain": self.kinetic_chain,
            "ohb_contact": self.contact,
            "ohb_extension": self.extension,
            "ohb_balance": self.balance,
        }


# =====================================================================
# Master Configuration
# =====================================================================

@dataclass(frozen=True)
class BackhandConfig:
    """单反评估总配置。"""
    preparation: OHB_PreparationConfig = field(default_factory=OHB_PreparationConfig)
    backswing: OHB_BackswingConfig = field(default_factory=OHB_BackswingConfig)
    kinetic_chain: OHB_KineticChainConfig = field(default_factory=OHB_KineticChainConfig)
    contact: OHB_ContactConfig = field(default_factory=OHB_ContactConfig)
    extension: OHB_ExtensionConfig = field(default_factory=OHB_ExtensionConfig)
    balance: OHB_BalanceConfig = field(default_factory=OHB_BalanceConfig)
    scoring: OHB_ScoringWeights = field(default_factory=OHB_ScoringWeights)


DEFAULT_BACKHAND_CONFIG = BackhandConfig()
