"""诊断引擎 — 完整因果推理引擎。

接收 VLM 的纯视觉观察 + 量化数据，通过知识图谱进行因果推理，
输出包含根因追溯、量化验证、用户历史对比和训练建议的完整诊断。

核心流程：
  1. 观察→概念映射：把 VLM 视觉描述映射到知识图谱概念节点
  2. 概念→根因追溯：沿 causes 边追溯到最上游根因
  3. 量化验证：用量化数据验证/修正推理
  4. 用户历史对比：检查是否为反复出现的问题
  5. 训练建议：从图谱的 drills_for/diagnostic chains 找到对应 drill
  6. 生成最终诊断文本：中文自然语言诊断叙述

输入：
  - vlm_result: VLM 返回的分析（observation_v1 或 semi_structured_v5 格式）
  - metrics: 量化指标（同步性、scooping 深度、穿透值等）

输出：
  - 增强后的诊断 dict，包含完整因果推理结果
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ══════════════════════════════════════════════════════════════════════
#  观察→概念映射规则表
# ══════════════════════════════════════════════════════════════════════

OBSERVATION_TO_CONCEPT: List[Dict[str, Any]] = [
    # ── 肩膀相关 ──
    {"keywords": ["右肩低", "右肩比左肩低", "肩膀不平", "右肩下沉", "shoulder dip", "right shoulder lower"],
     "concept": "problem_p07", "frame_range": [1, 2],
     "severity": 0.8, "label": "Unit Turn时右肩下沉"},
    {"keywords": ["右肩低", "肩膀不平", "shoulder tilt"],
     "concept": "shoulder_tilt", "frame_range": [4],
     "severity": 0.3, "label": "击球时肩膀倾斜（正常）"},
    {"keywords": ["肩膀正对球网", "完全正面", "chest facing net", "正对", "chest open"],
     "concept": "problem_p11", "frame_range": [4, 5],
     "severity": 0.7, "label": "过度转体"},
    {"keywords": ["肩膀侧身", "侧对球网", "shoulder sideways", "shoulder closed"],
     "concept": "good_rotation", "frame_range": [4],
     "severity": 0.0, "label": "转体良好"},

    # ── 躯干相关 ──
    {"keywords": ["后仰", "身体往后倒", "上半身在后", "backward lean", "leaning back"],
     "concept": "trunk_leaning", "frame_range": None,
     "severity": 0.5, "label": "躯干后仰"},
    {"keywords": ["前倾", "身体前倾", "forward lean", "leaning forward"],
     "concept": "forward_lean", "frame_range": None,
     "severity": 0.4, "label": "躯干前倾"},

    # ── 手臂相关 ──
    {"keywords": ["手臂先动", "手臂在身体之前", "手先动", "arm leads", "arm initiating",
                   "手臂独立", "手臂主导", "arm driven", "小臂代偿", "小臂主动"],
     "concept": "problem_p03", "frame_range": None,
     "severity": 0.9, "label": "手臂独立动作/小臂代偿"},
    {"keywords": ["手臂紧贴", "肘部贴着", "贴着身体", "no elbow space", "arm pinned"],
     "concept": "problem_p14", "frame_range": None,
     "severity": 0.6, "label": "肘部空间不足"},
    {"keywords": ["手臂和身体一起", "同步", "arm body connected", "arm follows body"],
     "concept": "arm_body_connected", "frame_range": None,
     "severity": 0.0, "label": "手臂-身体连接良好"},

    # ── 球拍轨迹 ──
    {"keywords": ["V形", "V字", "急剧下降", "突然下坠", "scooping", "捞球", "V-shape",
                   "v shape", "拍头急降", "拍头下坠"],
     "concept": "problem_p02", "frame_range": None,
     "severity": 0.8, "label": "V形Scooping"},
    {"keywords": ["平滑弧线", "弧形", "smooth arc", "nike swoosh"],
     "concept": "smooth_arc", "frame_range": None,
     "severity": 0.0, "label": "挥拍弧线平滑"},
    {"keywords": ["向内收缩", "内收", "inward", "没有向外", "缺Out", "wrap around early"],
     "concept": "problem_p05", "frame_range": None,
     "severity": 0.6, "label": "缺乏向外延展"},
    {"keywords": ["向外延展", "向外", "outward", "extension", "穿透"],
     "concept": "swing_out", "frame_range": None,
     "severity": 0.0, "label": "有向外延展"},

    # ── 拍头下坠 ──
    {"keywords": ["拍头过度下坠", "pat the dog", "拍头急降再急升", "拍头下落过深"],
     "concept": "problem_p01", "frame_range": None,
     "severity": 0.7, "label": "拍头过度下坠"},

    # ── 下半身 ──
    {"keywords": ["膝盖直", "腿直", "没有弯曲", "straight legs", "no knee bend"],
     "concept": "straight_legs", "frame_range": None,
     "severity": 0.5, "label": "腿部未弯曲"},
    {"keywords": ["重心在后脚", "后脚承重", "weight back", "重心偏后"],
     "concept": "problem_p09", "frame_range": None,
     "severity": 0.7, "label": "重心在错误的脚上"},
    {"keywords": ["后脚离地", "跳起", "jumping", "airborne"],
     "concept": "jumping", "frame_range": None,
     "severity": 0.3, "label": "后脚离地"},
    {"keywords": ["后脚拧转", "脚尖", "pivot", "toe pivot"],
     "concept": "a6_pivot", "frame_range": None,
     "severity": 0.0, "label": "后脚拧转（正确）"},

    # ── 左手 ──
    {"keywords": ["左手垂", "左手放下", "左手在身侧", "left hand dropped", "左手早早放下"],
     "concept": "problem_p15", "frame_range": None,
     "severity": 0.6, "label": "左手未参与"},
    {"keywords": ["左手收胸", "左手在胸", "left hand braking", "左手刹车"],
     "concept": "non_hitting_hand_braking", "frame_range": None,
     "severity": 0.0, "label": "左手收胸刹车（正确）"},
    {"keywords": ["左手指向", "左手伸出", "left hand pointing"],
     "concept": "non_hitting_hand_pointing", "frame_range": None,
     "severity": 0.0, "label": "左手指向来球（正确）"},

    # ── 动力链 ──
    {"keywords": ["动力链断", "力量泄漏", "能量损失", "kinetic chain break", "脱节",
                   "力在中途", "力传不上"],
     "concept": "problem_p04", "frame_range": None,
     "severity": 0.8, "label": "动力链断裂"},
    {"keywords": ["转体不足", "没转体", "rotation insufficient", "no rotation", "转体小"],
     "concept": "unit_turn", "frame_range": None,
     "severity": 0.7, "label": "Unit Turn不足"},
    {"keywords": ["只转上半身", "髋没转", "肩转了髋没", "上半身转下半身没", "only upper body"],
     "concept": "upper_body_only_turn", "frame_range": None,
     "severity": 0.9, "label": "只转上半身不转髋（缺少核心蓄力）"},
    {"keywords": ["击球点偏后", "击球点靠后", "被挤压", "late contact", "cramped"],
     "concept": "problem_p05", "frame_range": None,
     "severity": 0.6, "label": "击球点偏后"},
    {"keywords": ["太急", "匆忙", "rushing", "时机太早"],
     "concept": "problem_p08", "frame_range": None,
     "severity": 0.5, "label": "击球太急"},
    {"keywords": ["背部松", "背部张力", "back tension lost"],
     "concept": "problem_p10", "frame_range": None,
     "severity": 0.7, "label": "背部张力松掉"},
    {"keywords": ["手主动引拍", "拉拍", "手往后拉", "arm pulling back"],
     "concept": "problem_p13", "frame_range": None,
     "severity": 0.6, "label": "手主动引拍"},

    # ══════════════════════════════════════════════════════════════════
    #  Preparation phase / Footwork concepts (L4-L5)
    #  Source: docs/research/coach_analysis/*.md
    # ══════════════════════════════════════════════════════════════════
    {"keywords": ["分腿晚", "split step晚", "split late", "落地晚", "对方击球时还没起跳"],
     "concept": "prep01_late_split_step", "frame_range": None,
     "severity": 0.85, "label": "分腿垫步过晚"},
    {"keywords": ["没有分腿", "没有split", "missing split", "完全没跳", "静止站立等球"],
     "concept": "prep02_no_split_step", "frame_range": None,
     "severity": 0.9, "label": "缺失分腿垫步"},
    {"keywords": ["跳得太高", "split太高", "stiff landing", "落地砸"],
     "concept": "prep03_split_step_too_high", "frame_range": None,
     "severity": 0.6, "label": "分腿过高/僵硬落地"},
    {"keywords": ["没pivot", "无pivot", "右脚没转", "no pivot"],
     "concept": "prep04_no_pivot", "frame_range": None,
     "severity": 0.85, "label": "缺失Pivot第一步"},
    {"keywords": ["pivot晚", "late pivot", "pivot过晚"],
     "concept": "prep05_late_pivot", "frame_range": None,
     "severity": 0.8, "label": "Pivot过晚"},
    {"keywords": ["碎步", "调整步太多", "choppy steps", "多次小步"],
     "concept": "prep06_choppy_steps", "frame_range": None,
     "severity": 0.5, "label": "碎步过多"},
    {"keywords": ["没有回复步", "no recovery", "击球后停留", "原地不动"],
     "concept": "prep07_no_recovery_step", "frame_range": None,
     "severity": 0.5, "label": "击球后无回复步"},
    {"keywords": ["unit turn晚", "准备晚", "late unit turn", "转身晚", "球落地后才转"],
     "concept": "prep08_late_unit_turn", "frame_range": None,
     "severity": 0.9, "label": "Unit Turn启动过晚"},
    {"keywords": ["手臂引拍", "只用手臂", "arm only turn", "单臂引拍", "肩没转手已动"],
     "concept": "prep09_arm_only_unit_turn", "frame_range": None,
     "severity": 0.9, "label": "只用手臂引拍（非整体转）"},
    {"keywords": ["肩胛没动", "肩胛骨未滑", "no scapular glide", "背没参与", "背部没褶皱"],
     "concept": "prep10_no_scapular_glide", "frame_range": None,
     "severity": 0.85, "label": "肩胛骨未滑动"},
    {"keywords": ["拍头早垂", "hold up失效", "拍头掉下来", "拍头没撑住"],
     "concept": "prep11_racket_head_dropped_early", "frame_range": None,
     "severity": 0.75, "label": "拍头过早下垂"},
    {"keywords": ["左手没搭", "左手没碰拍", "左手垂着", "left hand off racket"],
     "concept": "prep12_left_hand_dropped_in_unit_turn", "frame_range": None,
     "severity": 0.8, "label": "Unit Turn中左手未搭拍颈"},
    {"keywords": ["左肩没向前", "左肩未先动", "left shoulder not forward"],
     "concept": "prep13_left_shoulder_not_forward", "frame_range": None,
     "severity": 0.7, "label": "左肩未向前启动"},
    {"keywords": ["肩髋分离不足", "no separation", "no torque", "无扭矩", "肩髋没分开"],
     "concept": "prep14_insufficient_hip_shoulder_separation", "frame_range": None,
     "severity": 0.9, "label": "肩髋分离不足（无扭矩）"},
    {"keywords": ["假分离", "fake separation", "髋没动只转肩"],
     "concept": "prep15_fake_separation", "frame_range": None,
     "severity": 0.85, "label": "假分离（只转上身）"},
    {"keywords": ["顶点静止", "stop start", "引拍停顿", "顶点停住"],
     "concept": "prep16_stop_start_syndrome", "frame_range": None,
     "severity": 0.8, "label": "引拍顶点静止综合症"},
    {"keywords": ["球落地准备没完成", "球弹起还在引拍", "racket back late"],
     "concept": "prep17_prep_not_done_by_bounce", "frame_range": None,
     "severity": 0.9, "label": "球落地时准备未完成"},
    {"keywords": ["没有等待", "没有暂停", "一气呵成", "no wait", "no pause"],
     "concept": "prep18_no_wait_after_prep", "frame_range": None,
     "severity": 0.7, "label": "准备完未等待"},
    {"keywords": ["站姿窄", "narrow stance", "双脚太近"],
     "concept": "prep19_narrow_stance", "frame_range": None,
     "severity": 0.7, "label": "站姿过窄"},
    {"keywords": ["triple bend缺失", "踝膝髋未弯", "no triple bend", "腿没加载"],
     "concept": "prep20_no_triple_bend", "frame_range": None,
     "severity": 0.8, "label": "缺Triple Bend"},
    {"keywords": ["land then start", "落地后才动", "no landing in motion"],
     "concept": "prep21_no_landing_in_motion", "frame_range": None,
     "severity": 0.8, "label": "落地后才启动"},
    {"keywords": ["补步", "中性转半开放失败", "extra step"],
     "concept": "prep22_neutral_to_semi_open_failed", "frame_range": None,
     "severity": 0.55, "label": "中性站位转半开放失败"},
    {"keywords": ["press slot", "槽位没到", "slot未到达"],
     "concept": "prep23_no_press_slot", "frame_range": None,
     "severity": 0.75, "label": "未到达Press Slot"},
    {"keywords": ["主动后挥", "backswinging", "主动往后拉拍"],
     "concept": "prep24_backswinging_not_placing", "frame_range": None,
     "severity": 0.75, "label": "主动后挥而非Place"},
    {"keywords": ["multi split", "球机时只跳一次"],
     "concept": "prep25_no_multi_split_on_machine", "frame_range": None,
     "severity": 0.4, "label": "球机训练时无连续分腿"},
    {"keywords": ["手臂用来平衡", "持拍臂摆动", "arm balance"],
     "concept": "prep26_arm_used_for_balance", "frame_range": None,
     "severity": 0.6, "label": "跑动中用持拍臂平衡"},
    {"keywords": ["x stretch", "X拉伸缺失", "对角拉伸"],
     "concept": "prep27_no_x_stretch", "frame_range": None,
     "severity": 0.65, "label": "无X对角拉伸"},
    {"keywords": ["握拍松", "握拍死", "innervation缺失", "手指未支配"],
     "concept": "prep28_grip_not_innervated", "frame_range": None,
     "severity": 0.6, "label": "握拍未建立神经支配"},
    {"keywords": ["第一步晚", "late first step"],
     "concept": "prep29_late_first_step", "frame_range": None,
     "severity": 0.75, "label": "第一步过晚"},
    {"keywords": ["准备整体晚", "late preparation", "准备迟滞"],
     "concept": "prep30_late_preparation_general", "frame_range": None,
     "severity": 0.85, "label": "整体准备迟滞"},
    {"keywords": ["挥拍被挤压", "cramped swing", "没时间伸展"],
     "concept": "prep31_cramped_swing_no_extension", "frame_range": None,
     "severity": 0.8, "label": "挥拍被挤压无时间延展"},
    {"keywords": ["匆忙击球", "击球点偏后被迫", "rushed contact"],
     "concept": "prep32_forced_rush_contact_behind", "frame_range": None,
     "severity": 0.85, "label": "被迫匆忙击球点偏后"},
]


# ══════════════════════════════════════════════════════════════════════
#  概念→层级映射（L1=Contact, L2=Rhythm/Timing, L3=Kinetic Chain,
#                L4=Preparation, L5=Footwork）
#  用于 top-down 诊断：找最早出现问题的层 = 根因
# ══════════════════════════════════════════════════════════════════════

_CONCEPT_LAYER: Dict[str, str] = {
    # L1 — Contact（症状显示器）
    "problem_p05": "L1",  # 击球点偏后/缺向外延展
    "problem_p14": "L1",  # 肘部空间不足（contact 时表现）
    "prep31_cramped_swing_no_extension": "L1",
    "prep32_forced_rush_contact_behind": "L1",
    "shoulder_tilt": "L1",
    # L2 — Rhythm / Timing
    "problem_p08": "L2",  # 击球太急
    "problem_p12": "L2",  # separation 缺失
    "prep16_stop_start_syndrome": "L2",
    "prep17_prep_not_done_by_bounce": "L2",
    "prep18_no_wait_after_prep": "L2",
    # L3 — Kinetic Chain
    "problem_p01": "L3",  # 拍头过度下坠
    "problem_p02": "L3",  # V形 Scooping
    "problem_p03": "L3",  # 小臂代偿
    "problem_p04": "L3",  # 动力链断裂
    "problem_p10": "L3",  # 背部张力松
    "problem_p11": "L3",  # 过度转体
    "problem_p13": "L3",  # 手主动引拍
    "problem_p20": "L3",  # 腹斜肌未激活
    "upper_body_only_turn": "L3",
    "prep14_insufficient_hip_shoulder_separation": "L3",
    # L4 — Preparation / Unit Turn
    "problem_p07": "L4",  # 右肩下沉（准备阶段）
    "problem_p15": "L4",  # 左手未参与
    "unit_turn": "L4",
    "prep08_late_unit_turn": "L4",
    "prep09_arm_only_unit_turn": "L4",
    "prep10_no_scapular_glide": "L4",
    "prep11_racket_head_dropped_early": "L4",
    "prep12_left_hand_dropped_in_unit_turn": "L4",
    "prep13_left_shoulder_not_forward": "L4",
    "prep15_fake_separation": "L4",
    "prep23_no_press_slot": "L4",
    "prep24_backswinging_not_placing": "L4",
    "prep26_arm_used_for_balance": "L4",
    "prep27_no_x_stretch": "L4",
    "prep28_grip_not_innervated": "L4",
    "prep30_late_preparation_general": "L4",
    # L5 — Footwork (最底层根因)
    "problem_p09": "L5",  # 重心错脚
    "straight_legs": "L5",
    "prep01_late_split_step": "L5",
    "prep02_no_split_step": "L5",
    "prep03_split_step_too_high": "L5",
    "prep04_no_pivot": "L5",
    "prep05_late_pivot": "L5",
    "prep06_choppy_steps": "L5",
    "prep07_no_recovery_step": "L5",
    "prep19_narrow_stance": "L5",
    "prep20_no_triple_bend": "L5",
    "prep21_no_landing_in_motion": "L5",
    "prep22_neutral_to_semi_open_failed": "L5",
    "prep25_no_multi_split_on_machine": "L5",
    "prep29_late_first_step": "L5",
    # ── v2 wave-2 video-watching concepts ──
    "v2_no_side_bending": "L3",          # rotation power source missing
    "v2_takeback_speed_mismatch": "L2",  # rhythm/tempo behavior
    "v2_slow_motion_practice": "L5",     # ready-state inactive
    "v2_late_split_recognition": "L5",   # split airborne pre-recognition
    "v2_pivot_at_butt_cap": "L4",        # grip / preparation structural
    "v2_chasing_not_intercepting": "L1", # contact-point geometry
}

# Layer order: earliest = highest priority as root cause
_LAYER_ORDER = ["L5", "L4", "L3", "L2", "L1"]


def _get_concept_layer(concept_id: str) -> Optional[str]:
    """Return the diagnostic layer of a concept, or None if unknown."""
    return _CONCEPT_LAYER.get(concept_id)


def _find_earliest_layer_problem(
    matched_concepts: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Top-down diagnostic flow: find the earliest layer with a problem.

    Order: L5 (footwork) → L4 (preparation) → L3 (chain) → L2 (rhythm) → L1 (contact).
    The 'earliest layer' wins as root cause; downstream layers are compensations.

    Returns the matched-concept dict that is the earliest-layer problem,
    or None if no problems have known layers.
    """
    problems = [m for m in matched_concepts if m.get("severity", 0) > 0]
    if not problems:
        return None

    by_layer: Dict[str, List[Dict[str, Any]]] = {}
    for m in problems:
        layer = _get_concept_layer(m["mapped_concept"])
        if layer:
            by_layer.setdefault(layer, []).append(m)

    for layer in _LAYER_ORDER:
        candidates = by_layer.get(layer, [])
        if candidates:
            # Within the earliest layer, pick highest severity
            candidates.sort(key=lambda m: m.get("severity", 0), reverse=True)
            return candidates[0]
    return None


# ══════════════════════════════════════════════════════════════════════
#  量化指标阈值和含义
# ══════════════════════════════════════════════════════════════════════

_METRIC_THRESHOLDS = {
    "arm_torso_synchrony": {
        "good": 0.7, "ok": 0.4,
        "meaning": {
            "good": "手臂跟随身体旋转",
            "ok": "手臂部分跟随身体",
            "bad": "手臂完全独立于身体",
        },
    },
    "scooping_depth": {
        "threshold": 0.3,
        "meaning": {
            "detected": "手腕轨迹出现 V 形下坠",
            "normal": "手腕轨迹正常",
        },
    },
    "forward_extension": {
        "good": 0.4, "ok": 0.2,
        "meaning": {
            "good": "击球后有向前穿透",
            "ok": "穿透一般",
            "bad": "几乎没有向前穿透",
        },
    },
    "shoulder_rotation": {
        "good": 60, "ok": 30,
        "meaning": {
            "good": "转体充分",
            "ok": "转体一般",
            "bad": "转体严重不足",
        },
    },
    "elbow_angle_at_contact": {
        "good_range": (100, 170),
        "meaning": {
            "good": "肘角在健康区间",
            "bad_low": "肘部过度弯曲（夹肘）",
            "bad_high": "手臂过度伸直",
        },
    },
    "swing_arc_ratio": {
        "threshold": 3.0,
        "meaning": {
            "outward": "有向外延展（Out 向量）",
            "inward": "向内收缩（缺 Out）",
        },
    },
}


# ══════════════════════════════════════════════════════════════════════
#  概念→量化指标的验证映射
# ══════════════════════════════════════════════════════════════════════

_CONCEPT_TO_METRIC_VALIDATION: Dict[str, List[Dict[str, Any]]] = {
    "problem_p03": [  # 小臂代偿
        {"metric": "arm_torso_synchrony", "check": lambda v: v is not None and v < 0.4,
         "confirm_text": "同步性{val:.2f}证实手臂独立于身体运动",
         "contradict_text": "同步性{val:.2f}显示手臂与身体仍有跟随，视觉判断的手臂独立可能被放大"},
    ],
    "problem_p02": [  # V形 Scooping
        {"metric": "scooping_depth", "check": lambda v: v is not None and v > 0.3,
         "confirm_text": "Scooping深度{val:.2f}证实V形下坠",
         "contradict_text": "Scooping深度{val:.2f}未达阈值，V形可能是轻微的"},
        {"metric": "scooping_detected", "check": lambda v: v is True,
         "confirm_text": "量化算法检测到V形scooping",
         "contradict_text": "量化算法未检测到显著scooping，可能是拍摄角度放大了视觉印象"},
    ],
    "problem_p01": [  # 拍头过度下坠
        {"metric": "scooping_depth", "check": lambda v: v is not None and v > 0.3,
         "confirm_text": "拍头下坠深度{val:.2f}证实过度下落",
         "contradict_text": "下坠深度{val:.2f}在正常范围"},
    ],
    "unit_turn": [  # Unit Turn不足
        {"metric": "shoulder_rotation", "check": lambda v: v is not None and v < 30,
         "confirm_text": "肩部转开只有{val:.0f}°（正常应>60°），转体确实严重不足",
         "contradict_text": "肩部转开{val:.0f}°，转体并非不足"},
    ],
    "straight_legs": [  # 下肢未承载（与 _KPI_INJECTION_RULES 阈值对齐）
        {"metric": "min_knee_angle", "check": lambda v: v is not None and v > 150,
         "confirm_text": "最小膝角{val:.0f}°（应≤140°），下肢承载偏弱",
         "contradict_text": "最小膝角{val:.0f}°，下肢承载充分"},
    ],
    "problem_p11": [  # 过度转体
        {"metric": "shoulder_rotation", "check": lambda v: v is not None and v > 90,
         "confirm_text": "肩部转开{val:.0f}°，确实过度",
         "contradict_text": "肩部转开{val:.0f}°在正常范围，过度转体可能不成立"},
    ],
    "problem_p05": [  # 击球点偏后/缺向外延展
        {"metric": "forward_extension", "check": lambda v: v is not None and v < 0.2,
         "confirm_text": "向前穿透只有{val:.2f}，击球后确实缺乏延展",
         "contradict_text": "向前穿透{val:.2f}，延展尚可"},
        {"metric": "swing_arc_ratio", "check": lambda v: v is not None and v > 5,
         "confirm_text": "挥拍弧度比{val:.1f}，轨迹严重内收",
         "contradict_text": "挥拍弧度比{val:.1f}在正常范围"},
    ],
    "problem_p14": [  # 肘部空间不足
        {"metric": "elbow_angle_at_contact",
         "check": lambda v: v is not None and v < 100,
         "confirm_text": "击球肘角{val:.0f}°，肘部过度弯曲",
         "contradict_text": "击球肘角{val:.0f}°在正常范围"},
    ],
}


# ══════════════════════════════════════════════════════════════════════
#  概念→修复建议映射（从 diagnostic chains + 知识图谱 drill 节点）
# ══════════════════════════════════════════════════════════════════════

_CONCEPT_TO_FIX: Dict[str, Dict[str, str]] = {
    "problem_p03": {
        "drill": "外旋+沉肩锁定练习",
        "method": "Unit Turn 时轻微外旋持拍手+沉肩。感觉到背部肩胛骨区域和胸侧三角区同时绷紧。然后只靠转体挥拍。",
        "why": "外旋激活冈下肌+小圆肌（后），沉肩激活背阔肌（下），胸大肌（前）同时工作。三面锁住手臂，手臂不可能独立动作。",
        "muscle_cue": "做对时你应该感觉到：背部肩胛骨区域绷紧（冈下肌+小圆肌），胸侧跨腋下的三角区绷紧（胸大肌）。如果只感觉到手臂在用力但背部没感觉，说明锁定没到位。",
    },
    "upper_body_only_turn": {
        "drill": "双手抱球整体转体",
        "method": "双手抱篮球在胸前做 Unit Turn。因为双手被球锁住，身体被迫整体转（包括髋部）。转到位后暂停 1 秒，感受右腿臀部后侧的拉伸感。每天 20 次。",
        "why": "只转上半身不转髋 = 没有核心蓄力 = 后续所有发力都得靠手臂代偿。抱球强制髋部跟着转，重建整体转动的肌肉记忆。",
        "muscle_cue": "转到位时你应该感觉到：臀部和大腿后侧有明显拉伸感（臀肌+髋关节离心收缩在蓄力）。如果只感觉到腰在扭但臀部没感觉，说明髋没有真正参与转体。",
    },
    "unit_turn": {
        "drill": "对镜 Unit Turn 深度检验",
        "method": "对着镜子做 Unit Turn，到位后暂停。低头看下巴是否贴到右肩——贴到=够了，看到胸口=远没转够。配合：右脚先后撤半步落稳重心→再整体转→右肩转到下巴下。每组 10 次。",
        "why": "肩部转开 < 60° 等于核心没拉开，腹斜肌没有弹性势能，蹬地后没力量可释放，下游必然出现手臂代偿。先转开度，再谈鞭打。",
        "muscle_cue": "转开到位时你应该感觉到：左侧腹斜肌被拉伸（像拉满的弓）+背阔肌张力上升+右大腿后侧绷紧。如果只感觉肩膀转但腹部没拉伸感，说明只转了上半身没转髋。",
    },
    "straight_legs": {
        "drill": "Triple Bend 蹲位准备",
        "method": "Unit Turn 时刻意把膝盖压过脚尖（≤140°），重心明显沉下去半个身位。大腿前侧应有承重感，臀部下沉。前挥时从这个低位蹬地起来。",
        "why": "腿直 = 没有蓄力的弹性势能 = 蹬地输出 0 = 只能用手臂。下肢承载是动力链最底层的发动机，腿不弯，上面所有的转体、Press、鞭打都是空架子。",
        "muscle_cue": "做对时你应该感觉到：股四头肌（大腿前）+臀大肌共同承重，膝关节有'压住弹簧'的感觉。如果膝盖几乎是直的、感觉不到大腿承重，说明腿没参与发力。",
    },
    "problem_p02": {
        "drill": "平推穿透练习",
        "method": "站在离网2米处，用拍面平推球过网，不允许任何向上动作。目标是让球平飞过网。",
        "why": "V形scooping的根因是手臂做了独立的下压+上翻。平推练习消除向上意图，重建水平穿透路径。",
        "muscle_cue": "正确的前挥你应该感觉到：背部有'先存后放'的切换感——引拍时背阔肌拉伸蓄力，前挥时从背部释放力量向前。如果感觉是手腕在主动翻转，说明背阔肌的离心→向心切换没有发生。",
    },
    "problem_p01": {
        "drill": "拍头高度控制练习",
        "method": "引拍后在击球手一侧放置一个标记物（高度约膝盖），前挥时拍头不能低于该标记。",
        "why": "拍头过度下坠源于小臂主动做pat the dog。限制下落幅度强制使用身体旋转而非手臂下压。",
    },
    "problem_p07": {
        "drill": "双肩端平引拍练习",
        "method": "Unit Turn时口令'扁担水平'，想象肩膀上放着一根扁担。对着镜子检查两肩是否保持水平。",
        "why": "右肩下沉会把手带到低位，导致重心偏移和后续的scooping。保持肩膀水平是从源头切断问题。",
    },
    "problem_p11": {
        "drill": "左手收胸刹车练习",
        "method": "击球瞬间左手主动收向右胸口，感受身体被'刹住'。随挥是惯性不是主动继续转。",
        "why": "过度转体的根因是缺少刹车机制。左手收胸提供反向力矩，让旋转脉冲终点=击球瞬间。",
        "muscle_cue": "刹车成功时你应该感觉到：腹部侧面有一股'刹车感'（腹斜肌在做离心制动）。如果转体停不住、整个人跟着转过去，说明腹斜肌的离心制动没有启动。",
    },
    "problem_p04": {
        "drill": "蹬地→转髋→腹斜肌串联练习",
        "method": "站立，右脚蹬地→感受力经髋部→左下腹（肚脐左侧）有拧转感→胸部被带走。从慢速开始。",
        "why": "动力链断裂通常在3个点：膝-髋、髋-腹斜肌、胸-臂。逐段串联让能量完整传递。",
        "muscle_cue": "串联做对时你应该依次感觉到：脚掌蹬地→臀部发力→肚脐左侧有拧转感（腹斜肌向心收缩）→胸部被带着转。哪个环节'断感觉'了，动力链就断在哪。",
    },
    "problem_p05": {
        "drill": "弹...落...打 三拍节奏练习",
        "method": "球弹起时说'弹'，最高点说'落'，下落到击球高度说'打'。退后半步给自己更多时间。",
        "why": "击球点偏后的根因通常是站太前+准备时间不足。三拍节奏强制建立时空余量。",
        "muscle_cue": "击球时向外延展做对了，你应该感觉到：胸部有'往前推'的感觉（前锯肌+胸大肌做向心收缩）。如果击球后手臂立刻往身体内侧收，说明胸部肌肉的向心推送不够。",
    },
    "problem_p09": {
        "drill": "后脚蹬地启动练习",
        "method": "引拍后先确认重心在右脚（右手持拍），然后从右脚蹬地开始前挥。感受力从地面传上来。",
        "why": "重心在后脚（左脚）无法蹬地启动动力链。通过明确右脚重心+蹬地，重建正确的力量传递起点。",
    },
    "problem_p10": {
        "drill": "背部张力保持练习",
        "method": "引拍后感受背阔肌张力，整个前挥过程保持这个张力不松。击球后才允许释放。",
        "why": "背部松掉导致手臂失去'根'，变成独立运动。背部张力是手臂连接身体的关键中转站。",
        "muscle_cue": "引拍到位时你应该感觉到：背部两侧（背阔肌）有拉伸感，像拉开的弹弓。根据运动生物力学研究，引拍阶段背阔肌做离心收缩来储存弹性势能。如果引拍时背部没有这种'被拉开'的感觉，说明背阔肌没有参与。",
    },
    "problem_p13": {
        "drill": "左手引导Unit Turn练习",
        "method": "引拍时只想着左手把拍子送到右边，右手完全被动。左手引导=整体转动，右手拉拍=手臂独立。",
        "why": "手主动引拍意味着手臂脱离了整体转动系统。左手引导让Unit Turn回归'整体'。",
    },
    "problem_p14": {
        "drill": "肘部空间感知练习",
        "method": "引拍完成后，检查肘部是否有一拳的空间。如果贴身，主动把肘部推离身体约15cm。",
        "why": "肘部空间不足限制了挥拍的自由度，导致小臂代偿。保持空间让手臂成为鞭子的自然一环。",
    },
    "problem_p15": {
        "drill": "左手指向来球练习",
        "method": "判断来球后左手主动指向来球方向，Unit Turn时左手引导转体，击球时收向右胸。",
        "why": "左手参与Unit Turn确保整体转动+提供刹车。左手缺失=右手被迫主动拉拍。",
    },
    "problem_p08": {
        "drill": "节奏延迟练习",
        "method": "有意识地在球到达前多等0.5秒。口令'等...等...打'。宁可晚也不要早。",
        "why": "太急导致separation缺失和准备不足。刻意延迟重建时间余量。",
    },
    "problem_p12": {
        "drill": "Separation意识练习",
        "method": "引拍完成后，在前挥开始前感受一个'停顿'——这个停顿就是separation。慢动作反复体会。",
        "why": "Separation是引拍到前挥的过渡，缺失会导致发力突然和拍头急降。",
    },
    "problem_p20": {
        "drill": "腹斜肌激活练习",
        "method": "站立，双手交叉放胸前，蹬地后感受肚脐左方有拧转感。这个拧转感=腹斜肌在工作。",
        "why": "腹斜肌是腿→胸的能量中转站。激活不了=力在腰部断掉。",
        "muscle_cue": "蹬地后你应该感觉到：肚脐左方有拧转感（腹斜肌向心收缩在传递旋转力）。如果蹬地后只感觉到腿在用力、腰部以上没感觉，说明腹斜肌没有被激活。",
    },

    # ── Preparation / Footwork drills (L4-L5) ──
    "prep01_late_split_step": {
        "drill": "Land-as-they-hit 节拍练习",
        "method": "对方挥拍引拍时你起跳；对方拍触球瞬间你正好落地。可对着镜子或视频反复跟练10次。",
        "why": "Tomaz/Tom 共识：split的目的是把你转入弹簧加载状态，不是装饰。落地必须与对手contact同帧。",
        "muscle_cue": "落地瞬间你应该感觉到：脚掌像踩到弹簧——前掌着地、跟腱被快速拉伸（腓肠肌+比目鱼肌的离心储能）。如果是脚跟先落或全脚掌'砸'下去，说明split变成了stiff jump。",
    },
    "prep02_no_split_step": {
        "drill": "强制三跳节奏",
        "method": "left-right-both 连续节奏，无停顿；对手每次挥拍前你都做一次小split。先在原地练1分钟。",
        "why": "完全不split=反应慢半拍。强制小幅垫步重建'地面是弹簧'的肌肉记忆。",
        "muscle_cue": "三跳做对时小腿应有连续的'弹起'感，膝盖不深屈；如果小腿没感觉就是站着等球。",
    },
    "prep03_split_step_too_high": {
        "drill": "矮split练习",
        "method": "故意把split做到'几乎看不见'的高度——双脚只离地2-3cm，落地立即向任意方向爆发。",
        "why": "Tomaz: 'small, low, loaded'。高split=误解为jump=丢失反应窗口。",
    },
    "prep04_no_pivot": {
        "drill": "Pivot & Hold",
        "method": "教练喊'now'，你立刻右脚外旋pivot+unit turn并保持1秒，再正常击球。隔离第一步动作。",
        "why": "Tom: pivot是动力链的物理起点；缺pivot=横向滑出=unit turn只剩手臂。",
        "muscle_cue": "pivot做对时右脚臀部外侧应有发力感（髋外旋肌群），右脚后跟轻微离地。如果只感觉脚掌在拖地，说明髋外旋肌没参与。",
    },
    "prep05_late_pivot": {
        "drill": "脚先动 shadow",
        "method": "无球shadow swing，强制顺序：右脚pivot→髋→肩→手。每段之间留半拍。慢速10次/快速10次。",
        "why": "Tom Allsopp: 脚带动髋带动肩。pivot晚=链条倒置=cramped swing。",
    },
    "prep06_choppy_steps": {
        "drill": "两步到位练习",
        "method": "限定从split到contact只能2步：pivot+一个发力步。如果到不了，就退后一步起跳。",
        "why": "碎步=第一步方向错。强制2步迫使你用预判而不是补救。",
    },
    "prep07_no_recovery_step": {
        "drill": "Recovery split drill",
        "method": "每次击球后必须完成一次小split才能等下一球。教练有意拖后下一球的喂球节奏。",
        "why": "Tom: 'recovery是下一拍split的开始'。无recovery=下一拍永远晚一步。",
    },
    "prep08_late_unit_turn": {
        "drill": "球落地前完成转身 drill",
        "method": "教练慢速喂球，要求'球过网时已完成unit turn'。眼盯球落点，但身体已经转好等待。",
        "why": "Tomaz铁律: racket back before the ball bounces. 晚turn→所有下游全错。",
        "muscle_cue": "做对时背部（背阔肌）应有'被拉开'的拉伸感；如果转完背部没感觉，说明只转了肩没拉到背。",
    },
    "prep09_arm_only_unit_turn": {
        "drill": "双手抱球unit turn",
        "method": "双手抱篮球做unit turn。因双手锁住，身体被迫整体转，右臂无法独立后拉。每天20次。",
        "why": "Tom + Tomaz共识：'turn as a unit'。抱球强制肩、髋、双手同步。",
        "muscle_cue": "做对时你应该感觉到：背部肩胛区紧绷（rhomboids+lats），左肩明显往前推。如果只感觉右肩在往后拉，说明unit turn未发生。",
    },
    "prep10_no_scapular_glide": {
        "drill": "肩胛滑动激活",
        "method": "面对墙站立，掌心贴墙，做肩胛后缩→前推的'滑动门'动作（不动手臂）。10×3组。然后带入shadow swing。",
        "why": "FTT: 'The back is the glue'。肩胛滑动是手臂连接动力链的桥梁，缺失=必然小臂代偿。",
        "muscle_cue": "做对时你应该感觉到：肩胛骨在肋骨上滑动（菱形肌+前锯肌交替工作）。如果只感觉肩膀在转动而背部无任何'滑动'感，说明肩胛被锁死了。",
    },
    "prep11_racket_head_dropped_early": {
        "drill": "Hold it Up shadow",
        "method": "Unit turn结束后，拍头必须停在比手腕高的位置3秒。然后才允许前挥。10次。",
        "why": "FTT口令: 'Hold it up'。拍头主动下垂=press slot错位+前挥距离过长。",
    },
    "prep12_left_hand_dropped_in_unit_turn": {
        "drill": "Left hand on the throat",
        "method": "Tom Allsopp经典：左手压住拍颈占引拍90%过程。强迫上身整体转。10次shadow + 10次喂球。",
        "why": "左手是unit turn的'同步锚'。左手垂着=右臂必然单独动。",
    },
    "prep13_left_shoulder_not_forward": {
        "drill": "左肩启动 shadow",
        "method": "Tomaz feel cue：只想'左肩往前推'，让球拍自然到准备位。右手完全被动。20次/天。",
        "why": "Tomaz: 'go forward with left shoulder first'。思维倒置避免单独的右臂后拉。",
        "muscle_cue": "做对时你应该感觉到：左侧胸大肌+前三角肌发力把左肩'推'向球网方向。如果是右肩在往后拉，说明启动顺序错了。",
    },
    "prep14_insufficient_hip_shoulder_separation": {
        "drill": "Golf separation feel",
        "method": "徒手模仿引拍，肩转到底但髋停住——感受'拉满弓'的对角线张力。10次保持5秒。",
        "why": "Tom: 35-55°分离=理想扭矩。无扭矩=无power。",
        "muscle_cue": "做对时你应该感觉到：左侧腹斜肌（肚脐左侧到肋下）有强烈拉伸感。如果腰部完全无感觉，说明肩髋一起转了。",
    },
    "prep15_fake_separation": {
        "drill": "髋必须先动 drill",
        "method": "Shadow swing时口令'髋先5°'：引拍到顶后，髋必须先反向动一点点再让肩松开。慢动作20次。",
        "why": "假分离=只转上身=没有真正蓄力。强制髋部参与。",
    },
    "prep16_stop_start_syndrome": {
        "drill": "Continuous loop shadow",
        "method": "Tom Allsopp: 从引拍顶点直接拉向球，不允许中间静止。想象'过山车自由落体'。10次。",
        "why": "顶点停顿=引擎熄火=后续choppy+小臂代偿。",
    },
    "prep17_prep_not_done_by_bounce": {
        "drill": "球弹起前完成 drill",
        "method": "教练慢喂，要求'球落地瞬间你的unit turn已完成'。盯球落点同时身体已侧。20球。",
        "why": "Tomaz铁律。完成晚=必然contact偏后。",
    },
    "prep18_no_wait_after_prep": {
        "drill": "切分版正手",
        "method": "做'转身→停1秒→挥拍'三段切分。对比一气呵成版的手臂感觉差异。10次切分+10次正常。",
        "why": "Tomaz: 'turn quickly, then wait'。准备和挥拍混在一起=总打晚。",
    },
    "prep19_narrow_stance": {
        "drill": "宽站姿习惯",
        "method": "在地上画两条线，间距=肩宽×1.5。每次准备脚必须落在线外。20球。",
        "why": "FTT: 站姿宽于肩才能triple bend并稳定重心。窄站=拍头轨迹不稳。",
    },
    "prep20_no_triple_bend": {
        "drill": "踝膝髋同弯",
        "method": "准备时口令'踝、膝、髋'三个关节同时弯到合适角度（约140°/130°/120°）。镜子检查。10次。",
        "why": "FTT: triple bend是动力链加载的统一视觉判据。",
        "muscle_cue": "做对时你应该感觉到：股四头肌+臀肌+小腿同时被加载（像三条弹簧一起压紧）。如果只感觉膝盖在弯，说明髋和踝没参与。",
    },
    "prep21_no_landing_in_motion": {
        "drill": "Landing-in-motion drill",
        "method": "split落地的同时第一步必须已经在出发。'落地=出发'同帧。20次shadow。",
        "why": "FTT: 'land then start'是错过窗口。落地必须已经在动。",
    },
    "prep22_neutral_to_semi_open_failed": {
        "drill": "1次pivot到位",
        "method": "限定从split到semi-open站位只用1次pivot+1次调位。不允许补步。慢速练10次。",
        "why": "Tom: 'less steps is more time'。补步=预判迟钝。",
    },
    "prep23_no_press_slot": {
        "drill": "推墙Press Slot测试",
        "method": "FTT经典：把手放到准备结束的位置，对墙轻推。如果能稳定传力到地面=slot对了。每天5次。",
        "why": "FTT: slot是动力链的物理入口。slot错位=后续盲发力。",
    },
    "prep24_backswinging_not_placing": {
        "drill": "Place, Pull Forward 口令",
        "method": "口令从'后挥'改成'放置→拉向前'。想象只是把手'放'到位，不主动后挥。20次shadow。",
        "why": "FTT: 'There is no backswing'。心理表象错=动作错。",
    },
    "prep25_no_multi_split_on_machine": {
        "drill": "球机多次小split",
        "method": "球机训练时每次击球间做2-3次连续小split保持神经兴奋。",
        "why": "FTT: 无对手节奏时单次split=神经兴奋下降=比赛找不到节奏。",
    },
    "prep26_arm_used_for_balance": {
        "drill": "Arm in Ready State跑动",
        "method": "shadow跑动时持拍臂始终保持在unit turn结构中。让腿和核心负责平衡。",
        "why": "FTT Apparatus原则：手臂离开准备状态=apparatus失刚性。",
    },
    "prep27_no_x_stretch": {
        "drill": "X对角拉伸感知",
        "method": "Unit turn到位后，主动感受'左肩↔右髋'对角线的拉长。保持3秒。10次。",
        "why": "FTT: X stretch是肩髋分离的可视证据。",
        "muscle_cue": "做对时你应该感觉到：左肩到右髋的对角线（穿过腹部）有明显拉伸感（腹外斜肌+背阔肌的对角线张力）。",
    },
    "prep28_grip_not_innervated": {
        "drill": "Innervate the racket",
        "method": "准备时主动用食指指节背、小指掌根'感受'拍柄。不死握、不松垮，建立触觉映射。",
        "why": "FTT: 神经支配未建立=动力链盲发力。",
    },
    "prep29_late_first_step": {
        "drill": "落地即出发",
        "method": "split落地后允许的最大延迟=1帧（60fps）。视频自检。",
        "why": "第一步晚=所有下游全晚。",
    },
    "prep30_late_preparation_general": {
        "drill": "整套节奏重建",
        "method": "组合：land-as-they-hit + pivot+unit-turn + racket-back-before-bounce + wait + accelerate。慢速20次。",
        "why": "整体准备迟滞通常由split晚+pivot晚+unit turn晚复合造成，需要整套重建。",
    },
    "prep31_cramped_swing_no_extension": {
        "drill": "退半步给空间",
        "method": "刻意退后半步给身体延展空间。配合'弹...落...打'三拍节奏。",
        "why": "挥拍被挤压的根因通常是站太前+pivot晚。退一步立刻有空间。",
    },
    "prep32_forced_rush_contact_behind": {
        "drill": "弹...落...打 + 早转身",
        "method": "球弹起时说'弹'，最高点说'落'，下落到击球高度说'打'。同时确保球弹起前unit turn已完成。",
        "why": "匆忙=准备晚的最终下游。修根因（早转身）+修节奏（三拍）。",
    },

    # ── v2 wave-2 concepts (from video-watching insights) ──
    "v2_no_side_bending": {
        "drill": "药球对拉 + 侧弯感受",
        "method": "Johnny FTT经典：双手持药球，做正手挥拍模拟，但脊柱必须主动向左侧倾15°。镜子检查：肩线和地面之间不再是垂直角。",
        "why": "直立旋转 = 机械僵硬。脊柱侧弯 + X-stretch 才是真正的旋转power来源。",
        "muscle_cue": "做对时左侧腹外斜肌应明显收缩、右侧腰方肌被拉长。如果只感觉肩在转、腰部毫无感觉，说明侧弯没发生。",
    },
    "v2_takeback_speed_mismatch": {
        "drill": "Slow ball, slow takeback",
        "method": "Tom Allsopp 口令：来球慢→引拍也慢；来球快→引拍才快。教练故意给慢喂球，强迫你把引拍速度也降下来。20球。",
        "why": "引拍远快于来球 = 早早做完动作然后停下等球 = stop-start syndrome 的根因。引拍速度是用来匹配来球节奏的，不是越早越好。",
    },
    "v2_slow_motion_practice": {
        "drill": "Multi-Split-Step on machine",
        "method": "FTT Johnny: 球机训练时每次击球间做2-3次连续小split垫步，禁止完全静止。配合用力呼气（听得见喷气声）。",
        "why": "等球时身体完全静止 = 神经兴奋下降 = 比赛级反应消失。业余练习与比赛之间的真正差距就在这30帧里。",
        "muscle_cue": "做对时小腿应一直有连续低幅度的'弹起'感（腓肠肌持续微张力）。完全松弛 = slow motion 模式。",
    },
    "v2_late_split_recognition": {
        "drill": "落地前先看球",
        "method": "Quentin/Federer footwork drill：起跳瞬间眼睛+躯干必须已经朝向来球方向，落地即第一步。镜前shadow 20次。",
        "why": "落地后才转 = 浪费1-2帧 = 第一步永远晚。Federer的秘诀是在空中就完成预判。",
    },
    "v2_pivot_at_butt_cap": {
        "drill": "两指挥拍 drill",
        "method": "Johnny FTT 经典：只用拇指+食指捏住拍柄上部做shadow swing。强迫支点上移到虎口。10次后回归全握，保留触觉。",
        "why": "支点在底盖 = lag 物理上不可能产生。这不是动作错，是结构性问题。修了它，lag 自动出现。",
    },
    "v2_chasing_not_intercepting": {
        "drill": "Intercept the ball",
        "method": "Tomaz Feel Tennis 口令：'拍头和球相向而去'。教练从背后抛球，强迫你主动上前拦截而不是退着等。20球。",
        "why": "拍头追着球走 = 击球点偏后 = 失去掌控。Tomaz: 'Don't respond to speed with speed'，主动拦截才是正解。",
    },
}


# ══════════════════════════════════════════════════════════════════════
#  知识图谱缓存和加载
# ══════════════════════════════════════════════════════════════════════

_GRAPH_SNAPSHOT_PATH = Path(__file__).parent.parent / "knowledge" / "extracted" / "_graph_snapshot.json"
_PREP_CONCEPTS_PATH = Path(__file__).parent.parent / "knowledge" / "extracted" / "preparation_footwork_concepts.json"
_DIAGNOSTIC_CHAINS_PATH = Path(__file__).parent.parent / "knowledge" / "extracted" / "ftt_video_diagnostic_chains.json"
_USER_HISTORY_PATH = Path(__file__).parent.parent / "knowledge" / "extracted" / "user_journey" / "learning_deep_analysis.json"

_cached_graph_data: Optional[Dict] = None
_cached_chains: Optional[List[Dict]] = None
_cached_user_history: Optional[Dict] = None


def _load_graph_data() -> Dict:
    """Load and cache the graph snapshot, merging in preparation/footwork concepts."""
    global _cached_graph_data
    if _cached_graph_data is None:
        try:
            _cached_graph_data = json.loads(_GRAPH_SNAPSHOT_PATH.read_text())
        except Exception:
            _cached_graph_data = {"nodes": [], "edges": []}
        # Merge in preparation/footwork concepts (additive, idempotent)
        try:
            prep_data = json.loads(_PREP_CONCEPTS_PATH.read_text())
            existing_ids = {n.get("id") for n in _cached_graph_data.get("nodes", [])}
            for node in prep_data.get("nodes", []):
                if node.get("id") and node["id"] not in existing_ids:
                    _cached_graph_data.setdefault("nodes", []).append(node)
                    existing_ids.add(node["id"])
            # Merge edges (normalize keys to match snapshot format)
            for edge in prep_data.get("edges", []):
                normalized = dict(edge)
                normalized.setdefault("source", edge.get("source_id", ""))
                normalized.setdefault("target", edge.get("target_id", ""))
                _cached_graph_data.setdefault("edges", []).append(normalized)
        except Exception:
            pass
    return _cached_graph_data


def _load_chains() -> List[Dict]:
    """Load and cache diagnostic chains."""
    global _cached_chains
    if _cached_chains is None:
        try:
            data = json.loads(_DIAGNOSTIC_CHAINS_PATH.read_text())
            _cached_chains = data.get("chains", [])
        except Exception:
            _cached_chains = []
    return _cached_chains


def _load_user_history() -> Dict:
    """Load and cache user training history analysis."""
    global _cached_user_history
    if _cached_user_history is None:
        try:
            _cached_user_history = json.loads(_USER_HISTORY_PATH.read_text())
        except Exception:
            _cached_user_history = {}
    return _cached_user_history


# ══════════════════════════════════════════════════════════════════════
#  Step 1: 观察→概念映射
# ══════════════════════════════════════════════════════════════════════

def _extract_observations_text(vlm_result: Dict) -> List[Dict[str, Any]]:
    """Extract observation texts from VLM result, tagged with frame info.

    Returns list of {"text": str, "frame": int_or_None, "field": str}
    """
    observations = []
    fmt = vlm_result.get("format", "")

    # observation_v1 format: per-frame observations
    frames = vlm_result.get("frames", {})
    if frames and isinstance(frames, dict):
        for frame_num, frame_data in frames.items():
            if isinstance(frame_data, dict):
                for field, text in frame_data.items():
                    if text and isinstance(text, str) and text.strip():
                        try:
                            fn = int(frame_num)
                        except (ValueError, TypeError):
                            fn = None
                        observations.append({
                            "text": text.strip(),
                            "frame": fn,
                            "field": field,
                        })
        # Also check overall section
        overall = vlm_result.get("overall", {})
        if isinstance(overall, dict):
            for field, text in overall.items():
                if text and isinstance(text, str) and text.strip():
                    observations.append({
                        "text": text.strip(),
                        "frame": None,
                        "field": f"overall_{field}",
                    })

    # semi_structured_v5: extract from root_cause_tree and issues
    tree = vlm_result.get("root_cause_tree")
    if isinstance(tree, dict):
        for key in ["root_cause", "root_cause_evidence", "causal_explanation"]:
            val = tree.get(key, "")
            if val and isinstance(val, str):
                observations.append({"text": val.strip(), "frame": None, "field": key})

    issues = vlm_result.get("issues", [])
    if isinstance(issues, list):
        for issue in issues:
            if isinstance(issue, dict):
                for key in ["name", "description"]:
                    val = issue.get(key, "")
                    if val and isinstance(val, str):
                        observations.append({"text": val.strip(), "frame": None, "field": f"issue_{key}"})

    # Also check raw text fields
    for key in ["core_diagnosis", "overall_narrative", "overall_assessment"]:
        val = vlm_result.get(key, "")
        if val and isinstance(val, str) and val.strip():
            observations.append({"text": val.strip(), "frame": None, "field": key})

    return observations


# ══════════════════════════════════════════════════════════════════════
#  Q 编号直接映射表（v4 精确映射）
# ══════════════════════════════════════════════════════════════════════

Q_DIRECT_MAPPING: Dict[str, Dict[str, Any]] = {
    "Q1": {  # 手臂身体同步
        "positive_signals": ["一起动", "同步", "跟着", "一起走", "连着", "follow"],
        "negative_signals": ["自己先", "先动", "独立", "手臂先", "先开始", "脱节", "不同步", "先于"],
        "positive_concept": "arm_body_connected",
        "negative_concept": "problem_p03",
        "severity": 0.9,
    },
    "Q3": {  # 胸臂间隙变化
        "positive_signals": ["没有变化", "一直贴着", "保持", "紧贴"],
        "negative_signals": ["分开", "变大", "脱离", "离开", "间隙增大"],
        "positive_concept": "arm_body_connected",
        "negative_concept": "problem_p03",
        "severity": 0.8,
    },
    "Q4": {  # 准备阶段肩膀水平
        "positive_signals": ["同一水平", "平的", "水平", "一样高"],
        "negative_signals": ["右肩低", "右肩比左肩低", "不平", "右边低"],
        "positive_concept": "good_shoulder_level",
        "negative_concept": "problem_p07",
        "severity": 0.7,
    },
    "Q5b": {  # 髋部是否跟着肩膀一起转
        "positive_signals": ["一起转", "跟着转", "髋部也转", "整体转", "都转了"],
        "negative_signals": ["只有肩膀", "髋没动", "髋部没转", "只转上半身", "肩膀在转髋部没", "差很多"],
        "positive_concept": "good_unit_turn",
        "negative_concept": "upper_body_only_turn",
        "severity": 0.9,
    },
    "Q6": {  # 躯干后仰
        "positive_signals": ["直立", "保持直立", "没有后仰"],
        "negative_signals": ["后仰", "往后", "后倾", "后倒"],
        "positive_concept": "good_posture",
        "negative_concept": "trunk_leaning",
        "severity": 0.5,
    },
    "Q7": {  # 击球时身体朝向
        "positive_signals": ["侧身", "45度", "保持侧身"],
        "negative_signals": ["正面", "完全正面", "朝网", "正对球网"],
        "positive_concept": "good_rotation",
        "negative_concept": "problem_p11",
        "severity": 0.7,
    },
    "Q8": {  # 手下降程度
        "positive_signals": ["没有", "轻微", "缓慢"],
        "negative_signals": ["急剧", "很多", "大幅", "突然下坠", "明显下降"],
        "positive_concept": "smooth_drop",
        "negative_concept": "problem_p01",
        "severity": 0.7,
    },
    "Q9": {  # 轨迹形状
        "positive_signals": ["平缓", "弧线", "(a)", "向上弧线"],
        "negative_signals": ["V形", "V字", "(b)", "急剧下坠再上升", "V-shape"],
        "positive_concept": "smooth_arc",
        "negative_concept": "problem_p02",
        "severity": 0.8,
    },
    "Q11": {  # 击球后手臂方向
        "positive_signals": ["向前", "延伸", "向前延伸"],
        "negative_signals": ["内侧", "向内", "收", "向身体", "立刻向"],
        "positive_concept": "swing_out",
        "negative_concept": "problem_p05",
        "severity": 0.6,
    },
    "Q12": {  # 左手准备位置
        "positive_signals": ["伸向前方", "指向", "在球拍上"],
        "negative_signals": ["垂在身侧", "放下", "垂着"],
        "positive_concept": "non_hitting_hand_pointing",
        "negative_concept": "problem_p15",
        "severity": 0.5,
    },
    "Q13": {  # 左手击球时动作
        "positive_signals": ["收回", "胸口", "收到胸", "刹车"],
        "negative_signals": ["垂着", "没动", "甩在身后", "一直垂"],
        "positive_concept": "non_hitting_hand_braking",
        "negative_concept": "problem_p15",
        "severity": 0.6,
    },
    "Q14": {  # 膝盖弯曲
        "positive_signals": ["明显弯曲", "弯曲蓄力", "弯了很多"],
        "negative_signals": ["几乎直立", "直的", "没弯", "很少弯"],
        "positive_concept": "good_knee_bend",
        "negative_concept": "straight_legs",
        "severity": 0.5,
    },
    "Q16": {  # 后脚状态
        "positive_signals": ["脚尖", "拧转", "pivot"],
        "negative_signals": ["平踩", "不动", "离地", "跳"],
        "positive_concept": "a6_pivot",
        "negative_concept": "problem_p09",
        "severity": 0.4,
    },
    "Q17": {  # 最先动的部位
        "positive_signals": ["下半身", "腿", "髋", "躯干", "肩膀"],
        "negative_signals": ["手臂", "手", "持拍手"],
        "positive_concept": "good_sequence",
        "negative_concept": "problem_p03",
        "severity": 0.8,
    },
    "Q19": {  # 躯干减速
        "positive_signals": ["有", "减速", "停顿", "击球之前"],
        "negative_signals": ["没有", "一直在转", "没有减速"],
        "positive_concept": "trunk_deceleration",
        "negative_concept": "problem_p11",
        "severity": 0.6,
    },

    # ══════════════════════════════════════════════════════════════════
    # ── L4 PREPARATION (Q21-Q26) — aligned with system_prompt.md.j2 v4 ──
    # ══════════════════════════════════════════════════════════════════
    "Q21": {  # 转身开始时机 vs 对方击球
        "positive_signals": ["之前", "同时", "对方击球时", "对方挥拍时"],
        "negative_signals": ["之后", "晚", "对方球过网才", "球过网后", "球落地后"],
        "positive_concept": "good_unit_turn",
        "negative_concept": "prep08_late_unit_turn",
        "severity": 0.95,
    },
    "Q22": {  # 转身完成时机 vs 球过网/落地
        "positive_signals": ["球过网之前", "过网时", "之前完成", "已完成"],
        "negative_signals": ["落地之后", "球已落地", "球弹起还", "未完成"],
        "positive_concept": "good_unit_turn",
        "negative_concept": "prep17_prep_not_done_by_bounce",
        "severity": 0.95,
    },
    "Q23": {  # 左肩先动 vs 右肩自拉
        "positive_signals": ["左肩先", "左肩明显先", "左肩带动", "left shoulder first"],
        "negative_signals": ["右肩自己", "右肩向后拉", "右肩先", "只有右肩"],
        "positive_concept": "good_unit_turn",
        "negative_concept": "prep13_left_shoulder_not_forward",
        "severity": 0.8,
    },
    "Q24": {  # 引拍最高点拍头
        "positive_signals": ["高于手腕", "拍头高", "举着", "与头部齐平", "比手腕高"],
        "negative_signals": ["拍头下垂", "低于手腕", "已经下垂", "拍头掉"],
        "positive_concept": "racket_held_up",
        "negative_concept": "prep11_racket_head_dropped_early",
        "severity": 0.75,
    },
    "Q25": {  # 背部褶皱 / 肩胛滑动
        "positive_signals": ["有皱褶", "背部拉伸", "看到褶皱", "肩胛滑动", "球衣拉伸"],
        "negative_signals": ["没有皱褶", "无拉伸", "背部平", "看不到褶皱"],
        "positive_concept": "scapular_glide",
        "negative_concept": "prep10_no_scapular_glide",
        "severity": 0.85,
    },
    "Q26": {  # 一气呵成 vs 顶点停顿
        "positive_signals": ["一气呵成", "连续", "无停顿", "流畅", "无明显等"],
        "negative_signals": ["停一下", "顶点停", "等一下", "停顿", "stop"],
        "positive_concept": "continuous_loop",
        "negative_concept": "prep16_stop_start_syndrome",
        "severity": 0.8,
    },

    # ══════════════════════════════════════════════════════════════════
    # ── L5 FOOTWORK (Q27-Q32) — aligned with system_prompt.md.j2 v4 ──
    # ══════════════════════════════════════════════════════════════════
    "Q27": {  # split step 是否可见
        "positive_signals": ["清晰", "清晰可见", "看到", "做了", "明显的split", "明显起跳"],
        "negative_signals": ["完全没", "没有", "missing", "站着不动", "静止"],
        "positive_concept": "good_split_step",
        "negative_concept": "prep02_no_split_step",
        "severity": 0.9,
    },
    "Q28": {  # split step 落地时机 vs 对方击球
        "positive_signals": ["之前", "同时", "对方击球时", "对方击球瞬间", "land as they hit"],
        "negative_signals": ["之后", "晚", "对方击球后才", "落地晚"],
        "positive_concept": "good_split_step",
        "negative_concept": "prep01_late_split_step",
        "severity": 0.85,
    },
    "Q29": {  # 第一只脚 / 方向
        "positive_signals": ["右脚", "持拍侧", "朝来球方向", "朝侧"],
        "negative_signals": ["晚", "犹豫", "方向错", "原地", "没动"],
        "positive_concept": "good_first_step",
        "negative_concept": "prep29_late_first_step",
        "severity": 0.7,
    },
    "Q30": {  # 右脚 pivot 外旋
        "positive_signals": ["有pivot", "外旋", "脚尖转", "后跟离地", "明显pivot"],
        "negative_signals": ["没pivot", "无pivot", "横向滑", "没转", "脚没动"],
        "positive_concept": "a6_pivot",
        "negative_concept": "prep04_no_pivot",
        "severity": 0.85,
    },
    "Q31": {  # 站位类型
        "positive_signals": ["半开放", "开放", "中性", "semi-open", "open"],
        "negative_signals": ["完全正面", "关闭", "closed", "正对球网"],
        "positive_concept": "good_stance",
        "negative_concept": "prep22_neutral_to_semi_open_failed",
        "severity": 0.5,
    },
    "Q32": {  # split→contact 步数 / 回位
        "positive_signals": ["1步", "2步", "干净", "有回位", "回位步"],
        "negative_signals": ["碎步", "4步", "5步", "调整步多", "无回位", "原地"],
        "positive_concept": "good_footwork",
        "negative_concept": "prep06_choppy_steps",
        "severity": 0.5,
    },

    # ══════════════════════════════════════════════════════════════════
    # ── Q33-Q38: New questions from wave-2 video-watching insights ──
    # 来源: docs/research/coach_analysis/video_watching_findings.md §4
    #       docs/research/coach_analysis/feel_tennis_video_findings.md §4
    # ══════════════════════════════════════════════════════════════════
    "Q33": {  # 击球瞬间脊柱倾斜（侧弯证据）
        "positive_signals": ["明显倾斜", "侧弯", "向左倾", "脊柱倾斜", "有侧弯"],
        "negative_signals": ["垂直", "直立", "完全垂直", "无侧弯", "笔直"],
        "positive_concept": "good_side_bend",
        "negative_concept": "v2_no_side_bending",
        "severity": 0.8,
    },
    "Q34": {  # 等球时双脚是否有持续小跳
        "positive_signals": ["有小跳", "持续小跳", "脚在动", "在律动", "弹性"],
        "negative_signals": ["完全静止", "脚不动", "没动", "站着不动", "stiff"],
        "positive_concept": "active_ready_state",
        "negative_concept": "v2_slow_motion_practice",
        "severity": 0.85,
    },
    "Q35": {  # split 空中相位躯干/视线已转
        "positive_signals": ["已经转", "落地前已", "落地前转向", "空中已朝向"],
        "negative_signals": ["落地后才转", "落地后", "转向晚", "落地后才看"],
        "positive_concept": "early_ball_recognition",
        "negative_concept": "v2_late_split_recognition",
        "severity": 0.85,
    },
    "Q36": {  # 握拍特写：底部三指 vs 虎口支点
        "positive_signals": ["虎口", "V字", "拇指食指", "握在虎口", "上部握"],
        "negative_signals": ["底部三指", "底盖", "握底", "底部紧握"],
        "positive_concept": "pivot_at_v_grip",
        "negative_concept": "v2_pivot_at_butt_cap",
        "severity": 0.7,
    },
    "Q37": {  # 拍头向量 vs 来球向量（拦截 vs 追）
        "positive_signals": ["相向", "迎向", "拦截", "对着球", "向前"],
        "negative_signals": ["同向", "追", "在球后面", "跟着球"],
        "positive_concept": "intercepting_ball",
        "negative_concept": "v2_chasing_not_intercepting",
        "severity": 0.85,
    },
    "Q38": {  # 引拍速度 vs 来球速度匹配
        "positive_signals": ["匹配", "差不多", "一致", "match"],
        "negative_signals": ["远快于", "快得多", "提前完成", "已完成在等", "stop-start"],
        "positive_concept": "matched_takeback_tempo",
        "negative_concept": "v2_takeback_speed_mismatch",
        "severity": 0.8,
    },
}


def _map_via_q_direct(vlm_result: Dict) -> List[Dict[str, Any]]:
    """Use Q-number direct mapping for precise concept matching.

    This is more accurate than keyword search because we know exactly
    what each Q is asking about.
    """
    matched = []
    raw_answers = vlm_result.get("raw_answers", {})
    if not raw_answers:
        return matched

    for q_num, mapping in Q_DIRECT_MAPPING.items():
        answer = raw_answers.get(q_num, "")
        if not answer or answer == "看不清":
            continue

        answer_lower = answer.lower()

        # Check negative signals (problem detected)
        is_negative = any(sig in answer_lower for sig in mapping["negative_signals"])
        is_positive = any(sig in answer_lower for sig in mapping["positive_signals"])

        if is_negative and not is_positive:
            # Use the actual VLM answer as the observation, not "Q1检测到问题"
            matched.append({
                "observation": answer[:150],
                "keyword_matched": q_num,
                "mapped_concept": mapping["negative_concept"],
                "frame": None,
                "field": q_num,
                "severity": mapping["severity"],
                "label": answer[:50],
                "source": "q_direct",
            })
        elif is_positive and not is_negative:
            matched.append({
                "observation": f"{q_num}: {answer[:100]}",
                "keyword_matched": q_num,
                "mapped_concept": mapping["positive_concept"],
                "frame": None,
                "field": q_num,
                "severity": 0.0,
                "label": f"{q_num}正常",
                "source": "q_direct",
            })

    return matched


def _cross_validate_q_answers(
    q_matched: List[Dict], raw_answers: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Cross-validate Q answers to detect contradictions.

    Example: Q1 says arm syncs with body, but Q8 says wrist drops sharply
    → contradiction → lower Q1 confidence.
    """
    contradictions = []

    # Q1(同步) vs Q8(手下降) 矛盾检测
    q1 = raw_answers.get("Q1", "")
    q8 = raw_answers.get("Q8", "")
    if q1 and q8:
        q1_positive = any(s in q1 for s in ["一起", "同步", "跟着"])
        q8_negative = any(s in q8 for s in ["急剧", "突然", "大幅"])
        if q1_positive and q8_negative:
            contradictions.append({
                "q_pair": "Q1 vs Q8",
                "detail": f"Q1说手臂同步({q1[:30]})，但Q8说手腕急剧下降({q8[:30]})",
                "action": "降低Q1置信度",
            })

    # Q5(转体深度) vs Q7(击球时朝向) 一致性
    q5 = raw_answers.get("Q5", "")
    q7 = raw_answers.get("Q7", "")
    if q5 and q7:
        q5_little = any(s in q5 for s in ["一点", "很少", "不够"])
        q7_full = any(s in q7 for s in ["正面", "完全"])
        if q5_little and q7_full:
            contradictions.append({
                "q_pair": "Q5 vs Q7",
                "detail": f"准备时转体少({q5[:30]})但击球时已正面朝网({q7[:30]})——转体不够但过度转了",
            })

    return contradictions


def _map_observations_to_concepts(
    observations: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Map VLM observations to knowledge graph concepts using keyword matching.

    This is the fallback when Q-direct mapping is not available.
    Returns list of matched concepts with evidence.
    """
    matched: List[Dict[str, Any]] = []
    seen_concepts: Dict[str, Dict] = {}

    for obs in observations:
        text = obs["text"]
        frame = obs["frame"]

        for rule in OBSERVATION_TO_CONCEPT:
            for kw in rule["keywords"]:
                if kw.lower() in text.lower():
                    concept_id = rule["concept"]
                    if rule["frame_range"] is not None and frame is not None:
                        if frame not in rule["frame_range"]:
                            continue

                    match_info = {
                        "observation": text[:200],
                        "keyword_matched": kw,
                        "mapped_concept": concept_id,
                        "frame": frame,
                        "field": obs["field"],
                        "severity": rule["severity"],
                        "label": rule["label"],
                        "source": "keyword",
                    }

                    if concept_id not in seen_concepts:
                        seen_concepts[concept_id] = match_info
                        matched.append(match_info)
                    elif rule["severity"] > seen_concepts[concept_id]["severity"]:
                        idx = next(i for i, m in enumerate(matched) if m["mapped_concept"] == concept_id)
                        matched[idx] = match_info
                        seen_concepts[concept_id] = match_info
                    break

    matched.sort(key=lambda m: m["severity"], reverse=True)
    return matched


# ══════════════════════════════════════════════════════════════════════
#  Step 2: 概念→根因追溯（沿 causes 边追溯）
# ══════════════════════════════════════════════════════════════════════

def _build_causes_graph() -> Dict[str, List[str]]:
    """Build a reverse causes lookup: child -> [parent causes].

    In our graph, edge A->B with relation=causes means A causes B.
    To trace root causes of a symptom, we go from B to A (reverse).
    """
    graph_data = _load_graph_data()
    reverse_causes: Dict[str, List[str]] = {}
    for edge in graph_data.get("edges", []):
        if edge.get("relation") == "causes":
            target = edge.get("target", edge.get("target_id", ""))
            source = edge.get("source", edge.get("source_id", ""))
            if target and source:
                reverse_causes.setdefault(target, []).append(source)
    return reverse_causes


_NAME_OVERRIDE_ZH: Dict[str, str] = {
    "straight_legs": "下肢未承载（腿没弯）",
    "upper_body_only_turn": "只转上半身（髋未参与）",
    "unit_turn": "Unit Turn 转开不足",
}


def _get_node_name_zh(concept_id: str) -> str:
    """Get Chinese name of a concept from graph or override map."""
    if concept_id in _NAME_OVERRIDE_ZH:
        return _NAME_OVERRIDE_ZH[concept_id]
    graph_data = _load_graph_data()
    for node in graph_data.get("nodes", []):
        if node.get("id") == concept_id:
            return node.get("name_zh", concept_id)
    return concept_id


def _trace_root_causes(
    concept_ids: List[str],
    matched_concepts_for_voting: Optional[List[Dict]] = None,
) -> Tuple[Optional[str], List[Dict[str, str]]]:
    """Trace from symptom concepts to root causes.

    Strategy:
    1. First try diagnostic chains (most accurate, human-verified)
    2. Then multi-path graph traversal with voting
    3. Root cause = the one with highest weighted votes

    Returns (root_cause_id, causal_chain_list).
    """
    if matched_concepts_for_voting is None:
        matched_concepts_for_voting = []

    # ── Top-down 5-layer flow (v4): earliest layer with a problem = root cause ──
    # Order: L5 footwork → L4 prep → L3 chain → L2 rhythm → L1 contact
    # Source: docs/research/coach_analysis/diagnostic_methodology.md
    earliest = _find_earliest_layer_problem(matched_concepts_for_voting)
    if earliest is not None:
        root_id = earliest["mapped_concept"]
        reverse_causes = _build_causes_graph()
        causal_chain: List[Dict[str, str]] = []
        # Build a forward chain from root to its descendants if any
        # (downstream concepts that are also matched and in deeper layers)
        downstream_matched_ids = {
            m["mapped_concept"] for m in matched_concepts_for_voting
            if m.get("severity", 0) > 0 and m["mapped_concept"] != root_id
        }
        # Forward edges where root_id is the source
        graph_data = _load_graph_data()
        forward: Dict[str, List[str]] = {}
        for edge in graph_data.get("edges", []):
            if edge.get("relation") == "causes":
                s = edge.get("source", edge.get("source_id", ""))
                t = edge.get("target", edge.get("target_id", ""))
                if s and t:
                    forward.setdefault(s, []).append(t)
        # 1-hop forward: show root → first matched downstream
        for child in forward.get(root_id, []):
            if child in downstream_matched_ids:
                causal_chain.append({
                    "from": root_id,
                    "from_name": _get_node_name_zh(root_id),
                    "to": child,
                    "to_name": _get_node_name_zh(child),
                    "relation": "causes",
                })
                break
        if not causal_chain:
            # Fall back to 1-hop reverse parent for context
            parents = reverse_causes.get(root_id, [])
            if parents:
                parent = parents[0]
                causal_chain.append({
                    "from": parent,
                    "from_name": _get_node_name_zh(parent),
                    "to": root_id,
                    "to_name": _get_node_name_zh(root_id),
                    "relation": "causes",
                })
        return root_id, causal_chain

    # Diagnostic chains skipped here intentionally: when KPI injection seeds
    # a clear L4/L5 root, the top-down branch above wins. Chains may still
    # be useful for symptom-driven lookup once their `root_causes` ordering
    # is tightened — see _try_diagnostic_chains for the (currently unused) path.

    # Step 2: Simple strategy — highest severity directly observed concept = root cause
    # Graph traversal tends to go too deep to abstract concepts like "重心"
    # Direct observation is more actionable and accurate
    if concept_ids:
        # Count frequency: concepts observed multiple times are stronger candidates
        from collections import Counter
        freq = Counter(concept_ids)
        # Pick: highest frequency first, then highest severity
        best_id = max(
            set(concept_ids),
            key=lambda c: (freq[c], next(
                (m["severity"] for m in matched_concepts_for_voting
                 if m.get("mapped_concept") == c), 0
            )),
        )
        # Build simple causal chain from graph (1 hop only for context)
        reverse_causes = _build_causes_graph()
        causal_chain = []
        parents = reverse_causes.get(best_id, [])
        if parents:
            parent = parents[0]
            causal_chain.append({
                "from": parent,
                "from_name": _get_node_name_zh(parent),
                "to": best_id,
                "to_name": _get_node_name_zh(best_id),
                "relation": "causes",
            })
        return best_id, causal_chain

    # Fallback: graph traversal (only when no direct observations)
    MAX_DEPTH = 3
    reverse_causes = _build_causes_graph()
    root_vote: Dict[str, int] = {}
    root_best_chain: Dict[str, List[str]] = {}

    for concept_id in concept_ids:
        visited = set()

        def _walk(node: str, path: List[str], depth: int) -> None:
            if depth >= MAX_DEPTH:
                root = path[-1]
                root_vote[root] = root_vote.get(root, 0) + 1
                if root not in root_best_chain or len(path) > len(root_best_chain[root]):
                    root_best_chain[root] = list(path)
                return
            parents = reverse_causes.get(node, [])
            if not parents:
                root = path[-1]
                root_vote[root] = root_vote.get(root, 0) + 1
                if root not in root_best_chain or len(path) > len(root_best_chain[root]):
                    root_best_chain[root] = list(path)
                return
            for parent in parents:
                if parent not in visited and parent != "c_unnamed":
                    visited.add(parent)
                    _walk(parent, path + [parent], depth + 1)

        visited.add(concept_id)
        _walk(concept_id, [concept_id], 0)

    # High-severity directly observed concepts are strong root cause candidates
    # Even if they have parents in the graph, they might be the most actionable root cause
    from collections import Counter
    direct_counts = Counter(concept_ids)
    for concept_id, count in direct_counts.items():
        # If this concept was observed multiple times (e.g., Q1 and Q17 both → problem_p03)
        # it's a strong root cause candidate
        if count >= 2:
            root_vote[concept_id] = root_vote.get(concept_id, 0) + count * 3
            if concept_id not in root_best_chain:
                root_best_chain[concept_id] = [concept_id]
        # Also boost concepts with no parents (true roots)
        if concept_id not in reverse_causes or not reverse_causes.get(concept_id):
            root_vote[concept_id] = root_vote.get(concept_id, 0) + 2
            if concept_id not in root_best_chain:
                root_best_chain[concept_id] = [concept_id]

    if not root_vote:
        if concept_ids:
            return concept_ids[0], []
        return None, []

    # Pick root cause: votes weighted by severity of the original observations
    # Concepts with higher severity observations pointing to them are better candidates
    directly_observed = set(concept_ids)
    # Boost directly observed concepts by their severity
    concept_severity = {}
    for cid in concept_ids:
        concept_severity[cid] = max(concept_severity.get(cid, 0), 0.5)
    for m in matched_concepts_for_voting:
        cid = m.get("mapped_concept", "")
        sev = m.get("severity", 0)
        concept_severity[cid] = max(concept_severity.get(cid, 0), sev)

    # Strongly prefer directly observed concepts over graph-discovered ones
    root_cause_id = max(
        root_vote,
        key=lambda k: (
            root_vote[k] * concept_severity.get(k, 0.1),  # votes × severity
            10 if k in directly_observed else 0,  # huge boost for direct observations
        ),
    )
    best_chain = root_best_chain.get(root_cause_id, [])

    causal_chain: List[Dict[str, str]] = []
    for i in range(len(best_chain) - 1):
        causal_chain.append({
            "from": best_chain[i + 1],
            "from_name": _get_node_name_zh(best_chain[i + 1]),
            "to": best_chain[i],
            "to_name": _get_node_name_zh(best_chain[i]),
            "relation": "causes",
        })

    return root_cause_id, causal_chain


def _try_diagnostic_chains(
    concept_ids: List[str], chains: List[Dict],
) -> Optional[Tuple[str, List[Dict[str, str]]]]:
    """Try to match symptoms against the 18 hand-verified diagnostic chains.

    These are the most accurate causal paths. If a match is found,
    use the chain's root_cause directly instead of graph traversal.
    """
    if not chains:
        return None

    best_match = None
    best_score = 0

    for chain in chains:
        symptom_id = chain.get("symptom_concept_id", "")
        root_causes = chain.get("root_causes", [])
        if not symptom_id or not root_causes:
            continue

        # Check if this chain's symptom matches any of our concepts
        for cid in concept_ids:
            if cid == symptom_id or symptom_id in cid or cid in symptom_id:
                score = chain.get("priority", 5)
                if best_match is None or score < best_score:
                    best_match = chain
                    best_score = score

    if best_match:
        root_id = best_match["root_causes"][0] if best_match["root_causes"] else None
        if root_id:
            chain_steps = best_match.get("check_sequence", [])
            causal_chain = [{
                "from": root_id,
                "from_name": _get_node_name_zh(root_id),
                "to": best_match.get("symptom_concept_id", ""),
                "to_name": best_match.get("symptom_zh", best_match.get("symptom", "")),
                "relation": "causes (diagnostic chain)",
            }]
            return root_id, causal_chain

    return None


# ══════════════════════════════════════════════════════════════════════
#  Step 3: 量化验证
# ══════════════════════════════════════════════════════════════════════

def _validate_with_metrics(
    matched_concepts: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    excluded_concepts: Optional[set] = None,
) -> Dict[str, List[str]]:
    """Validate concept matches against quantitative metrics.

    Returns {"confirmed": [...], "contradicted": [...]}.

    excluded_concepts: concepts already handled by arbitration layer;
    skip here to avoid double-reporting the same VLM-vs-metric conflict.
    """
    confirmed: List[str] = []
    contradicted: List[str] = []
    excluded = excluded_concepts or set()

    checked_concepts = set()
    for match in matched_concepts:
        concept_id = match["mapped_concept"]
        if concept_id in checked_concepts or concept_id in excluded:
            continue
        checked_concepts.add(concept_id)

        validations = _CONCEPT_TO_METRIC_VALIDATION.get(concept_id, [])
        for v in validations:
            metric_name = v["metric"]
            val = metrics.get(metric_name)
            if val is None:
                continue
            if v["check"](val):
                confirmed.append(v["confirm_text"].format(val=val))
            else:
                contradicted.append(v["contradict_text"].format(val=val))

    # Also check for issues detected by metrics but not by VLM
    matched_ids = {m["mapped_concept"] for m in matched_concepts}

    sync = metrics.get("arm_torso_synchrony")
    if sync is not None and sync < 0.4 and "problem_p03" not in matched_ids:
        confirmed.append(f"量化数据额外发现：同步性{sync:.2f}，手臂独立于身体（VLM未明确提及）")

    rot = metrics.get("shoulder_rotation")
    if rot is not None and rot < 30 and "unit_turn" not in matched_ids:
        confirmed.append(f"量化数据额外发现：肩部转开{rot:.0f}°，转体不足（VLM未明确提及）")

    scoop = metrics.get("scooping_depth")
    if scoop is not None and scoop > 0.3 and "problem_p02" not in matched_ids:
        confirmed.append(f"量化数据额外发现：Scooping深度{scoop:.2f}（VLM未明确提及）")

    return {"confirmed": confirmed, "contradicted": contradicted}


# ══════════════════════════════════════════════════════════════════════
#  KPI → 概念注入（让量化信号参与根因竞争）
# ══════════════════════════════════════════════════════════════════════
#
# 设计动机：v4.2 的 top-down "earliest layer wins" 推理只看
# matched_concepts，而 matched_concepts 只来自 VLM 观察映射。
# KPI 信号原先只用于"交叉验证文本"，从未进入候选池，导致：
# VLM 漏掉 L4/L5 问题 → engine 报 L3 为根因（违反 earliest-layer 原则）。
# 这里把 KPI 提供的硬证据注入 matched_concepts，让上游层有竞争资格。

_KPI_INJECTION_RULES: List[Dict[str, Any]] = [
    # —— L5 Footwork ——
    {
        "metric": "min_knee_angle",
        "concept": "straight_legs",
        "thresholds": [(160, 1.0, "下肢承载严重不足"),
                       (150, 0.7, "下肢承载偏弱")],
        "above_threshold": True,  # 数值大于阈值=问题（膝角越大=越直）
        "observation_template": "量化检测：最小膝角 {val:.0f}°，{label}（应 ≤140°）",
        "frame": "preparation",
    },
    # —— L4 Preparation ——
    {
        "metric": "shoulder_rotation",
        "concept": "unit_turn",
        "thresholds": [(30, 1.0, "肩部转开严重不足"),
                       (50, 0.7, "肩部转开偏弱")],
        "above_threshold": False,  # 数值小于阈值=问题
        "observation_template": "量化检测：肩部转开 {val:.0f}°，{label}（应 ≥60°）",
        "frame": "preparation",
    },
]


def _inject_kpi_problems(
    matched_concepts: List[Dict[str, Any]],
    metrics: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Inject KPI-evidenced concepts into matched_concepts.

    KPI signals provide independent evidence of preparation/footwork problems
    that VLM may miss. Without injection, _find_earliest_layer_problem would
    only see VLM-derived concepts and could not honor the L5→L1 priority.

    Concepts already in matched_concepts (from VLM) are upgraded if KPI
    confirms higher severity, but never downgraded.
    """
    if not metrics:
        return matched_concepts

    existing: Dict[str, Dict[str, Any]] = {
        m["mapped_concept"]: m for m in matched_concepts
    }

    for rule in _KPI_INJECTION_RULES:
        val = metrics.get(rule["metric"])
        if val is None:
            continue

        # Find the most severe threshold this value crosses
        triggered: Optional[Tuple[float, str]] = None
        for threshold, severity, label in rule["thresholds"]:
            crossed = (val > threshold) if rule["above_threshold"] else (val < threshold)
            if crossed:
                triggered = (severity, label)
                break  # thresholds ordered most-severe first

        if triggered is None:
            continue

        severity, label = triggered
        observation = rule["observation_template"].format(val=val, label=label)
        concept_id = rule["concept"]

        if concept_id in existing:
            # Upgrade severity if KPI is more confident than VLM
            current = existing[concept_id]
            if severity > current.get("severity", 0):
                current["severity"] = severity
                current["observation"] = observation + " | " + current.get("observation", "")
                current["source"] = "kpi+vlm"
        else:
            matched_concepts.append({
                "observation": observation,
                "mapped_concept": concept_id,
                "label": _get_node_name_zh(concept_id),
                "severity": severity,
                "frame": rule["frame"],
                "source": "kpi",
            })
            existing[concept_id] = matched_concepts[-1]

    matched_concepts.sort(key=lambda m: m.get("severity", 0), reverse=True)
    return matched_concepts


# ══════════════════════════════════════════════════════════════════════
#  Step 1.6: VLM 观察 ↔ 量化数据 仲裁
# ══════════════════════════════════════════════════════════════════════
#
# 设计动机：
#   VLM 和量化算法是两个独立的"检测器"，类似医院的 CT 影像 + 血液检查：
#     - VLM = 视觉观察（主观，易受拍摄角度/光线影响）
#     - 量化算法 = 数值检测（客观，基于坐标/时序）
#   知识库推理引擎才是"医生"，需要综合两类证据下诊断。
#
#   历史 bug（2026-04-22 用户发现）：
#     VLM 视觉上看到 "V 形轨迹" → 映射到 problem_p02 (Scooping)
#     量化算法检测到 scooping_depth < 阈值 / scooping_detected=False
#     旧系统仍把 problem_p02 列入问题 + 并列两个矛盾结论
#     实际上这是"现代正手的被动 lag drop"（拍头因惯性下落，非错误）
#
#   仲裁层职责：当量化数据给出明确反驳性证据时，撤销 VLM 的 False Positive。
#
# 规则设计原则：
#   1. 仅对 VLM 来源（q_direct / keyword）的概念生效，不对 KPI 注入的生效
#   2. 必须有明确的"反驳性数值阈值"（不是模糊判断）
#   3. resolution_text 用医生视角写清楚"为什么判定已解决"

_ARBITRATION_RULES: List[Dict[str, Any]] = [
    # —— Scooping / V 形轨迹 ——
    # 用户 2026-04-22 场景：VLM 看到 V 形，算法确认无 scooping
    # 判定为现代正手被动 lag drop，非业余主动捞球
    {
        "concept": "problem_p02",
        "vlm_only": True,  # 仅对纯 VLM 来源的概念生效，KPI 加持过的保留
        "arbitrate": lambda m: (
            # 算法给出明确的"非 scooping"证据：
            # scooping_detected=False 且 scooping_depth 未超阈值
            (m.get("scooping_detected") is False)
            and (m.get("scooping_depth") is None or m.get("scooping_depth") <= 0.3)
        ),
        "resolution_text": (
            "VLM 视觉观察到手腕 V 形轨迹，但算法量化检测未发现显著 scooping"
            "（scooping_depth 未超阈值）。判定为**现代正手的被动 lag drop**——"
            "拍头因重力和惯性自然下落，被身体旋转带回击球位，并非业余的"
            "小臂主动下压+上翻。所有现代职业选手（费德勒、纳达尔、辛纳）侧面"
            "轨迹也呈 V 形，属于正手标志性特征，不是问题。"
        ),
        "short_note": "V 形 = 现代正手被动 lag drop（非 scooping），已解决",
    },
    # —— 可扩展：其他 VLM-算法冲突规则 ——
    # 模板：
    # {
    #     "concept": "problem_pXX",
    #     "vlm_only": True,
    #     "arbitrate": lambda m: <明确反驳条件>,
    #     "resolution_text": "<医生视角解释>",
    #     "short_note": "<一句话结论>",
    # },
]


def _arbitrate_vlm_vs_metrics(
    matched_concepts: List[Dict[str, Any]],
    metrics: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """VLM 观察 ↔ 量化数据 仲裁层。

    当量化算法给出对 VLM 观察的明确反驳时，将 VLM 映射的问题从候选池移除，
    避免知识库推理引擎把"被动自然现象"误判为问题。

    Parameters
    ----------
    matched_concepts : list
        Step 1 + 1.5 合并后的概念列表（含 VLM 来源 + KPI 来源）
    metrics : dict
        量化数据

    Returns
    -------
    (cleaned_concepts, arbitration_log)
        cleaned_concepts : 经仲裁后保留的概念
        arbitration_log : 被撤销的 VLM 观察 + 医生视角的解释
    """
    if not matched_concepts or not metrics:
        return matched_concepts, []

    cleaned: List[Dict[str, Any]] = []
    arbitration_log: List[Dict[str, Any]] = []

    for m in matched_concepts:
        concept_id = m.get("mapped_concept")
        source = m.get("source", "unknown")
        rule = next(
            (r for r in _ARBITRATION_RULES if r["concept"] == concept_id),
            None,
        )

        # 无规则 → 保留
        if rule is None:
            cleaned.append(m)
            continue

        # 仅 VLM 来源规则，但当前 match 是 KPI 加持的 → 保留（KPI 独立证据不撤销）
        if rule.get("vlm_only") and source not in {"q_direct", "keyword"}:
            cleaned.append(m)
            continue

        # 执行仲裁判定（narrow catch：只吞 lambda 逻辑异常，不吞配置错误）
        try:
            arbitration_triggered = rule["arbitrate"](metrics)
        except (TypeError, KeyError, AttributeError) as e:
            import logging
            logging.warning(
                f"Arbitration rule for {concept_id} failed: {type(e).__name__}: {e}. "
                f"Falling back to non-arbitrated (VLM observation retained)."
            )
            arbitration_triggered = False

        if arbitration_triggered:
            # 撤销该 VLM 观察，记入仲裁日志
            arbitration_log.append({
                "concept": concept_id,
                "label": m.get("label", _get_node_name_zh(concept_id)),
                "vlm_observation": m.get("observation", ""),
                "resolution": rule["resolution_text"],
                "short_note": rule["short_note"],
            })
        else:
            cleaned.append(m)

    return cleaned, arbitration_log


# ══════════════════════════════════════════════════════════════════════
#  Step 4: 用户历史对比
# ══════════════════════════════════════════════════════════════════════

# Concept ID -> recurring issue keywords mapping
_CONCEPT_TO_RECURRING_KEYWORDS: Dict[str, List[str]] = {
    "problem_p02": ["scooping", "V形", "捞球"],
    "problem_p03": ["小臂代偿", "手臂主动", "手臂独立"],
    "problem_p01": ["拍头过度下坠", "Pat the Dog"],
    "problem_p07": ["右肩下沉", "Tilt 时机"],
    "problem_p11": ["过度转体", "Over-rotation"],
    "problem_p04": ["动力链断裂", "力量"],
    "problem_p05": ["击球点偏后"],
    "problem_p09": ["重心"],
    "problem_p10": ["背部张力"],
    "problem_p08": ["时机太早"],
    "problem_p14": ["肘部空间"],
    "problem_p15": ["左手"],
    "problem_p13": ["手主动引拍"],
    "unit_turn": ["转体不足", "Unit Turn"],
    "straight_legs": ["腿没弯", "下肢承载", "triple bend"],
    # Preparation phase recurring keywords
    "prep01_late_split_step": ["分腿晚", "split晚"],
    "prep02_no_split_step": ["没有分腿", "split缺失"],
    "prep04_no_pivot": ["pivot", "右脚没转"],
    "prep08_late_unit_turn": ["unit turn晚", "准备晚", "转身晚"],
    "prep09_arm_only_unit_turn": ["只用手臂", "单臂引拍"],
    "prep10_no_scapular_glide": ["肩胛", "背没参与"],
    "prep14_insufficient_hip_shoulder_separation": ["肩髋分离", "扭矩"],
    "prep17_prep_not_done_by_bounce": ["球落地准备", "racket back late"],
    "prep20_no_triple_bend": ["triple bend", "腿没加载"],
    "prep30_late_preparation_general": ["准备迟滞"],
}


def _check_user_history(matched_concepts: List[Dict[str, Any]]) -> Optional[str]:
    """Check if matched concepts correspond to recurring user issues.

    Returns a human-readable note, or None.
    """
    history = _load_user_history()
    recurring = history.get("recurring_issues", [])
    if not recurring:
        return None

    matched_ids = {m["mapped_concept"] for m in matched_concepts}
    matches: List[str] = []

    for issue in recurring:
        issue_text = issue.get("issue", "")
        dates = issue.get("dates", [])
        resolved = issue.get("resolved", False)

        for concept_id in matched_ids:
            keywords = _CONCEPT_TO_RECURRING_KEYWORDS.get(concept_id, [])
            if any(kw.lower() in issue_text.lower() for kw in keywords):
                date_str = "、".join(dates[-3:])  # Show last 3 dates
                if len(dates) > 3:
                    date_str = f"...{date_str}"
                status = "已解决" if resolved else "仍未解决"
                resolution = issue.get("resolution", "")
                note = f"「{issue_text}」在你的训练中出现过{len(dates)}次（{date_str}），{status}"
                if resolved and resolution:
                    note += f"。解决方案：{resolution[:80]}"
                elif not resolved and resolution:
                    note += f"。当前进展：{resolution[:80]}"
                matches.append(note)
                break  # One match per issue

    if not matches:
        return None
    return " | ".join(matches)


# ══════════════════════════════════════════════════════════════════════
#  Step 5: 训练建议
# ══════════════════════════════════════════════════════════════════════

def _get_fix(root_cause_id: Optional[str], matched_concepts: List[Dict]) -> Dict[str, str]:
    """Get fix/drill recommendation for the root cause."""
    # Try root cause first
    if root_cause_id and root_cause_id in _CONCEPT_TO_FIX:
        return _CONCEPT_TO_FIX[root_cause_id]

    # Try matched concepts by severity
    for match in matched_concepts:
        cid = match["mapped_concept"]
        if cid in _CONCEPT_TO_FIX:
            return _CONCEPT_TO_FIX[cid]

    # Fallback: check diagnostic chains
    chains = _load_chains()
    for match in matched_concepts:
        cid = match["mapped_concept"]
        for chain in chains:
            if cid in chain.get("root_causes", []) and chain.get("drills"):
                return {
                    "drill": chain["drills"][0],
                    "method": "",
                    "why": f"诊断链 {chain['id']} 推荐",
                }

    return {"drill": "", "method": "", "why": ""}


# ══════════════════════════════════════════════════════════════════════
#  概念→肌肉生物力学映射（基于运动生物力学研究 Ch1 图1.5）
# ══════════════════════════════════════════════════════════════════════

_CONCEPT_TO_MUSCLE: Dict[str, Dict[str, str]] = {
    "problem_p03": {
        "muscle": "背阔肌",
        "action": "离心收缩",
        "phase": "引拍阶段",
        "feel": "引拍时背部两侧有拉伸感",
        "absence": "如果引拍时背部完全没感觉、只感觉手臂在动，说明背阔肌没有参与，手臂就失去了与身体的连接",
        "science": "根据运动生物力学研究，引拍阶段背阔肌做离心收缩来储存弹性势能，是手臂连接身体旋转系统的'胶水'",
    },
    "problem_p02": {
        "muscle": "背阔肌（离心→向心切换）",
        "action": "离心→向心切换",
        "phase": "引拍→前挥过渡",
        "feel": "从背部'存力'到'释放力'的切换感，像弹弓从拉开到弹出",
        "absence": "如果前挥时感觉是手腕在主动翻转而不是背部在释放力量，说明离心→向心的切换失败了，手臂被迫自己做V形scooping来补偿",
        "science": "根据运动生物力学研究，背阔肌在引拍时做离心收缩（储能），前挥时切换为向心收缩（释放）——这个切换是平滑挥拍弧线的动力来源",
    },
    "problem_p05": {
        "muscle": "前锯肌 + 胸大肌",
        "action": "向心收缩",
        "phase": "击球→随挥阶段",
        "feel": "胸部有'往前推'的感觉",
        "absence": "如果击球后手臂立刻内收、没有向外延展，说明前锯肌和胸大肌的向心推送不足",
        "science": "根据运动生物力学研究，向前挥拍阶段胸大肌做向心收缩驱动上臂加速，前锯肌推动肩胛骨前伸实现向外延展",
    },
    "problem_p11": {
        "muscle": "腹斜肌",
        "action": "离心制动",
        "phase": "击球瞬间",
        "feel": "腹部侧面有'刹车感'",
        "absence": "如果转体停不住、整个人跟着转过去，说明腹斜肌的离心制动没有启动",
        "science": "根据运动生物力学研究，击球阶段腹斜肌需要从向心（驱动旋转）切换到离心（制动旋转），这个切换提供精确的旋转终点控制",
    },
    "upper_body_only_turn": {
        "muscle": "臀肌 + 髋关节",
        "action": "离心收缩",
        "phase": "引拍阶段",
        "feel": "臀部和大腿后侧有拉伸感",
        "absence": "如果引拍时只感觉腰在扭但臀部没感觉，说明髋关节没有参与转体，核心蓄力就不存在",
        "science": "根据运动生物力学研究，引拍阶段臀肌和髋关节做离心收缩来启动整体旋转和蓄力，是动力链的第一个重要环节",
    },
    "unit_turn": {
        "muscle": "腹外斜肌 + 背阔肌 + 髋外旋肌群",
        "action": "离心收缩储能",
        "phase": "Unit Turn",
        "feel": "左侧腹斜肌被拉开、背部张力上升、右大腿后侧绷紧",
        "absence": "如果只感觉肩膀在转但腹部没有拉伸感，说明转开幅度不够，核心弹性势能没建立",
        "science": "muscle_activation_guide preparation: 腹外斜肌+背阔肌通过离心收缩储能形成的肩髋分离角，是后续蹬地转髋能爆发出力量的物理前提",
    },
    "straight_legs": {
        "muscle": "股四头肌 + 臀大肌",
        "action": "离心收缩承重",
        "phase": "Unit Turn 后到蹬地前",
        "feel": "大腿前侧承重感、膝盖像压住弹簧",
        "absence": "如果膝盖几乎是直的、大腿前侧没承重感，说明腿没参与蓄力，蹬地时无弹性可释放",
        "science": "运动生物力学：下肢承载（triple bend）是动力链最底层的弹性势能来源，腿直则力量传递链断在源头",
    },
    "problem_p04": {
        "muscle": "腹斜肌（核心中转）",
        "action": "向心收缩",
        "phase": "前挥阶段",
        "feel": "肚脐左侧有拧转感",
        "absence": "如果蹬地后力量传不到上半身，说明腹斜肌这个'中转站'没有工作",
        "science": "根据运动生物力学研究，前挥阶段腹斜肌做向心+离心收缩实现躯干旋转，是腿→胸能量传递的关键中转",
    },
    "problem_p10": {
        "muscle": "背阔肌",
        "action": "离心收缩（维持张力）",
        "phase": "引拍→前挥全程",
        "feel": "背部两侧持续的绷紧感，像拉开的弹弓",
        "absence": "如果前挥过程中背部突然松掉，手臂就失去了'根'，变成独立运动",
        "science": "根据运动生物力学研究，背阔肌在整个引拍-前挥过程中维持张力，是手臂连接身体旋转系统的关键'胶水'肌肉",
    },

    # ── Preparation / Footwork concept muscle cues ──
    "prep01_late_split_step": {
        "muscle": "腓肠肌 + 比目鱼肌",
        "action": "离心→向心切换",
        "phase": "对手contact瞬间（split落地）",
        "feel": "落地时小腿像踩到弹簧，跟腱被快速拉伸再立刻反弹",
        "absence": "如果落地是脚跟先着地或全脚掌'砸'下去，说明小腿储能没发生，下一步会慢半拍",
        "science": "muscle_activation_guide preparation phase: 腓肠肌+比目鱼肌做离心收缩储存蹬地能量，是反应启动的第一个肌肉环节",
    },
    "prep04_no_pivot": {
        "muscle": "髋外旋肌群 + 臀中肌",
        "action": "向心收缩",
        "phase": "split落地后第一帧",
        "feel": "右脚臀部外侧有发力感，右脚后跟轻微离地",
        "absence": "如果只感觉脚掌在拖地、臀部无感觉，说明髋外旋肌没参与，pivot变成了横向滑步",
        "science": "muscle_activation_guide: pivot由髋外旋肌驱动右脚旋转，臀中肌稳定骨盆，是unit turn的物理起点",
    },
    "prep08_late_unit_turn": {
        "muscle": "背阔肌 + 腹外斜肌",
        "action": "离心收缩（启动旋转储能）",
        "phase": "球过网到落地之间",
        "feel": "背部两侧被拉开，腹部侧面有拧转感",
        "absence": "如果球都落地了背部还没拉开感觉，说明unit turn启动晚",
        "science": "Tomaz: racket back before bounce 对应背阔肌和腹外斜肌的离心储能必须在球落地前完成",
    },
    "prep09_arm_only_unit_turn": {
        "muscle": "三角肌 + 肱二头肌（错误使用）",
        "action": "向心收缩（不该工作的肌肉在工作）",
        "phase": "引拍阶段",
        "feel": "感觉只有右肩+右臂在拉拍，背部和左肩完全不参与",
        "absence": "正确的unit turn应该感觉到背部+左肩主导而非右臂；如果右臂酸说明用错了肌肉",
        "science": "FTT/Tom Allsopp: arm-only turn = 三角肌前束代偿，bypass了背阔肌→动力链失效",
    },
    "prep10_no_scapular_glide": {
        "muscle": "菱形肌 + 前锯肌 + 背阔肌",
        "action": "肩胛在肋骨上的滑动门式收缩",
        "phase": "Unit Turn全程",
        "feel": "肩胛骨在背部'滑动'的感觉，球衣背部出现褶皱",
        "absence": "如果只感觉肩膀在转但背部肩胛区无任何滑动感，说明肩胛被锁死，手臂将变成独立运动",
        "science": "FTT 'The back is the glue': 菱形肌后缩+前锯肌前推交替，让肩胛骨成为手臂连接动力链的桥梁",
    },
    "prep13_left_shoulder_not_forward": {
        "muscle": "左侧胸大肌 + 左前三角肌",
        "action": "向心收缩",
        "phase": "Unit Turn启动瞬间",
        "feel": "左侧胸前+前肩有'推'的感觉，把左肩推向球网方向",
        "absence": "如果左侧胸前完全没感觉、只感觉右肩在往后拉，说明启动顺序错了（右臂主导而非左肩主导）",
        "science": "Tomaz feel cue: 'forward with left shoulder first'——左肩前推由左胸大肌+左前三角肌驱动，自动产生unit turn",
    },
    "prep14_insufficient_hip_shoulder_separation": {
        "muscle": "腹外斜肌 + 腹内斜肌",
        "action": "对角离心拉伸",
        "phase": "引拍顶点",
        "feel": "肚脐左侧到肋下有强烈拉伸感（左腹外斜+右腹内斜的对角张力）",
        "absence": "如果引拍到顶时腰腹完全无感觉，说明肩髋一起转了，没有真正的扭矩储能",
        "science": "Tom Allsopp 45°+5° model: 腹斜肌系统在分离角达到35-55°时被拉伸到最大储能位置，是power的核心源头",
    },
    "prep20_no_triple_bend": {
        "muscle": "股四头肌 + 臀大肌 + 小腿三头肌",
        "action": "三关节离心同步加载",
        "phase": "split→unit turn 完成",
        "feel": "大腿前侧+臀部+小腿同时被加载，像三条弹簧一起压紧",
        "absence": "如果只感觉膝盖在弯但臀部和小腿没感觉，说明triple bend只完成了1/3",
        "science": "FTT: 踝/膝/髋同时弯曲让动力链三个环节同时蓄力，单关节弯曲只能贡献1/3的弹射能量",
    },
    "prep27_no_x_stretch": {
        "muscle": "腹外斜肌 + 背阔肌（对角线）",
        "action": "对角线离心拉长",
        "phase": "Unit Turn 到位",
        "feel": "左肩到右髋的对角线（穿过腹部）有明显拉伸感",
        "absence": "如果对角线无拉伸感，说明肩髋分离不够或方向错了",
        "science": "FTT X Stretch: 左肩↔右髋的对角张力是肩髋分离的可视证据，由腹外斜肌+背阔肌的交叉走行肌纤维产生",
    },
    "v2_no_side_bending": {
        "muscle": "腹外斜肌 + 腰方肌",
        "action": "侧弯（左侧向心 / 右侧离心）",
        "phase": "击球瞬间",
        "feel": "左侧腹外斜肌主动收缩、右侧腰方肌被拉长，脊柱不再与地面垂直",
        "absence": "如果只感觉肩在转、腰部毫无感觉，说明在做僵硬的直立旋转，丢失了侧弯发力",
        "science": "Johnny FTT: 侧弯+X-stretch 是真正的旋转power来源，纯水平旋转无法把腿地反作用力转化为拍头加速度",
    },
    "v2_slow_motion_practice": {
        "muscle": "腓肠肌（持续低张力）",
        "action": "持续微离心-向心循环",
        "phase": "等球间隙（contact之间约30帧）",
        "feel": "小腿一直有'弹起'感，脚不真正粘地",
        "absence": "如果脚完全静止粘地，说明已掉入slow-motion模式，反应速度等同站着发呆",
        "science": "FTT Multi-Split-Step: 持续低幅度跳动维持神经-肌肉的预激活状态（拉伸-缩短循环热度），是比赛级反应的物理基础",
    },
}


# ══════════════════════════════════════════════════════════════════════
#  Step 6: 生成诊断叙述
# ══════════════════════════════════════════════════════════════════════

def _compute_score(
    matched_concepts: List[Dict],
    quant_validation: Dict[str, List[str]],
    vlm_score: Optional[int] = None,
) -> int:
    """基于问题严重度计算评分（0-100，越高越好）。

    如果 VLM 给了分数，取诊断引擎计算分和 VLM 分的平均值。
    """
    base = 80  # Higher base; problems deduct 5-8 points each (unique concepts only)

    problems = [m for m in matched_concepts if m["severity"] > 0]
    good_things = [m for m in matched_concepts if m["severity"] == 0]

    # 收集被量化确认的概念 ID
    confirmed_texts = quant_validation.get("confirmed", [])
    # 简单判断：如果 confirm_text 里包含"证实"或"检测到"，对应概念被量化确认
    confirmed_concept_ids: set = set()
    for match in problems:
        concept_id = match["mapped_concept"]
        validations = _CONCEPT_TO_METRIC_VALIDATION.get(concept_id, [])
        for v in validations:
            for ct in confirmed_texts:
                if v.get("confirm_text", "").split("{")[0] in ct:
                    confirmed_concept_ids.add(concept_id)

    # 扣分（去重：同一概念只扣一次，取最高 severity）
    seen_deducted = set()
    for m in problems:
        cid = m["mapped_concept"]
        if cid in seen_deducted:
            continue
        seen_deducted.add(cid)
        deduction = min(8, m["severity"] * 10)
        if cid in confirmed_concept_ids:
            deduction = min(12, deduction * 1.5)
        base -= deduction

    # 加分
    base += len(good_things) * 5

    engine_score = max(20, min(100, int(base)))  # Floor at 20 (not 0)

    # 如果 VLM 也给了分，取平均
    if vlm_score is not None and isinstance(vlm_score, (int, float)):
        return int((engine_score + vlm_score) / 2)
    return engine_score


def _clean_narrative(text: str) -> str:
    """Strip stray markdown / format markers from a coach narrative.

    The narrative is meant to read like a coach talking. AI generators
    sometimes leak format characters (** for bold, [brackets], duplicate
    spaces, mixed CN/EN punctuation, redundant emoji). This pass tames
    them without changing the meaning.
    """
    import re as _re
    if not text:
        return text
    s = text

    # Remove markdown bold/italic markers but keep the text inside
    s = _re.sub(r"\*\*([^*\n]+)\*\*", r"\1", s)
    s = _re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", s)
    s = _re.sub(r"__([^_\n]+)__", r"\1", s)
    s = _re.sub(r"`([^`]+)`", r"\1", s)

    # Drop bracketed source markers like [VLM], [DATA] but keep content
    s = _re.sub(r"\[(?:VLM|DATA|REASONING|FRAME\s*\d+)\]\s*", "", s, flags=_re.IGNORECASE)

    # Normalize Chinese / English punctuation in mixed sentences
    s = s.replace("，，", "，").replace("。。", "。").replace("、、", "、")
    s = s.replace(" ，", "，").replace(" 。", "。").replace(" ：", "：")

    # Collapse runs of 3+ identical newlines / spaces
    s = _re.sub(r"\n{3,}", "\n\n", s)
    s = _re.sub(r"[ \t]{2,}", " ", s)

    # Strip a leading bullet/heading marker that was meant for markdown
    s = _re.sub(r"^\s*#+\s*", "", s, flags=_re.MULTILINE)
    s = _re.sub(r"^\s*[-•]\s+", "", s, flags=_re.MULTILINE)

    # Final tidy
    s = s.strip()
    return s


def _generate_narrative(
    matched_concepts: List[Dict],
    causal_chain: List[Dict[str, str]],
    root_cause_id: Optional[str],
    quant_validation: Dict[str, List[str]],
    user_history: Optional[str],
    fix: Dict[str, str],
    metrics: Dict[str, Any],
    arbitration_log: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Generate a coach-style diagnosis narrative in Chinese.

    Structure:
    - Para 1: What was observed (cite VLM observations)
    - Para 2: Why it happens (cite causal chain from knowledge graph)
    - Para 3: How to fix (drill + why it works)
    """
    parts: List[str] = []

    # ── Paragraph 1: What was observed ──
    problem_concepts = [m for m in matched_concepts if m["severity"] > 0]
    if problem_concepts:
        obs_lines = []
        for m in problem_concepts[:4]:  # Top 4 issues
            frame_info = f"（图{m['frame']}）" if m["frame"] else ""
            obs_lines.append(f"{m['label']}{frame_info}")
        para1 = f"从视频中观察到以下问题：{'、'.join(obs_lines)}。"

        # Add quant evidence naturally
        confirmed = quant_validation.get("confirmed", [])
        if confirmed:
            para1 += f"量化数据佐证：{'；'.join(confirmed[:2])}。"
        contradicted = quant_validation.get("contradicted", [])
        if contradicted:
            para1 += f"但需要注意：{'；'.join(contradicted[:2])}。"
    else:
        para1 = "从视频中未观察到明显的技术问题，各项指标表现尚可。"

    # v4.3: 仲裁日志 —— 被量化数据撤销的 VLM 观察（医生视角展示）
    if arbitration_log:
        arb_notes = []
        for item in arbitration_log:
            arb_notes.append(f"• {item['short_note']}")
        para1 += (
            f"\n\n🔬 **VLM-算法仲裁**（共 {len(arbitration_log)} 条 VLM 观察被量化数据撤销）：\n"
            + "\n".join(arb_notes)
        )

    # Add raw metric numbers if available
    metric_notes = []
    sync = metrics.get("arm_torso_synchrony")
    if sync is not None:
        metric_notes.append(f"同步性{sync:.2f}")
    scoop = metrics.get("scooping_depth")
    if scoop is not None:
        metric_notes.append(f"scooping深度{scoop:.2f}")
    ext = metrics.get("forward_extension")
    if ext is not None:
        metric_notes.append(f"穿透值{ext:.2f}")
    rot = metrics.get("shoulder_rotation")
    if rot is not None:
        metric_notes.append(f"转体{rot:.0f}°")
    if metric_notes:
        para1 += f"（关键数值：{'、'.join(metric_notes)}）"

    parts.append(para1)

    # ── Paragraph 2: Why it happens (causal chain + muscle insight) ──
    if causal_chain:
        root_name = _get_node_name_zh(root_cause_id) if root_cause_id else "未知"
        chain_desc = []
        for link in causal_chain:
            chain_desc.append(f"「{link['from_name']}」导致「{link['to_name']}」")
        para2 = f"根因分析：这些问题的最上游根因是「{root_name}」。"
        para2 += f"因果链路：{'→'.join(chain_desc)}。"
        para2 += "也就是说，你看到的表面症状（" + "、".join(
            m["label"] for m in problem_concepts[:2]
        ) + "）其实是上游问题的下游表现。"
    elif problem_concepts:
        # No causal chain found but have problems
        root_name = _get_node_name_zh(problem_concepts[0]["mapped_concept"])
        para2 = f"根据知识体系分析，核心问题是「{root_name}」。"
        if len(problem_concepts) > 1:
            para2 += "其余问题可能是这个核心问题的下游表现。"
    else:
        para2 = ""

    # ── Muscle insight: add to root cause + top 1 symptom (max 2) ──
    if para2 and problem_concepts:
        muscle_parts: List[str] = []
        # Collect concept IDs to add muscle cues for: root cause + top symptom
        muscle_cue_targets: List[str] = []
        if root_cause_id and root_cause_id in _CONCEPT_TO_MUSCLE:
            muscle_cue_targets.append(root_cause_id)
        for m in problem_concepts[:2]:
            cid = m["mapped_concept"]
            if cid not in muscle_cue_targets and cid in _CONCEPT_TO_MUSCLE:
                muscle_cue_targets.append(cid)
                if len(muscle_cue_targets) >= 2:
                    break

        for cid in muscle_cue_targets:
            mdata = _CONCEPT_TO_MUSCLE[cid]
            cue = (
                f"{mdata['science']}。"
                f"你{mdata['phase']}应该感觉到{mdata['feel']}"
                f"——{mdata['absence']}。"
            )
            muscle_parts.append(cue)

        if muscle_parts:
            para2 += "\n\n" + "\n\n".join(muscle_parts)

    if user_history:
        para2 += f"\n\n⚠ 训练历史提醒：{user_history}"

    if para2:
        parts.append(para2)

    # ── Paragraph 3: How to fix ──
    if fix and fix.get("drill"):
        para3 = f"推荐练习：{fix['drill']}。"
        if fix.get("method"):
            para3 += f"做法：{fix['method']}"
        if fix.get("why"):
            para3 += f"原理：{fix['why']}"
        if fix.get("muscle_cue"):
            para3 += f"\n肌肉感知检验：{fix['muscle_cue']}"
    else:
        para3 = "当前未找到针对性的训练建议，建议继续录制视频观察。"

    parts.append(para3)

    return _clean_narrative("\n\n".join(parts))


# ══════════════════════════════════════════════════════════════════════
#  主入口
# ══════════════════════════════════════════════════════════════════════

def diagnose(vlm_result: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
    """主入口：VLM 输出 + 量化数据 → 完整因果推理诊断。

    保持与旧接口完全兼容：
    - 接收 (vlm_result, metrics) 两个 dict
    - 返回增强后的 vlm_result dict
    - 保留 issues, root_cause_tree 等旧字段
    - 新增 evidence_chain, causal_chain, narrative 等字段

    Parameters
    ----------
    vlm_result : dict
        VLM 返回的分析结果（observation_v1 或 semi_structured_v5 格式）
    metrics : dict
        量化指标（arm_torso_synchrony, scooping_depth 等）

    Returns
    -------
    dict
        增强后的诊断结果
    """
    if not vlm_result:
        return vlm_result or {}
    if not metrics:
        metrics = {}

    result = dict(vlm_result)

    # ── Step 1: 提取 VLM 观察并映射到概念 ──
    # 优先使用 Q 编号直接映射（更精确），然后用关键词匹配补充
    q_matched = _map_via_q_direct(vlm_result)
    observations = _extract_observations_text(vlm_result)
    keyword_matched = _map_observations_to_concepts(observations)

    # 合并：Q 直接映射优先，关键词补充未覆盖的概念
    seen_ids = {m["mapped_concept"] for m in q_matched}
    matched_concepts = list(q_matched)
    for m in keyword_matched:
        if m["mapped_concept"] not in seen_ids:
            matched_concepts.append(m)
            seen_ids.add(m["mapped_concept"])
    matched_concepts.sort(key=lambda m: m["severity"], reverse=True)

    # Q 交叉验证
    raw_answers = vlm_result.get("raw_answers", {})
    q_contradictions = _cross_validate_q_answers(q_matched, raw_answers) if raw_answers else []
    result["q_contradictions"] = q_contradictions

    # ── Step 1.5: KPI 信号注入到 matched_concepts ──
    # 让 L4/L5 量化证据进入 earliest-layer 候选池，避免 VLM 漏报时
    # 系统把下游症状（L3 小臂代偿）误判为根因。
    matched_concepts = _inject_kpi_problems(matched_concepts, metrics)

    # ── Step 1.6: VLM ↔ 量化数据 仲裁 ──
    # 当 VLM 视觉观察与量化算法冲突时（如 V 形轨迹 vs scooping 未检测），
    # 以量化数据为准撤销 VLM 的 false positive。避免"矛盾并列"误导用户。
    matched_concepts, arbitration_log = _arbitrate_vlm_vs_metrics(
        matched_concepts, metrics
    )

    # ── Step 2: 沿知识图谱追溯根因 ──
    problem_concept_ids = [
        m["mapped_concept"] for m in matched_concepts if m["severity"] > 0
    ]
    root_cause_id, causal_chain = _trace_root_causes(
        problem_concept_ids,
        matched_concepts_for_voting=matched_concepts,
    )

    # ── Step 3: 量化验证 ──
    # 排除已被仲裁层处理的概念，避免双重展示同一 VLM-metric 冲突
    arbitrated_ids = {item["concept"] for item in arbitration_log}
    quant_validation = _validate_with_metrics(
        matched_concepts, metrics, excluded_concepts=arbitrated_ids
    )

    # ── Step 4: 用户历史对比 ──
    user_history = _check_user_history(matched_concepts)

    # ── Step 5: 训练建议 ──
    fix = _get_fix(root_cause_id, matched_concepts)

    # ── Step 6: 计算分数和生成叙述 ──
    vlm_score = vlm_result.get("score")
    if isinstance(vlm_score, str):
        try:
            vlm_score = int(re.search(r"\d+", vlm_score).group())
        except (AttributeError, ValueError, TypeError):
            vlm_score = None

    score = _compute_score(matched_concepts, quant_validation, vlm_score)
    narrative = _generate_narrative(
        matched_concepts, causal_chain, root_cause_id,
        quant_validation, user_history, fix, metrics,
        arbitration_log=arbitration_log,
    )

    # ── Assemble output ──
    root_cause_name = _get_node_name_zh(root_cause_id) if root_cause_id else ""

    result["root_cause"] = root_cause_name
    result["root_cause_concept_id"] = root_cause_id
    # v4.2: expose root cause layer for top-down reporting
    root_cause_layer = _get_concept_layer(root_cause_id) if root_cause_id else None
    result["root_cause_layer"] = root_cause_layer

    result["evidence_chain"] = [
        {
            "observation": m["observation"],
            "mapped_concept": m["mapped_concept"],
            "frame": m["frame"],
        }
        for m in matched_concepts if m["severity"] > 0
    ]

    result["causal_chain"] = causal_chain

    result["quant_validation"] = quant_validation

    # v4.3: 仲裁日志（VLM 观察被量化数据撤销的列表）
    result["arbitration_log"] = arbitration_log

    result["user_history"] = user_history

    result["fix"] = fix

    result["narrative"] = narrative

    result["score"] = score

    # ── Backward compatibility: keep old fields ──

    # quant_evidence (old): one-liner summary
    evidence_parts = _build_quant_evidence(metrics)
    result["quant_evidence"] = " ".join(evidence_parts) if evidence_parts else ""

    # contradictions (old): VLM vs metric contradictions
    result["contradictions"] = [
        {"name": text, "correction": text}
        for text in quant_validation.get("contradicted", [])
    ]

    # quant_summary (old)
    result["quant_summary"] = _build_quant_summary(metrics)

    # diagnosis_confidence (old)
    if quant_validation.get("contradicted"):
        result["diagnosis_confidence"] = "需要验证"
    elif matched_concepts:
        result["diagnosis_confidence"] = "高"
    else:
        result["diagnosis_confidence"] = "中"

    # root_cause_tree (old format): merge with existing or create
    existing_tree = result.get("root_cause_tree") or {}
    if isinstance(existing_tree, dict):
        existing_tree.setdefault("root_cause", root_cause_name)
        existing_tree.setdefault("root_cause_evidence", "")
        existing_tree.setdefault("causal_explanation", narrative)
        existing_tree.setdefault("downstream_symptoms", [])
        existing_tree.setdefault("fix", fix)
        if root_cause_layer:
            existing_tree.setdefault("layer", root_cause_layer)
    else:
        existing_tree = {
            "root_cause": root_cause_name,
            "root_cause_evidence": "",
            "causal_explanation": narrative,
            "downstream_symptoms": [],
            "fix": fix,
        }
    result["root_cause_tree"] = existing_tree

    # issues (old format): ensure it exists
    if not result.get("issues"):
        result["issues"] = []
    if root_cause_name and not any(
        i.get("name") == root_cause_name for i in result["issues"] if isinstance(i, dict)
    ):
        result["issues"].append({
            "name": root_cause_name,
            "severity": "高" if score < 50 else "中",
            "frame": "",
            "description": narrative.split("\n\n")[0] if narrative else "",
        })

    return result


# ══════════════════════════════════════════════════════════════════════
#  Legacy helper functions (kept for backward compat)
# ══════════════════════════════════════════════════════════════════════

def _build_quant_evidence(metrics: Dict) -> List[str]:
    """从量化数据中提取关键佐证文本。"""
    parts = []

    sync = metrics.get("arm_torso_synchrony")
    if sync is not None:
        if sync < 0.4:
            parts.append(f"手臂-躯干同步性只有 {sync:.2f}，手臂几乎完全独立于身体运动。")
        elif sync < 0.7:
            parts.append(f"手臂-躯干同步性 {sync:.2f}，手臂部分跟随身体但还不够紧密。")

    scoop = metrics.get("scooping_depth")
    if scoop is not None and scoop > 0.3:
        parts.append(f"Scooping 深度 {scoop:.2f}，手腕轨迹出现明显的 V 形下坠。")

    ext = metrics.get("forward_extension")
    if ext is not None:
        if ext < 0.2:
            parts.append(f"向前穿透只有 {ext:.2f}，击球后几乎没有向前延展。")
        elif ext > 0.8:
            parts.append(f"向前穿透 {ext:.2f}，击球后有不错的向前延展。")

    rot = metrics.get("shoulder_rotation")
    if rot is not None:
        if rot < 30:
            parts.append(f"肩部转开只有 {rot:.0f}°，转体严重不足。")

    arc = metrics.get("swing_arc_ratio")
    if arc is not None and arc > 5:
        parts.append(f"挥拍弧度比 {arc:.1f}，轨迹严重向内收缩。")

    return parts


def _build_quant_summary(metrics: Dict) -> str:
    """一句话总结关键量化发现。"""
    issues = []

    sync = metrics.get("arm_torso_synchrony")
    if sync is not None and sync < 0.4:
        issues.append("手臂独立")

    scoop = metrics.get("scooping_depth")
    if scoop is not None and scoop > 0.3:
        issues.append("V形捞球")

    ext = metrics.get("forward_extension")
    if ext is not None and ext < 0.2:
        issues.append("缺穿透")

    rot = metrics.get("shoulder_rotation")
    if rot is not None and rot < 30:
        issues.append("转体不足")

    if not issues:
        return "量化指标未发现明显异常。"
    return f"量化数据佐证：{' + '.join(issues)}。"
