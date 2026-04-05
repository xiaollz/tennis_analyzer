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
]


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
        "drill": "腋下贴住练习",
        "method": "将一个网球或毛巾夹在击球手臂腋下，做完整挥拍。如果掉落说明手臂在做独立动作。",
        "why": "腋下贴住强制手臂成为躯干的一部分，消除手臂独立发力的空间，让身体旋转自然带动手臂。",
    },
    "problem_p02": {
        "drill": "平推穿透练习",
        "method": "站在离网2米处，用拍面平推球过网，不允许任何向上动作。目标是让球平飞过网。",
        "why": "V形scooping的根因是手臂做了独立的下压+上翻。平推练习消除向上意图，重建水平穿透路径。",
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
    },
    "problem_p04": {
        "drill": "蹬地→转髋→腹斜肌串联练习",
        "method": "站立，右脚蹬地→感受力经髋部→左下腹（肚脐左侧）有拧转感→胸部被带走。从慢速开始。",
        "why": "动力链断裂通常在3个点：膝-髋、髋-腹斜肌、胸-臂。逐段串联让能量完整传递。",
    },
    "problem_p05": {
        "drill": "弹...落...打 三拍节奏练习",
        "method": "球弹起时说'弹'，最高点说'落'，下落到击球高度说'打'。退后半步给自己更多时间。",
        "why": "击球点偏后的根因通常是站太前+准备时间不足。三拍节奏强制建立时空余量。",
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
    },
}


# ══════════════════════════════════════════════════════════════════════
#  知识图谱缓存和加载
# ══════════════════════════════════════════════════════════════════════

_GRAPH_SNAPSHOT_PATH = Path(__file__).parent.parent / "knowledge" / "extracted" / "_graph_snapshot.json"
_DIAGNOSTIC_CHAINS_PATH = Path(__file__).parent.parent / "knowledge" / "extracted" / "ftt_video_diagnostic_chains.json"
_USER_HISTORY_PATH = Path(__file__).parent.parent / "knowledge" / "extracted" / "user_journey" / "learning_deep_analysis.json"

_cached_graph_data: Optional[Dict] = None
_cached_chains: Optional[List[Dict]] = None
_cached_user_history: Optional[Dict] = None


def _load_graph_data() -> Dict:
    """Load and cache the graph snapshot."""
    global _cached_graph_data
    if _cached_graph_data is None:
        try:
            _cached_graph_data = json.loads(_GRAPH_SNAPSHOT_PATH.read_text())
        except Exception:
            _cached_graph_data = {"nodes": [], "edges": []}
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


def _map_observations_to_concepts(
    observations: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Map VLM observations to knowledge graph concepts using keyword matching.

    Returns list of matched concepts with evidence.
    """
    matched: List[Dict[str, Any]] = []
    seen_concepts: Dict[str, Dict] = {}  # concept_id -> best match

    for obs in observations:
        text = obs["text"]
        frame = obs["frame"]

        for rule in OBSERVATION_TO_CONCEPT:
            # Check if any keyword matches
            for kw in rule["keywords"]:
                if kw.lower() in text.lower():
                    concept_id = rule["concept"]

                    # Check frame_range constraint if specified
                    if rule["frame_range"] is not None and frame is not None:
                        if frame not in rule["frame_range"]:
                            continue

                    match_info = {
                        "observation": text[:200],  # Truncate long texts
                        "keyword_matched": kw,
                        "mapped_concept": concept_id,
                        "frame": frame,
                        "field": obs["field"],
                        "severity": rule["severity"],
                        "label": rule["label"],
                    }

                    # Keep the best match per concept (highest severity, or first)
                    if concept_id not in seen_concepts:
                        seen_concepts[concept_id] = match_info
                        matched.append(match_info)
                    elif rule["severity"] > seen_concepts[concept_id]["severity"]:
                        # Replace with higher severity match
                        idx = next(i for i, m in enumerate(matched) if m["mapped_concept"] == concept_id)
                        matched[idx] = match_info
                        seen_concepts[concept_id] = match_info

                    break  # Only need one keyword to match per rule

    # Sort by severity descending
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


def _get_node_name_zh(concept_id: str) -> str:
    """Get Chinese name of a concept from graph."""
    graph_data = _load_graph_data()
    for node in graph_data.get("nodes", []):
        if node.get("id") == concept_id:
            return node.get("name_zh", concept_id)
    return concept_id


def _trace_root_causes(concept_ids: List[str]) -> Tuple[Optional[str], List[Dict[str, str]]]:
    """Trace from symptom concepts to root causes via causes edges.

    Returns (root_cause_id, causal_chain_list).
    """
    reverse_causes = _build_causes_graph()
    all_chains: List[List[str]] = []

    for concept_id in concept_ids:
        # Walk backwards through causes edges
        visited = set()

        def _walk(node: str, path: List[str]) -> None:
            parents = reverse_causes.get(node, [])
            if not parents:
                # Reached a root — record
                all_chains.append(list(path))
                return
            for parent in parents:
                if parent not in visited and parent != "c_unnamed":
                    visited.add(parent)
                    _walk(parent, path + [parent])

        visited.add(concept_id)
        _walk(concept_id, [concept_id])

    if not all_chains:
        # No causes found in graph; the concepts themselves are the "root"
        if concept_ids:
            return concept_ids[0], []
        return None, []

    # Find the longest chain (deepest root cause)
    best_chain = max(all_chains, key=len)
    root_cause_id = best_chain[-1]  # Last element is the deepest upstream cause

    # Build causal chain as list of edges
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


# ══════════════════════════════════════════════════════════════════════
#  Step 3: 量化验证
# ══════════════════════════════════════════════════════════════════════

def _validate_with_metrics(
    matched_concepts: List[Dict[str, Any]],
    metrics: Dict[str, Any],
) -> Dict[str, List[str]]:
    """Validate concept matches against quantitative metrics.

    Returns {"confirmed": [...], "contradicted": [...]}.
    """
    confirmed: List[str] = []
    contradicted: List[str] = []

    checked_concepts = set()
    for match in matched_concepts:
        concept_id = match["mapped_concept"]
        if concept_id in checked_concepts:
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
#  Step 6: 生成诊断叙述
# ══════════════════════════════════════════════════════════════════════

def _compute_score(
    matched_concepts: List[Dict],
    quant_validation: Dict[str, List[str]],
    vlm_score: Optional[int],
) -> int:
    """Compute a diagnosis score (0-100, higher = better).

    Based on severity of matched issues and quantitative validation.
    """
    if not matched_concepts:
        return vlm_score if vlm_score is not None else 70

    # Filter to actual problems (severity > 0)
    problems = [m for m in matched_concepts if m["severity"] > 0]
    if not problems:
        return vlm_score if vlm_score is not None else 75

    # Max severity determines base deduction
    max_severity = max(m["severity"] for m in problems)
    total_severity = sum(m["severity"] for m in problems)

    # Base score starts at 80, deducted by severity
    base = 80 - int(max_severity * 30) - int(min(total_severity - max_severity, 2.0) * 10)

    # Contradictions add back some points (maybe not as bad as thought)
    contradictions = len(quant_validation.get("contradicted", []))
    base += contradictions * 5

    # Clamp
    return max(15, min(85, base))


def _generate_narrative(
    matched_concepts: List[Dict],
    causal_chain: List[Dict[str, str]],
    root_cause_id: Optional[str],
    quant_validation: Dict[str, List[str]],
    user_history: Optional[str],
    fix: Dict[str, str],
    metrics: Dict[str, Any],
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

    # ── Paragraph 2: Why it happens (causal chain) ──
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
    else:
        para3 = "当前未找到针对性的训练建议，建议继续录制视频观察。"

    parts.append(para3)

    return "\n\n".join(parts)


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
    observations = _extract_observations_text(vlm_result)
    matched_concepts = _map_observations_to_concepts(observations)

    # ── Step 2: 沿知识图谱追溯根因 ──
    problem_concept_ids = [
        m["mapped_concept"] for m in matched_concepts if m["severity"] > 0
    ]
    root_cause_id, causal_chain = _trace_root_causes(problem_concept_ids)

    # ── Step 3: 量化验证 ──
    quant_validation = _validate_with_metrics(matched_concepts, metrics)

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
    )

    # ── Assemble output ──
    root_cause_name = _get_node_name_zh(root_cause_id) if root_cause_id else ""

    result["root_cause"] = root_cause_name
    result["root_cause_concept_id"] = root_cause_id

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
