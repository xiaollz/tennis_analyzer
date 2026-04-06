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


def _get_node_name_zh(concept_id: str) -> str:
    """Get Chinese name of a concept from graph."""
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
    # Note: diagnostic chains have data quality issues (p09 appears as root cause
    # for almost everything). Skipping chain lookup, using direct observation instead.

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

    # ── Step 2: 沿知识图谱追溯根因 ──
    problem_concept_ids = [
        m["mapped_concept"] for m in matched_concepts if m["severity"] > 0
    ]
    root_cause_id, causal_chain = _trace_root_causes(
        problem_concept_ids,
        matched_concepts_for_voting=matched_concepts,
    )

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
