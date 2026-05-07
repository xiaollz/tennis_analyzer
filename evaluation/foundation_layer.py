"""
Foundation Layer — 网球分析系统的"地基检查层"

这个模块 hard-wire 了 7 个 foundation 检查项，确保 runtime 视频分析时
按"地基优先"原则做。priority=0 的 foundation（FTT Foundation 4 项）失败时，
应阻断上层分析，先修地基。priority=1 的 foundation（个人圣经 2 项）次之。

设计原则：
- 数据硬：优先量化指标，其次 VLM 关键词
- 证据显式：每个判断都附带触发证据
- 失败级联：明确告知失败会导致哪些下游问题
- 立即可行：每条失败都有对应 drill + 出处

来源：
- FTT Foundation 4 项: docs/research/18_ftt_build_foundation.md
- 4/27 圣经: memory/project_right_foot_axis_bible.md
- 4/30 圣经: memory/project_scapular_slot_bible.md + docs/research/scapular_stabilization_root_cause.md
- 5/3 HSA: docs/research/hsa_master_index.md + memory/project_hsa_engine.md
"""

import re
from typing import Any, Dict, List, Optional


# =============================================================================
# FOUNDATIONS 数据结构
# =============================================================================

FOUNDATIONS: List[Dict[str, Any]] = [
    # -------------------------------------------------------------------------
    # Priority 0 — FTT Foundation 4 项（最基础）
    # -------------------------------------------------------------------------
    {
        "id": "F1_hold_up",
        "name": "架拍",
        "english": "Hold Up",
        "source": "docs/research/18_ftt_build_foundation.md §2.3 Holding Up, Pull Forward",
        "priority": 0,
        "vlm_question_ids": ["Q24", "Q9"],  # racket_hold_up + trajectory_shape
        "metric_keys": ["wrist_height_pattern"],
        "pass_criteria": [
            "拍头从引拍最高点保持位置，无明显下降",
            "Q9 trajectory 不出现 V 形（先下坠后上升）",
            "wrist_height_pattern 不是 'V形scooping'",
        ],
        "fail_evidence_keywords": [
            "拍头早垂", "先下坠再上升", "先下坠后上升", "V形", "V 形",
            "下沉到腰部", "急剧下坠", "拍头下垂", "scooping", "舀球",
        ],
        "fail_metric_thresholds": {
            "wrist_height_pattern": {"equals": "V形scooping"},
        },
        "downstream_cascade": [
            "拍头早垂",
            "挥拍距离过长",
            "时间不够",
            "手臂主动加速代偿",
            "arming（小臂代偿）",
        ],
        "drill": "Hold it Up shadow",
        "drill_source": "evaluation/diagnosis_engine.py prep11_racket_head_dropped_early (line 784)",
        "explanation": "FTT 视频开篇就是 'Hold Up'：架拍维持高位才能让前挥距离短、时间够。"
                       "拍头早垂是所有 arming 问题的源头。",
    },
    {
        "id": "F2_place_pull",
        "name": "放置-拉向前",
        "english": "Place, Pull Forward",
        "source": "docs/research/18_ftt_build_foundation.md §2.3",
        "priority": 0,
        "vlm_question_ids": ["Q26"],  # place_then_pull
        "metric_keys": [],
        "pass_criteria": [
            "Q26 描述为'一气呵成'或'连续后拉再向前'",
            "无明显的高位停顿等待",
        ],
        "fail_evidence_keywords": [
            "停顿", "举到高位等待", "高位等待", "stop-start", "stop start",
            "两步动作", "中间静止", "顶点停顿", "断开",
        ],
        "fail_metric_thresholds": {},
        "downstream_cascade": [
            "高位停顿",
            "胸大肌张力衰减",
            "胸推肘失败",
            "大臂飘",
        ],
        "drill": "Place, Pull Forward 口令",
        "drill_source": "evaluation/diagnosis_engine.py prep24_backswinging_not_placing (line 852)",
        "explanation": "FTT: 'There is no backswing'。把'后挥'改成'放置→拉向前'，"
                       "避免顶点停顿导致胸大肌张力丢失。",
    },
    {
        "id": "F3_back_glue",
        "name": "背部胶水",
        "english": "Back is the Glue (arm-body sync)",
        "source": "docs/research/18_ftt_build_foundation.md §2.1 + docs/research/22_ftt_scapular_glide.md",
        "priority": 0,
        "vlm_question_ids": ["Q1"],  # arm_body_sync
        "metric_keys": ["arm_body_sync_score"],
        "pass_criteria": [
            "手臂-躯干同步性指标 ≥ 0.5",
            "Q1 答案不出现脱节关键词",
        ],
        "fail_evidence_keywords": [
            "不同步", "独立", "脱节", "先于身体", "手臂先动", "手臂单独",
            "和身体分离", "手臂提前", "身体没跟上", "arming",
        ],
        "fail_metric_thresholds": {
            "arm_body_sync_score": {"lt": 0.4},
        },
        "downstream_cascade": [
            "手臂独立动作",
            "arming（小臂代偿）",
            "大臂飘",
            "球软无穿透",
        ],
        "drill": "双手扶拍 unit turn + 背部 2.5 磅哑铃感受",
        "drill_source": "FTT §2.1 + diagnosis_engine.py prep10_no_scapular_glide (line 778)",
        "explanation": "FTT: 'The back is the glue'。背部肩胛是手臂连接动力链的桥梁，"
                       "缺失=必然小臂代偿。",
    },
    {
        "id": "F4_unit_action",
        "name": "整体转",
        "english": "Unit Turn + Unit Swing",
        "source": "docs/research/18_ftt_build_foundation.md §2.5",
        "priority": 0,
        "vlm_question_ids": ["Q23", "Q5b"],  # left_shoulder_leads + hip-shoulder differential
        "metric_keys": ["shoulder_hip_differential_deg"],
        "pass_criteria": [
            "左肩主动后推带动右肩（Q23）",
            "肩转-髋转差 < 60°（unit 而非 over-rotation）",
        ],
        "fail_evidence_keywords": [
            "左肩没参与", "左肩没动", "右肩单独后拉", "只有右臂", "拉拍",
            "肩先于髋", "上下脱节", "上下半身分离",
        ],
        "fail_metric_thresholds": {
            "shoulder_hip_differential_deg": {"gt": 60.0},
        },
        "downstream_cascade": [
            "上下半身脱节",
            "unit turn 不足",
            "胸大肌没拉开",
            "press slot 错位",
        ],
        "drill": "双手扶拍 unit turn",
        "drill_source": "diagnosis_engine.py prep09_arm_only_unit_turn + prep12_left_hand_dropped_in_unit_turn",
        "explanation": "FTT 核心：左手扶拍喉到 forward swing 启动，"
                       "强制左肩主导整体转，避免右臂单独后拉。",
    },
    # -------------------------------------------------------------------------
    # Priority 1 — 个人圣经（4/27 + 4/30 晚突破）
    # -------------------------------------------------------------------------
    {
        "id": "F5_right_foot_axis",
        "name": "右脚为轴",
        "english": "Right Foot as Axis",
        "source": "memory/project_right_foot_axis_bible.md",
        "priority": 1,
        "vlm_question_ids": ["Q15", "Q16", "Q31"],  # weight_transfer + pivot + stance
        "metric_keys": ["weight_transfer_timing", "pivot_quality"],
        "pass_criteria": [
            "右脚 pivot 拧转保持承重（Q16）",
            "半开放站姿（Q31）",
            "weight transfer 不早于挥拍启动（Q15）",
        ],
        "fail_evidence_keywords": [
            "早期 weight transfer", "重心提前", "重心过早", "前脚承重",
            "双脚平移", "无 pivot", "右脚没拧", "右脚没动", "失去支点",
        ],
        "fail_metric_thresholds": {
            "weight_transfer_timing": {"equals": "early"},
        },
        "downstream_cascade": [
            "失去旋转支点",
            "全身没 anchor",
            "4 坐标错位",
            "球软 / 被挤 / 后倒",
        ],
        "drill": "右脚 pivot drill + 一脚击球",
        "drill_source": "memory/project_right_foot_axis_bible.md (用户 4/27 Top 1 顿悟)",
        "explanation": "用户 4/27 Top 1 顿悟：右脚为轴 = 一切。"
                       "11 字系统都服务于这一件事，所有过去痛点的统一根因。",
    },
    {
        "id": "F6_scapular_slot",
        "name": "肩胛骨槽",
        "english": "Scapular Slot",
        "source": "memory/project_scapular_slot_bible.md + docs/research/scapular_stabilization_root_cause.md",
        "priority": 1,
        "vlm_question_ids": ["Q24", "Q25", "Q3"],  # racket_hold_up + scapular_glide_jersey + arm_chest_gap
        "metric_keys": ["elbow_height_relative_chest", "arm_chest_gap_change"],
        "pass_criteria": [
            "架拍时肘高度在胸口-嘴之间（Q24）",
            "非持拍侧背部球衣有拉伸褶皱（Q25）",
            "大臂-胸侧间距稳定不扩大（Q3）",
        ],
        "fail_evidence_keywords": [
            "肘部低于胸口", "肘下沉", "肘掉到腰",
            "没观察到褶皱", "球衣无褶皱", "肩胛没滑动",
            "间距明显变大", "一个拳头以上", "大臂离开胸侧",
            "槽脱离", "肩胛失稳",
        ],
        "fail_metric_thresholds": {
            "arm_chest_gap_change": {"equals": "increase"},
        },
        "downstream_cascade": [
            "肩胛槽脱离",
            "胸大肌无 fixed point",
            "胸推肘失败（4/30 突破失守）",
            "4/29 大臂住肩窝失守",
        ],
        "drill": "抬肘到嘴高度 mirror test + 左手摸右肩胛骨自检",
        "drill_source": "memory/project_scapular_slot_bible.md (用户 4/30 晚 新圣经)",
        "explanation": "4/30 晚新圣经：肩胛骨槽是胸大肌发力的固定点（fixed point），"
                       "槽稳=胸推肘可行；槽脱=4/29 + 4/30 突破全部失守。",
    },
    # -------------------------------------------------------------------------
    # Priority 1 — 5/3 HSA 突破（驱动引擎，非地基）
    # -------------------------------------------------------------------------
    {
        "id": "F7_hsa",
        "name": "肩水平内收",
        "english": "Horizontal Shoulder Adduction (HSA)",
        "source": "docs/research/hsa_master_index.md + memory/project_hsa_engine.md "
                  "+ evaluation/hsa_detector.py",
        "priority": 1,
        # Q43/Q44 (5/8 加入) — ESR 早启动检测，是 HSA 的物理前置条件
        "vlm_question_ids": ["Q39", "Q40", "Q43", "Q44"],
        "metric_keys": [
            "hsa_total_closure_deg",
            "hsa_angle_at_contact",
            "hsa_closure_pattern",
            "cross_body_finish",
            "hsa_health_score",
            # 5/8 加入 — ESR 启动质量（HSA 物理前置条件）
            "esr_quality",  # one of: "early_progressive" / "late_burst" / "absent" / "uncertain"
        ],
        "pass_criteria": [
            "总闭合幅度 ≥ 25° (peak → contact)",
            "接触瞬间 HSA 角在 45°-80° 健康区间",
            "cross_body_finish == True (右腕越过左肩)",
            "hsa_closure_pattern == 'healthy'",
            "hsa_health_score ≥ 60",
            # ESR 前置条件（5/8）
            "esr_quality == 'early_progressive'（Unit Turn 第一帧即启动 + 早期渐进）",
        ],
        "fail_evidence_keywords": [
            "大臂没有跨过胸前", "胸肌未参与", "右臂保持外展",
            "随挥往侧方", "随挥未越过身体", "拍头未交叉",
            "无内收", "肘卡身侧", "推球", "推送",
            "纯靠手腕翻拍", "纯靠转体", "胸肱角未关闭",
            "随挥停在右侧", "随挥指向侧面",
            # ESR 偷懒证据（5/8）
            "前臂朝向不变", "拍头未朝天", "末段突变 ESR",
            "ESR 启动晚", "Unit Turn 没外旋", "外旋偷懒",
        ],
        "fail_metric_thresholds": {
            "hsa_total_closure_deg": {"lt": 15.0},
            "hsa_angle_at_contact": {"gt": 90.0},
            "hsa_closure_pattern": {"equals": "no_closure"},
            "cross_body_finish": {"equals": "False"},
            "hsa_health_score": {"lt": 40.0},
            # ESR 失败阈值（5/8）— 末段突变 / 缺失都判 FAIL
            "esr_quality": {"equals": "late_burst"},
        },
        "downstream_cascade": [
            "胸大肌未发力（chest 未 fire）",
            "ISR 无上游驱动 → 自动消失",
            "Pronation 孤立 → 球软无穿透",
            "辛纳式肘前推无下游释放",
            "RHS 损失 25-48%（HSA 是单一关节最大贡献者）",
            # ESR 偷懒下游（5/8）
            "ESR 偷懒 → IR 抢跑（食指吃力 / 大臂上抬）→ HSA 闭合时大臂位置已错",
        ],
        "drill": "1) 手按胸大肌徒手横拉空挥（FTT Am8j1Zw5KrE 视频开头）"
                 " 2) 静态无转体击球（FTT 5KdScDKxVSI [03:40]）"
                 " 3) 反向工程握拍（FTT 5KdScDKxVSI [01:05]）"
                 " 4) [ESR 前置] Unit Turn 第一帧主动外旋（冈下肌发力）+ 镜前 50 次",
        "drill_source": "docs/research/hsa_youtube_survey.md Tier S 视频 + "
                        "docs/research/hsa_biomechanics_deep_dive.md §7 训练协议",
        "explanation": "5/3 突破：HSA 是 chest fire 的物理本体——'关闭胸肱角' 是真正"
                       "的发力动作，press slot / chest engagement / 胸推肘 / 撕 等概念"
                       "都是 HSA 的不同视角描述。Sasaki 2022 IMU 实测 HSA 贡献 45-48% "
                       "前向 RHS（单一关节最大），Kovacs 综述 HSA+ISR 合占 65% 接触速度。"
                       "F7 不是地基，是驱动引擎；它依赖 F5（右脚轴 = 能量入口）+ "
                       "F6（肩胛槽 = 发射台）作为支撑。"
                       " 5/8 增强：ESR 早启动是 HSA 的物理前置条件——Unit Turn 第一帧"
                       "主动外旋抑制 IR 抢跑，HSA 才有正确的初始大臂角度可以闭合。"
                       "ESR 偷懒（末段突变）= IR 抢跑表现，作为 F7 的子检测。",
    },
]


# =============================================================================
# 内部工具函数
# =============================================================================

def _safe_get_text(d: Optional[Dict[str, Any]], key: str) -> str:
    """安全获取字符串值，缺失或 None 返回空串。"""
    if not d:
        return ""
    val = d.get(key, "")
    if val is None:
        return ""
    return str(val)


_NEGATION_PREFIXES = [
    "无明显", "无明显的", "没有明显的", "没有明显", "没有", "并未", "并非",
    "无", "不出现", "未出现", "未见", "未观察到", "没观察到", "看不到", "不存在",
]
# 强正面信号——这些词出现在同一句中，覆盖该句里的负面关键词
_STRONG_POSITIVE_OVERRIDES = [
    "一气呵成", "连贯", "流畅", "同时启动", "同时同步", "无停顿", "无脱节",
]


def _split_sentences(text: str) -> List[str]:
    """简单中文分句：按 。 ， ； 切分；保留全文作为最后一项。"""
    if not text:
        return []
    sents = re.split(r"[。；，,;]", text)
    return [s for s in sents if s.strip()]


def _check_keyword_hits(text: str, keywords: List[str]) -> List[str]:
    """返回命中的关键词列表（小写匹配也算）。

    带中文否定上下文识别：
    1. 关键词紧跟在「无 / 没有 / 无明显 / 不出现」等否定词之后（窗口 30 字符）→ 不算命中
    2. 关键词所在分句包含强正面信号（"一气呵成"、"流畅"等）→ 不算命中
    避免「没有明显停顿」「连续一气呵成」之类的误判。
    """
    if not text:
        return []
    text_lower = text.lower()
    sentences = _split_sentences(text)
    hits: List[str] = []
    for kw in keywords:
        if not kw:
            continue
        kw_lower = kw.lower()
        matched_once = False
        for haystack, needle in ((text, kw), (text_lower, kw_lower)):
            idx = haystack.find(needle)
            while idx != -1:
                # 否定窗口扩展到 30 字符（覆盖"没有明显的在后方'停顿"这种）
                prefix_window = haystack[max(0, idx - 30):idx]
                negated = any(neg in prefix_window for neg in _NEGATION_PREFIXES)
                # 检查所在分句是否有强正面信号覆盖
                sentence_with_kw = next(
                    (s for s in sentences if needle in s or needle in s.lower()), ""
                )
                positive_override = any(
                    pos in sentence_with_kw for pos in _STRONG_POSITIVE_OVERRIDES
                )
                if not negated and not positive_override:
                    hits.append(kw)
                    matched_once = True
                    break
                idx = haystack.find(needle, idx + 1)
            if matched_once:
                break
    return hits


def _check_metric_threshold(metric_value: Any, threshold: Dict[str, Any]) -> bool:
    """检查 metric 是否超过失败阈值。返回 True 表示失败。"""
    if metric_value is None:
        return False
    try:
        if "equals" in threshold:
            return str(metric_value).strip() == str(threshold["equals"]).strip()
        if "lt" in threshold:
            return float(metric_value) < float(threshold["lt"])
        if "gt" in threshold:
            return float(metric_value) > float(threshold["gt"])
        if "lte" in threshold:
            return float(metric_value) <= float(threshold["lte"])
        if "gte" in threshold:
            return float(metric_value) >= float(threshold["gte"])
    except (ValueError, TypeError):
        return False
    return False


def _is_uncertain_answer(text: str) -> bool:
    """判断 VLM 答案是否表达不确定（看不清/无法判断）。"""
    if not text or not text.strip():
        return True
    uncertain_markers = [
        "看不清", "无法判断", "不确定", "看不出", "难以判断",
        "无法观察", "画面不清", "unclear", "cannot tell", "n/a", "未知",
    ]
    text_lower = text.lower()
    for marker in uncertain_markers:
        if marker in text or marker.lower() in text_lower:
            return True
    return False


# =============================================================================
# 主函数：check_foundations
# =============================================================================

def check_foundations(
    vlm_answers: Optional[Dict[str, str]],
    metrics: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    检查 7 个 foundation 的状态。

    判断逻辑：
    1. 优先量化指标（数据硬）：metric 超阈值 → fail
    2. 其次 VLM 关键词：fail_evidence_keywords 命中任一 → fail
    3. 证据不足（VLM 答"看不清" 且无量化指标）→ uncertain
    4. priority == 0 失败 → should_block_downstream = True

    参数:
        vlm_answers: VLM 答案 dict，key 为 Q1/Q9/Q23 等，value 为答案文本
        metrics: 量化指标 dict，key 见 FOUNDATIONS[i].metric_keys

    返回:
        List[Dict]，每项对应一个 foundation 的检查结果
    """
    vlm_answers = vlm_answers or {}
    metrics = metrics or {}
    results: List[Dict[str, Any]] = []

    for foundation in FOUNDATIONS:
        evidence: List[str] = []
        fail = False
        has_signal = False  # 是否有任何可判断的信号

        # 1) 量化指标检查（优先）
        for metric_key in foundation["metric_keys"]:
            if metric_key in metrics and metrics[metric_key] is not None:
                has_signal = True
                threshold = foundation["fail_metric_thresholds"].get(metric_key)
                if threshold and _check_metric_threshold(metrics[metric_key], threshold):
                    fail = True
                    evidence.append(
                        f"指标 {metric_key} = {metrics[metric_key]} 触发阈值 {threshold}"
                    )

        # 2) VLM 关键词检查
        for q_id in foundation["vlm_question_ids"]:
            answer = _safe_get_text(vlm_answers, q_id)
            if not answer:
                continue
            if _is_uncertain_answer(answer):
                # 该 VLM 答案无效，但不计入 has_signal
                continue
            has_signal = True
            hits = _check_keyword_hits(answer, foundation["fail_evidence_keywords"])
            if hits:
                fail = True
                evidence.append(
                    f"{q_id} 命中失败关键词: {', '.join(hits)} | 原文片段: {answer[:80]}"
                )

        # 3) 状态判定
        if not has_signal:
            status = "uncertain"
            if not evidence:
                evidence.append("证据不足：VLM 答案缺失或表达不确定，且无对应量化指标")
        elif fail:
            status = "fail"
        else:
            status = "pass"
            if not evidence:
                evidence.append("通过：未触发任何失败关键词或量化阈值")

        should_block = (status == "fail" and foundation["priority"] == 0)

        results.append({
            "id": foundation["id"],
            "name": foundation["name"],
            "english": foundation["english"],
            "priority": foundation["priority"],
            "status": status,
            "evidence": evidence,
            "should_block_downstream": should_block,
            "drill": foundation["drill"] if status == "fail" else None,
            "drill_source": foundation["drill_source"] if status == "fail" else None,
            "downstream_cascade": foundation["downstream_cascade"] if status == "fail" else [],
            "explanation": foundation["explanation"],
        })

    return results


# =============================================================================
# 主函数：format_foundation_summary
# =============================================================================

def format_foundation_summary(statuses: List[Dict[str, Any]]) -> str:
    """
    把 check_foundations 的输出格式化为 Markdown 报告，作为分析报告第一段。
    """
    if not statuses:
        return "## 地基检查（Foundation Layer）\n\n（无数据）\n"

    status_icon = {"pass": "✅ PASS", "fail": "❌ FAIL", "uncertain": "⚠️ UNCERTAIN"}

    lines: List[str] = []
    lines.append("## 地基检查（Foundation Layer）")
    lines.append("")
    lines.append("| 项 | 状态 | 证据 | 下一步 |")
    lines.append("|---|---|---|---|")

    pass_count = 0
    p0_fail_count = 0
    p1_fail_count = 0

    for s in statuses:
        # 标题列：F1 架拍 Hold Up
        # 用 P0 / P1 标识优先级
        prio_tag = "P0" if s["priority"] == 0 else "P1"
        title = f"[{prio_tag}] {s['id'].split('_', 1)[0]} {s['name']} {s['english']}"

        icon = status_icon.get(s["status"], s["status"])

        # 证据列：取第一条证据，并截断
        if s["evidence"]:
            ev = s["evidence"][0]
            if len(ev) > 100:
                ev = ev[:97] + "..."
        else:
            ev = "—"
        # 防止 Markdown 表格被竖线/换行打断
        ev = ev.replace("|", "/").replace("\n", " ")

        # 下一步列
        if s["status"] == "fail" and s["drill"]:
            next_step = f"drill: {s['drill']}"
        elif s["status"] == "uncertain":
            next_step = "需更清晰素材或补充指标"
        else:
            next_step = "—"
        next_step = next_step.replace("|", "/").replace("\n", " ")

        lines.append(f"| {title} | {icon} | {ev} | {next_step} |")

        if s["status"] == "pass":
            pass_count += 1
        elif s["status"] == "fail":
            if s["priority"] == 0:
                p0_fail_count += 1
            else:
                p1_fail_count += 1

    total = len(statuses)
    p0_total = sum(1 for s in statuses if s["priority"] == 0)
    p1_total = total - p0_total

    lines.append("")
    lines.append(f"**地基状态**: {pass_count} / {total} 通过 "
                 f"(FTT {p0_total} 项 + 圣经 {p1_total} 项)")

    # 结论
    blocked = any(s["should_block_downstream"] for s in statuses)
    if blocked:
        failed_p0 = [s for s in statuses if s["priority"] == 0 and s["status"] == "fail"]
        failed_names = "、".join(s["name"] for s in failed_p0)
        lines.append("")
        lines.append(f"**结论**: ⛔ Priority 0 地基失败（{failed_names}），"
                     "**不分析上层**，先修地基。")
        # 列出 cascade
        lines.append("")
        lines.append("**失败级联（按因果顺序）**:")
        for s in failed_p0:
            cascade = " → ".join(s["downstream_cascade"]) if s["downstream_cascade"] else "—"
            lines.append(f"- {s['name']}: {cascade}")
    elif p1_fail_count > 0:
        failed_p1 = [s for s in statuses if s["priority"] == 1 and s["status"] == "fail"]
        failed_names = "、".join(s["name"] for s in failed_p1)
        lines.append("")
        lines.append(f"**结论**: ⚠️ FTT 4 项地基通过，但圣经层失败（{failed_names}），"
                     "可分析上层但优先修圣经层。")
    elif pass_count == total:
        lines.append("")
        lines.append("**结论**: ✅ 全部通过 → 进入上层分析。")
    else:
        # 有 uncertain 但无 fail
        lines.append("")
        lines.append("**结论**: ⚠️ 部分项证据不足，建议补充更清晰素材后再下结论；"
                     "已通过项可进入上层分析。")

    lines.append("")
    return "\n".join(lines)


# =============================================================================
# CLI 自检
# =============================================================================

if __name__ == "__main__":
    print(f"FOUNDATIONS 数量: {len(FOUNDATIONS)}")
    print("IDs:", [f["id"] for f in FOUNDATIONS])

    # 示例：fake VLM answers + metrics
    fake_vlm = {
        "Q1": "手臂和躯干基本同步，左肩有引导",
        "Q3": "大臂与胸侧间距在击球前明显变大，超过一个拳头",
        "Q9": "拍头先下坠后上升，呈明显 V 形",
        "Q15": "重心在挥拍前已经转到前脚",
        "Q16": "右脚 pivot 不明显",
        "Q23": "左肩有主动后推",
        "Q24": "拍头在引拍后下沉到腰部",
        "Q25": "看不清球衣褶皱",
        "Q26": "动作一气呵成，无明显停顿",
        "Q31": "半开放站姿",
        "Q5b": "肩髋同步",
    }
    fake_metrics = {
        "wrist_height_pattern": "V形scooping",
        "arm_body_sync_score": 0.55,
        "shoulder_hip_differential_deg": 35.0,
        "weight_transfer_timing": "early",
        "arm_chest_gap_change": "increase",
        "elbow_height_relative_chest": "below",
        "pivot_quality": "weak",
    }

    statuses = check_foundations(fake_vlm, fake_metrics)
    print("\n--- check_foundations 结果 ---")
    for s in statuses:
        print(f"{s['id']:25s} [{s['status']:9s}] block={s['should_block_downstream']}")
        for ev in s["evidence"]:
            print(f"    · {ev}")

    print("\n--- format_foundation_summary 输出 ---")
    print(format_foundation_summary(statuses))
