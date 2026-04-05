"""诊断引擎 — 连接 VLM 视觉输出和量化数据，产出最终诊断。

VLM 只负责看图说话（视觉观察）。
本模块拿到 VLM 的观察结果 + 量化指标，做交叉验证和根因推理。

输入：
  - vlm_result: VLM 返回的半结构化分析（根因、证据、因果解释）
  - metrics: 量化指标（同步性、scooping 深度、穿透值等）

输出：
  - 验证后的诊断（确认/修正/补充 VLM 的判断）
  - 量化佐证文本（融入因果解释的数字引用）
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any


# 量化指标的阈值和含义
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

# VLM 判断 vs 量化数据的矛盾检测规则
_CONTRADICTION_RULES = [
    {
        "name": "手臂独立但VLM没提到",
        "check": lambda vlm, m: (
            m.get("arm_torso_synchrony") is not None
            and m["arm_torso_synchrony"] < 0.4
            and "手臂" not in vlm.get("root_cause", "")
            and "连接" not in vlm.get("root_cause", "")
            and "脱节" not in vlm.get("root_cause", "")
        ),
        "correction": "量化数据显示手臂-躯干同步性极低（{sync:.2f}），手臂在独立于身体运动。这可能是 VLM 未充分识别的深层根因。",
    },
    {
        "name": "VLM说捞球但算法没检测到",
        "check": lambda vlm, m: (
            ("捞球" in vlm.get("root_cause", "") or "scooping" in vlm.get("root_cause", "").lower())
            and m.get("scooping_detected") is False
        ),
        "correction": "VLM 观察到捞球迹象，但量化算法未检测到显著的 V 形下坠。可能是轻微的捞球倾向，或者拍摄角度导致视觉放大了实际程度。",
    },
    {
        "name": "VLM说转体好但数据显示不足",
        "check": lambda vlm, m: (
            m.get("shoulder_rotation") is not None
            and m["shoulder_rotation"] < 30
            and "转体" not in vlm.get("root_cause", "")
            and "Unit Turn" not in vlm.get("root_cause", "")
        ),
        "correction": "量化数据显示肩部转开只有 {rotation:.0f}°（正常应 >60°），转体严重不足。这是 VLM 可能遗漏的一个上游问题。",
    },
]


def diagnose(vlm_result: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
    """主入口：VLM 输出 + 量化数据 → 最终诊断。

    返回增强后的 vlm_result，新增字段：
    - quant_evidence: 量化佐证文本（可融入因果解释）
    - contradictions: 发现的矛盾列表
    - quant_summary: 关键量化数据的一句话总结
    - diagnosis_confidence: 综合置信度（VLM + 量化一致 = 高，有矛盾 = 低）
    """
    if not vlm_result or not metrics:
        return vlm_result or {}

    result = dict(vlm_result)

    # 1. 生成量化佐证
    evidence_parts = _build_quant_evidence(metrics)
    result["quant_evidence"] = " ".join(evidence_parts) if evidence_parts else ""

    # 2. 检测 VLM 判断和量化数据的矛盾
    contradictions = _detect_contradictions(vlm_result, metrics)
    result["contradictions"] = contradictions

    # 3. 生成关键量化总结
    result["quant_summary"] = _build_quant_summary(metrics)

    # 4. 评估诊断置信度
    if contradictions:
        result["diagnosis_confidence"] = "需要验证"
        # 如果有严重矛盾，补充到因果解释
        correction_texts = [c["correction"] for c in contradictions]
        existing_causal = result.get("causal_explanation", "")
        if existing_causal:
            result["causal_explanation"] = existing_causal + "\n\n" + "\n".join(correction_texts)
    else:
        result["diagnosis_confidence"] = "高"

    # 5. 如果 VLM 没给出根因但量化数据有明确信号，自动补充
    if not result.get("root_cause"):
        auto_root = _infer_root_cause_from_metrics(metrics)
        if auto_root:
            result["root_cause"] = auto_root
            result["root_cause_source"] = "量化推断"

    return result


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


def _detect_contradictions(vlm_result: Dict, metrics: Dict) -> List[Dict]:
    """检测 VLM 判断和量化数据之间的矛盾。"""
    contradictions = []
    for rule in _CONTRADICTION_RULES:
        try:
            if rule["check"](vlm_result, metrics):
                correction = rule["correction"].format(
                    sync=metrics.get("arm_torso_synchrony", 0),
                    rotation=metrics.get("shoulder_rotation", 0),
                )
                contradictions.append({
                    "name": rule["name"],
                    "correction": correction,
                })
        except Exception:
            pass
    return contradictions


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


def _infer_root_cause_from_metrics(metrics: Dict) -> Optional[str]:
    """当 VLM 没给出根因时，尝试从量化数据推断。"""
    sync = metrics.get("arm_torso_synchrony")
    rot = metrics.get("shoulder_rotation")
    scoop = metrics.get("scooping_depth")

    if sync is not None and sync < 0.2:
        return "手臂完全脱离身体旋转系统（量化数据推断）"
    if rot is not None and rot < 20:
        return "Unit Turn 严重不足，转体角度极小（量化数据推断）"
    if scoop is not None and scoop > 0.5:
        return "严重的 V 形捞球（量化数据推断）"
    return None
