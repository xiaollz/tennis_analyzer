#!/usr/bin/env python3
"""
Retrofit Foundation Layer 检查到已经分析过的视频。

用途：把 evaluation/foundation_layer.py 的检查应用到老的 report markdown 上，
看看如果 4/30 那两段视频用新的 Foundation-first 框架重新分析，结论会变成什么。

输入：
  reports/2026-04-30/正手分析报告_*.md

输出（写入新文件）：
  reports/2026-04-30/foundation_retrofit_<basename>.md

依据：
  从老报告里 parse VLM Q1-Q38 + 量化指标 → 喂给 check_foundations() →
  format_foundation_summary() 渲染。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.foundation_layer import (  # noqa: E402
    check_foundations,
    format_foundation_summary,
)


# ============================================================
# Parsers
# ============================================================

SWING_HEADER_RE = re.compile(r"^## 第 (\d+) 球\s*$", re.MULTILINE)
QA_BLOCK_RE = re.compile(
    r"```\n(Q1:.*?)\n```",
    re.DOTALL,
)
Q_LINE_RE = re.compile(r"^(Q\d+[a-z]?):\s*(?:\[回答\]\s*)?(.+)$", re.MULTILINE)

METRIC_HEADERS = {
    "肩部转开幅度": ("shoulder_turn_degree", "deg"),
    "下肢承载": ("lower_limb_load", "deg"),
    "击球时手臂结构": ("arm_structure_at_contact", "deg"),
    "前方击球点": ("contact_point_forward", "score"),
    "向前穿透": ("forward_penetration", "score"),
    "击球时头部稳定": ("head_stability_at_contact", "score"),
    "脊柱一致性": ("spine_consistency", "score"),
    "手臂-躯干同步性": ("arm_body_sync", "score"),
    "手腕高度模式": ("wrist_height_pattern", "string"),
    "挥拍轨迹": ("swing_trajectory", "string"),
}


def parse_qa_block(report_text: str, swing_idx: int) -> dict:
    """
    Find the QA block immediately following '## 第 N 球' header.
    Returns dict {Q1: answer, Q2: answer, ...}.
    """
    # Split report by swing header
    parts = re.split(r"^## 第 (\d+) 球\s*$", report_text, flags=re.MULTILINE)
    # parts: [pre_text, swing_idx_1, content_1, swing_idx_2, content_2, ...]
    target_idx = str(swing_idx)
    answers: dict[str, str] = {}
    for i in range(1, len(parts), 2):
        if parts[i].strip() == target_idx:
            content = parts[i + 1]
            # find QA block inside (between ``` ``` after "VLM 视觉观察原始数据")
            qa_match = QA_BLOCK_RE.search(content)
            if not qa_match:
                return {}
            qa_text = qa_match.group(1)
            for m in Q_LINE_RE.finditer(qa_text):
                qid = m.group(1)
                ans = m.group(2).strip()
                # Strip leading [回答] if still there
                ans = re.sub(r"^\[回答\]\s*", "", ans)
                answers[qid] = ans
            return answers
    return {}


def parse_metrics_block(report_text: str, swing_idx: int) -> dict:
    """
    Pull metrics from the '## 量化辅助参考' tail section, scoped to swing N.
    """
    metrics_section_match = re.search(
        r"## 量化辅助参考(.+?)(?:\n---|\Z)", report_text, re.DOTALL
    )
    if not metrics_section_match:
        return {}
    section = metrics_section_match.group(1)

    # find sub-section per swing
    pattern = rf"\*\*第 {swing_idx} 球\*\*\n(.+?)(?=\n\*\*第|\Z)"
    sub = re.search(pattern, section, re.DOTALL)
    if not sub:
        return {}
    block = sub.group(1)

    metrics: dict = {}
    for cn_name, (en_key, kind) in METRIC_HEADERS.items():
        m = re.search(
            rf"^- {re.escape(cn_name)}:\s*(.+?)(?:\s*\([^)]*\))?\s*$",
            block,
            re.MULTILINE,
        )
        if not m:
            continue
        raw = m.group(1).strip()
        if kind == "string":
            metrics[en_key] = raw
        else:
            num_match = re.match(r"^(-?[\d.]+)", raw)
            if num_match:
                try:
                    metrics[en_key] = float(num_match.group(1))
                except ValueError:
                    metrics[en_key] = raw
    return metrics


def count_swings(report_text: str) -> int:
    return len(SWING_HEADER_RE.findall(report_text))


# ============================================================
# Retrofit
# ============================================================

def retrofit_report(report_path: Path) -> str:
    """Generate a retrofit foundation-first analysis from an old report."""
    text = report_path.read_text(encoding="utf-8")
    n_swings = count_swings(text)

    out = []
    out.append(f"# Foundation-Layer Retrofit · {report_path.stem}\n")
    out.append("> 用 evaluation/foundation_layer.py 重新检查这段视频的每一拍。")
    out.append("> 数据源：原报告里已有的 VLM Q1-Q38 答案 + 量化指标。")
    out.append("> 不重跑 VLM，纯回溯检查。\n\n")
    out.append("---\n\n")

    aggregate = {f["id"]: {"pass": 0, "fail": 0, "uncertain": 0} for f in __import__(
        "evaluation.foundation_layer", fromlist=["FOUNDATIONS"]
    ).FOUNDATIONS}

    p0_blocked_swings: list[int] = []
    swing_summaries = []

    for swing_idx in range(1, n_swings + 1):
        qa = parse_qa_block(text, swing_idx)
        metrics = parse_metrics_block(text, swing_idx)

        if not qa:
            swing_summaries.append(f"## 第 {swing_idx} 球\n\n⚠️ 无 VLM 数据（API 失败或被截断），foundation 检查跳过。\n\n")
            continue

        statuses = check_foundations(qa, metrics)
        for s in statuses:
            aggregate[s["id"]][s["status"]] += 1

        p0_failures = [s for s in statuses if s.get("should_block_downstream")]
        if p0_failures:
            p0_blocked_swings.append(swing_idx)

        swing_summaries.append(f"## 第 {swing_idx} 球\n\n")
        swing_summaries.append(format_foundation_summary(statuses))
        swing_summaries.append("\n\n---\n\n")

    # Aggregate header
    out.append("## 🏛️ 跨拍聚合 (Aggregate Across Swings)\n\n")
    out.append(f"**总球数**: {n_swings}\n\n")
    out.append("| Foundation | Pass | Fail | Uncertain | 通过率 |\n")
    out.append("|---|---|---|---|---|\n")
    for f in __import__("evaluation.foundation_layer", fromlist=["FOUNDATIONS"]).FOUNDATIONS:
        a = aggregate[f["id"]]
        rate = (a["pass"] / n_swings * 100) if n_swings else 0
        out.append(
            f"| **{f['id']}** {f['name']} (P{f['priority']}) | "
            f"{a['pass']} | {a['fail']} | {a['uncertain']} | {rate:.0f}% |\n"
        )
    out.append("\n")

    if p0_blocked_swings:
        out.append(
            f"**⛔ 结论**: {len(p0_blocked_swings)} / {n_swings} 球的 P0 地基失败"
            f"（球 {p0_blocked_swings}）。**地基没通过的球不进行上层分析**——"
            f"先把地基修好，unit turn 度数 / 胸推肘 / 肩胛槽等讨论先放一边。\n\n"
        )
    else:
        out.append("**✅ 结论**: 所有球地基通过，可进入上层分析。\n\n")

    out.append("---\n\n")
    out.extend(swing_summaries)

    return "".join(out)


def main():
    report_dir = PROJECT_ROOT / "reports" / "2026-04-30"
    if not report_dir.exists():
        print(f"reports dir not found: {report_dir}", file=sys.stderr)
        sys.exit(1)

    targets = sorted(report_dir.glob("正手分析报告_*.md"))
    if not targets:
        print(f"no reports in {report_dir}", file=sys.stderr)
        sys.exit(1)

    for report_path in targets:
        retrofit = retrofit_report(report_path)
        out_path = report_dir / f"foundation_retrofit_{report_path.stem.replace('正手分析报告_', '')}.md"
        out_path.write_text(retrofit, encoding="utf-8")
        print(f"✓ {out_path.relative_to(PROJECT_ROOT)} ({len(retrofit)} chars)")


if __name__ == "__main__":
    main()
