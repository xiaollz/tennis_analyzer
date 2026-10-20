#!/usr/bin/env python3
"""Compose trunk→branches→leaves report from frozen facts + trunk_branches map.

usage:
  python3 compose_trunk_report.py \\
    --branches ../ftt_tree/trunk_branches.json \\
    --facts    ../user_case/rear_20261019_facts_frozen.json \\
    [--diagnose diagnose_out.json] \\
    [--tree    ../ftt_tree/tree_v18.json] \\
    --out-json ../loop/reports/trunk_rear_20261019_<tag>.json \\
    --out-md   ../loop/reports/trunk_rear_20261019_<tag>.md

Pure rules; identical inputs → byte-identical JSON (sorted keys, no timestamps).
See TRUNK_REPORT_CONTRACT.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from common import CONFIG, load_json

BAN = CONFIG["ban_patterns"]
STATUS_HAS = "has_facts"
STATUS_SELF = "pending_self_report"
STATUS_UNTESTED = "pending_untested"
V5_PASSED = False  # until V5 gate passes; hard-coded per contract


def _frac(s: str) -> float:
    a, b = s.split("/")
    return int(a) / int(b)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scan_ban(text: str) -> list[str]:
    hits = []
    for label, pats in BAN.items():
        for p in pats:
            if re.search(p, text, flags=re.IGNORECASE):
                hits.append(label)
                break
    return hits


def qualifying_facts(facts_doc: dict) -> dict[str, dict]:
    """obs -> fact for facts that may drive has_facts."""
    used = {}
    for f in facts_doc.get("facts", []):
        if f.get("source") not in ("model", "pose", "manual"):
            continue
        if f.get("value") == "unclear":
            continue
        if _frac(f.get("agree", "0/1")) < 2 / 3:
            continue
        used[f["obs"]] = f
    return used


def index_by(items: list, key: str) -> dict:
    return {it[key]: it for it in items if key in it}


def match_leaf(leaf: dict, facts_doc: dict, qfacts: dict) -> dict:
    """Return leaf record with state."""
    out = {
        "id": leaf["id"],
        "role": leaf["role"],
        "zh": leaf.get("zh", ""),
        "obs": leaf.get("obs"),
        "camera": leaf.get("camera"),
    }
    if leaf.get("note"):
        out["note"] = leaf["note"]

    obs = leaf.get("obs")
    want = leaf.get("value")
    role = leaf["role"]

    # not_visible role forces not_visible unless we somehow have a clear fact
    # (contract: unclear facts → not_visible)
    unclear = None
    for f in facts_doc.get("facts", []):
        if f.get("obs") == obs and f.get("value") == "unclear":
            unclear = f
            break

    # candidate_facts (NEW): never counts as fact
    cand = None
    for c in facts_doc.get("candidate_facts", []):
        if obs and c.get("obs") == obs:
            cand = c
            break
        # also match by leaf id path when obs is NEW-* and value matches
        if obs and c.get("obs") == obs:
            cand = c

    # legacy
    legacy = None
    lk = leaf.get("legacy_key")
    if lk:
        for L in facts_doc.get("legacy_facts", []):
            if L.get("key") == lk:
                legacy = L
                break

    # self_report
    sr = None
    sk = leaf.get("self_report_key")
    if sk:
        for s in facts_doc.get("self_reports", []):
            if s.get("key") == sk:
                sr = s
                break

    # qualifying fact match (exact obs+value). NEW candidates never qualify.
    is_new = bool(obs and str(obs).startswith("NEW-"))
    fact = None
    if obs and not is_new and obs in qfacts:
        f = qfacts[obs]
        if want is None or f["value"] == want:
            fact = f

    if role == "not_visible" or unclear is not None:
        out["state"] = "not_visible"
        if unclear:
            out["agree"] = unclear.get("agree")
            out["value"] = "unclear"
        return out

    if fact is not None:
        out["state"] = "fact"
        out["value"] = fact["value"]
        out["agree"] = fact["agree"]
        out["source"] = fact["source"]
        return out

    if cand is not None:
        out["state"] = "candidate"
        out["value"] = cand.get("value")
        out["agree"] = cand.get("agree")
        return out

    if legacy is not None:
        out["state"] = "legacy"
        out["legacy_key"] = lk
        out["legacy_zh"] = legacy.get("zh")
        return out

    if sr is not None:
        out["state"] = "self_report"
        out["self_report_key"] = sk
        out["self_report_zh"] = sr.get("zh")
        return out

    out["state"] = "missing"
    return out


def branch_status(leaves_out: list, branch_def: dict) -> str:
    has_evidence_fact = any(
        L["role"] == "evidence" and L["state"] == "fact" for L in leaves_out
    )
    if has_evidence_fact:
        return STATUS_HAS
    has_sr = any(L["state"] == "self_report" for L in leaves_out)
    # also count self_report_key on evidence leaves even if matched above
    if has_sr:
        return STATUS_SELF
    # B2 may have self_report only via early_swing_start on L2_start_vs_bounce
    return STATUS_UNTESTED


def compose(branches_doc: dict, facts_doc: dict, diagnose_doc=None, tree_doc=None) -> dict:
    qfacts = qualifying_facts(facts_doc)
    trunk = branches_doc["trunk"]
    branches_out = []

    for b in branches_doc["branches"]:
        leaves_out = [match_leaf(leaf, facts_doc, qfacts) for leaf in b["leaves"]]
        status = branch_status(leaves_out, b)

        # excluded siblings from legacy
        excluded = []
        for es in b.get("excluded_siblings", []):
            lk = es.get("legacy_key")
            hit = any(L.get("key") == lk for L in facts_doc.get("legacy_facts", []))
            if hit:
                excluded.append({
                    "weakens": es.get("weakens"),
                    "legacy_key": lk,
                    "zh": es.get("zh", ""),
                })

        missing_facts = []
        if status.startswith("pending"):
            for m in b.get("missing_facts_if_pending", []):
                missing_facts.append({
                    "obs": m["obs"],
                    "camera": m.get("camera"),
                    "question": m.get("question"),
                    "values": m.get("values"),
                })

        conf = "medium" if status == STATUS_HAS else "low"
        if V5_PASSED is False and conf == "high":
            conf = "medium"

        # optional tree_nodes existence check (informational only)
        tree_note = None
        if tree_doc and b.get("tree_nodes"):
            ids = {n["id"] for n in tree_doc.get("nodes", [])}
            missing_nodes = [nid for nid in b["tree_nodes"] if nid not in ids]
            if missing_nodes:
                tree_note = f"tree_nodes missing: {missing_nodes}"

        entry = {
            "id": b["id"],
            "zh": b["zh"],
            "status": status,
            "confidence": conf,
            "validation_label": "待验证",
            "trunk_question": b.get("trunk_question"),
            "flt_refs": b.get("flt_refs", []),
            "sources": b.get("sources", []),
            "leaves": leaves_out,
            "excluded_siblings": excluded,
            "missing_facts": missing_facts,
        }
        if b.get("holdout_knowledge"):
            entry["holdout_knowledge"] = [
                {"id": h, "label": "holdout"} for h in b["holdout_knowledge"]
            ]
        if tree_note:
            entry["tree_note"] = tree_note
        branches_out.append(entry)

    next_thing = dict(branches_doc["next_one_thing"])
    # ban scan on prescription text
    ban_text = " ".join([
        next_thing.get("zh", ""),
        next_thing.get("breath", ""),
        next_thing.get("protocol", ""),
        " ".join(next_thing.get("verify", [])),
    ])
    hits = _scan_ban(ban_text)
    if hits:
        print(f"BAN terms in next_one_thing: {hits}", file=sys.stderr)
        sys.exit(1)

    retracted = [
        {"key": r["key"], "zh": r.get("zh", ""), "reason": r.get("reason", ""), "origin": r.get("origin", "")}
        for r in facts_doc.get("retracted_claims", [])
    ]

    report = {
        "fixture": facts_doc.get("clip_id", "").replace("_small", "") if facts_doc.get("clip_id", "").startswith("rear_") else "rear_20261019",
        "trunk": {"id": trunk["id"], "zh": trunk["zh"], "sources": trunk.get("sources", [])},
        "branches": branches_out,
        "branch_links": list(branches_doc.get("branch_links", [])),
        "retracted": retracted,
        "next_one_thing": {
            "breath": next_thing.get("breath", ""),
            "zh": next_thing.get("zh", ""),
            "source": next_thing.get("source", ""),
            "protocol": next_thing.get("protocol", ""),
            "verify": list(next_thing.get("verify", [])),
        },
        "diagnose_appendix": (diagnose_doc or {}).get("top", []) if diagnose_doc else [],
        "meta": {
            "v5_passed": V5_PASSED,
            "camera": facts_doc.get("camera"),
            "clip_id": facts_doc.get("clip_id"),
        },
    }
    # normalize fixture id
    if report["fixture"] == "rear_20261019" or "rear_20261019" in str(facts_doc.get("clip_id", "")):
        report["fixture"] = "rear_20261019"
    return report


def render_md(report: dict) -> str:
    lines = []
    lines.append(f"# 树干报告：{report['fixture']}（compose_trunk_report，待验证）")
    lines.append("")
    lines.append("> 由 `compose_trunk_report.py` 按 `trunk_branches.json` 重写。V5 未过关前全部「待验证」。")
    lines.append("> 禁令口令未写入任何处方。")
    lines.append("")
    lines.append("## 1. 树干")
    lines.append("")
    lines.append(f"**{report['trunk']['zh']}**")
    lines.append("")
    lines.append(f"id=`{report['trunk']['id']}` · sources={', '.join(report['trunk'].get('sources', []))}")
    lines.append("")

    status_zh = {
        STATUS_HAS: "有事实",
        STATUS_SELF: "待测 / 仅自述",
        STATUS_UNTESTED: "待测",
    }
    for i, b in enumerate(report["branches"], 1):
        lines.append(f"## {i + 1}. {b['zh']}（{status_zh.get(b['status'], b['status'])}）")
        lines.append("")
        lines.append(f"- status=`{b['status']}` · confidence=`{b['confidence']}` · {b['validation_label']}")
        if b.get("flt_refs"):
            lines.append(f"- FLT: {', '.join(b['flt_refs'])}")
        lines.append("")
        lines.append("| 叶子 | role | state | 证据 |")
        lines.append("|---|---|---|---|")
        for L in b["leaves"]:
            ev = ""
            if L["state"] == "fact":
                ev = f"{L.get('obs')}={L.get('value')} ({L.get('agree')})"
            elif L["state"] == "candidate":
                ev = f"候选 {L.get('obs')}={L.get('value')} ({L.get('agree')})"
            elif L["state"] == "legacy":
                ev = L.get("legacy_zh") or L.get("legacy_key")
            elif L["state"] == "self_report":
                ev = f"用户自述：{L.get('self_report_zh') or L.get('self_report_key')}"
            elif L["state"] == "not_visible":
                ev = f"{L.get('obs')}=unclear / 机位不可见"
            else:
                ev = "缺失"
            lines.append(f"| {L.get('zh','')} | {L['role']} | {L['state']} | {ev} |")
        lines.append("")
        if b.get("excluded_siblings"):
            lines.append("削弱的旁支：")
            for es in b["excluded_siblings"]:
                lines.append(f"- {es.get('weakens')}: {es.get('zh')}")
            lines.append("")
        if b.get("missing_facts"):
            lines.append("缺失的中立事实（待测）：")
            for m in b["missing_facts"]:
                cam = m.get("camera") or "?"
                lines.append(f"- {m['obs']}（机位 {cam}）" + (f"：{m['question']}" if m.get("question") else ""))
            lines.append("")
        if b.get("holdout_knowledge"):
            lines.append("holdout 知识（不进 tree evidence）：" + ", ".join(h["id"] for h in b["holdout_knowledge"]))
            lines.append("")

    lines.append("## 关系（推断，不是证据）")
    lines.append("")
    for link in report.get("branch_links", []):
        lines.append(f"- {link.get('from')} → {link.get('to')}（{link.get('kind')}）: {link.get('zh','')}")
    lines.append("")

    lines.append("## 撤回")
    lines.append("")
    for r in report.get("retracted", []):
        lines.append(f"- **{r.get('zh') or r['key']}**（`{r['key']}`）：{r.get('reason')}")
    lines.append("")

    lines.append("## 下一次只改一件事")
    lines.append("")
    not_ = report["next_one_thing"]
    lines.append(f"> {not_.get('breath','')}")
    lines.append("")
    lines.append(f"**外部目标：{not_.get('zh','')}**")
    lines.append("")
    lines.append(f"出处：{not_.get('source','')}")
    lines.append("")
    lines.append(f"协议：{not_.get('protocol','')}")
    lines.append("")
    if not_.get("verify"):
        lines.append("验证：")
        for v in not_["verify"]:
            lines.append(f"- {v}")
        lines.append("")

    if report.get("diagnose_appendix"):
        lines.append("## 附录：diagnose.py 扁平排名（不决定粗枝）")
        lines.append("")
        for t in report["diagnose_appendix"]:
            lines.append(f"- {t.get('node')} {t.get('name')} conf={t.get('confidence')} score={t.get('score')} evid={t.get('evidence')}")
        lines.append("")

    lines.append("---")
    lines.append(f"meta: clip={report['meta'].get('clip_id')} camera={report['meta'].get('camera')} v5_passed={report['meta'].get('v5_passed')}")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branches", required=True)
    ap.add_argument("--facts", required=True)
    ap.add_argument("--diagnose", default="")
    ap.add_argument("--tree", default="")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    a = ap.parse_args()

    branches_path = Path(a.branches)
    facts_path = Path(a.facts)
    branches_doc = load_json(branches_path)
    facts_doc = load_json(facts_path)
    diagnose_doc = load_json(a.diagnose) if a.diagnose else None
    tree_doc = load_json(a.tree) if a.tree else None

    report = compose(branches_doc, facts_doc, diagnose_doc, tree_doc)
    report["meta"]["branches_file_sha256"] = _sha256(branches_path)
    report["meta"]["facts_file_sha256"] = _sha256(facts_path)

    # ban scan whole MD body too (prescription sections)
    md = render_md(report)
    hits = _scan_ban(md)
    # allow holdout knowledge quotes that may contain English technical terms — ban is Chinese prescription
    # still fail hard on Chinese ban terms
    if hits:
        print(f"BAN terms in report MD: {hits}", file=sys.stderr)
        sys.exit(1)

    out_j = Path(a.out_json)
    out_m = Path(a.out_md)
    out_j.parent.mkdir(parents=True, exist_ok=True)
    out_j.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_m.write_text(md, encoding="utf-8")
    print(f"wrote {out_j}")
    print(f"wrote {out_m}")


if __name__ == "__main__":
    main()
