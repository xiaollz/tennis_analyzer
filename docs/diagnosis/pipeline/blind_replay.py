#!/usr/bin/env python3
"""V5 blind replay: can the tree find a KNOWN problem from a MUTED clip's neutral facts?

usage: blind_replay.py <tree.json> <cases_dir> <episodes_dir> [--gate G1|G2] [--holdout-only] [--json out.json]
A case = clip segment from an FTT episode where the coach labels what the demonstrator
does (label quote must exist and be verified), plus fact sheet(s) produced from the
audio-stripped clip with the neutral prompt. facts_path may be a glob (several runs).
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from common import CONFIG, Finding, fails, load_json, load_schema, validate_schema
from diagnose import diagnose


def run(tree, cases_dir, ep_dir, holdout_only=False, base=None):
    findings, rows = [], []
    schema = load_schema("replay_case.schema.json")
    facts_schema = load_schema("facts.schema.json")
    holdout = set(tree["holdout_episodes"])
    cards = {load_json(p)["video"]["id"]: load_json(p) for p in Path(ep_dir).glob("*.card.json")}
    base = Path(base or cases_dir)
    for p in sorted(Path(cases_dir).glob("*.json")):
        case = load_json(p)
        w = str(p)
        sf = validate_schema(case, schema)
        if fails(sf):
            findings += [Finding(f.level, w, f.msg) for f in sf]
            continue
        if holdout_only and case["episode"] not in holdout:
            continue
        # B01 label must be a verified coach quote about the demonstrator (or a correct demo)
        ep, qid = case["label_source"].split(":")
        card = cards.get(ep)
        q = next((x for x in card["quotes"] if x["id"] == qid), None) if card else None
        if not q or card["status"] not in ("stage1_verified", "stage2_verified"):
            findings.append(Finding("FAIL", w, f"B01 label {case['label_source']} missing or card not verified"))
            continue
        if case.get("label_kind") != "correct_demo" and not q["about"].startswith("demonstrator"):
            findings.append(Finding("FAIL", w, f"B01 label quote is about '{q['about']}', not the demonstrator"))
            continue
        runs = sorted((base / Path(case["facts_path"]).parent).glob(Path(case["facts_path"]).name)) if "*" in case["facts_path"] else [base / case["facts_path"]]
        results = []
        if not runs:
            findings.append(Finding("FAIL", w, f"B03 no facts files match {case['facts_path']}"))
            continue
        for fp in runs:
            facts = load_json(fp)
            sf = validate_schema(facts, facts_schema)
            if fails(sf):
                findings += [Finding("FAIL", str(fp), "B03 " + f.msg) for f in sf]
                continue
            # B02 leakage guard: clips must be muted, otherwise the coach's words give the answer
            if not facts.get("audio_stripped"):
                findings.append(Finding("FAIL", str(fp), "B02 facts not produced from an audio-stripped clip"))
                continue
            results.append(diagnose(tree, facts))
        if not results:
            continue
        r0 = results[0]
        top_ids = [c["node"] for c in r0["top"]]
        neg = case["expected"] == ["NONE"]
        top1s = Counter(r["top"][0]["node"] if r["top"] else "NONE" for r in results)
        rows.append({
            "case": case["case_id"], "episode": case["episode"], "negative": neg,
            "expected": case["expected"], "got": top_ids,
            "hit3": (not neg) and any(e in top_ids for e in case["expected"]),
            "hit1": (not neg) and bool(top_ids) and top_ids[0] in case["expected"],
            "neg_high_conf": neg and any(c["confidence"] == "high" for c in r0["top"]),
            "abstain": r0["insufficient"],
            "runs": len(results), "top1_stable": top1s.most_common(1)[0][1] / len(results),
            "holdout": case["episode"] in holdout,
        })
    return findings, rows


def metrics(rows):
    pos = [r for r in rows if not r["negative"]]
    neg = [r for r in rows if r["negative"]]
    multi = [r for r in rows if r["runs"] > 1]
    return {
        "cases": len(rows), "positives": len(pos), "negatives": len(neg),
        "episodes": len({r["episode"] for r in rows}),
        "recall_at_3": sum(r["hit3"] for r in pos) / len(pos) if pos else 0.0,
        "top1": sum(r["hit1"] for r in pos) / len(pos) if pos else 0.0,
        "abstain_rate": sum(r["abstain"] for r in pos) / len(pos) if pos else 0.0,
        "negative_high_conf": sum(r["neg_high_conf"] for r in neg) / len(neg) if neg else 0.0,
        "rerun_top1_stable": sum(r["top1_stable"] >= 2 / 3 for r in multi) / len(multi) if multi else None,
    }


def gate(m, name):
    g = CONFIG["replay_gate"][name]
    out = []
    chk = lambda ok, msg: out.append(Finding("PASS" if ok else "FAIL", f"gate {name}", msg))
    chk(m["cases"] >= g["min_cases"], f"cases {m['cases']} >= {g['min_cases']}")
    chk(m["episodes"] >= g["min_episodes"], f"episodes {m['episodes']} >= {g['min_episodes']}")
    chk(m["recall_at_3"] >= g["recall_at_3"], f"recall@3 {m['recall_at_3']:.2f} >= {g['recall_at_3']}")
    chk(m["negatives"] > 0 and m["negative_high_conf"] <= g["max_negative_high_conf"],
        f"negatives {m['negatives']} > 0 and high-conf FP {m['negative_high_conf']:.2f} <= {g['max_negative_high_conf']}")
    if "top1" in g:
        chk(m["top1"] >= g["top1"], f"top1 {m['top1']:.2f} >= {g['top1']}")
    if "rerun_top1_stable" in g:
        s = m["rerun_top1_stable"]
        chk(s is not None and s >= g["rerun_top1_stable"], f"rerun top1 stable {s} >= {g['rerun_top1_stable']}")
    return out


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("tree"); ap.add_argument("cases"); ap.add_argument("episodes")
    ap.add_argument("--gate"); ap.add_argument("--holdout-only", action="store_true"); ap.add_argument("--json")
    a = ap.parse_args(argv)
    findings, rows = run(load_json(a.tree), a.cases, a.episodes, a.holdout_only)
    m = metrics(rows)
    if a.gate:
        findings += gate(m, a.gate)
    for f in findings:
        print(f)
    for r in rows:
        print(f"  {r['case']:<24} exp={r['expected']} got={r['got']} hit3={r['hit3']} negFP={r['neg_high_conf']}")
    print("blind_replay:", json.dumps(m))
    if a.json:
        Path(a.json).write_text(json.dumps({"metrics": m, "rows": rows}, indent=1))
    return 1 if fails(findings) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
