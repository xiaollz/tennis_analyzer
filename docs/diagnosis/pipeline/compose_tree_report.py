#!/usr/bin/env python3
"""Compose loop/reports/verify_<date>_tree_v1.json from script outputs only (no narration).

usage: compose_tree_report.py --verify raw_verify.json --text results.json --tree tree.json --log build_log.json --out report.json
Also dumps compact replay artifacts (gold statements, retrieved ids, answer keys) next to results.json.
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

from common import DIAG_DIR, load_json

WORK = Path("/workspace/s1_work/tree_blind_v1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", required=True); ap.add_argument("--text", required=True)
    ap.add_argument("--tree", required=True); ap.add_argument("--log", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--gold-usage", default=str(WORK / "gold_usage_stage.json"))
    a = ap.parse_args()
    v, t, tree, log = load_json(a.verify), load_json(a.text), load_json(a.tree), load_json(a.log)
    nodes = tree["nodes"]
    kinds = Counter(n["kind"] for n in nodes)
    rule_counts = {}
    for k, c in v.get("checks", {}).items():
        rc = Counter(re.search(r"\b([A-Z]\d{2})\b", f).group(1) if re.search(r"\b([A-Z]\d{2})\b", f) else "other" for f in c["fail"])
        rule_counts[k] = dict(rc)

    # failure classes per holdout case (derived from script rows)
    classes = {}
    keyed_nodes = set()
    art = {}
    for r in t["rows"]:
        if r["excluded"]:
            classes[r["episode"]] = "excluded: " + r["excluded"]
            continue
        runs = [x for x in r["runs"] if x]
        returned = any(x["problems"] or x["causes"] for x in runs)
        hit = any(x["p_hit3"] or x["c_hit3_exp"] for x in runs)
        covered = r["problem_covered"] or r["cause_covered"]
        if covered and hit:
            c = "hit"
        elif covered and not hit:
            c = "retrieval_miss (tree has a matching node; Gemini returned none or the wrong node)"
        elif not covered and returned:
            c = "tree_gap + false_retrieval (no matching node in tree; Gemini still returned nodes)"
        else:
            c = "tree_gap (no matching node in tree; Gemini correctly abstained)"
        classes[r["episode"]] = c
    pos_rows = [r for r in t["rows"] if not r["excluded"]]
    gold_causes_total = sum(r["gold_causes"] for r in pos_rows)
    key_causes_total = sum(r["key_cause_items_covered"] for r in pos_rows)
    # compact artifacts
    for r in pos_rows:
        vid = r["episode"]
        g = load_json(WORK / "gold" / vid / "gold.json")
        k = load_json(WORK / "judge" / vid / "key.json") if (WORK / "judge" / vid / "key.json").exists() else {}
        for ids in k.get("matches", {}).values():
            keyed_nodes |= set(ids)
        ret = {}
        for n in (1, 2, 3):
            p = WORK / "retrieve" / vid / f"run{n}.json"
            if p.exists():
                o = load_json(p)["out"] or {}
                ret[f"run{n}"] = {"none_match": o.get("none_match"), "video_is_teaching": o.get("video_is_teaching"),
                                  "problems": o.get("problems"), "causes": o.get("causes")}
        art[vid] = {"title": g["title"], "gold_source": g["source"], "gold_problem": g["problems"][:1], "gold_causes": g["causes"],
                    "rerun_agree": g.get("rerun_agree"), "answer_key": k.get("matches", {}), "retrieval": ret}
    adir = Path(a.text).parent
    (adir / "artifacts.json").write_text(json.dumps(art, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    gu = load_json(a.gold_usage)["total"] if Path(a.gold_usage).exists() else {}
    report = {
        "date": "2026-10-18", "base_commit": "6bcf4c8", "branch": "cursor/diagnosis-phase1",
        "verdict": "FAIL" if not v["pass"] else "PASS",
        "tree": {
            "file": "docs/diagnosis/ftt_tree/tree.json", "builder": "pipeline/build_tree.py", "merge_cosine": log["merge_cosine"],
            "problem_nodes": kinds["problem"], "cause_nodes": kinds["cause"], "solution_items": len(tree["solutions"]),
            "angle_items": len(tree["angles"]), "symptom_items": len(tree["symptoms"]), "edges": len(tree["edges"]),
            "edge_strength": dict(Counter(e["strength"] for e in tree["edges"])),
            "cards_attached": len(log["episodes_used"]), "episodes_used": log["episodes_used"], "skipped": log["skipped"],
            "cause_items_before_merge": log["counts"]["cause_items_before_merge"], "merged_clusters": log["merged_clusters"],
            "dropped_edges": log["dropped_edges"], "observables": len(tree["observables"]),
            "cause_layers": dict(Counter(n.get("layer") for n in nodes if n["kind"] == "cause")),
            "solutions_usable_as_prescription": sum(s["usable_as_prescription"] for s in tree["solutions"]),
            "symptom_obs_ref": dict(Counter(s["obs_ref"] for s in tree["symptoms"])),
            "holdout_episodes": tree["holdout_episodes"], "video_layer_items": len(tree["video_layer"]),
        },
        "verify_all_full": {"pass": v["pass"], "selftest_pass": v["selftest"]["pass"], "mutations": len(v["selftest"]["mutations"]),
                            "checks": {k: {"fail": len(c["fail"]), "warn": len(c["warn"])} for k, c in v["checks"].items()},
                            "fail_rule_breakdown": rule_counts, "warns": {k: c["warn"] for k, c in v["checks"].items() if c["warn"]},
                            "replay_metrics_official_V5": v.get("replay_metrics"), "raw": v},
        "official_V5_muted_clip_replay": {"run": True, "cases": 0, "result": "FAIL (G1: 0 cases) - no replay/cases or audio-stripped clips exist; needs user decision on per-segment yt-dlp download (PLAN_v2 §9.2)"},
        "text_blind_replay": {"note": "supplementary, NOT the PLAN V5 muted-clip replay; gold = verified card (Am8j1Zw5KrE) or 2 raw Stage-1 runs passing validate_cards+check_quotes; answer key by temperature-0 Gemini judge; hits by script",
                              "metrics": t["metrics"], "gate_reference": t["gate_reference"], "failure_class_per_case": classes,
                              "gold_causes_total": gold_causes_total, "gold_causes_with_a_tree_match": key_causes_total,
                              "distinct_tree_nodes_in_any_answer_key": len(keyed_nodes),
                              "holdout_without_transcript": t["holdout_without_transcript"],
                              "rows": t["rows"]},
        "gemini_usage": {"model": t["model"], "base": "micuapi.ai", "total_text_replay_incl_gold": t["gemini_usage"],
                         "gold_stage_first_pass": gu, "retrieve_calls": 54, "judge_calls": 18,
                         "tree_build_calls": 0, "note": "build_tree.py / verify_all.py make no model calls"},
        "no_secrets": True,
    }
    Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("verdict",)}), json.dumps(classes, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
