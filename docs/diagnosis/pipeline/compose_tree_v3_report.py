#!/usr/bin/env python3
"""Compose loop/reports/verify_<date>_tree_v3.json (script outputs only) with before (v2) / after (v3) comparison.

usage: compose_tree_v3_report.py --verify raw_verify.json --out report.json [--commit-pre HASH]
"""
import argparse
import glob
import json
import re
from collections import Counter
from pathlib import Path

from common import DIAG_DIR, load_json
from blind_replay_text_v2 import menu_variants

W3 = Path("/workspace/s1_work/tree_v3")
KEYS = ("promptTokenCount", "candidatesTokenCount", "thoughtsTokenCount", "totalTokenCount", "calls")


def usage_of(paths):
    u = Counter()
    for p in paths:
        r = load_json(p)
        u.update({k: v for k, v in (r.get("usage") or {}).items() if k in KEYS})
    return {k: u.get(k, 0) for k in KEYS}


def tree_stats(t, bind, t1_nodes):
    nodes = t["nodes"]
    cnt = lambda k, f: sum(n["kind"] == k and f(n) for n in nodes)
    src = {}
    nm = t["build"]["mechanism_merge"]["node_map_v1_to_v2"]
    for it in bind["items"].values():
        if it["kind"] in ("problem", "cause"):
            src.setdefault(nm.get(it["node"], it["node"]), []).append(it)
    gap = Counter()
    for n in nodes:
        if n["must_fix"] and not n["signature"]["required"]:
            gap["has_NEW_OBS_candidate" if any(i["new_obs"] for i in src[n["id"]]) else "unobservable_per_model"] += 1
    return {"nodes_total": len(nodes), "problem_with_required": cnt("problem", lambda n: bool(n["signature"]["required"])),
            "cause_with_required": cnt("cause", lambda n: bool(n["signature"]["required"])),
            "cause_total": cnt("cause", lambda n: True), "problem_total": cnt("problem", lambda n: True),
            "nodes_with_required": sum(bool(n["signature"]["required"]) for n in nodes),
            "nodes_with_any_signature": sum(bool(n["signature"]["required"] or n["signature"]["supporting"]) for n in nodes),
            "must_fix_without_required": dict(gap) | {"total": sum(gap.values())},
            "binding_items_with_required": dict(Counter(i["kind"] for i in bind["items"].values() if i["required"])),
            "symptom_obs_ref": dict(Counter(s["obs_ref"] for s in t["symptoms"])),
            "new_obs_candidates": len(bind["new_obs_candidates"]), "new_obs_clusters": len(bind["new_obs_clusters"])}


def rule_counts(v, check):
    return dict(Counter(re.search(r"\b([A-Z]\d{2})\b", f).group(1) for f in v["checks"][check]["fail"] if re.search(r"\b([A-Z]\d{2})\b", f)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--commit-pre", default="")
    a = ap.parse_args()
    ft = DIAG_DIR / "ftt_tree"
    t1, t2, t3 = load_json(ft / "tree.json"), load_json(ft / "tree_v2.json"), load_json(ft / "tree_v3.json")
    b2, b3 = load_json(ft / "obs_binding.json"), load_json(ft / "obs_binding_v3.json")
    m3 = load_json(ft / "mech_merge_v3.json")
    prom = load_json(ft / "new_obs_promotion.json")
    v3 = load_json(a.verify)
    r2 = load_json(DIAG_DIR / "loop" / "reports" / "verify_2026-10-18_tree_v2.json")
    v2raw = r2["verify_all_full"]["raw"]
    s2, s3 = tree_stats(t2, b2, None), tree_stats(t3, b3, None)

    # T08 pairs after
    t08 = [f.split("T08 ")[1] for f in v3["checks"]["check_tree"]["fail"] if "T08" in f]
    # which OBS conditions drive the T08 pairs
    nodes3 = {n["id"]: n for n in t3["nodes"]}
    pair_sig = Counter()
    for f in t08:
        x = f.split(" and ")[0]
        pair_sig[json.dumps([(c["obs"], c["value"]) for c in nodes3[x]["signature"]["required"]])] += 1
    # OBS usage
    nr, ns, sr = Counter(), Counter(), Counter()
    for n in t3["nodes"]:
        for c in n["signature"]["required"]:
            nr[c["obs"]] += 1
        for o in {x["obs"] for x in n["signature"]["supporting"]}:
            ns[o] += 1
    for it in b3["items"].values():
        if it["kind"] == "symptom":
            for c in it["required"]:
                sr[c["obs"]] += 1
    usage_tbl = {o["id"]: {"layer": o["layer"], "camera": o["camera"], "nodes_required": nr[o["id"]], "nodes_supporting": ns[o["id"]], "symptoms_required": sr[o["id"]]} for o in t3["observables"]}
    # NEW-OBS list (v3)
    cand = {c["cid"]: c for c in b3["new_obs_candidates"]}
    clusters = []
    for c in b3["new_obs_clusters"]:
        rep = cand[c["rep"]]
        its = sorted({cand[m]["item"] for m in c["members"]})
        clusters.append({"id": c["id"], "description": rep["description"], "values": rep["values"], "layer": rep["layer"], "camera": rep["camera"],
                         "candidates": len(c["members"]), "episodes": sorted({b3["items"][i]["episode"] for i in its}), "from_items": its})
    clusters.sort(key=lambda c: (-len(c["episodes"]), -len(c["from_items"]), c["id"]))
    # replay: menu identity v2 vs v3 (same node ids + names + also-worded lines => the v2 retrieval/judge inputs are byte-identical)
    mm2 = load_json(ft / "mech_merge.json")
    same_menu = menu_variants(t2, mm2["groups"]) == menu_variants(t3, m3["groups"])
    rm = load_json(DIAG_DIR / "replay" / "text_blind" / "results_v2_indep_main.json")
    u = {
        "promote_new_obs family grouping (2 calls)": usage_of(glob.glob(str(W3 / "families" / "run*.json"))),
        "obs_bind round1 v3 (44 calls)": usage_of(glob.glob(str(W3 / "bind" / "*" / "run*.json"))),
        "obs_gapfill v3 (2 calls)": usage_of(glob.glob(str(W3 / "gapfill" / "run*.json"))),
        "newobs_cluster v3 (2 calls)": usage_of(glob.glob(str(W3 / "newobs" / "run*.json"))),
        "mech_merge v3 (6 calls)": usage_of(glob.glob(str(W3 / "merge" / "*" / "run*.json"))),
    }
    tot = Counter()
    for x in u.values():
        tot.update(x)
    u["TOTAL_this_round"] = dict(tot)
    report = {
        "date": "2026-10-18", "base_commit": "1dd015f", "pre_commit": a.commit_pre, "branch": "cursor/diagnosis-phase1",
        "verdict": "PASS" if v3["pass"] else "FAIL",
        "tree_v3": {"file": "docs/diagnosis/ftt_tree/tree_v3.json (tree.json = v1 baseline and tree_v2.json = previous version, both untouched)",
                    "builder": "pipeline/build_tree_v2.py with --binding obs_binding_v3.json --merge mech_merge_v3.json",
                    "observables": len(t3["observables"]), "problem_nodes": s3["problem_total"], "cause_nodes_v1": 97, "cause_nodes_v3": s3["cause_total"],
                    "edges": len(t3["edges"]), "core_edges": sum(e["strength"] == "core" for e in t3["edges"])},
        "new_observables": {"rule": prom["rule"], "families_total": prom["families_total"], "families_merged_from_several_clusters": prom["families_multi_cluster"],
                            "eligible": prom["eligible"], "promoted": [{k: p[k] for k in ("obs", "description", "values", "layer", "camera", "episodes", "family", "file")} for p in prom["promoted"]],
                            "eligible_not_promoted": prom["not_promoted_eligible"], "ineligible_summary": prom["ineligible_summary"],
                            "note": "question + values are copied verbatim from the family's representative candidate (model-written); no discriminator was invented; "
                                    "OBS-25 (shoulders vs hips first) overlaps in meaning with OBS-18 (which part starts first) and may collide with it - flagged for review"},
        "before_after": {
            "v2_before (18 observables)": s2, "v3_after (25 observables)": s3,
            "T05_fail": {"before": v2raw["checks"]["check_tree"]["fail"].__len__() - sum("T08" in f for f in v2raw["checks"]["check_tree"]["fail"]), "after": sum("T05" in f for f in v3["checks"]["check_tree"]["fail"])},
            "T08_fail": {"before": sum("T08" in f for f in v2raw["checks"]["check_tree"]["fail"]), "after": len(t08)},
            "check_tree_fail_total": {"before": len(v2raw["checks"]["check_tree"]["fail"]), "after": len(v3["checks"]["check_tree"]["fail"])},
            "T09_warn": {"before": sum("T09" in f for f in v2raw["checks"]["check_tree"]["warn"]), "after": sum("T09" in f for f in v3["checks"]["check_tree"]["warn"])},
            "T08_pairs_after": t08, "T08_pairs_by_shared_required_signature": dict(pair_sig),
            "t08_note": "more nodes now carry a required signature, so more pairs share an identical one; no differential was added (no model-invented discriminators, rule unchanged)"},
        "obs_usage_v3": usage_tbl, "obs_never_used": sorted(i for i, x in usage_tbl.items() if not (x["nodes_required"] or x["nodes_supporting"])),
        "new_obs_cluster_list_v3": clusters,
        "mechanism_merge_v3": {"groups": len(m3["groups"]), "group_sizes": [len(g["members"]) for g in m3["groups"]],
                               "cause_nodes": f"{m3['cause_nodes_before']} -> {m3['cause_nodes_after']}",
                               "same_groups_as_v2_run": [sorted(x["node"] for x in g["members"]) for g in m3["groups"]] == [sorted(x["node"] for x in g["members"]) for g in mm2["groups"]],
                               "review_file": "docs/diagnosis/ftt_tree/mech_merge_review.md (groups identical to the v2 run, so unchanged)"},
        "verify_all_full": {"pass": v3["pass"], "selftest_pass": v3["selftest"]["pass"], "mutations": len(v3["selftest"]["mutations"]),
                            "checks": {k: {"fail": len(c["fail"]), "warn": len(c["warn"])} for k, c in v3["checks"].items()},
                            "fail_rule_breakdown": {k: rule_counts(v3, k) for k in v3["checks"]},
                            "blind_replay_V5_muted_clips": "FAIL (G1: 0 cases) - no clips downloaded, no muted-clip test (user decision)",
                            "t05_note": "must_fix never changed; check_tree rules unchanged", "raw": v3},
        "text_blind_replay": {"rerun": False,
                              "why": "tree_v3 has the same node ids, node texts, merge groups and 'also worded' lines as tree_v2, so the retrieval and judge inputs are byte-identical to the v2 run; a rerun would only add sampling noise on the holdout",
                              "library_menu_identical_to_v2": same_menu,
                              "latest_result_still_valid": {"file": "replay/text_blind/results_v2_indep_main.json", "case_hit3": rm["metrics"]["case_hit3 (problem OR cause-with-upstream)"], "coverage_any": rm["metrics"]["coverage_any"]}},
        "gemini_usage": {"model": b3["model"], "base": "micuapi.ai", "by_stage": u, "note": "this round only; earlier rounds in verify_2026-10-18_tree_v2.json. Key read from env only."},
        "remaining_gaps": {
            "problem_nodes_without_required": f"{s3['problem_total'] - s3['problem_with_required']}/{s3['problem_total']}",
            "must_fix_nodes_without_required": s3["must_fix_without_required"],
            "T05_after": sum("T05" in f for f in v3["checks"]["check_tree"]["fail"]), "T08_after": len(t08),
            "new_obs_clusters_still_open": {"total": len(clusters), "spanning_>=2_episodes": sum(len(c["episodes"]) >= 2 for c in clusters)},
            "obs_never_used_by_any_node": sorted(i for i, x in usage_tbl.items() if not (x["nodes_required"] or x["nodes_supporting"])),
            "holdout_text_replay_coverage": "3/18 unchanged (tree content gap, not signature gap)",
            "V5_muted_clip_replay": "0 cases; needs per-segment download approval (PLAN_v2 sec 9.2)"},
        "no_secrets": True,
    }
    Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("verdict",)}), json.dumps(report["before_after"]["T05_fail"]), json.dumps(report["before_after"]["T08_fail"]), same_menu, report["mechanism_merge_v3"]["same_groups_as_v2_run"], u["TOTAL_this_round"])


if __name__ == "__main__":
    main()
