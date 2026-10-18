#!/usr/bin/env python3
"""Compose loop/reports/verify_<date>_tree_v4.json (script outputs only) with before (v3) / after (v4) comparison.

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

W3 = Path("/workspace/s1_work/tree_v4")
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
    t1, t2, t3 = load_json(ft / "tree.json"), load_json(ft / "tree_v3.json"), load_json(ft / "tree_v4.json")
    b2, b3 = load_json(ft / "obs_binding_v3.json"), load_json(ft / "obs_binding_v4.json")
    m3 = load_json(ft / "mech_merge_v4.json")
    prom = load_json(ft / "new_obs_promotion.json")
    v3 = load_json(a.verify)
    r2 = load_json(DIAG_DIR / "loop" / "reports" / "verify_2026-10-18_tree_v3.json")
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
    # replay: menu identity vs the tree_v2 run that produced the holdout result
    mm2 = load_json(ft / "mech_merge_v3.json")
    mm_v2 = load_json(ft / "mech_merge.json"); t_v2 = load_json(ft / "tree_v2.json")
    same_menu = menu_variants(t_v2, mm_v2["groups"]) == menu_variants(t3, m3["groups"])
    rm = load_json(DIAG_DIR / "replay" / "text_blind" / "results_v2_indep_main.json")
    u = {
        "obs_bind round1 v4 (44 calls)": usage_of(glob.glob(str(W3 / "bind" / "*" / "run*.json"))),
        "obs_gapfill v4": usage_of(glob.glob(str(W3 / "gapfill" / "run*.json"))),
        "newobs_cluster v4": usage_of(glob.glob(str(W3 / "newobs" / "run*.json"))),
        "mech_merge v4 (6 calls)": usage_of(glob.glob(str(W3 / "merge" / "*" / "run*.json"))),
    }
    tot = Counter()
    for x in u.values():
        tot.update(x)
    u["TOTAL_this_round"] = dict(tot)
    # T08 groups of the human review (t08_review_obs11_22_24.md): before (v3) vs after (v4)
    t08_before = [f.split("T08 ")[1] for f in v2raw["checks"]["check_tree"]["fail"] if "T08" in f]
    key = lambda f: tuple(sorted(f.split(" have identical")[0].split(" and ")))
    sb, sa = {key(f) for f in t08_before}, {key(f) for f in t08}
    review = {
        "OBS-11 group (5-Ia7c4XzfU:C02 / FxDmVi3EFnE:C01 / C03)": [list(k) for k in sorted(sb) if any("5_IA7C4XZFU_C02" in x or "FXDMVI3EFNE_C0" in x for x in k) and k in sb],
        "OBS-24 group (lwS7:C02 / NEpD:C01 / pQ79:C01)": [list(k) for k in sorted(sb) if any(x in ("N-C_LWS7XXQRRDW_C02", "N-C_NEPD7FIM7HI_C01", "N-C_PQ793MBQE50_C01") for x in k)],
        "OBS-22 group (lwS7:C03 / UJPU:C01 / WFIR:C02)": [list(k) for k in sorted(sb) if any(x in ("N-C_LWS7XXQRRDW_C03", "N-C_UJPUDUX0JIA_C01", "N-C_WFIRPMUTZRO_C02") for x in k)],
    }
    review_after = {g: [list(k) for k in sorted(sa) if any(x in set(sum(review[g], [])) for x in k)] for g in review}
    nodes_v4 = {n["id"]: n for n in t3["nodes"]}
    approved = prom["user_approved_single_episode_exceptions"]
    demoted = sorted(k for k, it in b3["items"].items() if it.get("demoted_conflicting_required"))
    report = {
        "date": "2026-10-18", "base_commit": "1053946", "pre_commit": a.commit_pre, "branch": "cursor/diagnosis-phase1",
        "verdict": "PASS" if v3["pass"] else "FAIL",
        "tree_v4": {"file": "docs/diagnosis/ftt_tree/tree_v4.json (tree.json = v1 baseline, tree_v2.json / tree_v3.json = previous versions, all untouched)",
                    "builder": "pipeline/build_tree_v2.py with --binding obs_binding_v4.json --merge mech_merge_v4.json",
                    "observables": len(t3["observables"]), "problem_nodes": s3["problem_total"], "cause_nodes_v1": 97, "cause_nodes_v4": s3["cause_total"],
                    "edges": len(t3["edges"]), "core_edges": sum(e["strength"] == "core" for e in t3["edges"])},
        "new_observables": {"rule": "NOT the cross-episode promotion rule: each is a single-episode candidate the user approved by hand (approved_by_user 2026-10-18); the >=2 episode rule itself is unchanged and recorded as exception in ftt_tree/new_obs_promotion.json -> user_approved_single_episode_exceptions",
                            "promoted_by_hand": [{k: p[k] for k in ("obs", "description", "values", "layer", "camera", "episodes", "source", "candidate", "approved_by_user", "file")} for p in approved],
                            "note": "question + values copied verbatim from model-written candidates (no discriminator invented; no prescription). OBS-28 is racket-face tilt at contact (Qszz0N4fRb4 candidate); "
                                    "the wrist-break candidates (NEW-OBS-28/39 of the v2 run) were NOT used because they restate OBS-24. OBS-27's second value is named wrist_flexes_or_chops in the source candidate."},
        "before_after": {
            "v3_before (25 observables)": s2, "v4_after (29 observables)": s3,
            "T05_fail": {"before": sum("T05" in f for f in v2raw["checks"]["check_tree"]["fail"]), "after": sum("T05" in f for f in v3["checks"]["check_tree"]["fail"])},
            "T08_fail": {"before": len(t08_before), "after": len(t08)},
            "check_tree_fail_total": {"before": len(v2raw["checks"]["check_tree"]["fail"]), "after": len(v3["checks"]["check_tree"]["fail"])},
            "T09_warn": {"before": sum("T09" in f for f in v2raw["checks"]["check_tree"]["warn"]), "after": sum("T09" in f for f in v3["checks"]["check_tree"]["warn"])},
            "T08_pairs_after": t08, "T08_pairs_resolved": [list(k) for k in sorted(sb - sa)], "T08_pairs_new": [list(k) for k in sorted(sa - sb)],
            "T08_review_groups_before": review, "T08_review_groups_after": review_after,
            "t08_note": "no differential added, rules unchanged; remaining/new pairs are listed as they are"},
        "obs_usage_v4": usage_tbl, "obs_never_used": sorted(i for i, x in usage_tbl.items() if not (x["nodes_required"] or x["nodes_supporting"])),
        "new_obs_cluster_list_v4": clusters,
        "aggregation_fix": {"what": "obs_bind.agg: if both runs require two different values of the SAME observable (model meant 'any of', e.g. OBS-16 long/wide/net), that cannot all hold as 'required' (check_obs_binding O3 FAILed it) -> those conditions are demoted to supporting. check_obs_binding O10 re-derives exactly this. Stricter, never adds a required; no rule relaxed.",
                            "items_demoted": demoted,
                            "side_effect": "tree_v3 artifacts no longer pass check_obs_binding O9 against the extended 29-observable catalog (history only; selftest now runs on v4 artifacts)"},
        "mechanism_merge_v4": {"groups": len(m3["groups"]), "group_sizes": [len(g["members"]) for g in m3["groups"]],
                               "cause_nodes": f"{m3['cause_nodes_before']} -> {m3['cause_nodes_after']}",
                               "same_groups_as_v3_run": [sorted(x["node"] for x in g["members"]) for g in m3["groups"]] == [sorted(x["node"] for x in g["members"]) for g in mm2["groups"]],
                               "differences_vs_v3_run": "model sampling: group 'N-C_PQ793MBQE50_C03 + N-C_DDYKUNZTDYU_C04' now represented by DDYK:C04; new group N-C_UJPUDUX0JIA_C05 + N-C_WZLMV4ASG9S_C05 (both runs agreed; scapula-stabilisation causes, forehand)",
                               "groups_detail": [{"rep": g["rep"], "stroke": g["stroke"], "members": [x["node"] for x in g["members"]], "lexical_cos_pairs": g["lexical_cos_pairs"], "why_run1": g["why_run1"], "why_run2": g["why_run2"]} for g in m3["groups"]]},
        "verify_all_full": {"pass": v3["pass"], "selftest_pass": v3["selftest"]["pass"], "mutations": len(v3["selftest"]["mutations"]),
                            "checks": {k: {"fail": len(c["fail"]), "warn": len(c["warn"])} for k, c in v3["checks"].items()},
                            "fail_rule_breakdown": {k: rule_counts(v3, k) for k in v3["checks"]},
                            "blind_replay_V5_muted_clips": "FAIL (G1: 0 cases) - no clips downloaded, no muted-clip test (user decision)",
                            "t05_note": "must_fix never changed; check_tree rules unchanged", "raw": v3},
        "text_blind_replay": {"rerun": False, "why": "not requested this round; holdout result stays the tree_v2 one (no prompt tuning)",
                              "library_menu_identical_to_tree_v2_run": same_menu,
                              "latest_result": {"file": "replay/text_blind/results_v2_indep_main.json", "case_hit3": rm["metrics"]["case_hit3 (problem OR cause-with-upstream)"], "coverage_any": rm["metrics"]["coverage_any"]}},
        "gemini_usage": {"model": b3["model"], "base": "micuapi.ai", "by_stage": u, "note": "this round only (cached gapfill/clustering included when present); key read from env only."},
        "remaining_gaps": {
            "problem_nodes_without_required": f"{s3['problem_total'] - s3['problem_with_required']}/{s3['problem_total']}",
            "must_fix_nodes_without_required": s3["must_fix_without_required"],
            "T05_after": sum("T05" in f for f in v3["checks"]["check_tree"]["fail"]), "T08_after": len(t08),
            "new_obs_clusters_still_open": {"total": len(clusters), "spanning_>=2_episodes": sum(len(c["episodes"]) >= 2 for c in clusters)},
            "obs_never_used_by_any_node": sorted(i for i, x in usage_tbl.items() if not (x["nodes_required"] or x["nodes_supporting"])),
            "V5_muted_clip_replay": "0 cases; needs per-segment download approval (PLAN_v2 sec 9.2)"},
        "no_secrets": True,
    }
    Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(report["verdict"], json.dumps(report["before_after"]["T05_fail"]), json.dumps(report["before_after"]["T08_fail"]), same_menu, u["TOTAL_this_round"])


if __name__ == "__main__":
    main()
