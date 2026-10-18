#!/usr/bin/env python3
"""Compose loop/reports/verify_2026-10-18_tree_v5.json (script outputs only): v4 (22 verified episodes, 29 OBS) -> v5 (38 episodes, 32 OBS).

usage: compose_tree_v5_report.py --verify raw_verify.json --out report.json [--commit-pre HASH]
"""
import argparse
import glob
import json
import re
from collections import Counter
from pathlib import Path

from common import DIAG_DIR, load_json
from compose_tree_v4_report import rule_counts, tree_stats, usage_of

KEYS = ("promptTokenCount", "candidatesTokenCount", "thoughtsTokenCount", "totalTokenCount", "calls")
W5, W5A = Path("/workspace/s1_work/tree_v5"), Path("/workspace/s1_work/tree_v5a")
NEW_EPS = "PwkGLkRm73o Z9XwaXDVkZg m1NBdd3Bigg T5meVwBOKR4 LoSrMj-Qxkc 1W1rope5l0k 9NirSL59an0 cIIEYn42o4g RVfmDk-iEwo WoVEWh7fFfc x_WIg59Wr9Y dZadhzVEsds PfDqnff_vYA pWzyP-xfLfU AppdZqxH_2k fjpr9PS02tM toArb3WkJcs 6ak4ZScH_gc PjvOQkqH9DA FBJnFBtZozE Vg8lbXOhM3E omsLx7F2Wdc".split()


def s1_usage():
    u = Counter()
    for v in NEW_EPS:
        for p in glob.glob(str(DIAG_DIR / "episodes" / "raw" / v / "s1_run*.log.json")):
            x = load_json(p)
            u.update({k: val for k, val in (x.get("usage_total") or {}).items() if k in KEYS})
            u["calls"] += x.get("calls", 0) if "calls" not in (x.get("usage_total") or {}) else 0
    return {k: u.get(k, 0) for k in KEYS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", required=True); ap.add_argument("--out", required=True); ap.add_argument("--commit-pre", default="")
    a = ap.parse_args()
    ft = DIAG_DIR / "ftt_tree"
    t4, t5 = load_json(ft / "tree_v4.json"), load_json(ft / "tree_v5.json")
    b4, b5 = load_json(ft / "obs_binding_v4.json"), load_json(ft / "obs_binding_v5.json")
    m5 = load_json(ft / "mech_merge_v5.json")
    base5 = load_json(ft / "tree_base_v5.json")
    prom = load_json(ft / "new_obs_promotion_v5.json")
    cc = load_json(ft / "t08_cc_review_v5.json")
    v5 = load_json(a.verify)
    r4 = load_json(DIAG_DIR / "loop" / "reports" / "verify_2026-10-18_tree_v4.json")["verify_all_full"]["raw"]
    s4, s5 = tree_stats(t4, b4, None), tree_stats(t5, b5, None)
    f5, f4 = v5["checks"]["check_tree"]["fail"], r4["checks"]["check_tree"]["fail"]
    t08_5 = [f.split("T08 ")[1] for f in f5 if "T08" in f]
    t08_4 = [f.split("T08 ")[1] for f in f4 if "T08" in f]
    N = {n["id"]: n for n in t5["nodes"]}
    pair_req = Counter()
    for f in t08_5:
        pair_req[json.dumps([f"{c['obs']}={c['value']}" for c in N[f.split(' and ')[0]]["signature"]["required"]])] += 1
    nr, ns = Counter(), Counter()
    for n in t5["nodes"]:
        for c in n["signature"]["required"]:
            nr[c["obs"]] += 1
        for o in {x["obs"] for x in n["signature"]["supporting"]}:
            ns[o] += 1
    usage_tbl = {o["id"]: {"layer": o["layer"], "camera": o["camera"], "nodes_required": nr[o["id"]], "nodes_supporting": ns[o["id"]]} for o in t5["observables"]}
    cand = {c["cid"]: c for c in b5["new_obs_candidates"]}
    clusters = []
    for c in b5["new_obs_clusters"]:
        rep = cand[c["rep"]]
        its = sorted({cand[m]["item"] for m in c["members"]})
        clusters.append({"id": c["id"], "description": rep["description"], "values": rep["values"], "layer": rep["layer"], "camera": rep["camera"],
                         "candidates": len(c["members"]), "episodes": sorted({b5["items"][i]["episode"] for i in its})})
    clusters.sort(key=lambda c: (-len(c["episodes"]), c["id"]))
    rm = load_json(DIAG_DIR / "replay" / "text_blind" / "results_v5_indep_main.json")
    rm2 = load_json(DIAG_DIR / "replay" / "text_blind" / "results_v2_indep_main.json")
    mk = "case_hit3 (problem OR cause-with-upstream)"
    u = {"stage1_extraction round7 (22 episodes, 2-3 runs each)": s1_usage(),
         "obs_bind v5a (29 OBS) + gapfill + newobs clustering": usage_of(glob.glob(str(W5A / "bind" / "*" / "run*.json")) + glob.glob(str(W5A / "gapfill" / "run*.json")) + glob.glob(str(W5A / "newobs" / "run*.json"))),
         "mech_merge v5a": usage_of(glob.glob(str(W5A / "merge" / "*" / "run*.json"))),
         "family grouping for promotion": usage_of(glob.glob(str(W5A / "families" / "run*.json"))),
         "obs_bind v5 (32 OBS) + gapfill + newobs clustering": usage_of(glob.glob(str(W5 / "bind" / "*" / "run*.json")) + glob.glob(str(W5 / "gapfill" / "run*.json")) + glob.glob(str(W5 / "newobs" / "run*.json"))),
         "mech_merge v5": usage_of(glob.glob(str(W5 / "merge" / "*" / "run*.json"))),
         "text blind replay v5 (retrieve 54 + judge 18)": rm["gemini_usage"],
         "video probes (10 + 2 ad-hoc calls, YouTube URL fileData)": {"totalTokenCount": 69009 + 10500, "calls": 12, "note": "10 probe calls measured (69,009 tokens); 2 ad-hoc calls (60 s description test 6,601 tokens; audio check ~3.9k tokens) + 1 rejected request (HTTP 400, no tokens)"}}
    tot = Counter()
    for k, x in u.items():
        if k.startswith("video"):
            continue
        tot.update({kk: vv for kk, vv in x.items() if kk in KEYS})
    u["TOTAL_excluding_video_probes"] = dict(tot)
    gap = s5["must_fix_without_required"]
    report = {
        "date": "2026-10-18", "pre_commit": a.commit_pre, "branch": "cursor/diagnosis-phase1", "verdict": "PASS" if v5["pass"] else "FAIL",
        "stage1_round7": {"episodes_attempted": len(NEW_EPS), "verified": 16, "draft_rerun_fail": 6,
                          "verified_ids": [v for v in NEW_EPS if load_json(DIAG_DIR / "episodes" / f"{v}.card.json")["status"] == "stage1_verified"],
                          "draft_ids": [v for v in NEW_EPS if load_json(DIAG_DIR / "episodes" / f"{v}.card.json")["status"] != "stage1_verified"],
                          "prompt": "ep_s1_v6 unchanged", "excluded_as_evidence": ["ExkBtFRhUWY", "JIMgI3jiVns"], "holdout_episodes_extracted": 0},
        "tree_v5": {"file": "docs/diagnosis/ftt_tree/tree_v5.json (skeleton: tree_base_v5.json via build_tree.py; tree.json v1 and tree_v2..v4 untouched)",
                    "episodes_in_tree": len(base5["build"]["episodes_used"]), "observables": len(t5["observables"]),
                    "problem_nodes": s5["problem_total"], "cause_items_before_merge": base5["build"].get("cause_items_before_merge"), "cause_nodes_v5": s5["cause_total"],
                    "edges": len(t5["edges"]), "core_edges": sum(e["strength"] == "core" for e in t5["edges"])},
        "new_observables": {"rule": "unchanged cross-episode rule (>=2 source episodes, layer A/B, explicit camera, no forbidden term); promote_new_obs.py apply with TV_MIN_PROMOTE=1 (only the minimum COUNT of 5 was lowered; no eligibility condition changed)",
                            "promoted": [{k: p[k] for k in ("obs", "description", "values", "layer", "camera", "episodes")} for p in prom["promoted"]],
                            "families_total": prom["families_total"], "eligible": prom["eligible"], "ineligible_summary": prom["ineligible_summary"]},
        "before_after": {"v4 (22 episodes, 29 OBS)": s4, "v5 (38 episodes, 32 OBS)": s5,
                         "must_fix_nodes": {"v4": sum(1 for n in t4["nodes"] if n["must_fix"]), "v5": sum(1 for n in t5["nodes"] if n["must_fix"])},
                         "T05_fail": {"v4": sum("T05" in f for f in f4), "v5": sum("T05" in f for f in f5)},
                         "T05_share_of_must_fix": {"v4": f"{sum('T05' in f for f in f4)}/{sum(1 for n in t4['nodes'] if n['must_fix'])}", "v5": f"{sum('T05' in f for f in f5)}/{sum(1 for n in t5['nodes'] if n['must_fix'])}"},
                         "T08_fail": {"v4": len(t08_4), "v5": len(t08_5)},
                         "check_tree_fail_total": {"v4": len(f4), "v5": len(f5)},
                         "T08_pairs_v5": t08_5, "T08_required_signature_counts_v5": {json.loads(k)[0]: c for k, c in pair_req.items()},
                         "note": "must_fix never changed, check_tree rules unchanged, no differential added to the tree"},
        "t08_claude_code_review": {"file": "ftt_tree/t08_cc_review_v5.json", "tool": "local Claude Code via relays/cc.sh cheap tennis_gap (same topic; neutral prompt: only node texts + verbatim quotes + OBS catalog, no user conclusions / KB concepts / defect list)",
                                   "verdicts": cc["verdicts"], "gate_pass": cc["gate_pass"], "gate_fail_pairs": cc["gate_fail"],
                                   "proposed_new_observables": cc["proposed_new_observables"],
                                   "applied_to_tree": False, "why_not": "proposed observables are not cross-episode-promoted and are not user-approved; 19 same_mechanism verdicts would need a merge decision (the merge step did not group them); 1 existing_obs verdict failed the gate (vB missing)"},
        "obs_usage_v5": usage_tbl, "obs_never_used": sorted(i for i, x in usage_tbl.items() if not (x["nodes_required"] or x["nodes_supporting"])),
        "new_obs_cluster_list_v5": clusters[:40], "new_obs_clusters_total": len(clusters),
        "mechanism_merge_v5": {"groups": len(m5["groups"]), "group_sizes": [len(g["members"]) for g in m5["groups"]], "cause_nodes": f"{m5['cause_nodes_before']} -> {m5['cause_nodes_after']}",
                               "groups_detail": [{"rep": g["rep"], "stroke": g["stroke"], "members": [x["node"] for x in g["members"]], "lexical_cos_pairs": g["lexical_cos_pairs"]} for g in m5["groups"]]},
        "verify_all_full": {"pass": v5["pass"], "selftest_pass": v5["selftest"]["pass"], "mutations": len(v5["selftest"]["mutations"]),
                            "checks": {k: {"fail": len(c["fail"]), "warn": len(c["warn"])} for k, c in v5["checks"].items()},
                            "fail_rule_breakdown": {k: rule_counts(v5, k) for k in v5["checks"]},
                            "blind_replay_V5_muted_clips": "FAIL (G1: 0 cases) - no clips downloaded; URL-mode video cannot be muted (loop/video_layer_feasibility.md)",
                            "raw": v5},
        "text_blind_replay": {"rerun": True, "times": 1, "prompt_tuned_on_holdout": False, "retrieval_prompt": "prompts/tree_retrieve_v2.md (unchanged)",
                              "file": "replay/text_blind/results_v5_indep_main.json",
                              "v2": {"case_hit3": rm2["metrics"][mk], "coverage_any": rm2["metrics"]["coverage_any"], "conditional_case_hit3": rm2["metrics"]["conditional_on_tree_coverage"]},
                              "v5": {"case_hit3": rm["metrics"][mk], "coverage_any": rm["metrics"]["coverage_any"], "conditional_case_hit3": rm["metrics"]["conditional_on_tree_coverage"]},
                              "reading": "new cards did not add holdout coverage (still 3/18 holdout episodes have any gold cause/problem matched in the tree); case_hit3 moved with the larger library menu"},
        "gemini_usage": {"model": "gemini-3.8-flash", "base": "micuapi.ai", "by_stage": u, "note": "key from env only; Claude Code usage: 3 short cheap-tier calls (1 ping, 2 T08 review attempts; the first attempt had an empty prompt because of a spawn error and produced no content)"},
        "remaining_gaps": {"must_fix_without_required": gap, "T05_after": sum("T05" in f for f in f5), "T08_after": len(t08_5),
                           "new_obs_clusters_spanning_>=2_episodes": sum(len(c["episodes"]) >= 2 for c in clusters),
                           "V5_muted_clip_replay": "0 cases; needs per-segment download approval", "draft_cards": 12},
        "no_secrets": True}
    Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(report["verdict"], report["before_after"]["T05_fail"], report["before_after"]["T08_fail"], u["TOTAL_excluding_video_probes"])


if __name__ == "__main__":
    main()
