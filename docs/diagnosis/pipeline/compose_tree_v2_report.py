#!/usr/bin/env python3
"""Compose loop/reports/verify_<date>_tree_v2.json from script outputs only (no narration), and write ftt_tree/mech_merge_review.md.

usage: compose_tree_v2_report.py --verify raw_verify.json --out report.json [--commit-pre HASH]
Inputs: ftt_tree/{tree,tree_v2,obs_binding,mech_merge}.json, replay/text_blind/{results,results_v2_indep_main,results_v2_indep_ref}.json,
        usage fields in /workspace/s1_work/tree_v2 and tree_blind_v2 result files (outside the repo).
"""
import argparse
import glob
import json
import re
from collections import Counter
from pathlib import Path

from common import DIAG_DIR, load_json
from tv2_common import WORK

W2 = Path("/workspace/s1_work/tree_blind_v2")
KEYS = ("promptTokenCount", "candidatesTokenCount", "thoughtsTokenCount", "totalTokenCount", "calls")


def usage_of(paths):
    u = Counter()
    for p in paths:
        r = load_json(p)
        u.update({k: v for k, v in (r.get("usage") or {}).items() if k in KEYS})
    return {k: u.get(k, 0) for k in KEYS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--commit-pre", default="")
    a = ap.parse_args()
    ft = DIAG_DIR / "ftt_tree"
    t1, t2, bind, mm = load_json(ft / "tree.json"), load_json(ft / "tree_v2.json"), load_json(ft / "obs_binding.json"), load_json(ft / "mech_merge.json")
    v = load_json(a.verify)
    rd = DIAG_DIR / "replay" / "text_blind"
    r1, rm, rr = load_json(rd / "results.json"), load_json(rd / "results_v2_indep_main.json"), load_json(rd / "results_v2_indep_ref.json")

    # ---- OBS usage (node level on tree v2; item level on the binding)
    node_req, node_sup = Counter(), Counter()
    for n in t2["nodes"]:
        for c in n["signature"]["required"]:
            node_req[c["obs"]] += 1
        for c in {x["obs"] for x in n["signature"]["supporting"]}:
            node_sup[c] += 1
    sym_req, sym_sup = Counter(), Counter()
    for it in bind["items"].values():
        if it["kind"] == "symptom":
            for c in it["required"]:
                sym_req[c["obs"]] += 1
            for c in {x["obs"] for x in it["supporting"]}:
                sym_sup[c] += 1
    obs_usage = {o["id"]: {"layer": o["layer"], "camera": o["camera"], "nodes_required": node_req[o["id"]], "nodes_supporting": node_sup[o["id"]],
                           "symptoms_required": sym_req[o["id"]], "symptoms_supporting": sym_sup[o["id"]]} for o in t2["observables"]}
    # ---- coverage
    nodes = t2["nodes"]
    must = [n for n in nodes if n["must_fix"] and n["kind"] != "style"]
    by_kind = {k: {"total": sum(n["kind"] == k for n in nodes), "with_required": sum(n["kind"] == k and bool(n["signature"]["required"]) for n in nodes),
                   "with_any_signature": sum(n["kind"] == k and bool(n["signature"]["required"] or n["signature"]["supporting"]) for n in nodes)} for k in ("problem", "cause")}
    node_map = t2["build"]["mechanism_merge"]["node_map_v1_to_v2"]
    items = bind["items"]
    src = {}
    for it in items.values():
        if it["kind"] in ("problem", "cause"):
            src.setdefault(node_map.get(it["node"], it["node"]), []).append(it)
    gaps = Counter()
    gap_nodes = {"unobservable": [], "new_obs_candidate": []}
    for n in must:
        if n["signature"]["required"]:
            continue
        its = src[n["id"]]
        if any(i["new_obs"] for i in its):
            gaps["new_obs_candidate"] += 1
            gap_nodes["new_obs_candidate"].append(n["id"])
        else:
            gaps["unobservable"] += 1
            gap_nodes["unobservable"].append(n["id"])
    # ---- NEW-OBS candidate list
    cand = {c["cid"]: c for c in bind["new_obs_candidates"]}
    clusters = []
    for c in bind["new_obs_clusters"]:
        rep = cand[c["rep"]]
        clusters.append({"id": c["id"], "description": rep["description"], "values": rep["values"], "layer": rep["layer"], "camera": rep["camera"],
                         "candidates": len(c["members"]), "from_items": sorted({cand[m]["item"] for m in c["members"]}),
                         "other_descriptions": [cand[m]["description"] for m in c["members"] if m != c["rep"]]})
    clusters.sort(key=lambda c: (-len(c["from_items"]), c["id"]))
    # ---- merge review (first 15 groups; fewer exist)
    groups = mm["groups"][:15]
    q = t2["quote_sources"]
    review = []
    for g in groups:
        review.append({"group_id": g["group_id"], "stroke": g["stroke"], "rep": g["rep"], "why_run1": g["why_run1"], "why_run2": g["why_run2"],
                       "lexical_cos_pairs": g["lexical_cos_pairs"],
                       "members": [{"node": m["node"], "episodes": m["episodes"], "card_causes": m["card_causes"], "text": m["text"],
                                    "evidence": [{"ref": e, "start": q[e]["start"], "text": q[e]["text"]} for e in m["evidence"][:3]]} for m in g["members"]]})
    lines = ["# 机械合并抽检（自动生成，供人工复核）", "", f"- 合并前 {mm['cause_nodes_before']} 个根因节点 → 合并后 {mm['cause_nodes_after']}；最终分组 {len(mm['groups'])} 组（只有一次运行同意的候选对 {len(mm['pairs_grouped_by_only_one_run'])} 个，未合并）。",
             "- 规则：两次独立运行都把它们分在一起才合并；组代表文字 = 组内某成员原文，不改写；同一视频的两条根因不合并。", "- 出处列的是该节点证据引用按编号排序的前 3 条（原文逐字，来自卡片），不一定是最贴近该根因的那句；完整证据见 tree_v2.json / mech_merge.json。", ""]
    for g in review:
        lines += [f"## {g['group_id']}（{g['stroke']}，代表 `{g['rep']}`）", f"- 模型理由 run1：{g['why_run1']}", f"- 模型理由 run2：{g['why_run2']}", f"- 词面余弦（对照，阈值 0.40 未改）：{g['lexical_cos_pairs']}"]
        for m in g["members"]:
            lines.append(f"- **{m['node']}**（{','.join(m['episodes'])}；卡片根因 {','.join(m['card_causes'])}）{m['text']}")
            for e in m["evidence"]:
                lines.append(f"  - 出处 {e['ref']} @ {e['start']}s：“{e['text']}”")
        lines.append("")
    if mm["pairs_grouped_by_only_one_run"]:
        lines.append("## 仅一次运行同意、未合并的候选对")
        by = {m["node"]: m for g in [] for m in g}
        for p in mm["pairs_grouped_by_only_one_run"]:
            a_, b_ = p["pair"]
            na, nb = next(n for n in t1["nodes"] if n["id"] == a_), next(n for n in t1["nodes"] if n["id"] == b_)
            lines += [f"- 仅 run{p['in_run']} 同意（{p['stroke']}）：", f"  - {a_}：{na['name']}", f"  - {b_}：{nb['name']}"]
    (DIAG_DIR / "ftt_tree" / "mech_merge_review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- replay comparison
    k3 = "case_hit3 (problem OR cause-with-upstream)"
    k1 = "case_hit1 (problem OR cause top-1)"
    pick = lambda m: {"cases_scored": m["cases_scored"], "positives": m["positives"], "negatives": m["negatives"], "coverage_any": m["coverage_any"],
                      "coverage_problem": m["coverage_problem"], "coverage_cause_any": m["coverage_cause_any"],
                      "p_hit3": m["p_hit3"], "c_hit3_with_upstream": m["c_hit3_with_upstream"], "case_hit3": m[k3], "case_hit1": m[k1],
                      "gold_cause_recall_with_upstream": m["gold_cause_recall_with_upstream"], "conditional_on_tree_coverage": m["conditional_on_tree_coverage"],
                      "uncovered_positives_abstain_rate": m["uncovered_positives_abstain_rate"], "negative_high_conf_fp": m["negative_high_conf_fp"],
                      "covered_cases_of_positives": f"{round(m['coverage_any'] * m['positives'])}/{m['positives']}" if m["coverage_any"] is not None else None,
                      "lexical_gold_cause_cos_ge_0.40_any_tree_node": m["lexical_gold_cause_cos_ge_0.40_any_tree_node"]}
    per_case = {}
    for row in rm["rows"]:
        if row["excluded"]:
            continue
        o1 = next((x for x in r1["rows"] if x["episode"] == row["episode"]), None)
        orf = next((x for x in rr["rows"] if x["episode"] == row["episode"]), None)
        hit = lambda r: (sum(bool(x and (x["p_hit3"] or x["c_hit3_exp"])) for x in r["runs"]), r["valid_runs"], bool(r["problem_covered"] or r["cause_covered"])) if r and not r["excluded"] else None
        per_case[row["episode"]] = {"v1_text_replay(hit_runs,valid,covered)": hit(o1), "v1_tree+new_retrieval(ref)": hit(orf), "v2_tree+new_retrieval(main)": hit(row)}
    u = {
        "obs_bind_round1 (44 calls)": usage_of(glob.glob(str(WORK / "bind" / "*" / "run*.json"))),
        "obs_gapfill (2 calls)": usage_of(glob.glob(str(WORK / "gapfill" / "run*.json"))),
        "newobs_cluster_final (2 calls)": usage_of([p for p in glob.glob(str(WORK / "newobs" / "run*.json"))]),
        "newobs_cluster_discarded_round1 (2 calls; superseded after gapfill produced no new candidates)": usage_of(glob.glob(str(WORK / "newobs_discarded_round1" / "run*.json"))),
        "mech_merge (6 calls)": usage_of(glob.glob(str(WORK / "merge" / "*" / "run*.json"))),
        "replay_main_retrieve (54 calls)": usage_of(glob.glob(str(W2 / "retrieve_main" / "*" / "run*.json"))),
        "replay_main_judge (answer key on v2 tree)": usage_of(glob.glob(str(W2 / "judge_main" / "*" / "key.json"))),
        "replay_ref_retrieve (54 calls)": usage_of(glob.glob(str(W2 / "retrieve_ref" / "*" / "run*.json"))),
    }
    tot = Counter()
    for x in u.values():
        tot.update(x)
    u["TOTAL"] = dict(tot)
    # ---- verify summary
    rule_counts = {}
    for k, c in v.get("checks", {}).items():
        rc = Counter(re.search(r"\b([A-Z]\d{2})\b", f).group(1) if re.search(r"\b([A-Z]\d{2})\b", f) else "other" for f in c["fail"])
        rule_counts[k] = dict(rc)
    report = {
        "date": "2026-10-18", "base_commit": "8cece00", "pre_replay_commit": a.commit_pre, "branch": "cursor/diagnosis-phase1",
        "verdict": "PASS" if v["pass"] else "FAIL",
        "tree_v2": {"file": "docs/diagnosis/ftt_tree/tree_v2.json (tree.json = v1, untouched)", "builder": "pipeline/build_tree_v2.py (parent: build_tree.py, cosine merge 0.40 unchanged)",
                    "problem_nodes": by_kind["problem"]["total"], "cause_nodes_v1": mm["cause_nodes_before"], "cause_nodes_v2": mm["cause_nodes_after"],
                    "cause_items_before_any_merge": 99, "edges_v1": len(t1["edges"]), "edges_v2": len(t2["edges"]),
                    "core_edges_v1": sum(e["strength"] == "core" for e in t1["edges"]), "core_edges_v2": sum(e["strength"] == "core" for e in t2["edges"]),
                    "observables": len(t2["observables"]), "video_layer_items": len(t2["video_layer"]), "holdout_cited": 0},
        "obs_binding": {
            "model": bind["model"], "runs": 2, "prompt": bind["prompt"], "gapfill_prompt": "pipeline/prompts/obs_gapfill_v1.md",
            "items": {"problem": sum(i["kind"] == "problem" for i in items.values()), "symptom": sum(i["kind"] == "symptom" for i in items.values()), "cause_card_items": sum(i["kind"] == "cause" for i in items.values())},
            "invalid_obs_or_value_dropped_by_script": bind["stats"]["invalid_dropped"],
            "node_coverage_tree_v2": by_kind, "nodes_with_required": sum(n["signature"]["required"] != [] for n in nodes), "nodes_total": len(nodes),
            "must_fix_nodes_without_required": {"total": sum(gaps.values()), "unobservable_per_model(feel/rule/tactic, both runs)": gaps["unobservable"], "has_NEW_OBS_candidate": gaps["new_obs_candidate"]},
            "symptom_obs_ref_v1": dict(Counter(s["obs_ref"] for s in t1["symptoms"])), "symptom_obs_ref_v2": dict(Counter(s["obs_ref"] for s in t2["symptoms"])),
            "obs_usage": obs_usage, "obs_never_used_by_any_node": sorted(i for i, x in obs_usage.items() if not x["nodes_required"] and not x["nodes_supporting"]),
            "new_obs_candidates": len(bind["new_obs_candidates"]), "new_obs_clusters": len(clusters), "new_obs_cluster_list": clusters,
            "gap_nodes": gap_nodes,
        },
        "mechanism_merge": {
            "model": mm["model"], "runs": 2, "prompt": mm["prompt"], "lexical_layer": "unchanged: TF-IDF cosine >= 0.40 within stroke (build_tree.py)",
            "cause_items_before_any_merge": 99, "cause_nodes_after_lexical_merge(v1)": mm["cause_nodes_before"], "cause_nodes_after_semantic_merge(v2)": mm["cause_nodes_after"],
            "strokes_run": mm["strokes_run"], "groups": len(mm["groups"]), "group_sizes": [len(g["members"]) for g in mm["groups"]],
            "pairs_grouped_by_only_one_run_not_merged": mm["pairs_grouped_by_only_one_run"], "repair_log": mm["repair_log"],
            "review_first_15_groups(only_this_many_exist)": review, "review_file": "docs/diagnosis/ftt_tree/mech_merge_review.md"},
        "verify_all_full": {"pass": v["pass"], "selftest_pass": v["selftest"]["pass"], "mutations": len(v["selftest"]["mutations"]),
                            "checks": {k: {"fail": len(c["fail"]), "warn": len(c["warn"])} for k, c in v["checks"].items()}, "fail_rule_breakdown": rule_counts,
                            "warns": {k: c["warn"] for k, c in v["checks"].items() if c["warn"]},
                            "t05_note": "must_fix was NOT changed; T05 stays FAIL for every must_fix node without a required observable (problem nodes and causes the catalog cannot cover)",
                            "blind_replay_V5_muted_clips": {"run": True, "cases": 0, "result": "FAIL (G1: 0 cases) - no replay/cases or audio-stripped clips; per user decision no clip download and no muted-clip blind test"},
                            "raw": v},
        "text_blind_replay": {
            "note": "same 18 holdout episodes / same gold / same scorer as v1; retrieval variant = independent causes (tree_retrieve_v2.md), prompt fixed + committed before any holdout call, each config run once; no tuning on this holdout",
            "v1_baseline": pick(r1["metrics"]) | {"tree": "tree.json v1 + tree_retrieve_v1 (problem-first, none_match abstain)"},
            "ref_v1_tree_new_retrieval": pick(rr["metrics"]) | {"tree": "tree.json v1 + tree_retrieve_v2 (plain menu); answer key = v1 key reused"},
            "main_v2_tree_new_retrieval": pick(rm["metrics"]) | {"tree": "tree_v2.json + tree_retrieve_v2 (menu with also-worded); answer key re-judged on v2 menu"},
            "gate_reference_G1(text replay, reference only)": {"v1": r1["gate_reference"], "ref": rr["gate_reference"], "main": rm["gate_reference"]},
            "per_case(hit_runs_of_3,valid_runs,tree_covers_gold)": per_case,
            "excluded": rm["metrics"]["excluded"], "holdout_without_transcript": rm["holdout_without_transcript"],
            "failure_classes_main": None},
        "gemini_usage": {"model": bind["model"], "base": "micuapi.ai", "by_stage": u,
                         "note": "tokens as reported by the API (thoughts counted inside totalTokenCount); retries counted. build_tree_v2 / check_* / verify_all make no model calls. Key read from env only."},
        "no_secrets": True,
    }
    # failure class per case for main
    cls = {}
    for row in rm["rows"]:
        if row["excluded"]:
            cls[row["episode"]] = "excluded"
            continue
        runs = [x for x in row["runs"] if x]
        returned = any(x["causes"] or x["problems"] for x in runs)
        hit = any(x["p_hit3"] or x["c_hit3_exp"] for x in runs)
        cov = row["problem_covered"] or row["cause_covered"]
        cls[row["episode"]] = "hit" if cov and hit else ("retrieval_miss" if cov else ("tree_gap+false_retrieval" if returned else "tree_gap"))
    report["text_blind_replay"]["failure_classes_main"] = cls
    Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": report["verdict"], "replay": {k: report["text_blind_replay"][k] for k in ("v1_baseline", "ref_v1_tree_new_retrieval", "main_v2_tree_new_retrieval")}, "usage_total": u["TOTAL"]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
