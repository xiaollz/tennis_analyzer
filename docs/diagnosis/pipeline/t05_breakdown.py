#!/usr/bin/env python3
"""Deterministic T05 breakdown (no model): which must_fix nodes have no required observable, and why.

usage: t05_breakdown.py --tree ftt_tree/tree_vN.json --binding ftt_tree/obs_binding_vN.json --merge ftt_tree/mech_merge_vN.json --out loop/t05_breakdown_vN.md [--json out.json]
Only the nodes check_tree reports as T05 are classified (path A already exempts problem nodes with a signed cause).
Tiers: problem_node | has_NEW_OBS_candidate (cross_episode / single_episode) | feel_or_tactics | unobservable_per_model.
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import check_tree
from common import DIAG_DIR, load_json

FEEL = re.compile(r"\b(feel|feeling|sensation|tension|relax|tight|conscious|mindset|mental|tactic|tactical|strategy|strategic|pressure|choke|aware|awareness|intention|decision|confidence|nervous|risk|margin|pattern|point|match|opponent|play(ing)? (the )?percent)", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True); ap.add_argument("--binding", required=True); ap.add_argument("--merge", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--json")
    a = ap.parse_args()
    t = load_json(a.tree); b = load_json(a.binding); mm = load_json(a.merge)
    N = {n["id"]: n for n in t["nodes"]}
    members = defaultdict(set)
    for g in mm["groups"]:
        for m in g["members"]:
            members[g["rep"]].add(m["node"])
    cand = {c["cid"]: c for c in b["new_obs_candidates"]}
    cl_of = {}
    for c in b["new_obs_clusters"]:
        for m in c["members"]:
            cl_of[m] = c["id"]
    cl_eps = defaultdict(set)
    for c in b["new_obs_candidates"]:
        cl_eps[cl_of.get(c["cid"], c["cid"])].add(b["items"][c["item"]]["episode"])
    by_node = defaultdict(list)
    for iid, it in b["items"].items():
        by_node[it["node"]].append((iid, it))
    flagged = []
    import re as _re
    for f in check_tree.check(t, DIAG_DIR / "episodes"):
        m = _re.search(r"T05 (\S+) must_fix", f.msg)
        if m:
            flagged.append(m.group(1))
    rows = []
    for nid in sorted(flagged):
        n = N[nid]
        mem = members.get(nid) or {nid}
        items = [x for m_ in mem for x in by_node.get(m_, [])]
        nc = [cid for _, it in items for cid in [c["cid"] for c in b["new_obs_candidates"] if c["item"] == _] ] if items else []
        nc = [c["cid"] for c in b["new_obs_candidates"] if c["item"] in {i for i, _ in items}]
        clusters = sorted({cl_of[c] for c in nc if c in cl_of})
        cross = [c for c in clusters if len(cl_eps[c]) >= 2]
        reasons = " ".join(it.get("reason", "") for _, it in items)
        if n["kind"] == "problem":
            tier = "problem_node"
        elif nc:
            tier = "has_NEW_OBS_candidate"
        elif FEEL.search(n["name"] + " " + reasons):
            tier = "feel_or_tactics"
        else:
            tier = "unobservable_per_model"
        rows.append({"node": nid, "kind": n["kind"], "tier": tier, "name": n["name"], "new_obs_clusters": clusters, "cross_episode_clusters": cross,
                     "reason": (items[0][1].get("reason", "") if items else "")[:200]})
    cnt = Counter(r["tier"] for r in rows)
    L = [f"# T05 结构拆解（{Path(a.tree).stem}）", "", f"> 自动生成（`pipeline/t05_breakdown.py`，无模型）。T05 FAIL = **{len(rows)}**（路径 A 已生效）；must_fix 节点 {sum(n['must_fix'] for n in t['nodes'])}，有 required 的节点 {sum(bool(n['signature']['required']) for n in t['nodes'])}。", "",
         "| 档 | 数量 | 含义 |", "|---|---:|---|"]
    desc = {"problem_node": "问题节点，其下没有任何带 required 的根因", "has_NEW_OBS_candidate": "绑定产生了 NEW-OBS 候选但尚未升级为观察点",
            "feel_or_tactics": "体感/意图/战术/心理类描述，相机看不到", "unobservable_per_model": "其余：模型两次都没给 required，也没有候选"}
    for k in desc:
        L.append(f"| {k} | {cnt.get(k, 0)} | {desc[k]} |")
    cc = sum(1 for r in rows if r["tier"] == "has_NEW_OBS_candidate" and r["cross_episode_clusters"])
    L += ["", f"has_NEW_OBS_candidate 中，候选簇跨 ≥2 期的节点：**{cc}**；仅单期：**{cnt.get('has_NEW_OBS_candidate', 0) - cc}**。", ""]
    for k in desc:
        L += [f"## {k}（{cnt.get(k, 0)}）"]
        for r in rows:
            if r["tier"] == k:
                extra = f" · 簇 {','.join(r['new_obs_clusters'])}{' (跨期:' + ','.join(r['cross_episode_clusters']) + ')' if r['cross_episode_clusters'] else ''}" if r["new_obs_clusters"] else ""
                L.append(f"- `{r['node']}` ({r['kind']}) {r['name'][:90]}{extra}")
        L.append("")
    Path(a.out).write_text("\n".join(L), encoding="utf-8")
    if a.json:
        Path(a.json).write_text(json.dumps({"counts": dict(cnt), "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(dict(cnt), "cross-episode candidate nodes", cc)


if __name__ == "__main__":
    main()
