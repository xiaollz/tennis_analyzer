#!/usr/bin/env python3
"""Classify the remaining T08 pairs of a tree by WHY they cannot be resolved by the existing flow (deterministic; no model).

usage: t08_remaining.py --tree ftt_tree/tree_vN.json --review ftt_tree/t08_cc_review_vN.json --merge ftt_tree/mech_merge_vN.json --out loop/t08_remaining_vN.md
Input pairs = the T08 findings of check_tree on the tree. Each pair gets the verdict of the gated Claude Code review (if any) and a blocker:
  same_mechanism + same video          -> M02 (a merge may not combine two nodes of one episode)
  same_mechanism + different stroke    -> merge rule (same stroke only)
  same_mechanism + node in a group     -> merge rule (a node already merged cannot join another group) / M08
  same_mechanism + confirmed           -> flow merge was tried; check_mech_merge M08 rejects groups the two original runs did not form
  not_visible                          -> difference is effort/intent, not camera-visible
  existing_obs, gate failed            -> review did not give two different legal values
  new_obs_needed                       -> needs a new observable (cross-episode rule or user approval)
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path

import check_tree
from common import DIAG_DIR, load_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True); ap.add_argument("--review", required=True); ap.add_argument("--merge", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t = load_json(a.tree); rev = load_json(a.review); mm = load_json(a.merge)
    N = {n["id"]: n for n in t["nodes"]}
    cards = {load_json(p)["video"]["id"]: load_json(p) for p in (DIAG_DIR / "episodes").glob("*.card.json")}
    stroke = lambda nid: cards[N[nid]["evidence"][0].split(":")[0]]["classification"]["stroke"]
    eps = lambda nid: {e.split(":")[0] for e in N[nid]["evidence"]}
    ingroup = {m["node"] for g in mm["groups"] for m in g["members"]}
    R = {}
    for e in rev["entries"]:
        x, y = [s.strip() for s in e["pair"].split("|")]
        R[frozenset((x, y))] = e
    pairs = []
    for f in check_tree.check(t, DIAG_DIR / "episodes"):
        m = re.search(r"T08 (\S+) and (\S+) have", f.msg)
        if m:
            pairs.append(m.groups())
    rows = defaultdict(list)
    for x, y in pairs:
        e = R.get(frozenset((x, y)))
        if not e:
            rows["no review verdict"].append((x, y, "")); continue
        v = e["verdict"]; why = e["reason"][:140]
        if v == "same_mechanism":
            if eps(x) & eps(y):
                k = "same_mechanism but same video (M02: cannot merge)"
            elif N[x]["kind"] != "cause" or N[y]["kind"] != "cause":
                k = "same_mechanism but not cause-cause (problem node involved)"
            elif stroke(x) != stroke(y):
                k = "same_mechanism but different stroke label (merge rule: same stroke only)"
            elif x in ingroup or y in ingroup:
                k = "same_mechanism but a node is already in a merge group (extension needs M08 rule change)"
            else:
                k = "same_mechanism, confirmed by 2 Gemini runs, but check_mech_merge M08 rejects it (members were not grouped together by both original merge runs)"
        elif v == "not_visible":
            k = "not_visible (effort / intent, not camera-visible)"
        elif v == "existing_obs":
            k = "existing_obs verdict, gate failed (no two different legal values)" if not e.get("gate_pass") else "existing_obs applied"
        else:
            k = "new_obs_needed (single episode / user decision)"
        rows[k].append((x, y, why))
    L = [f"# T08 剩余 {len(pairs)} 对：为什么现有流程解决不了（{Path(a.tree).stem}）", "", "> 自动生成（`pipeline/t08_remaining.py`，无模型）。裁决来自 `" + Path(a.review).name + "`（Claude Code，已过 gate_cc_t08），不是结论，只是提案。", "",
         "| 阻塞原因 | 对数 |", "|---|---:|"]
    for k, v in sorted(rows.items(), key=lambda kv: -len(kv[1])):
        L.append(f"| {k} | {len(v)} |")
    for k, v in sorted(rows.items(), key=lambda kv: -len(kv[1])):
        L += ["", f"## {k}（{len(v)}）"]
        for x, y, why in v:
            L.append(f"- `{x}` | `{y}` — {why}")
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print({k: len(v) for k, v in rows.items()})


if __name__ == "__main__":
    main()
