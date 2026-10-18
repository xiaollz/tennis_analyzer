#!/usr/bin/env python3
"""Confirm CC same_mechanism T08 pairs with 2 independent Gemini runs; intersect; append groups to a mech_merge.json.

usage: mech_merge_from_review.py --tree ftt_tree/tree_v7.json --review ftt_tree/t08_cc_review_v7.json \
         --merge-in ftt_tree/mech_merge_v7.json --merge-out ftt_tree/mech_merge_v8.json

Rules (deterministic after the 2 model runs; no text rewrite):
  - Only verdict=same_mechanism AND gate_pass entries.
  - Only cause-cause pairs; same classification.stroke; distinct source videos (M02).
  - Connected components of those pairs are the candidate groups.
  - Model sees only node ids + original statements (no user conclusions / KB / defect list).
  - A candidate group is kept only if BOTH runs place all its members in one group (intersection).
  - Kept groups are appended to merge-in (rep = lexicographically first member id); member text/evidence
    copied from the tree; also_worded empty. Existing groups untouched.
Key from env MICU_GEMINI_KEY only.
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import blind_replay_text as brt
from common import DIAG_DIR, load_json
from tv2_common import RUN_TEMPS, WORK, up

HERE = Path(__file__).resolve().parent


def stroke_of(nid, nodes, cards):
    ep = nodes[nid]["evidence"][0].split(":")[0]
    return cards[ep]["classification"]["stroke"]


def episodes_of(nid, nodes):
    return sorted({e.split(":")[0] for e in nodes[nid]["evidence"]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True); ap.add_argument("--review", required=True)
    ap.add_argument("--merge-in", required=True); ap.add_argument("--merge-out", required=True)
    a = ap.parse_args()
    t = load_json(a.tree); rev = load_json(a.review); mm = load_json(a.merge_in)
    nodes = {n["id"]: n for n in t["nodes"]}
    cards = {load_json(p)["video"]["id"]: load_json(p) for p in (DIAG_DIR / "episodes").glob("*.card.json")}
    # candidate edges
    par = {}
    def f(x):
        par.setdefault(x, x)
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    skipped = []
    for e in rev["entries"]:
        if e["verdict"] != "same_mechanism" or not e.get("gate_pass"):
            continue
        a_id, b_id = [x.strip() for x in e["pair"].split("|")]
        if nodes[a_id]["kind"] != "cause" or nodes[b_id]["kind"] != "cause":
            skipped.append({"pair": e["pair"], "why": "not cause-cause"}); continue
        if stroke_of(a_id, nodes, cards) != stroke_of(b_id, nodes, cards):
            skipped.append({"pair": e["pair"], "why": "different stroke"}); continue
        if set(episodes_of(a_id, nodes)) & set(episodes_of(b_id, nodes)):
            skipped.append({"pair": e["pair"], "why": "shared source video (M02)"}); continue
        par[f(a_id)] = f(b_id)
    comps = defaultdict(list)
    for x in list(par):
        comps[f(x)].append(x)
    cands = [sorted(v) for v in comps.values() if len(v) >= 2]
    # already-merged nodes cannot join a new group
    taken = {m["node"] for g in mm["groups"] for m in g["members"]}
    cands2, blocked = [], []
    for c in cands:
        hit = [i for i in c if i in taken]
        if hit:
            blocked.append({"group": c, "why": "already in a merge group", "hit": hit})
        else:
            cands2.append(c)
    # prompt
    lines = []
    for i, c in enumerate(cands2, 1):
        lines.append(f"CANDIDATE {i}:")
        for nid in c:
            lines.append(f"  {nid} | {nodes[nid]['name']}")
    prompt = ("You are grouping tennis coaching cause statements by underlying MECHANISM only.\n"
              "Below are candidate groups proposed by an earlier review. For each candidate, say whether ALL members\n"
              "describe the SAME mechanism (yes) or not (no). Do not rewrite any text. Do not invent members.\n"
              "Output ONLY JSON: {\"decisions\":[{\"candidate\":1,\"same_mechanism\":true|false,\"why\":\"<=20 words\"},...]}\n\n"
              + "\n".join(lines))
    work = WORK / "mech_from_review"; work.mkdir(parents=True, exist_ok=True)
    (work / "prompt.txt").write_text(prompt, encoding="utf-8")
    decisions = []
    for n in (1, 2):
        out, u, txt = brt.gcall(prompt, RUN_TEMPS[n], f"mfr{n}")
        (work / f"run{n}.raw.txt").write_text(txt, encoding="utf-8")
        (work / f"run{n}.json").write_text(json.dumps({"ok": out is not None, "usage": u, "out": out}, ensure_ascii=False, indent=1))
        dec = {}
        if isinstance(out, dict):
            for d in out.get("decisions") or []:
                try:
                    dec[int(d["candidate"])] = bool(d["same_mechanism"])
                except Exception:
                    pass
        decisions.append(dec)
    kept = []
    for i, c in enumerate(cands2, 1):
        if decisions[0].get(i) is True and decisions[1].get(i) is True:
            kept.append(c)
        else:
            skipped.append({"group": c, "why": f"not both-runs-true ({decisions[0].get(i)},{decisions[1].get(i)})"})
    # append groups
    nxt = len(mm["groups"]) + 1
    new_groups = []
    for c in kept:
        rep = c[0]  # lexicographic first
        stroke = stroke_of(rep, nodes, cards)
        members = []
        for nid in c:
            n = nodes[nid]
            members.append({"node": nid, "text": n["name"], "also_worded": [], "evidence": n["evidence"],
                            "episodes": episodes_of(nid, nodes),
                            "card_causes": [f"{ep}:{cid}" for ep in episodes_of(nid, nodes)
                                            for cid in [next((x['id'] for x in cards[ep]['causes'] if x['statement']==n['name']), None)]
                                            if cid]})
        # fix card_causes properly
        members = []
        for nid in c:
            n = nodes[nid]
            ccs = []
            for e in n["evidence"]:
                ep = e.split(":")[0]
                card = cards[ep]
                cid = next((x["id"] for x in card["causes"] if x["statement"] == n["name"]), None)
                if cid and f"{ep}:{cid}" not in ccs:
                    ccs.append(f"{ep}:{cid}")
            members.append({"node": nid, "text": n["name"], "also_worded": [], "evidence": list(n["evidence"]),
                            "episodes": episodes_of(nid, nodes), "card_causes": ccs})
        g = {"group_id": f"MG{nxt:02d}", "rep": rep, "stroke": stroke, "members": members,
             "lexical_cos_pairs": [], "why_run1": "confirmed by mech_merge_from_review (CC same_mechanism + 2 Gemini yes)",
             "why_run2": "confirmed by mech_merge_from_review (CC same_mechanism + 2 Gemini yes)",
             "source": "mech_merge_from_review"}
        mm["groups"].append(g); new_groups.append(g); nxt += 1
    before = mm["cause_nodes_after"]
    mm["cause_nodes_after"] = before - sum(len(g["members"]) - 1 for g in new_groups)
    mm["from_review"] = {"candidates": cands2, "kept": kept, "skipped": skipped, "blocked": blocked,
                         "new_groups": [g["group_id"] for g in new_groups]}
    Path(a.merge_out).write_text(json.dumps(mm, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"candidates": len(cands2), "kept": len(kept), "new_groups": [g["rep"] for g in new_groups],
                      "skipped": len(skipped), "cause_nodes_after": mm["cause_nodes_after"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
