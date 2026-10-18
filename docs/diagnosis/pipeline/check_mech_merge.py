#!/usr/bin/env python3
"""V3c mechanism-merge check (deterministic; no model): merge never loses provenance, never rewrites text, never creates a cycle.

usage: check_mech_merge.py <mech_merge.json> <tree_v1.json> <tree_v2.json> <episodes_dir>
Rules
  M01 groups: every member is a v1 cause node; >= 2 members; a node is in at most one group; rep is a member.
  M02 same stroke type inside a group (classification.stroke of the member cards); no two members share a source video.
  M03 no rewriting: member text == v1 node name == the card cause statement; also_worded and evidence == v1/card; v2 node name == rep text.
  M04 counts: v2 cause nodes == v1 - sum(|group|-1); v2 node ids == v1 ids minus non-rep members; problem nodes unchanged.
  M05 provenance: each v1 cause node's evidence refs are all in its v2 node; every member card cause (ep:Cnn) is listed in
      tree_v2.build.mechanism_merge (groups) or sits in an unmerged v2 node with identical evidence; node_map covers exactly the non-rep members.
  M06 edges: every v1 edge maps to a v2 edge between the mapped endpoints carrying all its evidence, or is listed as self-loop / cycle-dropped.
  M07 tree v2 edges are acyclic (own DFS, independent of check_tree T11).
  M08 each final group lies inside one group of each of the two model runs (merge never goes beyond what both runs said).
  M09 solutions/symptoms/angles of v2 link only to existing v2 nodes; counts of symptoms/solutions/angles equal v1.
"""
import sys
from collections import defaultdict
from pathlib import Path

from common import Finding, fails, load_json


def check(mm, t1, t2, ep_dir):
    out = []
    F = lambda rule, msg: out.append(Finding("FAIL", "mech_merge", f"{rule} {msg}"))
    cards = {}
    for p in Path(ep_dir).glob("*.card.json"):
        c = load_json(p)
        cards[c["video"]["id"]] = c
    n1 = {n["id"]: n for n in t1["nodes"]}
    n2 = {n["id"]: n for n in t2["nodes"]}
    seen, rep_of = set(), {i: i for i in n1}
    for g in mm["groups"]:
        ids = [m["node"] for m in g["members"]]
        if len(ids) < 2:
            F("M01", f"{g['group_id']} has < 2 members")
        if g["rep"] not in ids:
            F("M01", f"{g['group_id']} rep {g['rep']} not a member")
        strokes, eps = set(), []
        for m in g["members"]:
            i = m["node"]
            if i not in n1 or n1[i]["kind"] != "cause":
                F("M01", f"{g['group_id']} member {i} is not a v1 cause node")
                continue
            if i in seen:
                F("M01", f"{i} is in more than one group")
            seen.add(i)
            rep_of[i] = g["rep"]
            eps += m["episodes"]
            for cc in m["card_causes"]:
                ep, cid = cc.split(":")
                c = cards.get(ep)
                st = next((x["statement"] for x in c["causes"] if x["id"] == cid), None) if c else None
                if st is None:
                    F("M03", f"{cc} not found in cards")
                    continue
                strokes.add(c["classification"]["stroke"])
                if cc == m["card_causes"][0] and m["text"] != n1[i]["name"]:
                    F("M03", f"{i} member text differs from v1 node name")
                if st != m["text"] and st not in m["also_worded"]:
                    F("M03", f"{cc} card statement is neither the member text nor in also_worded")
            if m["evidence"] != n1[i]["evidence"]:
                F("M03", f"{i} evidence differs from v1")
        if len(strokes) > 1:
            F("M02", f"{g['group_id']} mixes stroke types {sorted(strokes)}")
        if len(eps) != len(set(eps)):
            F("M02", f"{g['group_id']} has two members from the same video")
        if g["rep"] in n2 and n2[g["rep"]]["name"] != next((m["text"] for m in g["members"] if m["node"] == g["rep"]), None):
            F("M03", f"v2 node {g['rep']} name is not the rep's original text")
        # M08
        for k in ("run1", "run2"):
            raw = mm["runs_cleaned_partitions"].get(g["stroke"], {}).get(k, [])
            if not any(set(ids) <= set(r) for r in raw):
                F("M08", f"{g['group_id']} is not contained in one group of {k}")
    # M04
    c1 = [i for i, n in n1.items() if n["kind"] == "cause"]
    c2 = [i for i, n in n2.items() if n["kind"] == "cause"]
    merged = sum(len(g["members"]) - 1 for g in mm["groups"])
    if len(c2) != len(c1) - merged:
        F("M04", f"cause nodes {len(c1)} -> {len(c2)} but groups remove {merged}")
    if set(c2) != {i for i in c1 if rep_of[i] == i}:
        F("M04", "v2 cause ids are not exactly v1 ids minus non-rep members")
    if {i for i, n in n1.items() if n["kind"] == "problem"} != {i for i, n in n2.items() if n["kind"] == "problem"}:
        F("M04", "problem nodes changed")
    if mm["cause_nodes_before"] != len(c1) or mm["cause_nodes_after"] != len(c2):
        F("M04", "mech_merge.json cause counts disagree with the trees")
    # M05
    prov = {g["node"]: set(g["members"]) for g in t2["build"]["mechanism_merge"]["groups"]}
    nmap = t2["build"]["mechanism_merge"]["node_map_v1_to_v2"]
    if set(nmap) != {i for i in c1 if rep_of[i] != i} or any(nmap[i] != rep_of[i] for i in nmap):
        F("M05", "node_map_v1_to_v2 does not match the groups")
    for i in c1:
        r = rep_of[i]
        if r in n2 and not set(n1[i]["evidence"]) <= set(n2[r]["evidence"]):
            F("M05", f"evidence of {i} lost in {r}")
        if r == i and r not in prov and r in n2 and n2[r]["evidence"] != n1[i]["evidence"]:
            F("M05", f"unmerged node {i} evidence changed")
    for g in mm["groups"]:
        if prov.get(g["rep"]) != {m["node"] for m in g["members"]}:
            F("M05", f"{g['group_id']} not recorded in tree_v2.build.mechanism_merge")
    # M06
    e2 = {(e["from"], e["to"]): e for e in t2["edges"]}
    skipped = {(rep_of.get(x["from"], x["from"]), rep_of.get(x["to"], x["to"])) for x in t2["build"]["mechanism_merge"]["edges_dropped_cycle"]}
    loops = {(x["from"], x["to"]) for x in t2["build"]["mechanism_merge"]["self_loop_edges_removed"]}
    for e in t1["edges"]:
        f, t = rep_of[e["from"]], rep_of[e["to"]]
        if f == t:
            if (e["from"], e["to"]) not in loops:
                F("M06", f"edge {e['from']}->{e['to']} became a self-loop but is not listed")
        elif (f, t) in skipped:
            continue
        elif (f, t) not in e2:
            F("M06", f"edge {e['from']}->{e['to']} lost (no v2 edge {f}->{t}, not listed as dropped)")
        elif not set(e["evidence"]) <= set(e2[(f, t)]["evidence"]):
            F("M06", f"edge {f}->{t} lost evidence")
    # M07
    adj = defaultdict(list)
    for e in t2["edges"]:
        if e["from"] not in n2 or e["to"] not in n2:
            F("M06", f"v2 edge {e['from']}->{e['to']} has unknown endpoint")
        adj[e["from"]].append(e["to"])
    state = {}

    def dfs(u):
        state[u] = 1
        for v in adj[u]:
            if state.get(v) == 1:
                F("M07", f"cycle through {u}->{v}")
            elif not state.get(v):
                dfs(v)
        state[u] = 2
    for u in list(n2):
        if not state.get(u):
            dfs(u)
    # M09
    for kind in ("symptoms", "solutions", "angles"):
        if len(t2[kind]) != len(t1[kind]):
            F("M09", f"{kind} count changed {len(t1[kind])} -> {len(t2[kind])}")
        for it in t2[kind]:
            for nid in [it["problem"]] + it.get("caused_by", []) + it.get("targets_causes", []):
                if nid not in n2:
                    F("M09", f"{kind[:-1]} {it['id']} links missing node {nid}")
    return out


def main(argv):
    if len(argv) != 4:
        print(__doc__)
        return 2
    f = check(load_json(argv[0]), load_json(argv[1]), load_json(argv[2]), argv[3])
    for x in f:
        print(x)
    print(f"check_mech_merge: {len(fails(f))} FAIL")
    return 1 if fails(f) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
