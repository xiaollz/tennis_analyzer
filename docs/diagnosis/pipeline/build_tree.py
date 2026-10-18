#!/usr/bin/env python3
"""Deterministic skeleton builder: verified episode cards -> ftt_tree/tree.json (symptom -> cause -> solution/angle).

usage: build_tree.py [--episodes DIR] [--manifest FILE] [--out tree.json] [--log build_log.json] [--write-node-ids]

Rules (script-only; no model call, no hand-written node text; every string in the tree is copied verbatim from a card):
  B1  Only cards with status == stage1_verified. Skipped: manifest holdout videos, manifest low_confidence_cards,
      draft cards. Skipped episodes are listed in tree.build.skipped.
  B2  One problem node per episode (name = card problem.title_as_question).
  B3  Cause nodes: causes from different episodes MERGE when (same classification.stroke) AND
      TF-IDF cosine(statement_a, statement_b) >= MERGE_COSINE against the cluster representative
      (first member in sorted (episode, cause id) order). Causes of one episode never merge with each other.
      Cluster name = representative statement; evidence = union of member evidence.
  B4  Edges come from card chain: cause->cause and cause->PROBLEM as written. A cause->symptom chain edge becomes a
      cause->problem edge of that episode (symptoms belong to the episode problem). Edges dedupe by (from, to);
      strength = core if >= 2 distinct episodes, else single_episode (inferred if every source edge is 'inferred').
      An edge that would close a cycle is skipped and logged.
  B5  Symptoms / solutions / angles are kept per episode (not merged), linked to nodes by card ids.
      A solution is usable_as_prescription only if the card ban flag is false and the ban scan finds nothing.
  B6  must_fix=true and empty signatures: no card carries an observable (obs_ref is NEW-OBS/NONE everywhere), so the
      skeleton has no 'required' observables. check_tree T05 will FAIL on this; that is intentional and reported.
  B7  video_layer stays empty (Stage 2 not run).
Evidence = "EPISODE:Qnn"; quote_sources maps each ref to {video_id, start, about, text} copied from the card.
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from common import DIAG_DIR, ban_hits, load_json
from textsim import Sim

MERGE_COSINE = 0.40  # chosen from the pairwise distribution on non-holdout cards before any blind test (see README)
BUILDER_VERSION = "build_tree_v1"


def up(ep):
    return re.sub(r"[^A-Z0-9]", "_", ep.upper())


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default=str(DIAG_DIR / "episodes"))
    ap.add_argument("--manifest", default=str(DIAG_DIR / "loop" / "manifest.json"))
    ap.add_argument("--out", default=str(DIAG_DIR / "ftt_tree" / "tree.json"))
    ap.add_argument("--log", default=str(DIAG_DIR / "ftt_tree" / "build_log.json"))
    ap.add_argument("--write-node-ids", action="store_true", help="back-fill problem/causes node_id into the card files")
    a = ap.parse_args(argv)

    man = load_json(a.manifest)
    holdout = sorted(v["id"] for v in man["videos"] if v.get("holdout"))
    lowconf = sorted(x["id"] for x in man.get("low_confidence_cards", []))
    paths = sorted(Path(a.episodes).glob("*.card.json"))
    cards = {}
    for p in paths:
        c = load_json(p)
        cards[c["video"]["id"]] = (p, c)

    skipped, use = [], []
    for ep in sorted(cards):
        c = cards[ep][1]
        if ep in holdout:
            skipped.append({"episode": ep, "reason": "holdout", "status": c["status"]})
        elif ep in lowconf:
            skipped.append({"episode": ep, "reason": "low_confidence", "status": c["status"]})
        elif c["status"] != "stage1_verified":
            skipped.append({"episode": ep, "reason": f"status {c['status']}", "status": c["status"]})
        else:
            use.append(ep)

    refs = set()

    def ref(ep, q):
        r = f"{ep}:{q}"
        refs.add(r)
        return r

    # ---------------- cause clusters (B3)
    items = []  # (ep, cid, stroke, statement)
    for ep in use:
        c = cards[ep][1]
        for x in sorted(c["causes"], key=lambda x: x["id"]):
            items.append((ep, x["id"], c["classification"]["stroke"], x["statement"]))
    sim = Sim([i[3] for i in items])
    clusters = []  # {"rep": item, "members": [item], "sims": []}
    for it in items:
        placed = False
        for cl in clusters:
            r = cl["rep"]
            if r[0] == it[0] or any(m[0] == it[0] for m in cl["members"]) or r[2] != it[2]:
                continue
            s = sim.cos(r[3], it[3])
            if s >= MERGE_COSINE:
                cl["members"].append(it)
                cl["sims"].append(round(s, 3))
                placed = True
                break
        if not placed:
            clusters.append({"rep": it, "members": [it], "sims": []})
    node_of = {}  # (ep, cid) -> node id
    cause_nodes = {}
    for cl in clusters:
        r = cl["rep"]
        nid = f"N-C_{up(r[0])}_{r[1]}"
        for m in cl["members"]:
            node_of[(m[0], m[1])] = nid
        cause_nodes[nid] = cl
    prob_node = {ep: f"N-P_{up(ep)}" for ep in use}
    ids = list(cause_nodes) + list(prob_node.values())
    assert len(ids) == len(set(ids)), "node id collision"

    # ---------------- helper lookups
    def chain_syms(ep):
        out = defaultdict(list)  # cause node -> [(symptom id, chain strength)]
        for e in cards[ep][1]["chain"]:
            if e["from"].startswith("C") and e["to"].startswith("S"):
                out[node_of[(ep, e["from"])]].append(e["to"])
        return out

    # ---------------- nodes
    nodes = []
    for ep in use:
        c = cards[ep][1]
        nodes.append({
            "id": prob_node[ep], "name": c["problem"]["title_as_question"], "kind": "problem", "phase": "whole",
            "must_fix": True, "flt_ref": "NONE",
            "signature": {"required": [], "supporting": [], "excluding": []}, "differentials": [],
            "evidence": sorted(ref(ep, q) for q in c["problem"]["evidence"]),
        })
    sym_phase = {}
    for ep in use:
        for s in cards[ep][1]["symptoms"]:
            sym_phase[(ep, s["id"])] = s["phase"]
    sol_by_node = defaultdict(list)
    for ep in use:
        for s in sorted(cards[ep][1]["solutions"], key=lambda x: x["id"]):
            for t in s["targets"]:
                if t.startswith("C"):
                    sol_by_node[node_of[(ep, t)]].append(s)
    for nid, cl in cause_nodes.items():
        r = cl["rep"]
        rc = next(x for x in cards[r[0]][1]["causes"] if x["id"] == r[1])
        phases = Counter()
        ev = set()
        for ep, cid, _, _ in cl["members"]:
            c = cards[ep][1]
            for sid in chain_syms(ep).get(nid, []):
                phases[sym_phase[(ep, sid)]] += 1
            ev |= {ref(ep, q) for q in next(x for x in c["causes"] if x["id"] == cid)["evidence"]}
        top = phases.most_common(2)
        phase = top[0][0] if top and (len(top) == 1 or top[0][1] > top[1][1]) else "whole"
        node = {
            "id": nid, "name": r[3], "kind": "cause", "phase": phase, "must_fix": True, "flt_ref": "NONE",
            "layer": rc["layer"],
            "signature": {"required": [], "supporting": [], "excluding": []}, "differentials": [],
            "evidence": sorted(ev),
        }
        for s in sol_by_node.get(nid, []):
            g = s.get("external_goal", "")
            if g and s.get("intuition_first_ok") and not s["ban_conflict"]["flag"] and not ban_hits(g):
                node["next_one_thing"] = g
                break
        nodes.append(node)

    # ---------------- edges (B4)
    raw = defaultdict(lambda: {"ev": set(), "eps": set(), "str": []})
    for ep in use:
        for e in cards[ep][1]["chain"]:
            if not e["from"].startswith("C"):
                continue
            a_ = node_of[(ep, e["from"])]
            if e["to"].startswith("C"):
                b_ = node_of[(ep, e["to"])]
            else:  # PROBLEM or a symptom of this episode
                b_ = prob_node[ep]
            if a_ == b_:
                continue
            k = (a_, b_)
            raw[k]["ev"] |= {ref(ep, q) for q in e["evidence"]}
            raw[k]["eps"].add(ep)
            raw[k]["str"].append(e["strength"])
    adj = defaultdict(set)

    def reaches(u, v):
        stack, seen = [u], set()
        while stack:
            x = stack.pop()
            if x == v:
                return True
            if x in seen:
                continue
            seen.add(x)
            stack += list(adj[x])
        return False

    edges, dropped_edges = [], []
    for (f, t) in sorted(raw):
        info = raw[(f, t)]
        if reaches(t, f):
            dropped_edges.append({"from": f, "to": t, "reason": "would close a cycle", "episodes": sorted(info["eps"])})
            continue
        adj[f].add(t)
        strength = "core" if len(info["eps"]) >= 2 else ("inferred" if set(info["str"]) == {"inferred"} else "single_episode")
        edges.append({"from": f, "to": t, "strength": strength, "evidence": sorted(info["ev"])})

    # ---------------- symptoms / solutions / angles (B5)
    symptoms, solutions, angles = [], [], []
    for ep in use:
        c = cards[ep][1]
        by_sym = defaultdict(set)
        for e in c["chain"]:
            if e["from"].startswith("C") and e["to"].startswith("S"):
                by_sym[e["to"]].add(node_of[(ep, e["from"])])
        for s in c["symptoms"]:
            symptoms.append({
                "id": f"SYM_{up(ep)}_{s['id']}", "episode": ep, "problem": prob_node[ep], "description": s["description"],
                "kind": s["kind"], "layer": s["layer"], "camera": s.get("camera", "any"), "phase": s["phase"],
                "obs_ref": s["obs_ref"], "caused_by": sorted(by_sym.get(s["id"], [])),
                "evidence": sorted(ref(ep, q) for q in s["evidence"]),
            })
        for s in sorted(c["solutions"], key=lambda x: x["id"]):
            usable = (not s["ban_conflict"]["flag"]) and not ban_hits(s["summary"] + " " + s.get("external_goal", ""))
            solutions.append({
                "id": f"SOL_{up(ep)}_{s['id']}", "episode": ep, "problem": prob_node[ep], "kind": s["kind"],
                "summary": s["summary"], "external_goal": s.get("external_goal", ""), "scenario_fit": s["scenario_fit"],
                "ban_conflict": s["ban_conflict"]["flag"], "intuition_first_ok": s["intuition_first_ok"],
                "usable_as_prescription": usable,
                "targets_causes": sorted({node_of[(ep, t)] for t in s["targets"] if t.startswith("C")}),
                "targets_symptoms": sorted(f"SYM_{up(ep)}_{t}" for t in s["targets"] if t.startswith("S")),
                "evidence": sorted(ref(ep, q) for q in s["evidence"]),
            })
        for s in sorted(c["angles"], key=lambda x: x["id"]):
            angles.append({
                "id": f"ANG_{up(ep)}_{s['id']}", "episode": ep, "problem": prob_node[ep], "type": s["type"],
                "summary": s["summary"], "evidence": sorted(ref(ep, q) for q in s["evidence"]),
            })

    # ---------------- quote sources
    qs = {}
    for r in sorted(refs):
        ep, q = r.split(":")
        qq = next(x for x in cards[ep][1]["quotes"] if x["id"] == q)
        qs[r] = {"video_id": ep, "start": qq["start"], "about": qq["about"], "text": qq["text"]}

    tree = {
        "schema_version": "2.0",
        "holdout_episodes": holdout,
        "observables": [],
        "nodes": sorted(nodes, key=lambda n: (n["kind"] != "problem", n["id"])),
        "edges": edges,
        "symptoms": symptoms, "solutions": solutions, "angles": angles,
        "video_layer": [],
        "quote_sources": qs,
        "build": {
            "builder": BUILDER_VERSION, "merge_cosine": MERGE_COSINE, "episodes_used": use, "skipped": skipped,
            "note": "skeleton: observables empty, signatures empty (no card carries an OBS ref); video layer empty (Stage 2 not run)",
        },
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(tree, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    merged = [{"node": nid, "members": [f"{m[0]}:{m[1]}" for m in cl["members"]], "cosines": cl["sims"],
               "statements": [m[3] for m in cl["members"]]} for nid, cl in cause_nodes.items() if len(cl["members"]) > 1]
    log = {
        "builder": BUILDER_VERSION, "merge_cosine": MERGE_COSINE, "episodes_used": use, "skipped": skipped,
        "counts": {"problem_nodes": len(use), "cause_nodes": len(cause_nodes), "cause_items_before_merge": len(items),
                   "symptoms": len(symptoms), "solutions": len(solutions), "angles": len(angles), "edges": len(edges)},
        "merged_clusters": merged, "dropped_edges": dropped_edges,
    }
    Path(a.log).write_text(json.dumps(log, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    if a.write_node_ids:
        for ep in use:
            p, c = cards[ep]
            c["problem"]["node_id"] = prob_node[ep]
            for x in c["causes"]:
                x["node_id"] = node_of[(ep, x["id"])]
            p.write_text(json.dumps(c, ensure_ascii=False, indent=2) + "\n" if p.read_text().endswith("\n") else json.dumps(c, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(log["counts"], ensure_ascii=False), f"merged_clusters={len(merged)} dropped_edges={len(dropped_edges)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
