#!/usr/bin/env python3
"""Deterministic builder for tree v2 = tree v1 (build_tree.py output, untouched) + OBS binding + mechanism merge. No model call.

usage: build_tree_v2.py [--v1 ftt_tree/tree.json] [--binding ftt_tree/obs_binding.json] [--merge ftt_tree/mech_merge.json]
                        [--out ftt_tree/tree_v2.json] [--log ftt_tree/build_log_v2.json]
Rules (all script; every string in the tree is still copied verbatim from a card or from obs_catalog.json):
  V2-1  observables = obs_catalog.json (OBS-01..18; id/question/values/layer/camera/reliability).
  V2-2  v1 node signature: problem -> its binding item; cause -> its member card causes' items: required = conditions required by ALL
        members (intersection), supporting = every other condition any member named. must_fix is never changed.
  V2-3  mechanism merge: group members collapse into the representative node (id/name/layer/next_one_thing of the rep, original text only).
        evidence = union; phase = most common member phase (tie -> rep's); signature required = rep's required, else the first member (sorted id)
        that has one (recorded in build.mechanism_merge[*].signature_required_from); supporting = union of everything else.
  V2-4  edges remapped to merged ids; self-loops and duplicate edges are merged (evidence union, strength recomputed: core iff >= 2 episodes);
        an edge that would close a cycle is dropped and logged. Symptom.caused_by / solution.targets_causes remapped.
  V2-5  symptom.obs_ref = first required OBS of its binding, else first supporting OBS, else NEW-OBS (visible + new candidate), else NONE.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict

from common import DIAG_DIR, ban_hits, load_json
from tv2_common import load_catalog


def conds_key(c):
    return (c["obs"], c["value"])


def dedupe(lst):
    out, seen = [], set()
    for c in lst:
        if conds_key(c) not in seen:
            seen.add(conds_key(c))
            out.append({"obs": c["obs"], "value": c["value"]})
    return out


def node_signature(items):
    """items: binding items of one node's members -> (required, supporting)."""
    req = [c for c in items[0]["required"] if all(conds_key(c) in {conds_key(x) for x in it["required"]} for it in items)]
    sup = dedupe([c for it in items for c in it["required"] + it["supporting"] if conds_key(c) not in {conds_key(r) for r in req}])
    return dedupe(req), sup


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", default=str(DIAG_DIR / "ftt_tree" / "tree.json"))
    ap.add_argument("--binding", default=str(DIAG_DIR / "ftt_tree" / "obs_binding.json"))
    ap.add_argument("--merge", default=str(DIAG_DIR / "ftt_tree" / "mech_merge.json"))
    ap.add_argument("--out", default=str(DIAG_DIR / "ftt_tree" / "tree_v2.json"))
    ap.add_argument("--log", default=str(DIAG_DIR / "ftt_tree" / "build_log_v2.json"))
    a = ap.parse_args(argv)
    t1, bind, mm = load_json(a.v1), load_json(a.binding), load_json(a.merge)
    cat = load_catalog()
    items = bind["items"]
    nodes1 = {n["id"]: n for n in t1["nodes"]}

    # ---- V2-2 v1-level signatures
    by_node = defaultdict(list)
    for key, it in items.items():
        if it["kind"] in ("problem", "cause"):
            by_node[it["node"]].append(it)
    sig1 = {nid: node_signature(its) for nid, its in by_node.items()}
    assert set(sig1) == set(nodes1), "binding does not cover exactly the v1 problem/cause nodes"

    # ---- V2-3 merge map
    rep_of = {nid: nid for nid in nodes1}
    groups = {}
    for g in mm["groups"]:
        ids = [m["node"] for m in g["members"]]
        assert g["rep"] in ids and all(i in nodes1 for i in ids)
        for i in ids:
            assert rep_of[i] == i, f"{i} in two groups"
            rep_of[i] = g["rep"]
        groups[g["rep"]] = sorted(ids)

    nodes, merge_prov = [], []
    for n in t1["nodes"]:
        nid = n["id"]
        if rep_of[nid] != nid:
            continue
        mem = groups.get(nid, [nid])
        m = dict(n)
        if len(mem) > 1:
            ms = [nodes1[i] for i in mem]
            m["evidence"] = sorted({e for x in ms for e in x["evidence"]})
            ph = Counter(x["phase"] for x in ms).most_common()
            m["phase"] = ph[0][0] if len(ph) == 1 or ph[0][1] > ph[1][1] else n["phase"]
            if "next_one_thing" not in n:
                nxt = next((x["next_one_thing"] for x in sorted(ms, key=lambda x: x["id"]) if x.get("next_one_thing")), None)
                if nxt:
                    m["next_one_thing"] = nxt
            order = [nid] + [i for i in sorted(mem) if i != nid]
            src = next((i for i in order if sig1[i][0]), nid)
            req = sig1[src][0]
            rk = {conds_key(c) for c in req}
            sup = dedupe([c for i in mem for c in sig1[i][0] + sig1[i][1] if conds_key(c) not in rk])
            merge_prov.append({"node": nid, "members": mem, "signature_required_from": src})
        else:
            req, sup = sig1[nid]
        m["signature"] = {"required": req, "supporting": sup, "excluding": []}
        m["differentials"] = []
        nodes.append(m)
    node_ids = {n["id"] for n in nodes}

    # ---- V2-4 edges
    raw = defaultdict(lambda: {"ev": set(), "eps": set(), "str": []})
    for e in t1["edges"]:
        f, t = rep_of[e["from"]], rep_of[e["to"]]
        if f == t:
            continue
        k = (f, t)
        raw[k]["ev"] |= set(e["evidence"])
        raw[k]["eps"] |= {r.split(":")[0] for r in e["evidence"] if not r.startswith("BOOK:")}
        raw[k]["str"].append(e["strength"])
    adj = defaultdict(set)

    def reaches(u, v):
        st, seen = [u], set()
        while st:
            x = st.pop()
            if x == v:
                return True
            if x not in seen:
                seen.add(x)
                st += list(adj[x])
        return False
    edges, dropped = [], []
    self_loops = [{"from": e["from"], "to": e["to"], "evidence": e["evidence"]} for e in t1["edges"] if rep_of[e["from"]] == rep_of[e["to"]]]
    for (f, t) in sorted(raw):
        info = raw[(f, t)]
        if reaches(t, f):
            dropped.append({"from": f, "to": t, "reason": "would close a cycle after merge", "evidence": sorted(info["ev"])})
            continue
        adj[f].add(t)
        strength = "core" if len(info["eps"]) >= 2 else ("inferred" if set(info["str"]) == {"inferred"} else "single_episode")
        edges.append({"from": f, "to": t, "strength": strength, "evidence": sorted(info["ev"])})

    # ---- symptoms / solutions / angles
    prim = {}
    symptoms = []
    for s in t1["symptoms"]:
        it = items[s["id"]]
        ref = it["required"][0]["obs"] if it["required"] else (it["supporting"][0]["obs"] if it["supporting"] else ("NEW-OBS" if it["visible"] and it["new_obs"] else "NONE"))
        s2 = dict(s)
        s2["obs_ref"] = ref
        s2["caused_by"] = sorted({rep_of[x] for x in s["caused_by"]})
        symptoms.append(s2)
    solutions = []
    for s in t1["solutions"]:
        s2 = dict(s)
        s2["targets_causes"] = sorted({rep_of[x] for x in s["targets_causes"]})
        solutions.append(s2)

    obs = [{**{k: o[k] for k in ("id", "question", "values", "layer", "camera")}, "reliability": o.get("reliability", "untested")} for o in cat]
    build = dict(t1["build"])
    build.update({
        "builder": "build_tree_v2", "parent_builder": t1["build"]["builder"], "merge_cosine": t1["build"]["merge_cosine"],
        "note": "v2 = v1 skeleton + OBS-01..18 binding (obs_binding.json, model-bound, 2-run agreement) + semantic mechanism merge (mech_merge.json); video layer still empty",
        "obs_binding": {"file": "ftt_tree/obs_binding.json", "model": bind["model"], "prompt": bind["prompt"], "catalog_sha256": bind["catalog_sha256"]},
        "mechanism_merge": {"file": "ftt_tree/mech_merge.json", "model": mm["model"], "prompt": mm["prompt"], "groups": merge_prov,
                            "cause_nodes_before": mm["cause_nodes_before"], "cause_nodes_after": mm["cause_nodes_after"],
                            "node_map_v1_to_v2": {i: r for i, r in sorted(rep_of.items()) if i != r},
                            "self_loop_edges_removed": self_loops, "edges_dropped_cycle": dropped},
    })
    tree = {"schema_version": "2.0", "holdout_episodes": t1["holdout_episodes"], "observables": obs,
            "nodes": sorted(nodes, key=lambda n: (n["kind"] != "problem", n["id"])), "edges": edges,
            "symptoms": symptoms, "solutions": solutions, "angles": t1["angles"], "video_layer": [],
            "quote_sources": t1["quote_sources"], "build": build}
    for n in tree["nodes"]:
        if n.get("next_one_thing") and ban_hits(n["next_one_thing"]):
            raise SystemExit(f"banned cue in {n['id']}")
    open(a.out, "w", encoding="utf-8").write(json.dumps(tree, ensure_ascii=False, indent=1) + "\n")
    cn = [n for n in tree["nodes"] if n["kind"] == "cause"]
    log = {"builder": "build_tree_v2", "problem_nodes": len(tree["nodes"]) - len(cn), "cause_nodes": len(cn), "edges": len(edges),
           "edges_v1": len(t1["edges"]), "self_loops_removed": len(self_loops), "edges_dropped_cycle": len(dropped),
           "core_edges": sum(e["strength"] == "core" for e in edges),
           "nodes_with_required": sum(bool(n["signature"]["required"]) for n in tree["nodes"]),
           "nodes_with_signature": sum(bool(n["signature"]["required"] or n["signature"]["supporting"]) for n in tree["nodes"])}
    open(a.log, "w", encoding="utf-8").write(json.dumps(log, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(log))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
