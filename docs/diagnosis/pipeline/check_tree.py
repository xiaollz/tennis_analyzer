#!/usr/bin/env python3
"""V3 tree integrity: every node/edge is constrained and backed by verified evidence.

usage: check_tree.py <tree.json> <episodes_dir>
"""
import sys
from collections import defaultdict
from pathlib import Path

from common import Finding, ban_hits, fails, load_json, load_schema, validate_schema

VERIFIED = ("stage1_verified", "stage2_verified")


def check(tree, ep_dir, where="tree"):
    out = validate_schema(tree, load_schema("tree.schema.json"))
    if fails(out):
        return out
    F = lambda rule, msg, level="FAIL": out.append(Finding(level, where, f"{rule} {msg}"))
    cards = {}
    for p in Path(ep_dir).glob("*.card.json"):
        c = load_json(p)
        cards[c["video"]["id"]] = c
    obs = {o["id"]: o for o in tree["observables"]}
    nodes = {n["id"]: n for n in tree["nodes"]}
    holdout = set(tree["holdout_episodes"])
    if len(nodes) != len(tree["nodes"]):
        F("T01", "duplicate node ids")

    def resolve(ref, ctx):
        """Return episode id the ref cites (or None for BOOK)."""
        if ref.startswith("BOOK:"):
            return None
        ep, item = ref.split(":")
        if ep in holdout:
            F("T02", f"{ctx} cites holdout episode {ep} (leaks the blind replay set)")
        c = cards.get(ep)
        if not c:
            F("T03", f"{ctx} cites {ref}: no card for {ep}")
            return ep
        if c["status"] not in VERIFIED:
            F("T03", f"{ctx} cites {ref}: card status {c['status']} is not verified")
        if item.startswith("Q") and item not in {q["id"] for q in c["quotes"]}:
            F("T03", f"{ctx} cites {ref}: quote missing")
        if item.startswith("V"):
            v = {x["id"]: x for x in c["video_layer"]}.get(item)
            if not v or v["status"] != "accepted":
                F("T03", f"{ctx} cites {ref}: video item missing or not accepted")
        return ep

    # T04 signatures: obs exist, values legal, no C-layer in required, must_fix nodes are constrained
    sig_keys = {}
    for n in tree["nodes"]:
        nid = n["id"]
        for part in ("required", "supporting", "excluding"):
            for cond in n["signature"][part]:
                o = obs.get(cond["obs"])
                if not o:
                    F("T04", f"{nid} {part} uses unknown {cond['obs']}")
                    continue
                if cond["value"] not in o["values"]:
                    F("T04", f"{nid} {part} {cond['obs']}={cond['value']!r} not in {o['values']}")
                if part == "required" and o["layer"] == "C":
                    F("T04", f"{nid} required uses C-layer {cond['obs']} (feel cannot be seen in video)")
        req = n["signature"]["required"]
        if n["kind"] != "style" and n["must_fix"] and not req:
            F("T05", f"{nid} must_fix node has no required observable: it can never be confirmed from video")
        key = tuple(sorted((c["obs"], c["value"]) for c in req))
        if req:
            sig_keys.setdefault(key, []).append(nid)
        for e in n["evidence"]:
            resolve(e, nid)
        if n.get("next_one_thing") and ban_hits(n["next_one_thing"]):
            F("T06", f"{nid} next_one_thing contains banned cue {ban_hits(n['next_one_thing'])}")
        for d in n["differentials"]:
            if d["vs"] not in nodes:
                F("T07", f"{nid} differential vs unknown {d['vs']}")
    # T08 indistinguishable nodes: same required signature and no differential between them
    for key, ids in sig_keys.items():
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                da = {d["vs"] for d in nodes[a]["differentials"]}
                db = {d["vs"] for d in nodes[b]["differentials"]}
                if b not in da and a not in db:
                    F("T08", f"{a} and {b} have identical required signatures and no differential")
    # T09 nodes sharing a required condition should name each other as differentials
    by_cond = defaultdict(set)
    for n in tree["nodes"]:
        for c in n["signature"]["required"]:
            by_cond[(c["obs"], c["value"])].add(n["id"])
    for cond, ids in by_cond.items():
        for a in ids:
            missing = ids - {a} - {d["vs"] for d in nodes[a]["differentials"]}
            if missing:
                F("T09", f"{a} shares {cond[0]}={cond[1]} with {sorted(missing)} but lists no differential", "WARN")
    # T10 edges: endpoints, evidence, strength vs episode count, acyclic
    adj = defaultdict(list)
    for e in tree["edges"]:
        ctx = f"edge {e['from']}->{e['to']}"
        if e["from"] not in nodes or e["to"] not in nodes:
            F("T10", f"{ctx} unknown endpoint")
            continue
        adj[e["from"]].append(e["to"])
        eps = {resolve(r, ctx) for r in e["evidence"]} - {None}
        if e["strength"] == "core" and len(eps) < 2:
            F("T10", f"{ctx} strength=core needs evidence from >= 2 episodes (has {len(eps)})")
        if e["strength"] == "inferred" and eps:
            F("T10", f"{ctx} marked inferred but cites episode evidence; use single_episode/core", "WARN")
    state = {}

    def dfs(u, stack):
        state[u] = 1
        for v in adj[u]:
            if state.get(v) == 1:
                F("T11", f"cycle through {' -> '.join(stack + [v])}")
            elif not state.get(v):
                dfs(v, stack + [v])
        state[u] = 2
    for n in nodes:
        if not state.get(n):
            dfs(n, [n])
    # T12 coverage: every verified card is mapped to a node (or explicitly pending)
    for ep, c in cards.items():
        if c["status"] not in VERIFIED or ep in holdout:
            continue
        nid = c["problem"]["node_id"]
        if nid == "PENDING":
            F("T12", f"{ep} verified but problem not mapped to a node", "WARN")
        elif nid not in nodes:
            F("T12", f"{ep} maps to unknown node {nid}")
        else:
            cited = any(r.startswith(ep + ":") for r in nodes[nid]["evidence"])
            if not cited:
                F("T12", f"{ep} maps to {nid} but the node does not cite it", "WARN")
    return out


def main(args):
    if len(args) != 2:
        print(__doc__)
        return 2
    findings = check(load_json(args[0]), args[1], args[0])
    for f in findings:
        print(f)
    n = len(fails(findings))
    print(f"check_tree: {n} FAIL")
    return 1 if n else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
