#!/usr/bin/env python3
"""Step 2 of tree v2: semantic mechanism merge of cause nodes within one stroke type (model groups only; no text is rewritten).

usage: mech_merge.py run [--workers N]   # 2 runs per stroke (prompts/mech_group_v1.md), raw -> WORK/merge (outside repo)
       mech_merge.py build               # -> ftt_tree/mech_merge.json
Final groups = partition intersection of the two runs (two nodes are together only if BOTH runs put them together).
Script-side repairs (logged, never silent): ids unknown / of another stroke are ignored; an id in several groups of one run keeps
its first group; two nodes sharing a source video are never kept together (the later one is split off); groups < 2 dissolve.
This is an extra layer on top of build_tree.py's lexical merge (TF-IDF cosine >= 0.40, unchanged).
"""
import argparse
from pathlib import Path
import itertools
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import blind_replay_text as brt
from common import DIAG_DIR, load_json
from textsim import Sim
from tv2_common import PIPE, RUN_TEMPS, TREE_V1, WORK, cause_item_map, load_cards

import os
OUT = Path(os.environ["TV_MERGE_OUT"]) if os.environ.get("TV_MERGE_OUT") else DIAG_DIR / "ftt_tree" / "mech_merge.json"
MW = WORK / "merge"


def nodes_by_stroke():
    tree = load_json(TREE_V1)
    cards = load_cards(tree["build"]["episodes_used"])
    node_of, members = cause_item_map(tree, cards)
    info = {}
    for n in tree["nodes"]:
        if n["kind"] != "cause":
            continue
        ms = sorted(members[n["id"]])
        stroke = {cards[ep]["classification"]["stroke"] for ep, _ in ms}
        assert len(stroke) == 1, n["id"]
        stm = {(ep, cid): next(x["statement"] for x in cards[ep]["causes"] if x["id"] == cid) for ep, cid in ms}
        info[n["id"]] = {"id": n["id"], "stroke": stroke.pop(), "episodes": sorted({ep for ep, _ in ms}), "members": [f"{ep}:{cid}" for ep, cid in ms],
                         "name": n["name"], "layer": n.get("layer", "?"), "evidence": n["evidence"],
                         "also": [stm[m] for m in ms if stm[m] != n["name"]]}
    return tree, info


def render_items(ids, info):
    L = []
    for i in sorted(ids):
        x = info[i]
        L.append(f"{i} | {','.join(x['episodes'])} | layer {x['layer']} | {x['name']}")
        for a in x["also"]:
            L.append(f"    also worded: {a}")
    return "\n".join(L)


def run_one(args):
    stroke, n, ids, info = args
    d = MW / stroke
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"run{n}.json"
    if p.exists():
        return load_json(p)
    tpl = (PIPE / "prompts" / "mech_group_v1.md").read_text(encoding="utf-8").split("\n---\n", 1)[1]
    prompt = tpl.replace("{ITEMS}", render_items(ids, info))
    (d / f"run{n}.prompt.txt").write_text(prompt, encoding="utf-8")
    out, u, txt = brt.gcall(prompt, RUN_TEMPS[n], f"merge{n}")
    (d / f"run{n}.raw.txt").write_text(txt, encoding="utf-8")
    res = {"stroke": stroke, "run": n, "ok": out is not None, "usage": u, "out": out}
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return res


def strokes_to_run(info):
    by = defaultdict(list)
    for i, x in info.items():
        by[x["stroke"]].append(i)
    return {s: sorted(ids) for s, ids in by.items() if len({e for i in ids for e in info[i]["episodes"]}) >= 2 and len(ids) >= 2}


def cmd_run(workers):
    _, info = nodes_by_stroke()
    st = strokes_to_run(info)
    jobs = [(s, n, ids, info) for s, ids in sorted(st.items()) for n in (1, 2)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(run_one, jobs))
    u = Counter()
    for r in res:
        u.update(r["usage"])
    print({s: len(v) for s, v in st.items()}, "runs ok", sum(r["ok"] for r in res), "/", len(res), dict(u))


def clean_partition(out, ids, info, log, tag):
    """model groups -> {id: label}, applying the logged repairs described in the module docstring."""
    idset, lab, k, why, rep = set(ids), {}, 0, {}, {}
    groups = out.get("groups") if isinstance(out, dict) else None
    for g in groups if isinstance(groups, list) else []:
        if not isinstance(g, dict):
            continue
        raw = [x for x in (g.get("members") or []) if isinstance(x, str)]
        kept, eps = [], set()
        for x in raw:
            if x not in idset:
                log.append(f"{tag}: ignored unknown/other-stroke id {x}")
            elif x in lab:
                log.append(f"{tag}: {x} already in an earlier group; ignored in later group")
            elif eps & set(info[x]["episodes"]):
                log.append(f"{tag}: {x} shares a source video with an earlier member; split off")
            elif x in kept:
                continue
            else:
                kept.append(x)
                eps |= set(info[x]["episodes"])
        if len(kept) >= 2:
            for x in kept:
                lab[x] = k
            why[k] = str(g.get("why", ""))[:200]
            rep[k] = g.get("rep") if g.get("rep") in kept else None
            k += 1
    return lab, why, rep


def cmd_build():
    tree, info = nodes_by_stroke()
    st = strokes_to_run(info)
    sim = Sim([x["name"] for x in info.values()])
    cards = load_cards(tree["build"]["episodes_used"])
    groups, log, usage = [], [], Counter()
    runs_raw = {}
    one_run_pairs = []
    for stroke, ids in sorted(st.items()):
        runs = [load_json(MW / stroke / f"run{n}.json") for n in (1, 2)]
        for r in runs:
            usage.update(r["usage"])
        parts = [clean_partition(r["out"], ids, info, log, f"{stroke}/run{i + 1}") if r["ok"] else ({}, {}, {}) for i, r in enumerate(runs)]
        runs_raw[stroke] = {f"run{n + 1}": [sorted(i for i in parts[n][0] if parts[n][0][i] == g) for g in sorted(set(parts[n][0].values()))] for n in (0, 1)}
        by = defaultdict(list)
        for i in ids:
            if i in parts[0][0] and i in parts[1][0]:
                by[(parts[0][0][i], parts[1][0][i])].append(i)
        pairs = [set(frozenset(p) for p in itertools.combinations(sorted(i for i in parts[n][0] if parts[n][0][i] == g), 2))
                 for n in (0, 1) for g in set(parts[n][0].values())]
        p1 = set().union(*pairs[:len({*parts[0][0].values()})]) if parts[0][0] else set()
        p2 = set().union(*pairs[len({*parts[0][0].values()}):]) if parts[1][0] else set()
        one_run_pairs += [{"stroke": stroke, "pair": sorted(p), "in_run": 1 if p in p1 else 2} for p in sorted(p1 ^ p2, key=sorted)]
        for key, ms in by.items():
            if len(ms) < 2:
                continue
            ms = sorted(ms)
            reps = [parts[0][2].get(key[0]), parts[1][2].get(key[1])]
            rep = next((r for r in reps if r in ms), ms[0])
            mem = []
            for i in ms:
                x = info[i]
                mem.append({"node": i, "episodes": x["episodes"], "card_causes": x["members"], "layer": x["layer"], "text": x["name"],
                            "also_worded": x["also"], "evidence": x["evidence"]})
            groups.append({
                "stroke": stroke, "rep": rep, "members": mem,
                "why_run1": parts[0][1].get(key[0], ""), "why_run2": parts[1][1].get(key[1], ""),
                "rep_run1": reps[0], "rep_run2": reps[1],
                "lexical_cos_pairs": {f"{a}|{b}": round(sim.cos(info[a]["name"], info[b]["name"]), 3) for a, b in itertools.combinations(ms, 2)},
            })
    groups.sort(key=lambda g: (-len(g["members"]), g["stroke"], g["rep"]))
    for gi, g in enumerate(groups, 1):
        g["group_id"] = f"MG{gi:02d}"
    merged_nodes = sum(len(g["members"]) - 1 for g in groups)
    doc = {
        "version": "mech_group_v1", "model": brt.S1.MODEL, "prompt": "pipeline/prompts/mech_group_v1.md", "runs": 2,
        "rule": "final group = both runs group the nodes together (partition intersection); rep = run1 rep if inside the final group, else run2 rep, else first id; no text rewritten",
        "lexical_layer_unchanged": {"merge_cosine": 0.40, "note": "build_tree.py cosine merge is not touched; this is an extra layer applied to its 97 cause nodes"},
        "cause_nodes_before": len(info), "cause_nodes_after": len(info) - merged_nodes,
        "strokes_run": {s: len(v) for s, v in st.items()}, "groups": groups, "runs_cleaned_partitions": runs_raw,
        "pairs_grouped_by_only_one_run": one_run_pairs, "repair_log": log,
        "usage": dict(usage), "calls": 2 * len(st),
    }
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("groups", len(groups), "sizes", [len(g["members"]) for g in groups], "nodes", len(info), "->", doc["cause_nodes_after"],
          "one-run-only pairs", len(one_run_pairs), "repairs", len(log))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "build"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    cmd_run(a.workers) if a.cmd == "run" else cmd_build()
