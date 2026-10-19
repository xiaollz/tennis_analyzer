#!/usr/bin/env python3
"""Step 1 of tree v2: bind symptoms / problems / causes to observable items (OBS-01..18) with a model; deterministic afterwards.

usage: obs_bind.py run   [--workers N]     # 2 runs per episode (prompts/obs_bind_v1.md), raw -> WORK/bind (outside repo)
       obs_bind.py build                   # aggregate runs -> ftt_tree/obs_binding.json (+ NEW-OBS clustering call)
Aggregation (script only): an item's condition is 'required' only if BOTH runs list it as required; any other condition
named by either run is 'supporting'. Conditions with an unknown OBS id / value are dropped and counted (stats.invalid_dropped).
The model never edits node text. Prompt = obs_catalog + node texts only (no transcript, no user conclusions, no KB concepts).
"""
import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import blind_replay_text as brt
from common import DIAG_DIR, load_json
from tv2_common import PIPE, RUN_TEMPS, TREE_V1, WORK, cause_item_map, load_cards, load_catalog, sha256_file

import os
OUT = Path(os.environ["TV_BIND_OUT"]) if os.environ.get("TV_BIND_OUT") else DIAG_DIR / "ftt_tree" / "obs_binding.json"
BW = WORK / "bind"


def render_catalog(cat):
    return "\n".join(f"{o['id']} | {o['question']} | values: {', '.join(o['values'])} | layer {o['layer']} | camera {o['camera']}" for o in cat)


def episode_items(tree, cards, node_of):
    """-> {ep: {"problem": (nid, text), "symptoms": [(local, sym_id, text, kind, layer, camera)], "causes": [(local, key, nid, text, layer, [syms])]}}"""
    sym_by = defaultdict(list)
    for s in tree["symptoms"]:
        sym_by[s["episode"]].append(s)
    out = {}
    nodes = {n["id"]: n for n in tree["nodes"]}
    for ep in tree["build"]["episodes_used"]:
        c = cards[ep]
        pn = f"N-P_{ep.upper().replace('-', '_')}"
        pnode = next(n for n in tree["nodes"] if n["kind"] == "problem" and n["id"] == f"N-P_{__import__('tv2_common').up(ep)}")
        syms = []
        for s in sorted(sym_by[ep], key=lambda s: s["id"]):
            syms.append(("S" + s["id"].rsplit("_S", 1)[1], s["id"], s["description"], s["kind"], s["layer"], s.get("camera", "any")))
        chain = defaultdict(list)
        for e in c["chain"]:
            if e["from"].startswith("C") and e["to"].startswith("S"):
                chain[e["from"]].append(e["to"])
        causes = []
        for x in sorted(c["causes"], key=lambda x: x["id"]):
            causes.append((x["id"], f"{ep}:{x['id']}", node_of[(ep, x["id"])], x["statement"], x["layer"], sorted(chain.get(x["id"], []))))
        out[ep] = {"problem": (pnode["id"], pnode["name"]), "symptoms": syms, "causes": causes}
    return out


def render_episode(e):
    L = ["PROBLEM", f"P | {e['problem'][1]}", "", "SYMPTOMS (as stated; kind and layer are the extractor's own labels)"]
    for local, _, text, kind, layer, cam in e["symptoms"]:
        L.append(f"{local} | ({kind}, layer {layer}, camera {cam}) | {text}")
    L += ["", "CAUSES"]
    for local, _, _, text, layer, syms in e["causes"]:
        L.append(f"{local} | (layer {layer}; explains {', '.join(syms) if syms else 'no listed symptom'}) | {text}")
    return "\n".join(L)


def sanitize(obj, cat, local_ids):
    """Return ({local: {...clean}}, dropped_count, missing_ids). Never invents content; drops what is not in the catalog."""
    valid = {o["id"]: set(o["values"]) for o in cat}
    items = obj.get("items") if isinstance(obj, dict) else None
    items = items if isinstance(items, dict) else {}
    out, dropped = {}, 0
    for lid in local_ids:
        r = items.get(lid)
        if not isinstance(r, dict):
            continue

        def conds(lst):
            nonlocal dropped
            res = []
            for c in lst if isinstance(lst, list) else []:
                if isinstance(c, dict) and c.get("obs") in valid and c.get("value") in valid[c["obs"]]:
                    k = {"obs": c["obs"], "value": c["value"]}
                    if k not in res:
                        res.append(k)
                else:
                    dropped += 1
            return res
        req = conds(r.get("required"))
        sup = [c for c in conds(r.get("supporting")) if c not in req]
        new = []
        for n in r.get("new_obs") if isinstance(r.get("new_obs"), list) else []:
            if isinstance(n, dict) and isinstance(n.get("description"), str) and n["description"].strip() and isinstance(n.get("values"), list) and len(n["values"]) >= 2:
                new.append({"description": n["description"].strip(), "values": [str(v) for v in n["values"]][:6],
                            "layer": n.get("layer") if n.get("layer") in ("A", "B") else "B",
                            "camera": n.get("camera") if n.get("camera") in ("S", "B", "R", "any") else "any"})
        out[lid] = {"visible": bool(r.get("visible", True)), "required": req, "supporting": sup, "new_obs": new[:2], "reason": str(r.get("reason", ""))[:200]}
    return out, dropped, [l for l in local_ids if l not in out]


def run_one(args):
    ep, n, e, cat = args
    d = BW / ep
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"run{n}.json"
    if p.exists():
        return load_json(p)
    tpl = (PIPE / "prompts" / "obs_bind_v1.md").read_text(encoding="utf-8").split("\n---\n", 1)[1]
    prompt = tpl.replace("{CATALOG}", render_catalog(cat)).replace("{EPISODE}", render_episode(e))
    for bad in ("屈书毅", "用户", "learning.md", "FLT-", "HSA", "ESR"):
        assert bad not in prompt.replace(render_episode(e), ""), bad
    (d / f"run{n}.prompt.txt").write_text(prompt, encoding="utf-8")
    out, u, txt = brt.gcall(prompt, RUN_TEMPS[n], f"bind{n}")
    (d / f"run{n}.raw.txt").write_text(txt, encoding="utf-8")
    local = ["P"] + [s[0] for s in e["symptoms"]] + [c[0] for c in e["causes"]]
    clean, dropped, missing = sanitize(out, cat, local) if out is not None else ({}, 0, local)
    res = {"episode": ep, "run": n, "ok": out is not None, "usage": u, "items": clean, "invalid_dropped": dropped, "missing_ids": missing}
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return res


def agg(runs):
    """runs: list of clean item dicts (one per run that answered). -> aggregated item fields."""
    rs = [r for r in runs if r]
    if not rs:
        return {"visible": None, "required": [], "supporting": [], "new_obs": [], "reason": "", "agreement": "no_valid_run", "runs": []}
    req = [c for c in rs[0]["required"] if all(c in r["required"] for r in rs)] if len(rs) == len(runs) else []
    # 'required' means ALL listed conditions hold together; two different values of one OBS can never both hold (the model meant "any of").
    # Such a set cannot be expressed as required -> demote the conflicting conditions to supporting (stricter, never adds a required).
    seen = {}
    for c in req:
        seen.setdefault(c["obs"], set()).add(c["value"])
    conflict = {o for o, v in seen.items() if len(v) > 1}
    demoted = [c for c in req if c["obs"] in conflict]
    req = [c for c in req if c["obs"] not in conflict]
    anyc = list(demoted)
    for r in rs:
        for c in r["required"] + r["supporting"]:
            if c not in anyc and c not in req:
                anyc.append(c)
    visible = any(r["visible"] for r in rs)
    new = []
    for i, r in enumerate(rs):
        for n in r["new_obs"]:
            new.append(n)
    return {"visible": visible, "required": req, "supporting": anyc, "new_obs": new, "reason": rs[0]["reason"],
            "agreement": f"required {len(req)}/{len({json.dumps(c, sort_keys=True) for r in rs for c in r['required']})} agreed; runs={len(rs)}/{len(runs)}",
            "required_any_run": [c for r in rs for c in r["required"]],
            **({"demoted_conflicting_required": demoted} if demoted else {}),
            "runs": [{"visible": r["visible"], "required": r["required"], "supporting": r["supporting"], "new_obs": r["new_obs"]} for r in rs]}


def cmd_run(workers):
    tree, cat = load_json(TREE_V1), load_catalog()
    cards = load_cards(tree["build"]["episodes_used"])
    node_of, _ = cause_item_map(tree, cards)
    items = episode_items(tree, cards, node_of)
    jobs = [(ep, n, items[ep], cat) for ep in sorted(items) for n in (1, 2)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(run_one, jobs))
    print("runs ok:", sum(r["ok"] for r in res), "/", len(res), "invalid dropped:", sum(r["invalid_dropped"] for r in res),
          "missing ids:", sum(len(r["missing_ids"]) for r in res))
    u = Counter()
    for r in res:
        u.update(r["usage"])
    print("usage", dict(u))


def gap_items(items):
    """problem/cause items that are visible, have no required condition and no NEW-OBS candidate after round 1."""
    return sorted(k for k, it in items.items() if it["kind"] in ("problem", "cause") and not it["required"] and it["visible"] and not it["new_obs"])


def gapfill_prompt(items, cat):
    lines = []
    for k in gap_items(items):
        it = items[k]
        fit = "; ".join(f"{c['obs']}={c['value']}" for c in it["supporting"]) or "none"
        lines.append(f"{k} | {it['text']} | partly fits: {fit}")
    tpl = (PIPE / "prompts" / "obs_gapfill_v1.md").read_text(encoding="utf-8").split("\n---\n", 1)[1]
    return tpl.replace("{CATALOG}", render_catalog(cat)).replace("{ITEMS}", "\n".join(lines))


def cmd_gapfill():
    """Round 2 for round-1 gaps (2 runs, one call each). Needs round-1 aggregation, so it reads the draft built by `build`."""
    cat = load_catalog()
    doc = load_json(OUT)
    items = {k: v for k, v in doc["items"].items() if "gapfill" not in v}
    if not gap_items(items):
        print("no gap items")
        return
    prompt = gapfill_prompt(items, cat)
    d = WORK / "gapfill"
    d.mkdir(parents=True, exist_ok=True)
    (d / "prompt.txt").write_text(prompt, encoding="utf-8")
    for n in (1, 2):
        p = d / f"run{n}.json"
        if p.exists():
            continue
        out, u, txt = brt.gcall(prompt, RUN_TEMPS[n], f"gap{n}")
        (d / f"run{n}.raw.txt").write_text(txt, encoding="utf-8")
        p.write_text(json.dumps({"ok": out is not None, "usage": u, "out": out}, ensure_ascii=False, indent=1))
    print("gapfill runs saved for", len(gap_items(items)), "items")


def apply_gapfill(out_items, usage):
    """Merge the gapfill runs into round-1 items (script only): new_obs from both runs are appended; visible=False only if BOTH runs say so."""
    d = WORK / "gapfill"
    if not (d / "run1.json").exists():
        return None
    runs = [load_json(d / f"run{n}.json") for n in (1, 2)]
    for r in runs:
        usage.update(r["usage"])
    keys = gap_items(out_items)
    cat = load_catalog()
    for k in keys:
        per = []
        for r in runs:
            x = ((r["out"] or {}).get("items") or {}).get(k) if r["ok"] and isinstance(r["out"], dict) else None
            per.append(x if isinstance(x, dict) else None)
        news, vis = [], []
        for x in per:
            if x is None:
                continue
            vis.append(bool(x.get("visible", True)))
            for n in x.get("new_obs") if isinstance(x.get("new_obs"), list) else []:
                if isinstance(n, dict) and isinstance(n.get("description"), str) and n["description"].strip() and isinstance(n.get("values"), list) and len(n["values"]) >= 2:
                    news.append({"description": n["description"].strip(), "values": [str(v) for v in n["values"]][:6],
                                 "layer": n.get("layer") if n.get("layer") in ("A", "B") else "B",
                                 "camera": n.get("camera") if n.get("camera") in ("S", "B", "R", "any") else "any", "source": "gapfill"})
        it = out_items[k]
        it["gapfill"] = {"prompt": "pipeline/prompts/obs_gapfill_v1.md", "runs_answered": len(vis), "visible_per_run": vis}
        it["new_obs"] = it["new_obs"] + news[:4]
        if len(vis) == 2 and not any(vis):
            it["visible"] = False
            it["visible_source"] = "gapfill_both_runs"
        elif not news and not vis:
            it["gapfill"]["note"] = "no valid gapfill answer"
    return {"gap_items": len(keys), "calls": 2}


def cluster_new_obs(cands):
    """One model grouping call per run (2 runs); a cid pair is grouped only if BOTH runs group it (partition intersection)."""
    lines = [f"{c['cid']} | {c['description']} | {', '.join(c['values'])} | {c['layer']} | {c['camera']}" for c in cands]
    tpl = (PIPE / "prompts" / "obs_newcluster_v1.md").read_text(encoding="utf-8").split("\n---\n", 1)[1]
    prompt = tpl.replace("{ITEMS}", "\n".join(lines))
    d = WORK / "newobs"
    d.mkdir(parents=True, exist_ok=True)
    parts, usage = [], Counter()
    h = hashlib.sha256(prompt.encode()).hexdigest()[:10]
    for n in (1, 2):
        p = d / f"run{n}.{h}.json"
        if p.exists():
            r = load_json(p)
        else:
            out, u, txt = brt.gcall(prompt, RUN_TEMPS[n], f"newobs{n}")
            (d / f"run{n}.{h}.raw.txt").write_text(txt, encoding="utf-8")
            r = {"ok": out is not None, "usage": u, "out": out}
            p.write_text(json.dumps(r, ensure_ascii=False, indent=1))
        usage.update(r["usage"])
        parts.append(r)
    return parts, dict(usage)


def partition(groups, ids):
    """list of group dicts -> {id: group index}; an id in several groups keeps its first; groups need >=2 known ids."""
    idset, lab, k = set(ids), {}, 0
    for g in groups if isinstance(groups, list) else []:
        m = [x for x in (g.get("members") if isinstance(g, dict) else []) or [] if x in idset and x not in lab]
        if len(m) >= 2:
            for x in m:
                lab[x] = k
            k += 1
    return lab


def cmd_build(draft=False):
    tree, cat = load_json(TREE_V1), load_catalog()
    cards = load_cards(tree["build"]["episodes_used"])
    node_of, members = cause_item_map(tree, cards)
    items = episode_items(tree, cards, node_of)
    out_items, usage, invalid, missing = {}, Counter(), 0, []
    for ep in sorted(items):
        e = items[ep]
        runs = [load_json(BW / ep / f"run{n}.json") for n in (1, 2)]
        for r in runs:
            usage.update(r["usage"])
            invalid += r["invalid_dropped"]
        entries = [("problem", "P", e["problem"][0], e["problem"][1])]
        entries += [("symptom", s[0], s[1], s[2]) for s in e["symptoms"]]
        entries += [("cause", c[0], c[1], c[3]) for c in e["causes"]]
        for kind, local, key, text in entries:
            per = [r["items"].get(local) if r["ok"] else None for r in runs]
            if any(x is None for x in per):
                missing.append(f"{ep}:{local}")
            a = agg(per)
            nid = e["problem"][0] if kind == "problem" else (node_of[(ep, key.split(":")[1])] if kind == "cause" else None)
            out_items[key] = {"kind": kind, "episode": ep, "local_id": local, "node": nid, "text": text, **a}
    gap_stat = apply_gapfill(out_items, usage)
    # NEW-OBS candidates: every candidate from every run, then model grouping (both runs must agree)
    cands = []
    for key, it in sorted(out_items.items()):
        for i, n in enumerate(it["new_obs"]):
            cands.append({"cid": f"NC{len(cands) + 1:03d}", "item": key, **n})
    clusters, cl_usage = [], {}
    if cands and not draft:
        parts, cl_usage = cluster_new_obs(cands)
        ids = [c["cid"] for c in cands]
        labs = [partition(((p["out"] or {}).get("groups") if isinstance(p["out"], dict) else []), ids) for p in parts]
        by = defaultdict(list)
        for cid in ids:
            if cid in labs[0] and cid in labs[1]:
                by[(labs[0][cid], labs[1][cid])].append(cid)
        grouped = set()
        cd = {c["cid"]: c for c in cands}
        for gi, ms in enumerate(sorted(by.values(), key=lambda m: (-len(m), m[0]))):
            if len(ms) < 2:
                continue
            grouped |= set(ms)
            clusters.append({"id": f"NEW-OBS-{len(clusters) + 1:02d}", "rep": ms[0], "members": ms})
        for cid in ids:
            if cid not in grouped:
                clusters.append({"id": f"NEW-OBS-{len(clusters) + 1:02d}", "rep": cid, "members": [cid]})
        usage.update(cl_usage)
    doc = {
        "version": "obs_bind_v1", "model": brt.S1.MODEL, "prompt": "pipeline/prompts/obs_bind_v1.md", "runs": 2,
        "catalog_sha256": sha256_file(PIPE / "obs_catalog.json"), "source_tree": "docs/diagnosis/ftt_tree/tree.json (v1)",
        "aggregation": "required = in both runs' required; supporting = any other condition either run named; invalid obs/values dropped",
        "items": out_items, "new_obs_candidates": cands, "new_obs_clusters": clusters,
        "stats": {"gapfill": gap_stat, "items": len(out_items), "invalid_dropped": invalid, "items_with_missing_run": sorted(set(missing)),
                  "usage": dict(usage), "bind_calls": 2 * len(items), "newobs_cluster_calls": 2 if cands else 0},
    }
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    kinds = Counter((v["kind"], bool(v["required"])) for v in out_items.values())
    print("items", len(out_items), "with required by kind:", dict(kinds), "invalid dropped", invalid, "new_obs cands", len(cands), "clusters", len(clusters))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "build", "gapfill"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--draft", action="store_true", help="build: aggregate round-1 only, no NEW-OBS clustering call (input for gapfill)")
    a = ap.parse_args()
    {"run": lambda: cmd_run(a.workers), "build": lambda: cmd_build(a.draft), "gapfill": cmd_gapfill}[a.cmd]()
