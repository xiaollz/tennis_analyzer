#!/usr/bin/env python3
"""Prepare the input files for a Claude Code T08 review (deterministic; no model).

usage: t08_prep.py --tree ftt_tree/tree_vN.json --outdir /workspace/s1_work/tree_vN [--pairs-json pairs.json]
Writes outdir/t08_pairs.txt ("idA | idB"), t08_groups.md (connected components of the flagged pairs: node text, signature,
verbatim quotes), obs_list.txt (catalog). The flagged pairs are recomputed from check_tree's T08 rule unless --pairs-json is given.
The review prompt itself is outdir/t08_prompt.txt (same wording as the v5/v7 reviews, paths substituted).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import check_tree
from common import DIAG_DIR, load_json

PROMPT = """Task: review pairs of diagnosis-tree nodes that a checker flags as indistinguishable, and say, for each pair, whether the provided material supports a way to tell them apart from tennis video.

Files (read them): {d}/t08_groups.md (each node: id, original text, current observable signature, verbatim coach quotes with ids), {d}/t08_pairs.txt (the flagged pairs, "idA | idB"), {d}/obs_list.txt (the observable catalog: id, question, allowed values).

For EACH pair in t08_pairs.txt choose exactly one verdict:
 - "existing_obs": a catalog observable OBS-xx exists where, according to the two nodes' own texts and quotes, node A would show value vA and node B would show a different value vB. Give obs, vA, vB, and quote ids (from the quotes listed for that node) that support each side.
 - "same_mechanism": the two texts describe the same underlying thing in different words; no discrimination is needed. Quote the wording from both that shows this.
 - "not_visible": the texts differ, but the difference is not something a camera can show (e.g. only a sensation or an intention); say why in one sentence.
 - "new_obs_needed": the texts differ in something a camera could show, but no catalog observable covers it. Propose ONE neutral English question with 2-3 allowed values (no advice, no judgement words, no coaching terms, no conclusions about what is right or wrong), the side-view/back-view camera it needs, and quote ids from each node that motivate it.
Rules: use only the files above and nothing you know from elsewhere; every claim must cite quote ids that really appear under the node you cite them for; do not invent quotes; do not propose prescriptions or fixes; if the material is insufficient, say "not_visible" or "same_mechanism" rather than guessing.
Output: ONLY one JSON array, one object per pair, in the order of t08_pairs.txt: {{"pair":"idA | idB","verdict":"...","obs":"OBS-xx or null","vA":null,"vB":null,"quotes_A":[],"quotes_B":[],"new_question":null,"new_values":[],"camera":null,"reason":"one sentence"}}. quotes_A / quotes_B use the full ref form shown in the file (e.g. "aPcKJDadzek:Q11").
"""


def flagged(tree):
    fs = check_tree.check(tree, DIAG_DIR / "episodes")
    out = []
    import re
    for f in fs:
        m = re.search(r"T08 (\S+) and (\S+) have", f.msg)
        if m:
            out.append((m.group(1), m.group(2)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True); ap.add_argument("--outdir", required=True); ap.add_argument("--pairs-json")
    a = ap.parse_args()
    t = load_json(a.tree)
    N = {n["id"]: n for n in t["nodes"]}
    pairs = [tuple(p) for p in load_json(a.pairs_json)] if a.pairs_json else flagged(t)
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    (out / "t08_pairs.txt").write_text("\n".join(f"{x} | {y}" for x, y in pairs) + "\n", encoding="utf-8")
    parent = {}
    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for x, y in pairs:
        parent[find(x)] = find(y)
    comps = defaultdict(set)
    for x, y in pairs:
        comps[find(x)] |= {x, y}
    lines = []
    for i, ids in enumerate(sorted(comps.values(), key=lambda s: sorted(s)[0]), 1):
        n0 = N[sorted(ids)[0]]
        shared = ", ".join(f"{c['obs']}={c['value']}" for c in n0["signature"]["required"])
        lines += [f"## GROUP {i}", f"shared required: {shared}", ""]
        for nid in sorted(ids):
            n = N[nid]; s = n["signature"]
            f = lambda L: ", ".join(f"{c['obs']}={c['value']}" for c in L) or "-"
            lines += [f"### {nid} ({n['kind']}, phase={n['phase']})", f"text: {n['name']}", f"supporting: {f(s['supporting'])} | excluding: {f(s['excluding'])}"]
            for ref in n["evidence"][:4]:
                q = t["quote_sources"].get(ref)
                if q:
                    lines.append(f"- quote {ref} @{int(q['start'])}s ({q['about']}): \"{q['text']}\"")
            lines.append("")
    (out / "t08_groups.md").write_text("\n".join(lines), encoding="utf-8")
    cat = load_json(Path(__file__).resolve().parent / "obs_catalog.json")["observables"]
    (out / "obs_list.txt").write_text("\n".join(f"{o['id']} [{o['layer']}/{o['camera']}]: {o['question']}  values: {', '.join(o['values'])}" for o in cat) + "\n", encoding="utf-8")
    (out / "t08_prompt.txt").write_text(PROMPT.format(d=str(out)), encoding="utf-8")
    print(len(pairs), "pairs;", len(comps), "groups ->", out)


if __name__ == "__main__":
    main()
