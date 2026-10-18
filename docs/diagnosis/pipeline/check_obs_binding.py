#!/usr/bin/env python3
"""V3b OBS binding check (deterministic; no model): is the symptom/problem/cause -> observable mapping legal and honestly accounted for?

usage: check_obs_binding.py <obs_binding.json> <tree_v1.json> <tree_v2.json> [obs_catalog.json]
Rules
  O1  every v1 problem/cause node and every symptom has a binding item (cause nodes via their member card causes).
  O2  every (obs, value) is legal: OBS id exists in the catalog and the value is in that observable's value list.
  O3  no condition appears twice in one item; no contradictory 'required' (two values of the same observable / same sub-item).
  O4  signature not silently empty: an item with no required and no supporting must be visible=false (unobservable) or carry a NEW-OBS candidate.
  O5  required has <= 3 conditions; required never uses a C-layer observable.
  O6  every must_fix node of tree v2 has >= 1 required condition OR is an accounted gap (item visible=false, or NEW-OBS candidate on file).
      Silent gaps FAIL here. Accounted gaps are WARN here and still FAIL check_tree T05 (must_fix is never relaxed).
  O7  NEW-OBS candidates: >= 2 values, non-empty description; every candidate is in exactly one cluster; cluster rep is one of its members.
  O8  prompts + catalog contain no user conclusions / knowledge-base concepts (forbidden-term scan).
  O9  tree v2 observables == catalog (id, question, values, layer, camera); catalog has exactly the OBS-*.md files.
  O10 aggregation is reproducible: required == conditions required in ALL runs; supporting == other conditions of any run.
  O11 tree v2 node signatures are traceable: every condition comes from a binding item of one of the node's source nodes;
      symptom.obs_ref equals the rule in build_tree_v2 (V2-5).
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from common import DIAG_DIR, Finding, fails, load_json

FORBIDDEN = ["FTT", "Fault Tolerant", "HSA", "ESR", "FLT-", "屈书毅", "用户", "learning.md", "problem_register", "学员问题", "推肘", "降拍", "旋后", "伸肘", "BOOK:", "Tennis Science"]
FORBIDDEN_RE = [r"\buser('s)?\b", r"\bconclusion", r"\bour (project|notes|knowledge)", r"\bthe book\b"]
PROMPTS = ["obs_bind_v1.md", "mech_group_v1.md", "obs_newcluster_v1.md"]
PIPE = Path(__file__).resolve().parent


def sub(v):
    return v.split(":")[0] if re.match(r"^[abc]:", v) else ""


def check(bind, t1, t2, cat_doc=None, prompts_dir=None, obs_dir=None):
    out = []
    F = lambda rule, msg, level="FAIL": out.append(Finding(level, "obs_binding", f"{rule} {msg}"))
    cat_doc = cat_doc or load_json(PIPE / "obs_catalog.json")
    cat = {o["id"]: o for o in cat_doc["observables"]}
    items = bind["items"]
    # O1
    node_items = defaultdict(list)
    for k, it in items.items():
        if it["kind"] in ("problem", "cause"):
            node_items[it["node"]].append(k)
    for n in t1["nodes"]:
        if n["id"] not in node_items:
            F("O1", f"v1 node {n['id']} has no binding item")
    for s in t1["symptoms"]:
        if s["id"] not in items:
            F("O1", f"symptom {s['id']} has no binding item")
    known = {n["id"] for n in t1["nodes"]} | {s["id"] for s in t1["symptoms"]}
    for k, it in items.items():
        if it["kind"] != "cause" and k not in known:
            F("O1", f"binding item {k} is not a v1 node/symptom")
        if it["kind"] == "cause" and it["node"] not in known:
            F("O1", f"binding cause item {k} points to unknown node {it['node']}")
    cluster_of = defaultdict(int)
    for c in bind["new_obs_clusters"]:
        for m in c["members"]:
            cluster_of[m] += 1
    cand = {c["cid"]: c for c in bind["new_obs_candidates"]}
    for k, it in items.items():
        req, sup = it["required"], it["supporting"]
        for part, lst in (("required", req), ("supporting", sup)):
            for c in lst:
                o = cat.get(c["obs"])
                if not o:
                    F("O2", f"{k} {part} unknown observable {c['obs']}")
                elif c["value"] not in o["values"]:
                    F("O2", f"{k} {part} {c['obs']}={c['value']!r} not an allowed value")
                elif part == "required" and o["layer"] == "C":
                    F("O5", f"{k} required uses C-layer {c['obs']}")
        keys = [(c["obs"], c["value"]) for c in req + sup]
        if len(keys) != len(set(keys)):
            F("O3", f"{k} repeats a condition")
        seen = {}
        for c in req:
            g = (c["obs"], sub(c["value"]))
            if g in seen and seen[g] != c["value"]:
                F("O3", f"{k} contradictory required {c['obs']}: {seen[g]} vs {c['value']}")
            seen[g] = c["value"]
        if len(req) > 3:
            F("O5", f"{k} has {len(req)} required conditions (max 3)")
        if not req and not sup and it["visible"] is not False and not it["new_obs"]:
            F("O4", f"{k} has an empty signature without being marked unobservable or proposing a NEW-OBS candidate")
        # O10
        runs = it.get("runs", [])
        if runs:
            exp_req = [c for c in runs[0]["required"] if all(c in r["required"] for r in runs)]
            exp_sup = []
            for r in runs:
                for c in r["required"] + r["supporting"]:
                    if c not in exp_req and c not in exp_sup:
                        exp_sup.append(c)
            if exp_req != req or exp_sup != sup:
                F("O10", f"{k} aggregated signature differs from the rule applied to its runs")
    # O7
    for c in bind["new_obs_candidates"]:
        if not c["description"].strip() or len(c["values"]) < 2:
            F("O7", f"candidate {c['cid']} has an empty description or < 2 values")
        if cluster_of[c["cid"]] != 1:
            F("O7", f"candidate {c['cid']} is in {cluster_of[c['cid']]} clusters (needs exactly 1)")
    for c in bind["new_obs_clusters"]:
        if c["rep"] not in c["members"] or any(m not in cand for m in c["members"]):
            F("O7", f"cluster {c['id']} rep/members invalid")
    # O6 + O11 on tree v2
    if t2:
        node_of = t2["build"]["mechanism_merge"]["node_map_v1_to_v2"]
        rep = lambda i: node_of.get(i, i)
        src = defaultdict(list)  # v2 node -> binding items of every v1 source node
        for it in items.values():
            if it["kind"] in ("problem", "cause"):
                src[rep(it["node"])].append(it)
        gaps = {"visible_false": 0, "new_obs_candidate": 0}
        for n in t2["nodes"]:
            its = src.get(n["id"], [])
            if not its:
                F("O11", f"{n['id']} has no binding items")
                continue
            allc = {(c["obs"], c["value"]) for it in its for c in it["required"] + it["supporting"]}
            reqc = {(c["obs"], c["value"]) for it in its for c in it["required"]}
            for part in ("required", "supporting", "excluding"):
                for c in n["signature"][part]:
                    k = (c["obs"], c["value"])
                    if part == "required" and k not in reqc:
                        F("O11", f"{n['id']} required {k} is not required by any source item")
                    if part != "required" and k not in allc:
                        F("O11", f"{n['id']} {part} {k} is not named by any source item")
            if n["must_fix"] and n["kind"] != "style" and not n["signature"]["required"]:
                if any(it["visible"] is False for it in its) and not any(it["new_obs"] for it in its):
                    gaps["visible_false"] += 1
                elif any(it["new_obs"] for it in its):
                    gaps["new_obs_candidate"] += 1
                else:
                    F("O6", f"{n['id']} is must_fix with no required observable and no NEW-OBS / unobservable account")
        F("O6", f"accounted gaps (still FAIL in check_tree T05): {gaps}", "WARN")
        for s in t2["symptoms"]:
            it = items.get(s["id"])
            if it:
                exp = it["required"][0]["obs"] if it["required"] else (it["supporting"][0]["obs"] if it["supporting"] else ("NEW-OBS" if it["visible"] and it["new_obs"] else "NONE"))
                if s["obs_ref"] != exp:
                    F("O11", f"symptom {s['id']} obs_ref {s['obs_ref']} != binding-derived {exp}")
        # O9
        obs2 = {o["id"]: o for o in t2["observables"]}
        if set(obs2) != set(cat):
            F("O9", "tree v2 observables ids differ from catalog")
        for i, o in obs2.items():
            c = cat.get(i)
            if c and any(o[f] != c[f] for f in ("question", "values", "layer", "camera")):
                F("O9", f"tree v2 {i} differs from catalog")
    obs_dir = Path(obs_dir or DIAG_DIR / "observables")
    if obs_dir.exists():
        files = {p.name.split("_")[0] for p in obs_dir.glob("OBS-*.md")}
        if files != set(cat):
            F("O9", f"catalog ids {sorted(set(cat) ^ files)} not matching observables/OBS-*.md")
    # O8
    pdir = Path(prompts_dir or PIPE / "prompts")
    texts = {"obs_catalog.json": json.dumps([{k: o[k] for k in ("question", "values")} for o in cat_doc["observables"]], ensure_ascii=False)}
    for p in PROMPTS:
        if (pdir / p).exists():
            texts[p] = (pdir / p).read_text(encoding="utf-8")
    for name, text in texts.items():
        if "\n---\n" in text:
            text = text.split("\n---\n", 1)[1]
        for t in FORBIDDEN:
            if t in text:
                F("O8", f"{name} contains forbidden term {t!r}")
        for pat in FORBIDDEN_RE:
            m = re.search(pat, text, re.I)
            if m:
                F("O8", f"{name} contains forbidden wording {m.group(0)!r}")
    return out


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    cat = load_json(argv[3]) if len(argv) > 3 else None
    f = check(load_json(argv[0]), load_json(argv[1]), load_json(argv[2]), cat)
    for x in f:
        print(x)
    print(f"check_obs_binding: {len(fails(f))} FAIL")
    return 1 if fails(f) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
