#!/usr/bin/env python3
"""Write blind-replay case files from a plan (deterministic; no model).

usage: make_blind_cases.py --plan plan.json --facts-root DIR --tree ftt_tree/tree_vN.json --out replay/muted_vN
plan = [{"episode","segment":[s,e],"quote":"Qxx","kind":"demonstrator_induced|demonstrator_natural|correct_demo"}]
expected = nodes of the tree whose evidence cites <episode>:<quote> (positive kinds), or ["NONE"] for correct_demo (negative control).
Copies each <facts-root>/<episode>_<s>-<e>/facts.json to <out>/facts/<episode>_<s>-<e>_facts.json (only if the facts schema is valid).
A case is skipped (and listed) when no node cites the quote or the facts file is missing.
"""
import argparse
import json
from pathlib import Path

from common import load_json, load_schema, validate_schema, fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True); ap.add_argument("--facts-root", required=True); ap.add_argument("--tree", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="v6")
    a = ap.parse_args()
    t = load_json(a.tree); plan = load_json(a.plan)
    cites = {}
    for n in t["nodes"]:
        for e in n["evidence"]:
            cites.setdefault(e, []).append(n["id"])
    out = Path(a.out); (out / "cases").mkdir(parents=True, exist_ok=True); (out / "facts").mkdir(parents=True, exist_ok=True)
    fs = load_schema("facts.schema.json")
    skipped = []
    for p in plan:
        s, e = p["segment"]; name = f"{p['episode']}_{s}-{e}"
        src = Path(a.facts_root) / name / "facts.json"
        if not src.exists():
            skipped.append((name, "no facts file")); continue
        facts = load_json(src)
        if fails(validate_schema(facts, fs)):
            skipped.append((name, "facts schema invalid")); continue
        neg = p["kind"] == "correct_demo"
        exp = ["NONE"] if neg else sorted(cites.get(f"{p['episode']}:{p['quote']}", []))
        if not exp:
            skipped.append((name, "no node cites the label quote")); continue
        (out / "facts" / f"{name}_facts.json").write_text(json.dumps(facts, ensure_ascii=False, indent=1))
        case = {"case_id": f"{a.prefix}_{name}", "episode": p["episode"], "segment": [s, e], "label_source": f"{p['episode']}:{p['quote']}",
                "label_kind": p["kind"], "expected": exp, "facts_path": f"../facts/{name}_facts.json"}
        (out / "cases" / f"{a.prefix}_{name}.json").write_text(json.dumps(case, ensure_ascii=False, indent=2) + "\n")
    print("cases written:", len(plan) - len(skipped), "skipped:", skipped)


if __name__ == "__main__":
    main()
