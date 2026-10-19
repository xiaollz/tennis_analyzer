#!/usr/bin/env python3
"""Apply gated existing_obs verdicts from a T08 CC review as node differentials (deterministic; no model).

usage: apply_t08_differentials.py --tree ftt_tree/tree_vN.json --review ftt_tree/t08_cc_review_vN.json --out ftt_tree/tree_vN.json
Only entries with verdict=existing_obs AND gate_pass=true are applied. discriminator = "OBS-xx=value" (the value that
distinguishes the cited side). Bidirectional. Skips pairs whose required signatures already differ, or whose nodes
are missing. Does not invent OBS, does not change must_fix / required / supporting.
"""
import argparse
import json
from pathlib import Path

from common import load_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True); ap.add_argument("--review", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t = load_json(a.tree); rev = load_json(a.review)
    N = {n["id"]: n for n in t["nodes"]}
    applied, skipped = [], []
    for e in rev["entries"]:
        if e["verdict"] != "existing_obs" or not e.get("gate_pass"):
            continue
        a_id, b_id = [x.strip() for x in e["pair"].split("|")]
        if a_id not in N or b_id not in N:
            skipped.append({"pair": e["pair"], "why": "node missing"}); continue
        ra = {(c["obs"], c["value"]) for c in N[a_id]["signature"]["required"]}
        rb = {(c["obs"], c["value"]) for c in N[b_id]["signature"]["required"]}
        if ra != rb:
            skipped.append({"pair": e["pair"], "why": "required signatures already differ"}); continue
        obs, vA, vB = e["obs"], e["vA"], e["vB"]
        da = {d["vs"] for d in N[a_id]["differentials"]}
        db = {d["vs"] for d in N[b_id]["differentials"]}
        if b_id not in da:
            N[a_id]["differentials"].append({"vs": b_id, "discriminator": f"{obs}={vA}"})
        if a_id not in db:
            N[b_id]["differentials"].append({"vs": a_id, "discriminator": f"{obs}={vB}"})
        applied.append({"pair": e["pair"], "obs": obs, "vA": vA, "vB": vB, "quotes_A": e["quotes_A"], "quotes_B": e["quotes_B"]})
    t.setdefault("build", {}).setdefault("t08_differentials_from_cc", []).extend(applied)
    Path(a.out).write_text(json.dumps(t, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"applied": len(applied), "skipped": skipped, "detail": applied}, ensure_ascii=False))


if __name__ == "__main__":
    main()
