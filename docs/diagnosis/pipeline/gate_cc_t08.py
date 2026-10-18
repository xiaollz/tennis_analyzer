#!/usr/bin/env python3
"""Deterministic gate for a Claude Code review of T08 pairs (it only PROPOSES; nothing here edits a tree).

usage: gate_cc_t08.py --tree ftt_tree/tree_v5.json --raw cc_raw.txt --pairs t08_pairs.txt --out ftt_tree/t08_cc_review_v5.json
Gate per entry (all must hold for gate_pass=true):
  G1 pair equals the flagged pair (same order as pairs file)
  G2 every cited quote id is in the evidence of the node it is cited for (no invented / misplaced quotes)
  G3 verdict existing_obs: obs exists in the tree's observable list, vA and vB are legal and different, and both sides cite >= 1 quote
  G4 verdict new_obs_needed: question has no forbidden term (config neutral_prompt_forbidden), 2-3 values, camera in S/B, both sides cite >= 1 quote
  G5 proposals never enter the tree here: `enters_tree` is always false (new observables need the cross-episode rule or a user approval)
Also reports, per distinct proposed question, the set of episodes of the cited quotes (so the cross-episode rule can be read off).
"""
import argparse
import json
import re
from collections import defaultdict

from common import CONFIG, load_json


def forbidden_hit(text):
    for t in CONFIG["neutral_prompt_forbidden"]:
        pat = re.escape(t) if not t.isascii() else r"(?<![A-Za-z])" + re.escape(t) + r"(?![A-Za-z])"
        if re.search(pat, text, re.I):
            return t
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True); ap.add_argument("--raw", required=True); ap.add_argument("--pairs", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t = load_json(a.tree)
    N = {n["id"]: n for n in t["nodes"]}
    cat = {o["id"]: o for o in t["observables"]}
    pairs = [l.strip() for l in open(a.pairs, encoding="utf-8") if l.strip()]
    raw = re.sub(r"^```\w*\n|\n```\s*$", "", open(a.raw, encoding="utf-8").read().strip())
    J = json.loads(raw)
    out, props = [], defaultdict(lambda: {"pairs": [], "episodes": set(), "values": None, "camera": None})
    for i, j in enumerate(J):
        g = {}
        a_id, b_id = [x.strip() for x in j["pair"].split("|")]
        g["G1"] = i < len(pairs) and j["pair"].strip() == pairs[i]
        g["G2"] = all(q in N[a_id]["evidence"] for q in j["quotes_A"]) and all(q in N[b_id]["evidence"] for q in j["quotes_B"])
        v = j["verdict"]
        if v == "existing_obs":
            o = cat.get(j.get("obs"))
            g["G3"] = bool(o and j.get("vA") in o["values"] and j.get("vB") in o["values"] and j["vA"] != j["vB"] and j["quotes_A"] and j["quotes_B"])
        if v == "new_obs_needed":
            q = j.get("new_question") or ""
            g["G4"] = bool(q and not forbidden_hit(q + " " + " ".join(j.get("new_values") or [])) and 2 <= len(j.get("new_values") or []) <= 3
                           and str(j.get("camera", "")).strip()[:1].upper() in ("S", "B") and j["quotes_A"] and j["quotes_B"])
            p = props[q]
            p["pairs"].append(j["pair"]); p["values"] = j["new_values"]; p["camera"] = j["camera"]
            p["episodes"] |= {r.split(":")[0] for r in j["quotes_A"] + j["quotes_B"]}
        g["G5_enters_tree"] = False
        out.append({**j, "gate": g, "gate_pass": all(v_ for k, v_ in g.items() if k != "G5_enters_tree")})
    res = {"tree": a.tree, "n_pairs": len(out), "verdicts": {k: sum(o["verdict"] == k for o in out) for k in ("existing_obs", "same_mechanism", "not_visible", "new_obs_needed")},
           "gate_pass": sum(o["gate_pass"] for o in out), "gate_fail": [o["pair"] for o in out if not o["gate_pass"]],
           "proposed_new_observables": [{"question": q, "values": p["values"], "camera": p["camera"], "pairs": p["pairs"], "episodes_of_cited_quotes": sorted(p["episodes"])} for q, p in props.items()],
           "entries": out}
    open(a.out, "w", encoding="utf-8").write(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    print(res["verdicts"], "gate_pass", res["gate_pass"], "fail", res["gate_fail"][:5], "proposals", len(res["proposed_new_observables"]))


if __name__ == "__main__":
    main()
