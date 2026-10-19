#!/usr/bin/env python3
"""V4 rerun consistency.

usage:
  check_rerun.py cards <runA.card.json> <runB.card.json>     # two independent stage-1 extractions
  check_rerun.py facts <facts1.json> <facts2.json> [...]      # >= 2 fact runs of the same clip
cards: compares what is anchored in the transcript (quote time buckets), not wording.
facts: per-observable agreement; writes nothing, prints the reliability line for each OBS.
"""
import sys
from collections import Counter, defaultdict

from common import CONFIG, Finding, fails, load_json

BUCKET = 30  # seconds


def _buckets(card, items):
    qt = {q["id"]: q["start"] for q in card["quotes"]}
    return {int(qt[e] // BUCKET) for it in items for e in it["evidence"] if e in qt}


def _jac(a, b):
    return len(a & b) / len(a | b) if a | b else 1.0


def cards(a, b):
    out = []
    th = CONFIG["rerun_min_jaccard"]
    if a["video"]["id"] != b["video"]["id"]:
        return [Finding("FAIL", "rerun", "different videos")]
    if a["extraction"]["stage1_prompt_version"] != b["extraction"]["stage1_prompt_version"]:
        out.append(Finding("FAIL", "rerun", "prompt versions differ; runs are not comparable"))
    if a["classification"]["is_teaching"] != b["classification"]["is_teaching"]:
        out.append(Finding("FAIL", "rerun", "is_teaching differs between runs"))
    ja = _jac(_buckets(a, a["causes"]), _buckets(b, b["causes"]))
    jd = _jac({(d["kind"], k) for d in a["solutions"] for k in _buckets(a, [d])},
              {(d["kind"], k) for d in b["solutions"] for k in _buckets(b, [d])})
    ea = {(min(_buckets(a, [c])) if _buckets(a, [c]) else -1) for c in a["causes"]}
    eb = {(min(_buckets(b, [c])) if _buckets(b, [c]) else -1) for c in b["causes"]}
    je = _jac(ea, eb)
    for name, val, key in (("causes", ja, "causes_nodes"), ("chain roots", je, "chain_edges"), ("solutions", jd, "solutions_kind_targets")):
        out.append(Finding("FAIL" if val < th[key] else "PASS", "rerun", f"{name} transcript-anchor Jaccard {val:.2f} (min {th[key]})"))
    pa, pb = a["problem"]["node_id"], b["problem"]["node_id"]
    if pa != pb and "PENDING" not in (pa, pb):
        out.append(Finding("FAIL", "rerun", f"problem node differs: {pa} vs {pb}"))
    return out


def facts(runs):
    out = []
    vals = defaultdict(list)
    for r in runs:
        for f in r["facts"]:
            vals[f["obs"]].append(f["value"])
    need = CONFIG["fact_min_agree"]
    num, den = map(int, need.split("/"))
    for obs in sorted(vals):
        v, n = Counter(vals[obs]).most_common(1)[0]
        ok = v != "unclear" and n / len(runs) >= num / den
        out.append(Finding("PASS" if ok else "WARN", obs, f"{v!r} {n}/{len(runs)} runs -> {'reliable' if ok else 'unreliable'}"))
    return out


def main(args):
    if len(args) < 3 or args[0] not in ("cards", "facts"):
        print(__doc__)
        return 2
    data = [load_json(p) for p in args[1:]]
    res = cards(*data[:2]) if args[0] == "cards" else facts(data)
    for f in res:
        print(f)
    return 1 if fails(res) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
