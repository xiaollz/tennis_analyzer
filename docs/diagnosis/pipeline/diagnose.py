#!/usr/bin/env python3
"""Deterministic matcher: neutral facts + tree -> top-3 root causes with evidence and confidence.

usage: diagnose.py <tree.json> <facts.json>   (prints JSON)
No LLM here: identical inputs always give identical output, so all run-to-run
variation is attributable to fact extraction (measured by check_rerun.py).
"""
import json
import sys

from common import CONFIG, load_json

BREATH = "准备时鼻吸，启动前挥至触球时用嘴短促呼气。"


def _frac(s):
    a, b = s.split("/")
    return int(a) / int(b)


def usable_facts(tree, facts):
    """obs -> fact for facts that pass agreement, stroke-consistency and layer/camera rules."""
    obs = {o["id"]: o for o in tree["observables"]}
    min_agree = _frac(CONFIG["fact_min_agree"])
    used, dropped = {}, []
    for f in facts["facts"]:
        o = obs.get(f["obs"])
        why = None
        if not o:
            why = "unknown obs"
        elif f["value"] == "unclear":
            why = "unclear"
        elif o["layer"] == "C":
            why = "C-layer (feel) cannot come from video"
        elif _frac(f["agree"]) < min_agree:
            why = f"agree {f['agree']} below {CONFIG['fact_min_agree']}"
        elif o["layer"] == "B" and f["source"] == "model" and o["camera"] not in ("any", facts["camera"]):
            why = f"B-layer needs camera {o['camera']} or pose/manual"
        elif f.get("strokes") and _frac(f["strokes"]) < 0.8:
            why = f"only {f['strokes']} strokes (occasional, not a habit)"
        if why:
            dropped.append({"obs": f["obs"], "value": f["value"], "why": why})
        else:
            used[f["obs"]] = f
    return used, dropped


def diagnose(tree, facts):
    used, dropped = usable_facts(tree, facts)
    downstream = {}
    for e in tree["edges"]:
        downstream.setdefault(e["from"], set()).add(e["to"])
    cands, style = [], []
    for n in tree["nodes"]:
        sig = n["signature"]
        hit = lambda c: c["obs"] in used and used[c["obs"]]["value"] == c["value"]
        contra = lambda c: c["obs"] in used and used[c["obs"]]["value"] != c["value"]
        if any(hit(c) for c in sig["excluding"]) or any(contra(c) for c in sig["required"]):
            continue
        req_hit = [c for c in sig["required"] if hit(c)]
        sup_hit = [c for c in sig["supporting"] if hit(c)]
        missing = [c["obs"] for c in sig["required"] if c["obs"] not in used]
        full = bool(sig["required"]) and not missing
        if not req_hit and len(sup_hit) < 2:
            continue
        if n["kind"] == "style" or not n["must_fix"]:
            style.append(n["id"])
            continue
        matched = req_hit + sup_hit
        unanimous = [c for c in matched if _frac(used[c["obs"]]["agree"]) == 1.0]
        measured = [c for c in matched if used[c["obs"]]["source"] in ("pose", "manual")]
        if full and len(unanimous) >= 2 and measured:
            conf = "high"
        elif full:
            conf = "medium"
        else:
            conf = "low"
        cands.append({
            "node": n["id"], "name": n["name"], "confidence": conf, "full_match": full,
            "score": 2 * len(req_hit) + len(sup_hit),
            "evidence": [f"{c['obs']}={c['value']} ({used[c['obs']]['source']}, {used[c['obs']]['agree']})" for c in matched],
            "missing": missing,
            "falsify_if": [f"{c['obs']}={c['value']}" for c in sig["excluding"]] + [d["discriminator"] for d in n["differentials"]],
            "test10": n.get("test10", ""), "next_one_thing": n.get("next_one_thing", ""),
            "sources": n["evidence"][:3],
        })
    ids = {c["node"] for c in cands}
    for c in cands:  # FTT P3: prefer the upstream cause among co-occurring candidates
        c["upstream_of"] = sorted(downstream.get(c["node"], set()) & ids)
    order = {"high": 0, "medium": 1, "low": 2}
    cands.sort(key=lambda c: (order[c["confidence"]], -len(c["upstream_of"]), -c["score"], c["node"]))
    top = cands[:3]
    return {
        "clip_id": facts["clip_id"], "camera": facts["camera"],
        "top": top, "insufficient": not top,
        "style_not_fixed": style, "dropped_facts": dropped,
        "next_session": {
            "breath": BREATH,
            "one_thing": top[0]["next_one_thing"] if top and top[0]["confidence"] != "low" else "",
            "test10": top[0]["test10"] if top else "",
            "note": "low confidence: run the 10-ball test or re-shoot with the missing cameras before changing anything" if (not top or top[0]["confidence"] == "low") else "",
        },
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    print(json.dumps(diagnose(load_json(sys.argv[1]), load_json(sys.argv[2])), ensure_ascii=False, indent=1))
