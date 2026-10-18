#!/usr/bin/env python3
"""Update loop/manifest.json for a finished Stage-1 round from the episode cards + raw/<id>/rerun_adjudication.json (no judgement).

usage: s1_manifest.py <round_name> <prompt_version> ID [ID ...]
verified (card status stage1_verified) -> state stage1_verified; draft -> state stays subs_ready with the usual 'must not enter the tree' note.
"""
import json
import sys
from pathlib import Path

from common import load_json

DIAG = Path(__file__).resolve().parent.parent
EP, MAN = DIAG / "episodes", DIAG / "loop" / "manifest.json"


def note(vid, prompt, card, adj):
    c = card
    if c["status"] == "stage1_verified":
        cp = adj.get("chosen_pair")
        how = ("raw pair %s-%s" % tuple(cp) if cp and cp[0] != "consensus" else "3-run consensus vs %s" % cp[1]) if cp else "?"
        n = len(adj.get("runs", []))
        return f"{prompt} verified via {how} ({n} runs); {len(c['causes'])} causes / {len(c['solutions'])} solutions"
    return f"{prompt}: all raw pairs and 3-run consensus fail check_rerun; kept stage1_draft, must not enter the tree"


def main(a):
    rnd, prompt, ids = a[0], a[1], a[2:]
    man = load_json(MAN)
    byid = {v["id"]: v for v in man["videos"]}
    out = {"stage1_verified": [], "stage1_draft_rerun_fail": []}
    for vid in ids:
        card = load_json(EP / f"{vid}.card.json")
        adj = load_json(EP / "raw" / vid / "rerun_adjudication.json")
        v = byid[vid]
        v["state"] = "stage1_verified" if card["status"] == "stage1_verified" else "subs_ready"
        v["updated"] = "2026-10-18"
        v["round"] = rnd
        v["note"] = note(vid, prompt, card, adj)
        out["stage1_verified" if card["status"] == "stage1_verified" else "stage1_draft_rerun_fail"].append(vid)
    man["round"] = rnd
    man[f"{rnd}_selection"] = {"rule": "same as round6 (type_guess=教学长视频 + .en.vtt + no card + holdout=False + duration>=400s; technique titles first, tactics/psychology after; upload_date ascending); PwkGLkRm73o first by user request; holdout episodes are never extracted",
                               "prompt_version": prompt, "ids": ids, "outcome": out}
    MAN.write_text(json.dumps(man, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out))


if __name__ == "__main__":
    main(sys.argv[1:])
