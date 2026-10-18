#!/usr/bin/env python3
"""V7 loop manifest vs real artifacts (claims must match files, not narration).

usage: check_manifest.py <manifest.json> <episodes_dir> <tree.json|-> [--transcript-root DIR]
manifest: {"videos": [{"id","title","duration_s","state","reason"?,"updated"}]}
states (ordered): listed < subs_ready < stage1_verified < stage2_verified < merged ; or non_teaching / blocked
"""
import argparse
import sys
from pathlib import Path

from common import Finding, fails, load_json

ORDER = ["listed", "subs_ready", "stage1_verified", "stage2_verified", "merged"]
SIDE = ["non_teaching", "blocked"]
CARD_RANK = {"stage1_draft": 1, "stage1_verified": 2, "stage2_draft": 2, "stage2_verified": 3}


def check(man, ep_dir, tree, troot=None):
    out = []
    cards = {p.name.split(".")[0]: load_json(p) for p in Path(ep_dir).glob("*.card.json")}
    cited = set()
    holdout = set()
    if tree:
        holdout = set(tree["holdout_episodes"])
        for n in tree["nodes"]:
            cited |= {r.split(":")[0] for r in n["evidence"] if not r.startswith("BOOK:")}
    seen = set()
    for v in man["videos"]:
        vid, st = v["id"], v["state"]
        w = f"manifest:{vid}"
        if vid in seen:
            out.append(Finding("FAIL", w, "M01 duplicate entry"))
        seen.add(vid)
        if st not in ORDER + SIDE:
            out.append(Finding("FAIL", w, f"M02 unknown state {st}"))
            continue
        if st in SIDE:
            if not v.get("reason"):
                out.append(Finding("FAIL", w, f"M03 state {st} needs a reason"))
            if st == "non_teaching" and vid in cards:
                out.append(Finding("FAIL", w, "M03 non_teaching but a card exists"))
            continue
        rank = ORDER.index(st)
        if rank >= 1 and troot and not list(Path(troot).glob(f"{vid}.*")):
            out.append(Finding("FAIL", w, f"M04 state {st} but no transcript in {troot}"))
        if rank >= 2:
            c = cards.get(vid)
            need = 2 if st == "stage1_verified" else 3
            if not c or CARD_RANK.get(c["status"], 0) < need:
                out.append(Finding("FAIL", w, f"M05 state {st} but card is {c['status'] if c else 'missing'}"))
        if st == "merged" and vid not in cited and vid not in holdout:
            out.append(Finding("FAIL", w, "M06 state merged but no tree node cites this episode"))
    for vid in cards:
        if vid not in seen:
            out.append(Finding("FAIL", f"card:{vid}", "M07 card exists but episode missing from manifest"))
    done = sum(v["state"] in ("merged", "non_teaching") for v in man["videos"])
    out.append(Finding("INFO", "manifest", f"{done}/{len(man['videos'])} videos finished (merged or non_teaching)"))
    return out


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest"); ap.add_argument("episodes"); ap.add_argument("tree")
    ap.add_argument("--transcript-root")
    a = ap.parse_args(argv)
    tree = None if a.tree == "-" else load_json(a.tree)
    res = check(load_json(a.manifest), a.episodes, tree, a.transcript_root)
    for f in res:
        print(f)
    return 1 if fails(res) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
