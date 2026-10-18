#!/usr/bin/env python3
"""One command that decides PASS/FAIL for the whole loop. Agents report this output, not their own summary.

usage:
  verify_all.py --selftest
  verify_all.py --episodes DIR --tree tree.json --manifest manifest.json [--cases DIR]
                [--prompts rendered_prompt ...] [--transcript-root DIR] [--gate G1|G2] [--report out.json]

The self-test always runs first: it checks that the good fixtures pass AND that every
mutation in MUTATIONS is caught by the rule it targets ("test the tester").
"""
import argparse
import copy
import datetime
import json
import shutil
import sys
import tempfile
from pathlib import Path

import blind_replay
import check_manifest
import check_neutrality
import check_quotes
import check_rerun
import check_tree
import validate_cards
from common import HERE, fails, load_json, sha256_file
from diagnose import diagnose

FX = HERE / "fixtures"


def _card():
    return load_json(FX / "episodes" / "SYNTHETIC01.card.json")


def _tree():
    return load_json(FX / "tree.json")


def card_mut(fn):
    def run():
        c = _card()
        fn(c)
        where = "SYNTHETIC01.card.json"
        f = validate_cards.check_card(c, where)
        if not fails(f):
            f += check_quotes.check_card(c, where, FX / "subs")[0]
        return f
    return run


def tree_mut(fn):
    def run():
        t = _tree()
        fn(t)
        return check_tree.check(t, FX / "episodes")
    return run


def tree_mut_draft():
    with tempfile.TemporaryDirectory() as d:
        c = _card()
        c["status"] = "stage1_draft"
        (Path(d) / "SYNTHETIC01.card.json").write_text(json.dumps(c))
        return check_tree.check(_tree(), d)


def manifest_mut(fn, with_transcripts=True, empty_transcripts=False):
    def run():
        m = load_json(FX / "manifest.json")
        fn(m)
        with tempfile.TemporaryDirectory() as d:
            troot = d if empty_transcripts else (FX / "subs" if with_transcripts else None)
            return check_manifest.check(m, FX / "episodes", _tree(), troot)
    return run


def _vid(m, vid):
    return next(v for v in m["videos"] if v["id"] == vid)


def replay_mut(edit_case=None, edit_facts=None):
    def run():
        with tempfile.TemporaryDirectory() as d:
            shutil.copytree(FX / "replay", Path(d) / "replay")
            shutil.copytree(FX / "facts", Path(d) / "facts")
            for p in (Path(d) / "replay").glob("*.json"):
                case = load_json(p)
                if edit_case:
                    edit_case(case)
                p.write_text(json.dumps(case))
            for p in (Path(d) / "facts").glob("*.json"):
                fx = load_json(p)
                if edit_facts:
                    edit_facts(fx)
                p.write_text(json.dumps(fx))
            cards_dir = FX / "episodes"
            if edit_case and getattr(edit_case, "card_edit", None):
                cards_dir = Path(d) / "episodes"
                cards_dir.mkdir()
                c = _card()
                edit_case.card_edit(c)
                (cards_dir / "SYNTHETIC01.card.json").write_text(json.dumps(c))
            return blind_replay.run(_tree(), Path(d) / "replay", cards_dir)[0]
    return run


def neutrality_mut():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "facts_rendered.md"
        p.write_text("Watch the clip. Q1: is there lag behind the hand?", encoding="utf-8")
        return check_neutrality.check(p)


def rerun_mut():
    a, b = _card(), _card()
    for q in b["quotes"]:
        q["start"] = q["start"] + 300  # every anchor moves to another part of the video
    return check_rerun.cards(a, b)


def _general_label(case):
    pass


def _label_general_card(c):
    for q in c["quotes"]:
        if q["id"] == "Q04":
            q["about"] = "general"


_general_label.card_edit = _label_general_card


def _add_cycle(t):
    t["edges"].append({"from": "N-JAMMED", "to": "N-NO_ADJUST_STEPS", "strength": "single_episode", "evidence": ["SYNTHETIC01:Q02"]})


MUTATIONS = [
    ("fabricated quote", "Q04", card_mut(lambda c: c["quotes"][0].update(text="players should rotate the shoulders before the hips every single time"))),
    ("shifted timestamp", "Q03", card_mut(lambda c: c["quotes"][1].update(start=51))),
    ("transcript changed", "Q01", card_mut(lambda c: c["transcript"].update(sha256="0" * 64))),
    ("cause without subtitle quote", "R06", card_mut(lambda c: c["causes"][0].update(evidence=["BOOK:p116"]))),
    ("dangling evidence", "R04", card_mut(lambda c: c["symptoms"][0].update(evidence=["Q99"]))),
    ("depth not decreasing", "R07", card_mut(lambda c: c["causes"][1].update(depth=1))),
    ("feel labelled visible layer", "R08", card_mut(lambda c: c["symptoms"][3].update(layer="A"))),
    ("banned cue unflagged", "R09", card_mut(lambda c: c["solutions"][0].update(summary="主动推肘向前"))),
    ("user context leaked", "not in [False]", card_mut(lambda c: c["extraction"].update(user_context_in_prompt=True))),
    ("video item before stage 2", "R11", card_mut(lambda c: c["video_layer"].append({"id": "V01", "kind": "visual_detail", "target": "S01", "content": "elbow on ribs", "start": 25, "runs_agree": "2/2", "status": "accepted"}))),
    ("tree cites holdout", "T02", tree_mut(lambda t: t["nodes"][0]["evidence"].append("SYNTHETIC02:Q01"))),
    ("feel as required obs", "T04", tree_mut(lambda t: t["nodes"][0]["signature"]["required"].append({"obs": "OBS-20", "value": "high"}))),
    ("unconstrained must_fix node", "T05", tree_mut(lambda t: t["nodes"][1]["signature"].update(required=[]))),
    ("banned next_one_thing", "T06", tree_mut(lambda t: t["nodes"][0].update(next_one_thing="触球时伸直手臂"))),
    ("indistinguishable nodes", "T08", tree_mut(lambda t: [t["nodes"][1]["signature"].update(required=[{"obs": "OBS-02", "value": "touching"}]), t["nodes"][0].update(differentials=[]), t["nodes"][1].update(differentials=[])])),
    ("core edge from one episode", "T10", tree_mut(lambda t: t["edges"][0].update(strength="core"))),
    ("cycle", "T11", tree_mut(_add_cycle)),
    ("tree cites unverified card", "T03", tree_mut_draft),
    ("unmuted blind clip", "B02", replay_mut(edit_facts=lambda f: f.update(audio_stripped=False))),
    ("label not about demonstrator", "B01", replay_mut(edit_case=_general_label)),
    ("project term in fact prompt", "N01", neutrality_mut),
    ("rerun anchors disagree", "Jaccard", rerun_mut),
    ("manifest duplicate entry", "M01", manifest_mut(lambda m: m["videos"].append(dict(m["videos"][0])))),
    ("manifest unknown state", "M02", manifest_mut(lambda m: _vid(m, "SYNTHETIC01").update(state="done"))),
    ("non_teaching without reason", "M03", manifest_mut(lambda m: _vid(m, "SYNTHETIC03").pop("reason"))),
    ("state claims transcript that is missing", "M04", manifest_mut(lambda m: None, empty_transcripts=True)),
    ("merged claimed but card only stage1_verified", "M05", manifest_mut(lambda m: _vid(m, "SYNTHETIC01").update(state="merged"))),
    ("stage1_verified claimed but card missing", "M05", manifest_mut(lambda m: m["videos"].append({"id": "SYNTHETIC09", "title": "x", "duration_s": 60, "state": "stage1_verified", "updated": "2026-10-18"}))),
    ("merged but no tree node cites it", "M06", manifest_mut(lambda m: m["videos"].append({"id": "SYNTHETIC09", "title": "x", "duration_s": 60, "state": "merged", "updated": "2026-10-18"}))),
    ("card exists but episode not in manifest", "M07", manifest_mut(lambda m: m["videos"].remove(_vid(m, "SYNTHETIC01")))),
]


def selftest():
    report = {"good": {}, "mutations": []}
    ok = True
    good = {
        "validate_cards": validate_cards.check_card(_card(), "SYNTHETIC01.card.json"),
        "check_quotes": check_quotes.check_card(_card(), "SYNTHETIC01.card.json", FX / "subs")[0],
        "check_tree": check_tree.check(_tree(), FX / "episodes"),
        "check_manifest": check_manifest.check(load_json(FX / "manifest.json"), FX / "episodes", _tree(), FX / "subs"),
        "check_rerun": check_rerun.cards(_card(), _card()),
        "check_neutrality": [f for p in sorted((HERE / "prompts").glob("*.md")) for f in check_neutrality.check(p)],
    }
    bf, rows = blind_replay.run(_tree(), FX / "replay", FX / "episodes")
    good["blind_replay"] = bf
    m = blind_replay.metrics(rows)
    for k, f in good.items():
        report["good"][k] = [str(x) for x in fails(f)]
        if fails(f):
            ok = False
            print(f"SELFTEST good fixture FAILED in {k}:", *fails(f), sep="\n  ")
    if not (m["recall_at_3"] == 1.0 and m["negative_high_conf"] == 0.0 and m["rerun_top1_stable"] == 1.0):
        ok = False
        print("SELFTEST replay metrics wrong:", m)
    d1 = diagnose(_tree(), load_json(FX / "facts" / "pos_run1.json"))
    d2 = diagnose(_tree(), load_json(FX / "facts" / "pos_run1.json"))
    if d1 != d2 or [c["node"] for c in d1["top"]][:1] != ["N-NO_ADJUST_STEPS"] or "N-ELBOW_STYLE" not in d1["style_not_fixed"]:
        ok = False
        print("SELFTEST diagnose not deterministic or wrong ranking:", d1["top"])
    if "OBS-20" not in {x["obs"] for x in d1["dropped_facts"]}:
        ok = False
        print("SELFTEST diagnose used a C-layer fact")
    for name, code, fn in MUTATIONS:
        caught = [str(f) for f in fails(fn()) if code in f.msg]
        report["mutations"].append({"name": name, "expect": code, "caught": bool(caught)})
        if not caught:
            ok = False
            print(f"SELFTEST mutation NOT caught: {name} (expected {code})")
    print(f"selftest: good fixtures + {len(MUTATIONS)} mutations -> {'PASS' if ok else 'FAIL'}")
    report["pass"] = ok
    return ok, report


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--episodes"); ap.add_argument("--tree"); ap.add_argument("--manifest")
    ap.add_argument("--cases"); ap.add_argument("--prompts", nargs="*", default=[])
    ap.add_argument("--transcript-root"); ap.add_argument("--gate"); ap.add_argument("--report")
    a = ap.parse_args(argv)
    st_ok, st = selftest()
    report = {"time": datetime.datetime.now().isoformat(timespec="seconds"), "selftest": st, "checks": {}}
    ok = st_ok
    if not a.selftest:
        cards = sorted(Path(a.episodes).glob("*.card.json"))
        res = {"validate_cards": [], "check_quotes": []}
        for p in cards:
            c = load_json(p)
            res["validate_cards"] += validate_cards.check_card(c, str(p))
            res["check_quotes"] += check_quotes.check_card(c, str(p), a.transcript_root)[0]
        tree = load_json(a.tree)
        res["check_tree"] = check_tree.check(tree, a.episodes, a.tree)
        res["check_manifest"] = check_manifest.check(load_json(a.manifest), a.episodes, tree, a.transcript_root)
        res["check_neutrality"] = [f for p in a.prompts for f in check_neutrality.check(p)]
        if a.cases:
            bf, rows = blind_replay.run(tree, a.cases, a.episodes)
            m = blind_replay.metrics(rows)
            report["replay_metrics"] = m
            res["blind_replay"] = bf + (blind_replay.gate(m, a.gate) if a.gate else [])
        for k, f in res.items():
            report["checks"][k] = {"fail": [str(x) for x in fails(f)], "warn": [str(x) for x in f if x.level == "WARN"]}
            ok = ok and not fails(f)
            print(f"{k}: {len(fails(f))} FAIL, {sum(x.level == 'WARN' for x in f)} WARN")
            for x in fails(f)[:20]:
                print("  ", x)
        report["inputs"] = {"cards": len(cards), "tree_sha256": sha256_file(a.tree)}
    report["pass"] = ok
    if a.report:
        Path(a.report).write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print("VERIFY:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
