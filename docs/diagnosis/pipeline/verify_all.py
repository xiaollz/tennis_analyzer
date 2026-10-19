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
import check_mech_merge
import check_obs_binding
import check_neutrality
import check_quotes
import check_rerun
import check_tree
import validate_cards
from common import DIAG_DIR, HERE, fails, load_json, sha256_file
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



def _qsrc(t, wrong=False):
    cards = _card()
    qs = {}
    for r in [r for n in t["nodes"] for r in n["evidence"]] + [r for e in t["edges"] for r in e["evidence"]]:
        if not r.startswith("BOOK:"):
            ep, q = r.split(":")
            cq = next(x for x in cards["quotes"] if x["id"] == q)
            qs[r] = {"video_id": ep, "start": cq["start"], "about": cq["about"], "text": cq["text"]}
    if wrong:
        k = sorted(qs)[0]
        qs[k]["text"] = qs[k]["text"] + " (reworded)"
    t["quote_sources"] = qs


def _t13_missing(t):
    _qsrc(t)
    t["quote_sources"].pop(sorted(t["quote_sources"])[0])


def _t13_unknown_link(t):
    _qsrc(t)
    n0 = t["nodes"][0]
    t["solutions"] = [{"id": "SOL_X_D01", "episode": "SYNTHETIC01", "problem": "N-NOPE", "kind": "drill", "summary": "x",
                       "targets_causes": [], "targets_symptoms": [], "ban_conflict": False, "usable_as_prescription": True,
                       "evidence": [n0["evidence"][0]]}]


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
    # Path A: empty-req problem is still T05 when no edge-linked cause has required
    ("problem without signed cause", "T05", tree_mut(lambda t: (t["nodes"][0]["signature"].update(required=[]), t["nodes"][1]["signature"].update(required=[])))),
    ("banned next_one_thing", "T06", tree_mut(lambda t: t["nodes"][0].update(next_one_thing="触球时伸直手臂"))),
    ("indistinguishable nodes", "T08", tree_mut(lambda t: [t["nodes"][1]["signature"].update(required=[{"obs": "OBS-02", "value": "touching"}]), t["nodes"][0].update(differentials=[]), t["nodes"][1].update(differentials=[])])),
    ("core edge from one episode", "T10", tree_mut(lambda t: t["edges"][0].update(strength="core"))),
    ("cycle", "T11", tree_mut(_add_cycle)),
    ("tree cites unverified card", "T03", tree_mut_draft),
    ("cited quote has no quote_sources entry", "T13", tree_mut(_t13_missing)),
    ("quote_sources text differs from card", "T13", tree_mut(lambda t: _qsrc(t, wrong=True))),
    ("solution links unknown node", "T13", tree_mut(_t13_unknown_link)),
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


# ---- tree v2 artifacts (OBS binding + mechanism merge): mutations run on in-memory copies of the committed files
# selftest data = the CURRENT artifacts (v4, built against the current obs_catalog.json). v2 artifacts (18-observable catalog) stay in the repo as history.
V2 = {"t1": DIAG_DIR / "ftt_tree" / "tree_base_v9.json", "t2": DIAG_DIR / "ftt_tree" / "tree_v9.json", "bind": DIAG_DIR / "ftt_tree" / "obs_binding_v9.json",
      "mm": DIAG_DIR / "ftt_tree" / "mech_merge_v9.json", "eps": DIAG_DIR / "episodes"}


def v2_available():
    return all(V2[k].exists() for k in ("t1", "t2", "bind", "mm"))


def _v2():
    return {k: load_json(V2[k]) for k in ("t1", "t2", "bind", "mm")}


def ob_mut(fn, prompt_edit=None):
    def run():
        d = _v2()
        fn(d)
        pdir = None
        if prompt_edit:
            pdir = Path(tempfile.mkdtemp())
            for p in (HERE / "prompts").glob("*.md"):
                (pdir / p.name).write_text(p.read_text(encoding="utf-8"))
            prompt_edit(pdir)
        return check_obs_binding.check(d["bind"], d["t1"], d["t2"], prompts_dir=pdir)
    return run


def mm_mut(fn):
    def run():
        d = _v2()
        fn(d)
        return check_mech_merge.check(d["mm"], d["t1"], d["t2"], V2["eps"])
    return run


def _first(d, pred):
    return next(k for k, v in d["bind"]["items"].items() if pred(v))


def _m_unknown_obs(d):
    k = _first(d, lambda v: v["supporting"])
    d["bind"]["items"][k]["supporting"][0]["obs"] = "OBS-99"


def _m_bad_value(d):
    k = _first(d, lambda v: v["required"])
    d["bind"]["items"][k]["required"][0]["value"] = "not_a_value"


def _m_four_required(d):
    k = _first(d, lambda v: v["required"])
    d["bind"]["items"][k]["required"] = [{"obs": "OBS-01", "value": "clearly_bent"}, {"obs": "OBS-04", "value": "no_pause"}, {"obs": "OBS-05", "value": "no_steps_stands_still"}, {"obs": "OBS-06", "value": "open"}]


def _m_contradict(d):
    k = _first(d, lambda v: v["required"])
    d["bind"]["items"][k]["required"] = [{"obs": "OBS-18", "value": "arm_first"}, {"obs": "OBS-18", "value": "hips_first_then_trunk_and_arm"}]


def _m_silent_empty(d):
    k = _first(d, lambda v: v["kind"] == "cause" and v["visible"] and v["required"])
    it = d["bind"]["items"][k]
    it.update(required=[], supporting=[], new_obs=[], runs=[])


def _m_silent_must_fix(d):
    nid = next(n["id"] for n in d["t2"]["nodes"] if n["signature"]["required"] and n["kind"] == "cause")
    n2 = next(n for n in d["t2"]["nodes"] if n["id"] == nid)
    n2["signature"]["required"] = []
    for it in d["bind"]["items"].values():
        if it["kind"] == "cause" and d["t2"]["build"]["mechanism_merge"]["node_map_v1_to_v2"].get(it["node"], it["node"]) == nid:
            it.update(visible=True, new_obs=[])


def _m_orphan_candidate(d):
    d["bind"]["new_obs_clusters"][0]["members"] = d["bind"]["new_obs_clusters"][0]["members"][1:] or ["NC999"]
    d["bind"]["new_obs_clusters"][0]["rep"] = d["bind"]["new_obs_clusters"][0]["members"][0]


def _m_untraced_required(d):
    n = next(n for n in d["t2"]["nodes"] if n["kind"] == "cause" and n["signature"]["required"])
    n["signature"]["required"].append({"obs": "OBS-17", "value": "pointing_up"})


def _m_agg_wrong(d):
    k = _first(d, lambda v: v["required"] and v["runs"])
    d["bind"]["items"][k]["required"] = []


def _m_missing_item(d):
    d["bind"]["items"].pop(next(k for k, v in d["bind"]["items"].items() if v["kind"] == "symptom"))


def _m_catalog_drift(d):
    d["t2"]["observables"][0]["values"] = ["x", "y"]


def _p_leak(pdir):
    p = pdir / "obs_bind_v1.md"
    p.write_text(p.read_text(encoding="utf-8").replace("RULES_PLACEHOLDER", "") + "\nNote: the user's latest conclusion is that the elbow must not be pushed.\n", encoding="utf-8")


def _p_leak2(pdir):
    p = pdir / "mech_group_v1.md"
    p.write_text(p.read_text(encoding="utf-8") + "\nAccording to FLT-01 and HSA, group these.\n", encoding="utf-8")


def _g_other_stroke(d):
    """Swap first member of group 0 to an existing backhand cause node of t1 (M02: mixed strokes).

    Picks any t1 cause whose source card is classified backhand. Hard-coding a node id broke when that
    cause was lexically merged out of a later skeleton (e.g. LWS7:C03 into DX8A:C02 in tree_base_v7).
    """
    m = d["mm"]["groups"][0]["members"][0]
    back = None
    for n in d["t1"]["nodes"]:
        if n["kind"] != "cause":
            continue
        ep = n["evidence"][0].split(":")[0]
        card = load_json(DIAG_DIR / "episodes" / f"{ep}.card.json")
        if card["classification"]["stroke"] != "backhand":
            continue
        cid = next((x["id"] for x in card["causes"] if x["statement"] == n["name"]), None)
        if cid:
            back = (n["id"], ep, cid)
            break
    if back is None:
        raise RuntimeError("no backhand cause node in t1 for M02 mutation")
    nid, ep, cid = back
    m["node"], m["card_causes"], m["episodes"] = nid, [f"{ep}:{cid}"], [ep]


def _g_same_video(d):
    g = d["mm"]["groups"][0]
    g["members"].append(dict(g["members"][0], node="N-C_2NN9TYMECZU_C01", card_causes=["2Nn9tyMECZU:C01"]))


def _g_text_rewrite(d):
    d["mm"]["groups"][0]["members"][0]["text"] += " (rewritten)"


def _g_v2_name(d):
    g = d["mm"]["groups"][0]
    next(n for n in d["t2"]["nodes"] if n["id"] == g["rep"])["name"] = "A reworded statement."


def _g_lose_evidence(d):
    g = d["mm"]["groups"][0]
    n = next(n for n in d["t2"]["nodes"] if n["id"] == g["rep"])
    n["evidence"] = n["evidence"][:1]


def _g_lose_edge(d):
    d["t2"]["edges"].pop(0)


def _g_cycle(d):
    e = d["t2"]["edges"][0]
    d["t2"]["edges"].append({"from": e["to"], "to": e["from"], "strength": "single_episode", "evidence": e["evidence"]})


def _g_beyond_runs(d):
    g = d["mm"]["groups"][0]
    d["mm"]["runs_cleaned_partitions"][g["stroke"]]["run1"] = []


def _g_two_groups(d):
    d["mm"]["groups"][1]["members"].append(dict(d["mm"]["groups"][0]["members"][0]))


def _g_bad_rep(d):
    d["mm"]["groups"][0]["rep"] = "N-C_NOT_A_MEMBER"


def _g_count(d):
    d["mm"]["cause_nodes_after"] += 1


V2_MUTATIONS = [
    ("binding: unknown OBS id", "O2", ob_mut(_m_unknown_obs)),
    ("binding: value outside the observable's list", "O2", ob_mut(_m_bad_value)),
    ("binding: 4 required conditions", "O5", ob_mut(_m_four_required)),
    ("binding: contradictory required values", "O3", ob_mut(_m_contradict)),
    ("binding: silently empty signature", "O4", ob_mut(_m_silent_empty)),
    ("binding: must_fix node with no required and no account", "O6", ob_mut(_m_silent_must_fix)),
    ("binding: NEW-OBS candidate in no cluster", "O7", ob_mut(_m_orphan_candidate)),
    ("binding: tree required condition not from any binding item", "O11", ob_mut(_m_untraced_required)),
    ("binding: aggregated signature differs from its runs", "O10", ob_mut(_m_agg_wrong)),
    ("binding: symptom without binding item", "O1", ob_mut(_m_missing_item)),
    ("binding: tree observables drift from catalog", "O9", ob_mut(_m_catalog_drift)),
    ("binding: user conclusion in a prompt", "O8", ob_mut(lambda d: None, _p_leak)),
    ("binding: knowledge-base term in a prompt", "O8", ob_mut(lambda d: None, _p_leak2)),
    ("merge: members of different stroke types", "M02", mm_mut(_g_other_stroke)),
    ("merge: two members from one video", "M02", mm_mut(_g_same_video)),
    ("merge: member text rewritten", "M03", mm_mut(_g_text_rewrite)),
    ("merge: merged node name rewritten", "M03", mm_mut(_g_v2_name)),
    ("merge: evidence lost in merged node", "M05", mm_mut(_g_lose_evidence)),
    ("merge: v1 edge lost", "M06", mm_mut(_g_lose_edge)),
    ("merge: cycle in v2 edges", "M07", mm_mut(_g_cycle)),
    ("merge: group goes beyond what both runs said", "M08", mm_mut(_g_beyond_runs)),
    ("merge: node in two groups", "M01", mm_mut(_g_two_groups)),
    ("merge: rep not a member", "M01", mm_mut(_g_bad_rep)),
    ("merge: node counts disagree", "M04", mm_mut(_g_count)),
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
    muts = list(MUTATIONS)
    if v2_available():
        d = _v2()
        gb = check_obs_binding.check(d["bind"], d["t1"], d["t2"]) + check_mech_merge.check(d["mm"], d["t1"], d["t2"], V2["eps"])
        report["good"]["tree_v2_artifacts"] = [str(x) for x in fails(gb)]
        if fails(gb):
            ok = False
            print("SELFTEST committed tree v2 artifacts FAILED their own checkers:", *fails(gb), sep="\n  ")
        muts += V2_MUTATIONS
    else:
        print("selftest: tree v2 artifacts not present; skipping v2 mutations")
    for name, code, fn in muts:
        caught = [str(f) for f in fails(fn()) if code in f.msg]
        report["mutations"].append({"name": name, "expect": code, "caught": bool(caught)})
        if not caught:
            ok = False
            print(f"SELFTEST mutation NOT caught: {name} (expected {code})")
    print(f"selftest: good fixtures + {len(muts)} mutations -> {'PASS' if ok else 'FAIL'}")
    report["pass"] = ok
    return ok, report


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--episodes"); ap.add_argument("--tree"); ap.add_argument("--manifest")
    ap.add_argument("--cases"); ap.add_argument("--prompts", nargs="*", default=[])
    ap.add_argument("--transcript-root"); ap.add_argument("--gate"); ap.add_argument("--report")
    ap.add_argument("--tree-v1"); ap.add_argument("--binding"); ap.add_argument("--merge")
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
        if a.binding and a.merge and a.tree_v1:
            t1 = load_json(a.tree_v1)
            res["check_obs_binding"] = check_obs_binding.check(load_json(a.binding), t1, tree)
            res["check_mech_merge"] = check_mech_merge.check(load_json(a.merge), t1, tree, a.episodes)
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
