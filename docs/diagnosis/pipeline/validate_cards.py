#!/usr/bin/env python3
"""V1 schema + structural rules for episode cards.

usage: validate_cards.py <card.json | dir> ...
Each rule id (R01..) is referenced from PLAN_v2.md §4; most encode a pilot lesson.
"""
import sys
from pathlib import Path

from common import CONFIG, Finding, ban_hits, card_paths, fails, load_json, load_schema, tokens, validate_schema

SCHEMA = load_schema("episode_card.schema.json")


def check_card(card, where):
    out = validate_schema(card, SCHEMA)
    if fails(out):
        return out  # structural rules assume a schema-valid card
    F = lambda rule, msg, level="FAIL": out.append(Finding(level, where, f"{rule} {msg}"))

    status = card["status"]
    stage2 = status.startswith("stage2")
    quotes = {q["id"]: q for q in card["quotes"]}
    vlayer = {v["id"]: v for v in card["video_layer"]}
    ids = {"PROBLEM"} | {s["id"] for s in card["symptoms"]} | {c["id"] for c in card["causes"]}

    # R01 file name = video id (one card per episode)
    if Path(where).name != f"{card['video']['id']}.card.json":
        F("R01", "file must be named <video.id>.card.json")
    # R02 non-teaching episodes get no card
    if card["classification"]["is_teaching"] == "non_teaching":
        F("R02", "non_teaching episode must not have a card; record it in loop/manifest.json")
    # R03 unique ids
    all_ids = [x["id"] for k in ("quotes", "symptoms", "causes", "angles", "solutions", "video_layer", "conflicts") for x in card[k]]
    dup = {i for i in all_ids if all_ids.count(i) > 1}
    if dup:
        F("R03", f"duplicate ids {sorted(dup)}")
    # R04 every evidence ref resolves
    def refs(item_list, key):
        for it in item_list:
            for e in it["evidence"]:
                if e.startswith("BOOK:"):
                    continue
                if e not in quotes and e not in vlayer:
                    F("R04", f"{key} {it.get('id', it.get('from'))} cites missing {e}")
    for key in ("symptoms", "causes", "chain", "angles", "solutions"):
        refs(card[key], key)
    refs([card["problem"]], "problem")
    # R05 quotes: <= N words; video-sourced quotes only in stage 2 (subtitle is authority for speech)
    for q in card["quotes"]:
        if len(tokens(q["text"])) > CONFIG["quote_max_words"]:
            F("R05", f"{q['id']} longer than {CONFIG['quote_max_words']} words (store excerpts only)")
        if q["source"] == "video" and not stage2:
            F("R05", f"{q['id']} source=video not allowed before stage 2")
        if "end" in q and q["end"] < q["start"]:
            F("R05", f"{q['id']} end < start")
    # R06 causes must rest on at least one SUBTITLE quote (pilot: Gemini glosses are not evidence)
    for c in card["causes"]:
        if not any(e in quotes and quotes[e]["source"] == "subtitle" for e in c["evidence"]):
            F("R06", f"cause {c['id']} has no subtitle quote evidence")
    # R07 chain endpoints exist, depth decreases downstream, every cause feeds the chain
    depth = {c["id"]: c["depth"] for c in card["causes"]}
    used_from = set()
    for e in card["chain"]:
        if e["from"] not in ids or e["to"] not in ids:
            F("R07", f"chain {e['from']}->{e['to']} has unknown endpoint")
            continue
        used_from.add(e["from"])
        if e["to"] in depth and depth[e["from"]] <= depth[e["to"]]:
            F("R07", f"chain {e['from']}(d{depth[e['from']]})->{e['to']}(d{depth[e['to']]}) depth must decrease downstream")
        if e["strength"] == "inferred" and depth.get(e["from"], 1) == max(depth.values()):
            F("R07", f"top-depth root {e['from']} rests on an inferred link", "WARN")
    for c in card["causes"]:
        if c["id"] not in used_from:
            F("R07", f"cause {c['id']} is not linked into the chain")
    if not any(e["to"] in ("PROBLEM",) or e["to"].startswith("S") for e in card["chain"]):
        F("R07", "chain never reaches PROBLEM or a symptom")
    # R08 layer discipline: feel => C; visible => A/B with camera + obs_ref
    for s in card["symptoms"]:
        if s["kind"] == "feel" and s["layer"] != "C":
            F("R08", f"symptom {s['id']} is a feel but layer {s['layer']}")
        if s["kind"] == "visible":
            if s["layer"] == "C":
                F("R08", f"symptom {s['id']} is visible but layer C")
            if s.get("camera") in (None, "none") or s["obs_ref"] == "NONE":
                F("R08", f"visible symptom {s['id']} needs camera and obs_ref")
    if not any(s["kind"] in ("visible", "ball_outcome") for s in card["symptoms"]):
        F("R08", "no visible or ball-outcome symptom: the card cannot be matched against a video", "WARN")
    # R09 bans: banned active cue must be flagged, and flagged items cannot be intuition_first_ok
    for d in card["solutions"]:
        hits = ban_hits(d["summary"] + " " + d.get("external_goal", ""))
        if hits and not d["ban_conflict"]["flag"]:
            F("R09", f"solution {d['id']} contains banned cue {hits} but ban_conflict.flag=false")
        if d["ban_conflict"]["flag"] and d["intuition_first_ok"]:
            F("R09", f"solution {d['id']} is ban-flagged and cannot be intuition_first_ok")
        if d["intuition_first_ok"] and not d.get("external_goal"):
            F("R09", f"solution {d['id']} intuition_first_ok needs an external_goal")
        for t in d["targets"]:
            if t not in ids:
                F("R09", f"solution {d['id']} targets unknown {t}")
    # R10 demo episodes must say who shows the fault (pilot: '学员问题' labels were unreliable)
    if card["classification"]["format"] in ("demo_with_player", "lesson_review"):
        if not any(q["about"].startswith("demonstrator") for q in card["quotes"]):
            F("R10", "demo/lesson episode without any demonstrator-labelled quote", "WARN")
    # R11 stage-2 rules: video is authority only for what is SHOWN
    if stage2:
        ex = card["extraction"]
        if not ex.get("stage2_model") or not ex.get("stage2_prompt_version") or len(ex.get("stage2_run_ids", [])) < 2:
            F("R11", "stage 2 needs stage2_model, stage2_prompt_version and >= 2 run ids")
    elif card["video_layer"]:
        F("R11", "video_layer must be empty before stage 2")
    conflict_refs = {r for x in card["conflicts"] for r in x["between"]}
    for v in card["video_layer"]:
        agree = int(v["runs_agree"].split("/")[0])
        if v["status"] == "accepted":
            if agree < 2:
                F("R11", f"{v['id']} accepted with only {v['runs_agree']} runs agreeing")
            if v["kind"] in ("correction_proposal", "new_claim") and v["id"] not in conflict_refs:
                F("R11", f"{v['id']} ({v['kind']}) accepted without a conflicts entry")
        if v["status"] == "conflict" and v["id"] not in conflict_refs:
            F("R11", f"{v['id']} status=conflict but not listed in conflicts")
        if v["target"] != "NEW" and v["target"] not in ids | set(quotes) | {d["id"] for d in card["solutions"]}:
            F("R11", f"{v['id']} targets unknown {v['target']}")
    for x in card["conflicts"]:
        if x["resolution"] == "video_accepted_visual_only":
            for r in x["between"]:
                if r in vlayer and vlayer[r]["kind"] == "new_claim":
                    F("R11", f"{x['id']}: a spoken new_claim from video cannot override subtitles")
    # R12 transcript coverage
    if card["transcript"]["coverage_ratio"] < CONFIG["transcript_min_coverage"]:
        F("R12", f"transcript coverage {card['transcript']['coverage_ratio']} < {CONFIG['transcript_min_coverage']}")
    return out


def main(args):
    findings = []
    for p in card_paths(args):
        findings += check_card(load_json(p), str(p))
    for f in findings:
        print(f)
    n = len(fails(findings))
    print(f"validate_cards: {len(card_paths(args))} cards, {n} FAIL")
    return 1 if n else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
