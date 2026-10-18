#!/usr/bin/env python3
"""V2 quote existence + timestamp check against the transcript.

usage: check_quotes.py [--transcript-root DIR] <card.json | dir> ...
  --transcript-root  re-map transcript.path to DIR/<basename> (box vs Mac paths)
For each subtitle-sourced quote: the transcript must contain it (token LCS ratio >= threshold
for the transcript kind) and the matched location must be within +-tolerance of quote.start.
"""
import argparse
import sys
from pathlib import Path

from common import CONFIG, Finding, card_paths, fails, load_json, load_transcript, locate_quote, sha256_file


def check_card(card, where, root=None):
    out = []
    tr = card["transcript"]
    path = Path(root) / Path(tr["path"]).name if root else Path(tr["path"])
    if not path.exists():
        return [Finding("FAIL", where, f"Q00 transcript not found: {path}")], {}
    if sha256_file(path) != tr["sha256"]:
        out.append(Finding("FAIL", where, "Q01 transcript sha256 changed since the card was written; re-verify"))
    segs = load_transcript(path)
    if not segs:
        return out + [Finding("FAIL", where, "Q00 transcript has no segments")], {}
    need = CONFIG["quote_match_min"][tr["kind"]]
    tol = CONFIG["timestamp_tolerance_s"]
    dur = card["video"]["duration_s"]
    stats = {"quotes": 0, "text_ok": 0, "ts_ok": 0}
    for q in card["quotes"]:
        if q["source"] != "subtitle":
            continue
        stats["quotes"] += 1
        qid = q["id"]
        if q["start"] > dur:
            out.append(Finding("FAIL", where, f"Q02 {qid} start {q['start']} beyond duration {dur}"))
        near_score, near_t = locate_quote(q["text"], segs, around=q["start"], search_window=CONFIG["quote_search_window_s"])
        if near_score >= need:
            stats["text_ok"] += 1
            if abs(near_t - q["start"]) <= tol:
                stats["ts_ok"] += 1
            else:
                out.append(Finding("FAIL", where, f"Q03 {qid} found at {near_t:.1f}s, card says {q['start']}s (> {tol}s)"))
            continue
        any_score, any_t = locate_quote(q["text"], segs)
        if any_score >= need:
            out.append(Finding("FAIL", where, f"Q03 {qid} text exists at {any_t:.1f}s but card says {q['start']}s"))
        else:
            out.append(Finding("FAIL", where, f"Q04 {qid} not in transcript (best {any_score:.2f} < {need}): {q['text'][:60]!r}"))
    return out, stats


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript-root")
    ap.add_argument("paths", nargs="+")
    a = ap.parse_args(argv)
    findings, tot = [], {"quotes": 0, "text_ok": 0, "ts_ok": 0}
    for p in card_paths(a.paths):
        f, s = check_card(load_json(p), str(p), a.transcript_root)
        findings += f
        for k in tot:
            tot[k] += s.get(k, 0)
    for f in findings:
        print(f)
    n = len(fails(findings))
    print(f"check_quotes: {tot['quotes']} subtitle quotes, text_ok {tot['text_ok']}, ts_ok {tot['ts_ok']}, {n} FAIL")
    return 1 if n else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
