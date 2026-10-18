#!/usr/bin/env python3
"""Driver for one Stage-1 round (ep_s1_v6 flow, unchanged): per episode 2 independent runs -> `final` (check_rerun pair) ->
if no pair passes: run 3 + 3-run consensus (deterministic, no hand edits). Episodes run in parallel; each Gemini call is s1_extract.py run.

usage: S1_WORK=round7 s1_round.py ID [ID ...] [--workers N]
Also writes <ftt_subs>/<ID>.norm.tsv from the .en.vtt (same rule as the existing 57 files: cue start/end/text, 2 decimals) if missing.
Key only from env MICU_GEMINI_KEY (read by s1_extract). Subtitles / raw logs stay outside the repo (transcripts are not committed).
"""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import load_transcript

HERE = Path(__file__).resolve().parent
SUBS = Path("/workspace/ftt_subs")


def ensure_norm(vid):
    p = SUBS / f"{vid}.norm.tsv"
    if not p.exists():
        segs = load_transcript(SUBS / f"{vid}.en.vtt")
        p.write_text("\n".join(f"{a:.2f}\t{b:.2f}\t{t}" for a, b, t in segs if t.strip()) + "\n", encoding="utf-8")


def sh(*args):
    r = subprocess.run([sys.executable, str(HERE / "s1_extract.py"), *args], capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr)[-600:]


def one_run(job):
    vid, n = job
    for attempt in range(3):  # transient API errors only; the prompt/temperature stay identical
        rc, out = sh("run", vid, str(n))
        if rc == 0:
            return vid, n, "ok"
    return vid, n, "FAILED: " + out.replace("\n", " ")[-200:]


def main(argv):
    ids = [a for a in argv if not a.startswith("--")]
    w = int(argv[argv.index("--workers") + 1]) if "--workers" in argv else 6
    ids = [i for i in ids if not i.isdigit()]
    for v in ids:
        ensure_norm(v)
    with ThreadPoolExecutor(w) as ex:
        res = list(ex.map(one_run, [(v, n) for v in ids for n in (1, 2)]))
    bad = [r for r in res if r[2] != "ok"]
    print("runs 1-2:", len(res) - len(bad), "ok;", "failed:", bad)
    out = {}
    need3 = []
    for v in ids:
        rc, o = sh("final", v)
        out[v] = {"final_rc": rc}
        if rc != 0:
            need3.append(v)
    with ThreadPoolExecutor(w) as ex:
        res3 = list(ex.map(one_run, [(v, 3) for v in need3]))
    for v, _, st in res3:
        out[v]["run3"] = st
        if st == "ok":
            rc, o = sh("final", v)  # pairs 1-3 / 2-3 first (same order as earlier rounds)
            out[v]["final_rc_after_run3"] = rc
            if rc != 0:
                rc, o = sh("consensus", v)
                out[v]["consensus_rc"] = rc
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1:])
