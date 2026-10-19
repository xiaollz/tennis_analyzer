#!/usr/bin/env python3
"""Neutral fact extraction from a MUTED local clip (facts_v3), 3 independent runs -> facts JSON for diagnose.py / blind_replay.py.

usage: muted_facts.py <clip_muted.mp4> <clip_id> <outdir> [--runs 3] [--qids OBS-01,OBS-04,...]
Default questions = every observable in pipeline/obs_catalog.json. Option order is shuffled per run; the rendered prompt is saved and must pass
check_neutrality before it is sent. The clip must have no audio stream (checked with ffprobe) -> audio_stripped=true.
Aggregation (script only, no judgement): per run and per observable the majority value across that run's strokes; across runs the majority of those
(ties -> 'unclear'). agree = runs giving the final value / runs; strokes = (strokes showing the final value) / (all strokes) summed over runs.
camera = majority camera word of the runs (side->S, behind->B, other->other). Key only from env MICU_GEMINI_KEY. Raw runs stay outside the repo (outdir).
"""
import argparse
import base64
import json
import os
import random
import subprocess
import sys
import time
import urllib.request
import urllib.error
from collections import Counter
from pathlib import Path

import check_neutrality
from common import load_json

HERE = Path(__file__).resolve().parent
BASE, MODEL = "https://www.micuapi.ai", "gemini-3.8-flash"
TEMPS = {1: 0.2, 2: 0.4, 3: 0.3}


def render(qids, seed):
    cat = {o["id"]: o for o in load_json(HERE / "obs_catalog.json")["observables"]}
    tpl = (HERE / "prompts" / "facts_v3.md").read_text(encoding="utf-8").split("\n---\n", 1)[1].lstrip("\n")
    rnd = random.Random(seed)
    lines = []
    for q in qids:
        vals = cat[q]["values"] + ["unclear"]
        rnd.shuffle(vals)
        lines.append(f"{q}. {cat[q]['question']}\n   options: {' | '.join(vals)}")
    return tpl.replace("{QUESTIONS}", "\n".join(lines))


def call(mp4, prompt, temp, timeout=900):
    key = os.environ["MICU_GEMINI_KEY"]
    data = base64.b64encode(Path(mp4).read_bytes()).decode()
    body = {"contents": [{"parts": [{"inlineData": {"mimeType": "video/mp4", "data": data}}, {"text": prompt}]}],
            "generationConfig": {"temperature": temp, "maxOutputTokens": 32768, "responseMimeType": "application/json"}}
    req = urllib.request.Request(f"{BASE}/v1beta/models/{MODEL}:generateContent", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "x-goog-api-key": key, "Authorization": f"Bearer {key}", "User-Agent": "Mozilla/5.0"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            p = json.loads(r.read().decode())
        txt = "".join(x.get("text", "") for x in p["candidates"][0]["content"]["parts"])
        return {"ok": True, "text": txt, "usage": p.get("usageMetadata"), "elapsed_s": round(time.time() - t0, 1)}
    except urllib.error.HTTPError as e:
        return {"ok": False, "err": f"HTTP {e.code}: " + e.read().decode("utf-8", "replace")[:300]}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "err": f"{type(e).__name__}: {e}"}


def has_audio(mp4):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(mp4)], capture_output=True, text=True)
    return bool(r.stdout.strip())


def parse(txt):
    try:
        v = json.loads(txt)
        return v[0] if isinstance(v, list) and v and isinstance(v[0], dict) else (v if isinstance(v, dict) else None)
    except Exception:
        s, e = txt.find("{"), txt.rfind("}")
        try:
            return json.loads(txt[s:e + 1])
        except Exception:
            return None


def aggregate(clip_id, runs, qids):
    ok = [r for r in runs if r]
    cams = Counter({"side": "S", "behind": "B"}.get(str(r.get("camera", "")).lower(), "other") for r in ok)
    camera = cams.most_common(1)[0][0] if cams else "other"
    facts = []
    n_strokes = sum(len(r.get("strokes") or []) for r in ok)
    for q in qids:
        per_run, shown = [], Counter()
        total = 0
        for r in ok:
            vals = [(s.get("answers") or {}).get(q, {}).get("value") for s in (r.get("strokes") or [])]
            vals = [v for v in vals if v]
            total += len(r.get("strokes") or [])
            if vals:
                c = Counter(vals); per_run.append(c.most_common(1)[0][0]); shown.update(c)
        if not per_run:
            continue
        c = Counter(per_run); top = c.most_common(2)
        val = "unclear" if len(top) > 1 and top[0][1] == top[1][1] else top[0][0]
        facts.append({"obs": q, "value": val, "agree": f"{per_run.count(val)}/{len(ok)}", "source": "model", "strokes": f"{min(shown[val], total)}/{total}"})
    return {"clip_id": clip_id, "camera": camera, "audio_stripped": True, "prompt_version": "facts_v3", "model": MODEL, "runs": len(ok), "facts": facts}, n_strokes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mp4"); ap.add_argument("clip_id"); ap.add_argument("outdir")
    ap.add_argument("--runs", type=int, default=3); ap.add_argument("--qids", default="")
    a = ap.parse_args()
    if has_audio(a.mp4):
        print("REFUSED: clip still has an audio stream (muted copy required)"); return 2
    qids = a.qids.split(",") if a.qids else [o["id"] for o in load_json(HERE / "obs_catalog.json")["observables"]]
    out = Path(a.outdir) / a.clip_id; out.mkdir(parents=True, exist_ok=True)
    runs = []
    for n in range(1, a.runs + 1):
        pr = render(qids, seed=n * 7919 + sum(map(ord, a.clip_id)))
        pp = Path("/tmp") / f"facts_v3_{a.clip_id}_r{n}.md"; pp.write_text(pr, encoding="utf-8")
        fnd = check_neutrality.check(pp)
        if fnd:
            print("NEUTRALITY FAIL", [str(x) for x in fnd]); return 2
        (out / f"run{n}.prompt.md").write_text(pr, encoding="utf-8")
        p = out / f"run{n}.json"
        if p.exists() and json.loads(p.read_text()).get("ok"):
            r = json.loads(p.read_text())
        else:
            for attempt in range(4):
                r = call(a.mp4, pr, TEMPS.get(n, 0.3))
                if r["ok"]:
                    break
                time.sleep(15 * (attempt + 1))
            p.write_text(json.dumps(r, ensure_ascii=False, indent=1))
        runs.append(parse(r["text"]) if r.get("ok") else None)
    facts, n_strokes = aggregate(a.clip_id, runs, qids)
    facts["_summary"] = None
    summ = {"runs_ok": sum(bool(r) for r in runs), "strokes_per_run": [len((r or {}).get("strokes") or []) for r in runs],
            "no_swing_reason": [(r or {}).get("no_swing_reason", "") for r in runs], "camera_words": [(r or {}).get("camera") for r in runs],
            "swinger": [(r or {}).get("swinger") for r in runs]}
    facts.pop("_summary")
    (out / "facts.json").write_text(json.dumps(facts, ensure_ascii=False, indent=1))
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1))
    print(a.clip_id, json.dumps(summ, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
