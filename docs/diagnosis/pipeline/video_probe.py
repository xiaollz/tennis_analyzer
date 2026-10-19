#!/usr/bin/env python3
"""Video-layer probe: Gemini reads a YouTube URL directly (fileData + videoMetadata offsets); nothing is downloaded.

usage: video_probe.py <VIDEO_ID> <start_s> <end_s> <OBS-id,OBS-id,...> <run_n> [tag]
Renders prompts/facts_video_v1.md with the chosen catalog questions (option order shuffled by run_n seed), runs check_neutrality on the
rendered text, calls the model once, saves raw to /workspace/s1_work/video/<tag>/<ID>_<start>_<end>_run<n>.json (outside the repo).
Key only from env MICU_GEMINI_KEY.
"""
import json
import os
import random
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import check_neutrality
from common import load_json

HERE = Path(__file__).resolve().parent
BASE, MODEL = "https://www.micuapi.ai", "gemini-3.8-flash"


def render(qids, seed):
    cat = {o["id"]: o for o in load_json(HERE / "obs_catalog.json")["observables"]}
    tpl = (HERE / "prompts" / "facts_video_v1.md").read_text(encoding="utf-8").split("\n---\n", 1)[1].lstrip("\n")
    rnd = random.Random(seed)
    lines = []
    for q in qids:
        vals = cat[q]["values"] + ["unclear"]
        rnd.shuffle(vals)
        lines.append(f"{q}. {cat[q]['question']}\n   options: {' | '.join(vals)}")
    return tpl.replace("{QUESTIONS}", "\n".join(lines))


def call(url, prompt, start, end, temp, timeout=900):
    key = os.environ["MICU_GEMINI_KEY"]
    part = {"fileData": {"fileUri": url, "mimeType": "video/mp4"}, "videoMetadata": {"startOffset": f"{start}s", "endOffset": f"{end}s"}}
    body = {"contents": [{"parts": [part, {"text": prompt}]}], "generationConfig": {"temperature": temp, "maxOutputTokens": 32768, "responseMimeType": "application/json"}}
    req = urllib.request.Request(f"{BASE}/v1beta/models/{MODEL}:generateContent", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "x-goog-api-key": key, "Authorization": f"Bearer {key}", "User-Agent": "Mozilla/5.0"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            p = json.loads(r.read().decode())
        txt = "".join(x.get("text", "") for x in p["candidates"][0]["content"]["parts"])
        return {"ok": True, "text": txt, "usage": p.get("usageMetadata"), "elapsed_s": round(time.time() - t0, 1)}
    except urllib.error.HTTPError as e:
        return {"ok": False, "err": f"HTTP {e.code}: " + e.read().decode("utf-8", "replace")[:300], "elapsed_s": round(time.time() - t0, 1)}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "err": f"{type(e).__name__}: {e}", "elapsed_s": round(time.time() - t0, 1)}


def main(a):
    vid, s, e, qids, n = a[0], int(a[1]), int(a[2]), a[3].split(","), int(a[4])
    tag = a[5] if len(a) > 5 else "probe"
    prompt = render(qids, seed=n * 1000 + s)
    with tempfile.NamedTemporaryFile("w", suffix="", prefix="facts_video_rendered_", delete=False, dir="/tmp") as f:
        pass
    p = Path("/tmp") / f"facts_video_rendered_{vid}_{s}_{n}.md"
    p.write_text(prompt, encoding="utf-8")
    fnd = check_neutrality.check(p)
    if fnd:
        print("NEUTRALITY FAIL", [str(x) for x in fnd]); return 2
    out = Path("/workspace/s1_work/video") / tag
    out.mkdir(parents=True, exist_ok=True)
    r = {"video": vid, "start": s, "end": e, "qids": qids, "run": n, "prompt": prompt, **call(f"https://www.youtube.com/watch?v={vid}", prompt, s, e, {1: 0.2, 2: 0.4, 3: 0.3}.get(n, 0.3))}
    (out / f"{vid}_{s}_{e}_run{n}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1))
    print(vid, s, e, n, "ok" if r["ok"] else r["err"], (r.get("usage") or {}).get("totalTokenCount"))
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
