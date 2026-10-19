#!/usr/bin/env python3
"""Screen where (in an FTT episode window) a clear, single-camera, full-swing segment exists (URL mode; nothing downloaded).

usage: clip_screen.py <VIDEO_ID> <start_s> <end_s> [tag]
Sends prompts/facts_screen_v1.md with videoMetadata offsets, saves raw to /workspace/s1_work/video/<tag>/screen_<ID>_<start>_<end>.json
(outside the repo). Output is for LOCATING segments only: the URL mode carries the coach's audio, so it is never used as facts.
Key only from env MICU_GEMINI_KEY.
"""
import json
import sys
import time
from pathlib import Path

import check_neutrality
import video_probe

HERE = Path(__file__).resolve().parent


def main(a):
    vid, s, e = a[0], int(a[1]), int(a[2])
    tag = a[3] if len(a) > 3 else "screen"
    path = HERE / "prompts" / "facts_screen_v1.md"
    f = check_neutrality.check(path)
    if f:
        print("NEUTRALITY FAIL", [str(x) for x in f]); return 2
    prompt = path.read_text(encoding="utf-8").split("\n---\n", 1)[1].lstrip("\n")
    out = Path("/workspace/s1_work/video") / tag; out.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        r = video_probe.call(f"https://www.youtube.com/watch?v={vid}", prompt, s, e, 0.2)
        if r["ok"]:
            break
        time.sleep(10 * (attempt + 1))
    r.update({"video": vid, "start": s, "end": e, "prompt_version": "facts_screen_v1"})
    (out / f"screen_{vid}_{s}_{e}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1))
    print(vid, s, e, "ok" if r["ok"] else r["err"])
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
