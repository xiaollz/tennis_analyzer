#!/usr/bin/env python3
"""Download ONLY a short section (20-40 s) of an FTT video and strip the audio (user-approved per-segment download; never a whole episode).

usage: clip_fetch.py <VIDEO_ID> <start_s> <end_s> <outdir>
Writes <outdir>/raw/<ID>_<s>-<e>.mp4 and <outdir>/muted/<ID>_<s>-<e>_muted.mp4 (ffmpeg -an). Refuses sections shorter than 15 s or longer than 40 s.
Clips live outside the repo (copyright + size); only paths and sha256 go into the case files.
"""
import subprocess
import sys
from pathlib import Path

YTDLP = "/workspace/ftt_venv/bin/yt-dlp"


def main(a):
    vid, s, e, out = a[0], int(a[1]), int(a[2]), Path(a[3])
    if not 15 <= e - s <= 40:
        print("refused: section must be 15-40 s"); return 2
    (out / "raw").mkdir(parents=True, exist_ok=True); (out / "muted").mkdir(parents=True, exist_ok=True)
    name = f"{vid}_{s}-{e}"
    raw = out / "raw" / f"{name}.mp4"; muted = out / "muted" / f"{name}_muted.mp4"
    if not raw.exists():
        cmd = [YTDLP, "-q", "--no-warnings", "-f", "bv*[height<=720][ext=mp4]/b[height<=720]", "--extractor-args", "youtube:player_client=android",
               "--download-sections", f"*{s}-{e}", "--force-keyframes-at-cuts", "-o", str(out / "raw" / f"{name}.%(ext)s"), f"https://www.youtube.com/watch?v={vid}"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode or not raw.exists():
            print("download failed:", r.stderr[-300:]); return 1
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-an", "-c:v", "copy", str(muted)], capture_output=True, text=True)
    if r.returncode:
        print("ffmpeg failed:", r.stderr[-300:]); return 1
    print(name, "ok")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
