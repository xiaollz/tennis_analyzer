#!/usr/bin/env python3
"""Scrape any YouTube channel and filter for forehand-relevant videos.

Usage:
  python3 scripts/scrape_and_filter_channel.py \\
      --channel-url "https://www.youtube.com/@FeelTennis" \\
      --slug feel_tennis \\
      --already-done file_with_ids.txt

Produces:
  /tmp/{slug}_all_videos.tsv
  /tmp/{slug}_shorts.tsv
  docs/research/{slug}_video_analyses/_VIDEOS_TO_ANALYZE.json

Forehand-relevance filter is shared across all channels — it's the same
filter logic from filter_tpa_videos.py, with channel-agnostic skip rules.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Skip patterns — videos NOT about forehand
SKIP_KEYWORDS = [
    r"\bserves?\b", r"\bservice\b", r"\bserving\b",
    r"\btoss\b", r"\bball toss\b",
    r"\bbackhands?\b", r"\bbh\b", r"\btwo.{0,2}handed\b", r"\bone.{0,2}handed\b",
    r"\bvolleys?\b", r"\boverhead\b", r"\bsmash\b",
    r"\bdoubles\b", r"\breturn\b", r"\bdrop shot\b",
    r"\bvlog\b", r"\binterview\b", r"\bpodcast\b",
    r"\btennis bag\b", r"\bracket review\b", r"\bracquet review\b",
    r"\bequipment\b", r"\bracket comparison\b",
    r"\bcoach development\b",
    r"\binjury\b", r"\bnewsletter\b",
    r"\bwhat makes a great coach\b",
    r"\b\d\.0\b", r"\b\d\.5\b",  # NTRP rating videos
]

# Forehand-relevant keywords
INCLUDE_KEYWORDS = [
    r"\bforehands?\b", r"\bfh\b",
    r"\bgroundstrokes?\b",
    r"\bunit turn\b", r"\btake.{0,2}back\b", r"\bbackswing\b",
    r"\bopen stance\b", r"\bsemi.{0,3}open\b", r"\bclosed stance\b", r"\bneutral stance\b",
    r"\bstance\b",
    r"\bcontact point\b", r"\bcontact zone\b",
    r"\bspacing\b", r"\bjam\b", r"\bjammed\b",
    r"\bracket drop\b", r"\bracquet drop\b",
    r"\bweight transfer\b", r"\bweight shift\b", r"\bload\b", r"\bloading\b",
    r"\bhip\b", r"\bpivot\b",
    r"\bkinetic chain\b", r"\barming\b", r"\barm.{0,2}the.{0,2}shot\b",
    r"\bcompact\b", r"\beffortless\b",
    r"\bracket speed\b", r"\bracket head speed\b", r"\bswing path\b",
    r"\brotation\b", r"\bshoulder.{0,5}turn\b",
    r"\btiming\b",
    r"\brelease\b",
    r"\bspin\b", r"\btopspin\b",
    r"\bpower\b",
    r"\bchest\b", r"\blat\b",
    r"\bsupination\b", r"\bpronation\b",
    r"\bpassive\b.*\barm\b",
    r"\bfeel\b",
    r"\blow ball\b", r"\bhigh ball\b", r"\bshort ball\b",
    r"\bextension\b", r"\blag\b",
    # Single-handed BH and serve-related body mechanics that may apply to FH
    # —— intentionally NOT included since user wants forehand-only this round
]


def is_forehand_related(title: str) -> tuple[bool, str]:
    t = title.lower()
    for kw in SKIP_KEYWORDS:
        if re.search(kw, t):
            return False, f"skip:{kw}"
    for kw in INCLUDE_KEYWORDS:
        if re.search(kw, t):
            return True, f"match:{kw}"
    return False, "no_keyword"


def relevance_score(title: str) -> int:
    """0-10 score for relevance to user's CURRENT stage (post-Bible insight)."""
    t = title.lower()
    score = 0
    high_priority = [
        (r"\bweight\b", 3),
        (r"\bhip\b", 3),
        (r"\bload\b", 3),
        (r"\bjam\b|\bjammed\b", 3),
        (r"\brotation\b", 3),
        (r"\bracket drop\b", 4),
        (r"\bsupination\b|\bpronation\b", 3),
        (r"\bshoulder.{0,5}turn\b|\bunit turn\b", 3),
        (r"\beffortless\b", 3),
        (r"\barming\b", 3),
        (r"\blag\b", 2),
        (r"\bcompact\b", 2),
        (r"\bextension\b", 2),
        (r"\bcontact point\b", 3),
        (r"\bopen stance\b|\bsemi.{0,3}open\b", 3),
        (r"\bspin\b|\btopspin\b", 1),
        (r"\bpower\b", 1),
        (r"\bfeel\b", 1),
        (r"\bkinetic chain\b", 3),
    ]
    for pat, pts in high_priority:
        if re.search(pat, t):
            score += pts
    return min(10, score)


def parse_tsv(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or "\\t" not in line:
            continue
        parts = line.split("\\t")
        if len(parts) < 2:
            continue
        vid, title = parts[0], parts[1]
        duration = parts[2] if len(parts) >= 3 else "NA"
        try:
            duration_f = float(duration) if duration != "NA" else 0.0
        except ValueError:
            duration_f = 0.0
        out.append({
            "video_id": vid,
            "title": title,
            "duration_s": duration_f,
            "url": f"https://www.youtube.com/watch?v={vid}",
        })
    return out


def scrape_channel(channel_url: str, slug: str) -> tuple[Path, Path]:
    long_path = Path(f"/tmp/{slug}_all_videos.tsv")
    shorts_path = Path(f"/tmp/{slug}_shorts.tsv")

    if not long_path.exists() or long_path.stat().st_size < 100:
        print(f"[scrape] {channel_url}/videos")
        # Strip /videos or /shorts suffix and re-add to make sure both work
        base = channel_url.rstrip("/")
        if base.endswith("/videos") or base.endswith("/shorts"):
            base = base.rsplit("/", 1)[0]
        with open(long_path, "w") as f:
            subprocess.run(
                ["yt-dlp", "--flat-playlist",
                 "--print", "%(id)s\\t%(title)s\\t%(duration)s",
                 f"{base}/videos"],
                stdout=f, stderr=subprocess.DEVNULL, timeout=180,
            )

    if not shorts_path.exists() or shorts_path.stat().st_size < 100:
        try:
            with open(shorts_path, "w") as f:
                subprocess.run(
                    ["yt-dlp", "--flat-playlist",
                     "--print", "%(id)s\\t%(title)s\\t%(duration)s",
                     f"{base}/shorts"],
                    stdout=f, stderr=subprocess.DEVNULL, timeout=180,
                )
        except Exception:
            pass

    return long_path, shorts_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel-url", required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--already-done", default=None,
                    help="optional file with one video_id per line to skip")
    args = ap.parse_args()

    long_path, shorts_path = scrape_channel(args.channel_url, args.slug)

    long_videos = parse_tsv(long_path)
    shorts = parse_tsv(shorts_path)
    all_videos = long_videos + shorts

    already_done: set[str] = set()
    if args.already_done and Path(args.already_done).exists():
        for line in Path(args.already_done).read_text().splitlines():
            v = line.strip()
            if v and not v.startswith("#"):
                already_done.add(v)

    relevant = []
    rejected = 0
    for v in all_videos:
        if v["video_id"] in already_done:
            continue
        ok, _ = is_forehand_related(v["title"])
        if ok:
            v["score"] = relevance_score(v["title"])
            v["is_short"] = (v["duration_s"] < 100)
            relevant.append(v)
        else:
            rejected += 1

    relevant.sort(key=lambda v: (-v["score"], -v["duration_s"]))

    out_dir = PROJECT_ROOT / "docs" / "research" / f"{args.slug}_video_analyses"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "_VIDEOS_TO_ANALYZE.json"
    out_path.write_text(json.dumps({
        "channel_url": args.channel_url,
        "slug": args.slug,
        "total_scanned": len(all_videos),
        "already_done_count": len(already_done),
        "relevant_count": len(relevant),
        "rejected_count": rejected,
        "videos": relevant,
    }, ensure_ascii=False, indent=2))

    print(f"\nTotal scanned: {len(all_videos)}")
    print(f"Relevant (new): {len(relevant)}")
    print(f"Rejected: {rejected}")
    print(f"Saved: {out_path}")
    print()
    print("Top 20 candidates:")
    for i, v in enumerate(relevant[:20], 1):
        s = " [S]" if v["is_short"] else ""
        print(f"{i:2}. [{v['score']:2}] {v['video_id']:14}{s} {v['title'][:65]}")


if __name__ == "__main__":
    main()
