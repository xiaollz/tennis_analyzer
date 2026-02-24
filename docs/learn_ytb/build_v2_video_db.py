#!/usr/bin/env python3
"""
Build a structured video database from `网球学习指南_v2_综合版.md`.

This does NOT download anything from YouTube. It only:
1) parses the guide (stage/subsection/topic + link text + video id + inline notes)
2) enriches entries with metadata from `feeltennis_flat.json` (duration/view_count/description/uploader)
3) exports JSON + CSV for further analysis / note generation

Usage:
  python learn_ytb/build_v2_video_db.py
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


GUIDE_PATH = Path("learn_ytb/网球学习指南_v2_综合版.md")
META_PATH = Path("learn_ytb/feeltennis_flat.json")

OUT_JSON = Path("output/feeltennis_v2_video_db.json")
OUT_CSV = Path("output/feeltennis_v2_video_db.csv")


YTB_RE = re.compile(
    r"\[(?P<link_text>.+?)\]\((?P<url>https?://www\.youtube\.com/watch\?v=(?P<id>[A-Za-z0-9_-]{11}))\)"
)


@dataclass
class VideoEntry:
    stage: str | None
    subsection: str | None
    topic: str | None
    title: str
    video_id: str
    url: str
    notes: list[str] = field(default_factory=list)

    # metadata from feeltennis_flat.json
    duration_sec: float | None = None
    view_count: int | None = None
    uploader: str | None = None
    description: str | None = None


def _load_meta(path: Path) -> dict[str, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = raw.get("entries", []) if isinstance(raw, dict) else raw
    meta: dict[str, dict[str, Any]] = {}
    for e in entries or []:
        vid = (e.get("id") or "").strip()
        if not vid:
            continue
        meta[vid] = e
    return meta


def _parse_guide(path: Path) -> list[VideoEntry]:
    stage: str | None = None
    subsection: str | None = None
    topic: str | None = None

    out: list[VideoEntry] = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()

        m = re.match(r"^##\s+(.*)$", line)
        if m:
            stage = m.group(1).strip()
            subsection = None
            topic = None
            continue

        m = re.match(r"^###\s+(.*)$", line)
        if m:
            subsection = m.group(1).strip()
            topic = None
            continue

        # inline bold "topic" markers like: **平击发球：**
        m = re.match(r"^\*\*(.+?)：\*\*$", line)
        if m:
            topic = m.group(1).strip()
            continue

        m = YTB_RE.search(line)
        if m:
            out.append(
                VideoEntry(
                    stage=stage,
                    subsection=subsection,
                    topic=topic,
                    title=m.group("link_text").strip(),
                    video_id=m.group("id").strip(),
                    url=m.group("url").strip(),
                )
            )
            continue

        # attach guide notes to the previous video when present
        # examples:
        #   - **笔记**：...
        #   - **笔记**: ...
        if out and ("**笔记**" in line):
            note = line
            note = re.sub(r"^\s*[-*]\s*", "", note)  # remove list marker
            note = note.replace("**笔记**", "").lstrip(":：").strip()
            if note:
                out[-1].notes.append(note)

    return out


def _enrich(entries: list[VideoEntry], meta: dict[str, dict[str, Any]]) -> list[str]:
    missing: list[str] = []
    for e in entries:
        m = meta.get(e.video_id)
        if not m:
            missing.append(e.video_id)
            continue
        e.duration_sec = m.get("duration")
        e.view_count = m.get("view_count")
        e.uploader = m.get("uploader")
        e.description = m.get("description")
    return missing


def _write_json(path: Path, entries: list[VideoEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(e) for e in entries]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, entries: list[VideoEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "stage",
        "subsection",
        "topic",
        "title",
        "video_id",
        "url",
        "duration_sec",
        "view_count",
        "uploader",
        "notes",
        "description",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for e in entries:
            row = asdict(e)
            # make lists readable in spreadsheets
            row["notes"] = " | ".join(e.notes)
            w.writerow({k: row.get(k) for k in fieldnames})


def main() -> None:
    if not GUIDE_PATH.exists():
        raise SystemExit(f"missing guide: {GUIDE_PATH}")
    if not META_PATH.exists():
        raise SystemExit(
            f"missing meta: {META_PATH}\n"
            f"Hint: run youtube-dl to fetch it first, e.g.\n"
            f"  youtube-dl --flat-playlist -J https://www.youtube.com/@feeltennis/videos > {META_PATH.name}"
        )

    meta = _load_meta(META_PATH)
    entries = _parse_guide(GUIDE_PATH)
    missing = _enrich(entries, meta)

    _write_json(OUT_JSON, entries)
    _write_csv(OUT_CSV, entries)

    print("videos_in_guide", len(entries))
    print("unique_video_ids", len({e.video_id for e in entries}))
    print("missing_in_meta", len(missing))
    if missing:
        print("missing_sample", missing[:20])
    print("wrote", OUT_JSON)
    print("wrote", OUT_CSV)


if __name__ == "__main__":
    main()

