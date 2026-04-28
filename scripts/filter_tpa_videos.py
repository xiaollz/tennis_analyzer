#!/usr/bin/env python3
"""Filter TPA Tennis (Tom Allsopp) videos for forehand-related content
relevant to user's CURRENT learning stage (post-Bible insight).

Outputs JSON with prioritized list.
"""
import json
import re
from pathlib import Path

LONG = Path("/tmp/tpa_all_videos.tsv")
SHORTS = Path("/tmp/tpa_shorts.tsv")
OUT = Path("/Users/qsy/Desktop/tennis/docs/research/tpa_video_analyses/_VIDEOS_TO_ANALYZE.json")

# Already analyzed (from 14_tpa_videos_*.md + 17_kinetic_chain_synthesis.md)
ALREADY_DONE = {
    # Batch 1-3 (March 2026)
    "KuSlSkWyf70", "PeEkclg6SKE", "nVRFbq1jFEw", "r7YqEMWFf5g", "vd-hgPw_KYw",
    "1-0Qxye6P7w", "IhLcK-ScJ1k", "JvBZqVgTTrw", "Le0dgmeL-LE", "lcqN1ktnIXI",
    "BnmVSa9dWz0", "Z-XeR_wPrVk", "bIVKPlOMRn0", "pVNfT3pe0v0", "ral2cHTFcdY",
    "vpB8ToHvWb0",
    # Kinetic chain synthesis batch (also TPA)
    "5xcIshgCd-E", "Nmdk9qxrVHM", "RaRstufkeJ0", "gLAtxOnVJLc",
    "h9pDmmffTwk", "ogW0IiguqSA", "pAFDt_e5rQI", "qgO4PC2NZw8",
}

# SKIP: clearly not forehand
SKIP_KEYWORDS = [
    r"\bserves?\b", r"\bservice\b", r"\bserving\b", r"\bkick serves?\b", r"\bslice serves?\b",
    r"\btoss\b", r"\bball toss\b", r"\bpro ball tosser\b",
    r"\bbackhands?\b", r"\bbh\b", r"\btwo.{0,2}handed\b", r"\bone.{0,2}handed\b",
    r"\bvolleys?\b", r"\boverhead\b",
    r"\bdoubles\b", r"\bsingles\b",
    r"\breturn\b", r"\bdrop shot\b", r"\bslice\b",
    r"\bvlog\b", r"\binterview\b", r"\bpodcast\b",
    r"\btennis bag\b", r"\bracket review\b", r"\bracquet review\b",
    r"\bequipment\b", r"\bracket comparison\b",
    r"\bvs ", r"\bcompetition\b", r"\bmatch\b",
    r"\bcoach development\b", r"\btennis lesson\b",
    r"\bknee injury\b", r"\binjury\b",
    r"\b3\.0\b", r"\b3\.5\b", r"\b4\.0\b", r"\b4\.5\b",  # NTRP rating videos
    r"\bnewsletter\b", r"\bclient\b",
    r"\bstop hitting\b.*\bnet\b",  # generic
    r"\bcourt\b.*\bpositioning\b",  # tactical
    r"\bmental game\b",
    r"\bworld class\b",  # often general tennis psychology
    r"\bstop missing\b",  # too generic
    r"\bbeginner\b",
]

# INCLUDE: forehand-related signals
INCLUDE_KEYWORDS = [
    r"\bforehand\b", r"\bfh\b",
    r"\bgroundstroke\b", r"\bgroundstrokes\b",
    r"\bunit turn\b", r"\btake.{0,2}back\b", r"\bbackswing\b",
    r"\bopen stance\b", r"\bsemi.{0,3}open\b", r"\bclosed stance\b", r"\bneutral stance\b",
    r"\bstance\b",
    r"\bcontact point\b", r"\bcontact zone\b",
    r"\bspacing\b", r"\bjam\b", r"\bjammed\b",
    r"\bracket drop\b", r"\bracquet drop\b", r"\bdrop\b.*\bracket\b",
    r"\bweight transfer\b", r"\bweight shift\b", r"\bload\b", r"\bloading\b",
    r"\bhip\b", r"\bpivot\b",
    r"\bkinetic chain\b", r"\barming\b", r"\barm.{0,2}the.{0,2}shot\b",
    r"\bcompact\b", r"\beffortless\b",
    r"\bracket speed\b", r"\bracket head speed\b", r"\bswing path\b",
    r"\brotation\b", r"\bshoulder.{0,5}turn\b",
    r"\btiming\b",
    r"\bracket head\b", r"\brelease\b",
    r"\bspin\b", r"\btopspin\b",
    r"\bpower\b",  # broad — filter further
    r"\bchest\b", r"\blat\b", r"\bback muscle\b",
    r"\bsupination\b", r"\bpronation\b",
    r"\bpassive\b.*\barm\b",
    r"\bfeel\b",  # broad
    r"\blow ball\b", r"\bhigh ball\b", r"\bshort ball\b",
    r"\bdeeper\b.*\bracket drop\b",
    r"\bextension\b",
    r"\blag\b",
    r"\bracket lag\b",
]


def relevance_to_current_stage(title: str) -> tuple[int, list[str]]:
    """Score 0-10 for how directly this hits the user's CURRENT痛点.

    Highest priority (8-10): right foot axis, weight transfer, hip,
    pelvic tilt, hide elbow, shoulder loading, racket drop mechanics.

    Medium (5-7): unit turn, contact point, jamming, kinetic chain.

    Low (1-4): spin, power generation, generic forehand tips.
    """
    t = title.lower()
    score = 0
    reasons = []

    # Top priority — directly addresses current Bible-level concepts
    high_priority = [
        (r"\bweight\b", 4, "weight transfer"),
        (r"\bhip\b", 4, "hip work / pelvic tilt"),
        (r"\bload\b", 3, "loading"),
        (r"\bjam\b|\bjammed\b", 3, "spacing / jamming"),
        (r"\brotation\b", 3, "rotation mechanics"),
        (r"\bracket drop\b|\bracquet drop\b", 4, "racket drop mechanism"),
        (r"\bsupination\b|\bpronation\b", 3, "forearm rotation"),
        (r"\bshoulder.{0,5}turn\b|\bunit turn\b", 3, "unit turn"),
        (r"\beffortless\b|\beffortlessly\b", 3, "effortless = passive arm"),
        (r"\barming\b", 3, "arming = anti-pattern user struggles with"),
        (r"\blag\b", 2, "wrist lag"),
        (r"\bcompact\b", 2, "compact swing"),
        (r"\bextension\b", 2, "extension"),
        (r"\bracket head speed\b", 2, "racket head speed"),
        (r"\bkinetic chain\b", 3, "kinetic chain"),
        (r"\bcontact point\b|\bcontact zone\b", 3, "contact point"),
        (r"\bopen stance\b|\bsemi.{0,3}open\b", 3, "stance type"),
        (r"\bspin\b|\btopspin\b", 1, "spin"),
        (r"\bpower\b", 1, "power"),
        (r"\bfeel\b", 1, "feel"),
    ]

    for pat, pts, label in high_priority:
        if re.search(pat, t):
            score += pts
            reasons.append(label)

    return min(10, score), reasons


def is_forehand_related(title: str) -> tuple[bool, str]:
    t = title.lower()
    for kw in SKIP_KEYWORDS:
        if re.search(kw, t):
            return False, f"skip:{kw}"
    for kw in INCLUDE_KEYWORDS:
        if re.search(kw, t):
            return True, f"match:{kw}"
    return False, "no_keyword"


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


def main():
    long_videos = parse_tsv(LONG)
    shorts = parse_tsv(SHORTS)
    all_videos = long_videos + shorts

    relevant = []
    rejected = []
    for v in all_videos:
        if v["video_id"] in ALREADY_DONE:
            continue
        ok, reason = is_forehand_related(v["title"])
        if ok:
            score, sreasons = relevance_to_current_stage(v["title"])
            v["score"] = score
            v["score_reasons"] = sreasons
            v["is_short"] = (v["duration_s"] < 100 or "shorts" in v["title"].lower())
            relevant.append(v)
        else:
            v["rejection_reason"] = reason
            rejected.append(v)

    # Sort by score desc, then by duration (longer = more depth, usually)
    relevant.sort(key=lambda v: (-v["score"], -v["duration_s"]))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "channel": "https://www.youtube.com/@TomAllsopp",
        "channel_name": "TPA tennis (Tom Allsopp)",
        "total_scanned": len(all_videos),
        "already_analyzed": len(ALREADY_DONE),
        "relevant_count": len(relevant),
        "rejected_count": len(rejected),
        "videos": relevant,
    }
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2))

    print(f"Total scanned: {len(all_videos)}")
    print(f"Already analyzed (skipped): {len(ALREADY_DONE)}")
    print(f"Relevant new: {len(relevant)}")
    print(f"Rejected: {len(rejected)}")
    print()
    print("=== Top 25 by relevance to CURRENT stage ===")
    for i, v in enumerate(relevant[:25], 1):
        short = " [S]" if v["is_short"] else ""
        score = v["score"]
        title = v["title"][:70]
        print(f"{i:2}. [{score:2}] {v['video_id']:14}{short} {title}")
        if v["score_reasons"]:
            print(f"        → {' / '.join(v['score_reasons'][:4])}")

    print()
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
