"""Video state management for FTT YouTube video extraction pipeline.

Tracks the status of all 73 FTT channel videos through the extraction
pipeline: pending -> analyzed -> extracted -> failed.

The initial inventory is hardcoded from 03-RESEARCH.md verified enumeration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict


class VideoEntry(TypedDict):
    """Schema for a single video tracking entry."""
    video_id: str
    title: str
    url: str
    duration: int  # seconds
    status: str  # pending | analyzed | extracted | failed
    analysis_file: str | None
    extracted_file: str | None
    analyzed_at: str | None
    error: str | None


# ---------------------------------------------------------------------------
# Hardcoded video inventory from 03-RESEARCH.md
# ---------------------------------------------------------------------------

# 33 already-analyzed videos (from docs/research/09_ftt_videos_{1,2,3}.md)
# Duplicates in file 3 (gehRK0Y6AdQ, 9NirSL59an0) already removed
_ANALYZED_VIDEOS: list[dict] = [
    # File 1: 10 videos
    {"video_id": "lwS7Xxqrrdw", "title": "The Simple Foundation of the 2-Handed Backhand", "duration": 782, "analysis_file": "docs/research/09_ftt_videos_1.md"},
    {"video_id": "FBJnFBtZozE", "title": "How Sinner Choke-Proofed His Game", "duration": 605, "analysis_file": "docs/research/09_ftt_videos_1.md"},
    {"video_id": "LoSrMj-Qxkc", "title": "Their Mothers are Sisters, Their Forehands AREN'T", "duration": 718, "analysis_file": "docs/research/09_ftt_videos_1.md"},
    {"video_id": "Z9XwaXDVkZg", "title": "The Simple Foundation of the One-Handed Backhand", "duration": 1170, "analysis_file": "docs/research/09_ftt_videos_1.md"},
    {"video_id": "FjjvhNbk3q4", "title": "7 Tips for an Effortless Serve ft. Giovanni Mpetshi-Perricard", "duration": 1263, "analysis_file": "docs/research/09_ftt_videos_1.md"},
    {"video_id": "Bhzd6cM-oFU", "title": "How to Hit Crazy Slice Like Carlos Alcaraz", "duration": 227, "analysis_file": "docs/research/09_ftt_videos_1.md"},
    {"video_id": "emZxTmSVSa8", "title": "Master Pronation in 2 Minutes at Home", "duration": 379, "analysis_file": "docs/research/09_ftt_videos_1.md"},
    {"video_id": "x_WIg59Wr9Y", "title": "Why You Should Start Your Swing at Contact", "duration": 651, "analysis_file": "docs/research/09_ftt_videos_1.md"},
    {"video_id": "tzgAWr3QGv0", "title": "7 Steps for a Sinner-Like Forehand - #5: Tilt!", "duration": 1133, "analysis_file": "docs/research/09_ftt_videos_1.md"},
    {"video_id": "p41GGc_y5Xk", "title": "How Your Hand Controls Every Shot", "duration": 384, "analysis_file": "docs/research/09_ftt_videos_1.md"},
    # File 2: 10 videos
    {"video_id": "PfDqnff_vYA", "title": "Why Your Serve is So Slow", "duration": 534, "analysis_file": "docs/research/09_ftt_videos_2.md"},
    {"video_id": "hqganBxzQNM", "title": "Why I Give Every Student a Weight to Swing", "duration": 403, "analysis_file": "docs/research/09_ftt_videos_2.md"},
    {"video_id": "6j33LJfl46s", "title": "Scapular Glide on the Tennis Forehand", "duration": 12, "analysis_file": "docs/research/09_ftt_videos_2.md"},
    {"video_id": "lbdQpIHvXYA", "title": "The Stabilization Secret in Khachanov's Forehand", "duration": 347, "analysis_file": "docs/research/09_ftt_videos_2.md"},
    {"video_id": "7_JPqhG4G64", "title": "What He Does Next Separates Pros from Amateurs", "duration": 305, "analysis_file": "docs/research/09_ftt_videos_2.md"},
    {"video_id": "T5meVwBOKR4", "title": "How Alcaraz Hits Harder Than Everyone", "duration": 511, "analysis_file": "docs/research/09_ftt_videos_2.md"},
    {"video_id": "PjvOQkqH9DA", "title": "Never Lose Another Winnable Match", "duration": 564, "analysis_file": "docs/research/09_ftt_videos_2.md"},
    {"video_id": "gehRK0Y6AdQ", "title": "The Pros Brush Differently From You", "duration": 818, "analysis_file": "docs/research/09_ftt_videos_2.md"},
    {"video_id": "9NirSL59an0", "title": "The Paradoxical Nature of Modern Topspin", "duration": 620, "analysis_file": "docs/research/09_ftt_videos_2.md"},
    {"video_id": "cIIEYn42o4g", "title": "The #1 Backscratch Mistake -- How to Fix It", "duration": 505, "analysis_file": "docs/research/09_ftt_videos_2.md"},
    # File 3: 13 unique videos (excluding gehRK0Y6AdQ and 9NirSL59an0 duplicates)
    {"video_id": "1W1rope5l0k", "title": "Why Every Pro Serves In-to-Out", "duration": 692, "analysis_file": "docs/research/09_ftt_videos_3.md"},
    {"video_id": "-ZmnDrRFfjc", "title": "3 Brilliant Alcaraz Tactics ANYONE Can Use", "duration": 383, "analysis_file": "docs/research/09_ftt_videos_3.md"},
    {"video_id": "WoVEWh7fFfc", "title": "57-Year-Old Ripping Forehands (and Serves)", "duration": 457, "analysis_file": "docs/research/09_ftt_videos_3.md"},
    {"video_id": "EkzzGZkqCgc", "title": "Why Your Racket Doesn't Flip", "duration": 426, "analysis_file": "docs/research/09_ftt_videos_3.md"},
    {"video_id": "m1NBdd3Bigg", "title": "Simple Serve Progressions", "duration": 1013, "analysis_file": "docs/research/09_ftt_videos_3.md"},
    {"video_id": "dZadhzVEsds", "title": "How to Hit Hard Without Missing Long", "duration": 1260, "analysis_file": "docs/research/09_ftt_videos_3.md"},
    {"video_id": "RVfmDk-iEwo", "title": "Find Your Hand Slot, Everything Gets Easier", "duration": 1302, "analysis_file": "docs/research/09_ftt_videos_3.md"},
    {"video_id": "KU7FHy1qQOI", "title": "The Physics, Geometry, and Biomechanics of Power", "duration": 1088, "analysis_file": "docs/research/09_ftt_videos_3.md"},
    {"video_id": "Vg8lbXOhM3E", "title": "Training to Play Loose in Matches", "duration": 676, "analysis_file": "docs/research/09_ftt_videos_3.md"},
    {"video_id": "0m3BMfDDShI", "title": "Build This Foundation - The Rest Will Follow", "duration": 891, "analysis_file": "docs/research/09_ftt_videos_3.md"},
    {"video_id": "fZ-e7O7FTDE", "title": "Jannik Sinner Forehand Analysis", "duration": 694, "analysis_file": "docs/research/09_ftt_videos_3.md"},
    {"video_id": "azVf6CyDfVk", "title": "Weight Transfer Exercise For More Power", "duration": 480, "analysis_file": "docs/research/09_ftt_videos_3.md"},
    {"video_id": "1-g1OD8gh-I", "title": "This One Move Instantly Improves Your Forehand", "duration": 480, "analysis_file": "docs/research/09_ftt_videos_3.md"},
]

# 40 pending videos (from RESEARCH.md "Remaining Unanalyzed" table)
_PENDING_VIDEOS: list[dict] = [
    {"video_id": "pWzyP-xfLfU", "title": "The Secret to Lag is on Your Handle", "duration": 1262},
    {"video_id": "mOFtt9PllI0", "title": "The Geometry of Measured Aggression", "duration": 3105},
    {"video_id": "5jHCDc44SQM", "title": "The Abdominal Corkscrew ft. Carson Branstine", "duration": 458},
    {"video_id": "XXlndjnrA4E", "title": "Peripheral Vision Lets You Volley Like the Pros", "duration": 499},
    {"video_id": "hNVbbPEob3g", "title": "Chest Engagement Makes Controlling the Racket Face Easy", "duration": 380},
    {"video_id": "UB6SbA_KX9E", "title": "Proper Trunk Sequencing will Transform Your Tennis", "duration": 715},
    {"video_id": "w1FakobNq1Q", "title": "Breaking Down the Greatest Tiebreak Ever Played", "duration": 685},
    {"video_id": "FOmz8Wjv3DQ", "title": "Deeper Than Just Footwork - Movement Fundamentals", "duration": 1066},
    {"video_id": "8r09TliP-Ak", "title": "I Picked Draper to Win It All (3 patterns)", "duration": 684},
    {"video_id": "_Qu1LOwklAw", "title": "Patient but Ruthless - Alcaraz Broke Down Sinner", "duration": 394},
    {"video_id": "Qszz0N4fRb4", "title": "Why You Get Tight, and How to Fix It", "duration": 708},
    {"video_id": "Psidjei5BnI", "title": "4 Tips For Consistently Crushing Slow Balls", "duration": 476},
    {"video_id": "FxDmVi3EFnE", "title": "The Truth About the Topspin Pro", "duration": 929},
    {"video_id": "McCb-RfYd0w", "title": "The Magic of the Non-Dominant Side on the Forehand", "duration": 193},
    {"video_id": "JIMgI3jiVns", "title": "How Shoulder Rotation Syncs Your Contact", "duration": 449},
    {"video_id": "42BfbKsTGb4", "title": "You Aren't Practicing Half of Tennis - RECEIVING", "duration": 338},
    {"video_id": "Fu6DkHvZlGY", "title": "The Pure-Dextral Pinpoint - Serve Like Mensik", "duration": 436},
    {"video_id": "E_zmENJIj4g", "title": "Coiling/Spinal Motion/Eye Dominance/Arm Slot - 15 Serves", "duration": 2625},
    {"video_id": "BbGzWTp5pCM", "title": "Practicing in Slow Motion is Killing Match Play", "duration": 260},
    {"video_id": "JzcA_ku7Yhk", "title": "The Misunderstanding LOSING You Matches", "duration": 431},
    {"video_id": "dx8aGSIo24w", "title": "4 Tips for Two-Handers", "duration": 283},
    {"video_id": "wFIrPMutzRo", "title": "2 Secrets to Rotational Power - Side Bending + X", "duration": 766},
    {"video_id": "NEpD7fIM7HI", "title": "Learning from Federer's Slice - 4 Tips + 3 Drills", "duration": 1052},
    {"video_id": "GsHkML2mVEI", "title": "Nishioka's Unique Backswing Timing", "duration": 351},
    {"video_id": "wd4YRQW3TOc", "title": "4 Tips For Effortless, Controllable Topspin", "duration": 1339},
    {"video_id": "V-QkILd4V-w", "title": "8 Visual Return Strategies Tested by WTA Pro", "duration": 1022},
    {"video_id": "xLs469ZVMPU", "title": "Fix Your Forehand Over-Rotation - 3 Techniques", "duration": 735},
    {"video_id": "dDYKuNZtdyU", "title": "Find Your One-Handed Backhand Pull Slot", "duration": 666},
    {"video_id": "pQ793MBQE50", "title": "How Scap Retraction Powers One-Handed Backhand", "duration": 988},
    {"video_id": "YbLit9png2U", "title": "Fix Your Kick Serve by Throwing Sideways", "duration": 202},
    {"video_id": "XjJHA91HDbU", "title": "Don't Swing AT the Ball on the Overhead", "duration": 212},
    {"video_id": "wVa4XQPcaqs", "title": "Use The Wall to Find Your Perfect Contact", "duration": 276},
    {"video_id": "QXAtdSEUkfY", "title": "3 Tips to Rip the Low Backhand", "duration": 186},
    {"video_id": "wFOy0RKWBTg", "title": "Swing OUT on the Forehand", "duration": 31},
    {"video_id": "5KdScDKxVSI", "title": "Shoulder Adduction Will Transform Forehand Contact", "duration": 358},
    {"video_id": "dnNOOornvek", "title": "Retired WTA Pro Still Ripping Backhands", "duration": 48},
    {"video_id": "ExkBtFRhUWY", "title": "The Magic of Single-Foot Forehand Training", "duration": 497},
    {"video_id": "PUN6qIIYU-4", "title": "4 Tips for the Two-Handed Backhand", "duration": 887},
    {"video_id": "Am8j1Zw5KrE", "title": "Shoulder Adduction Unlocks the Tennis Forehand", "duration": 557},
    {"video_id": "OYf48k-cfNI", "title": "Fault Tolerance In Action", "duration": 233},
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_initial_state() -> dict:
    """Build the complete 73-video inventory from hardcoded research data.

    Returns a state dict with channel metadata and all video entries.
    """
    videos: dict[str, VideoEntry] = {}

    # Add analyzed videos
    for v in _ANALYZED_VIDEOS:
        vid_id = v["video_id"]
        videos[vid_id] = VideoEntry(
            video_id=vid_id,
            title=v["title"],
            url=f"https://www.youtube.com/watch?v={vid_id}",
            duration=v["duration"],
            status="analyzed",
            analysis_file=v["analysis_file"],
            extracted_file=None,
            analyzed_at="2026-03-17",
            error=None,
        )

    # Add pending videos
    for v in _PENDING_VIDEOS:
        vid_id = v["video_id"]
        videos[vid_id] = VideoEntry(
            video_id=vid_id,
            title=v["title"],
            url=f"https://www.youtube.com/watch?v={vid_id}",
            duration=v["duration"],
            status="pending",
            analysis_file=None,
            extracted_file=None,
            analyzed_at=None,
            error=None,
        )

    return {
        "channel_id": "@FaultTolerantTennis",
        "total_videos": len(videos),
        "enumerated_at": "2026-04-03",
        "videos": videos,
    }


def load_state(path: Path) -> dict:
    """Load state from JSON file. Creates initial state if file doesn't exist."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    # File doesn't exist -- generate and save
    state = generate_initial_state()
    save_state(state, path)
    return state


def save_state(state: dict, path: Path) -> None:
    """Persist state dict to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def mark_video(state: dict, video_id: str, status: str, **kwargs) -> None:
    """Update a video's status and optional fields.

    Args:
        state: The state dict to modify in-place.
        video_id: ID of the video to update.
        status: New status (pending, analyzed, extracted, failed).
        **kwargs: Optional fields to update (analysis_file, extracted_file,
                  analyzed_at, error).
    """
    if video_id not in state["videos"]:
        raise KeyError(f"Video {video_id} not found in state")
    entry = state["videos"][video_id]
    entry["status"] = status
    for key, value in kwargs.items():
        if key in entry:
            entry[key] = value


def get_videos_by_status(state: dict, status: str) -> list[dict]:
    """Return all video entries matching the given status."""
    return [v for v in state["videos"].values() if v["status"] == status]


def get_state_summary(state: dict) -> dict:
    """Return counts per status plus total."""
    from collections import Counter
    counts = Counter(v["status"] for v in state["videos"].values())
    counts["total"] = len(state["videos"])
    return dict(counts)
