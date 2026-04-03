"""Curated forehand-relevant video lists for secondary channels.

TomAllsopp (~45 videos) and Feel Tennis (~45 videos), filtered from
yt-dlp enumeration on 2026-04-03. Excludes FTT overlap, paid content,
volley-only, serve-only, backhand-only, and non-technique videos.

Video lists are hardcoded from yt-dlp --flat-playlist output, filtered
by title keywords for forehand technique relevance.
"""

from __future__ import annotations

from datetime import datetime, timezone
from knowledge.pipeline.video_state import VideoEntry


# ---------------------------------------------------------------------------
# Cross-channel dedup and exclusion sets
# ---------------------------------------------------------------------------

# Videos that appear in both FTT and TomAllsopp channels (collaboration)
FTT_OVERLAP_IDS: set[str] = {"1-g1OD8gh-I"}

# Feel Tennis paid/course content to skip
PAID_CONTENT_IDS: set[str] = {"CLEjGDGEGaA"}


# ---------------------------------------------------------------------------
# TomAllsopp curated forehand technique videos (~45)
# Source: yt-dlp --flat-playlist @TomAllsopp 2026-04-03 (307 total)
# Selection: forehand technique instruction, kinetic chain, biomechanics
# Excluded: volley, serve-only, analysis/makeover, child/beginner
# ---------------------------------------------------------------------------

TOMALLSOPP_FOREHAND_VIDEOS: list[dict] = [
    # Core forehand technique
    {"video_id": "ral2cHTFcdY", "title": "The Forehand Technique Nobody Teaches", "duration": 248},
    {"video_id": "IhLcK-ScJ1k", "title": "The Right Way To Jump Into Your Forehand", "duration": 311},
    {"video_id": "vd-hgPw_KYw", "title": "Stop Getting Jammed on Your Forehand", "duration": 87},
    {"video_id": "pVNfT3pe0v0", "title": "Effortless Tennis: Making a World-Class Forehand Even Better", "duration": 661},
    {"video_id": "KuSlSkWyf70", "title": "Unlock Effortless Forehands", "duration": 197},
    {"video_id": "vpB8ToHvWb0", "title": "The Secret to a Smooth & Effortless Forehand", "duration": 280},
    {"video_id": "Le0dgmeL-LE", "title": "Fix Your Forehand Unit Turn – Sync Your Arms!", "duration": 280},
    {"video_id": "PeEkclg6SKE", "title": "The #1 Mistake Killing Your Forehand Swing Path", "duration": 190},
    {"video_id": "nVRFbq1jFEw", "title": "The Throwing Drill That Fixes Your Forehand Timing", "duration": 238},
    {"video_id": "OBZaRtH7OQY", "title": "Struggling With High Forehands? Here's the Fix!", "duration": 315},
    # Pronation, supination, wrist
    {"video_id": "dw4hymptl9k", "title": "The Truth About Forehand Supination and Pronation", "duration": 482},
    {"video_id": "NG0rluboWLI", "title": "Forehand Wrist Action Fully Explained", "duration": 658},
    {"video_id": "ub4qPxEalI8", "title": "The Secret to Powerful Forehands: It's in Your Fingers!", "duration": 199},
    # Contact point
    {"video_id": "bevlgdgdrH4", "title": "One Simple Hack for Better Forehand Contact", "duration": 143},
    {"video_id": "OBjVdy1MS44", "title": "Simple Technique For a Perfect Forehand Contact Point", "duration": 162},
    {"video_id": "muxc0h0YAJg", "title": "How to Find the Perfect Forehand Contact Point", "duration": 248},
    {"video_id": "LU9yamZPOnw", "title": "Control The Ball Better With This Forehand Contact Point", "duration": 227},
    # Grip and grip pressure
    {"video_id": "-JU73Jxxj9c", "title": "Grip Pressure Timing: The Key to a Fluid Forehand", "duration": 190},
    {"video_id": "9qHURpX_sTw", "title": "Start Using A Continental Forehand Grip, Now!", "duration": 186},
    {"video_id": "KnFHS2YW_04", "title": "Find The Perfect Forehand Grip For You", "duration": 240},
    {"video_id": "ScH7MULsmjQ", "title": "Why Using an Eastern Grip Will Improve Your Forehand", "duration": 343},
    # Timing and takeback
    {"video_id": "UaPuZ5HIU1U", "title": "Fix This Forehand Mistake In 5 Minutes", "duration": 299},
    {"video_id": "8i8VJz74178", "title": "Forehand Takeback - 3 Reasons You Should Make It Bigger!", "duration": 169},
    {"video_id": "Gv7sF5DKK5E", "title": "Forehand Timing Masterclass - Strike a Cleaner Ball!", "duration": 215},
    {"video_id": "utZkaHi9XXM", "title": "The Perfect Forehand Takeback For Your Game", "duration": 434},
    {"video_id": "CmXxvX60TOI", "title": "Stop Hitting Your Forehand Late - Use This Takeback For Better Timing", "duration": 158},
    # Rotation, power, body mechanics
    {"video_id": "EPGLc9Ln8O4", "title": "Why Slowing Your Body Speeds Up Your Forehand", "duration": 190},
    {"video_id": "T4uvc_9ondc", "title": "Is Mouratoglou's Forehand Advice Wrong?", "duration": 219},
    {"video_id": "PgTEeHaLJ8U", "title": "Maximize Your Forehand Smash Factor!", "duration": 485},
    {"video_id": "FtGqOcmlWLY", "title": "Master Your Forehand Rotation and Weight Transfer", "duration": 216},
    {"video_id": "9td4hgpkSAA", "title": "This Shoulder Move Will Improve Your Forehand!", "duration": 563},
    {"video_id": "azVf6CyDfVk", "title": "Weight Transfer Exercise For More Power On Your Forehand", "duration": 344},
    {"video_id": "UVrZoQ70wxU", "title": "Relax Your Arm For More Forehand Power", "duration": 449},
    {"video_id": "zNxIL1US07o", "title": "Increase Your Forehand Power With This One Simple Move", "duration": 301},
    {"video_id": "ftyfZXr3Zcw", "title": "How to Transfer Weight and Rotate Into Your Forehand", "duration": 304},
    {"video_id": "Vcg_HcHaQ34", "title": "How to Generate More Power on Your Forehand", "duration": 289},
    # Follow through, swing path
    {"video_id": "lpFI1yxeok8", "title": "The 5 Mistakes Ruining Your Forehand Follow-Through (And How to Fix Them!)", "duration": 580},
    {"video_id": "5JKBxOpVEdc", "title": "Flawless Forehands - Two Steps For A Perfect Swing", "duration": 206},
    {"video_id": "wJITjKTcuMQ", "title": "Perfect Your Forehand Swing Path In Minutes!", "duration": 350},
    {"video_id": "WfrbPg1D5LM", "title": "More Forehand Racket Speed By Pulling or Throwing?", "duration": 375},
    {"video_id": "ZevtpeA_PAg", "title": "Finding The Perfect Forehand Arm Action - By Patting The Dog!?", "duration": 354},
    # Kinetic chain
    {"video_id": "ogW0IiguqSA", "title": "3 Tips To Master Your Forehand Kinetic Chain", "duration": 483},
    {"video_id": "pAFDt_e5rQI", "title": "Improve Your Forehand Kinetic Chain With This One Move!", "duration": 483},
    {"video_id": "Nmdk9qxrVHM", "title": "The 4 Reasons Your Forehand Kinetic Chain Is Breaking Down", "duration": 426},
    {"video_id": "qgO4PC2NZw8", "title": "The Forehand Kinetic Chain Made Simple...", "duration": 295},
    # Lag and racket speed
    {"video_id": "M1umUwuPe0w", "title": "A Simple Way To Create Forehand Lag and More Power!", "duration": 527},
    {"video_id": "wWWDqBKwO3U", "title": "Creating Forehand Lag - Next-Gen!!!!", "duration": 302},
    {"video_id": "O1i9y5NSoig", "title": "This Forehand Tip Will Give You More Lag and Racket Speed!", "duration": 302},
    {"video_id": "ubFJi2M3AMM", "title": "Improving Your Forehand Kinetic Chain - Creating Lag Without Wrist", "duration": 404},
]


# ---------------------------------------------------------------------------
# Feel Tennis curated forehand technique videos (~45)
# Source: yt-dlp --flat-playlist @feeltennis 2026-04-03 (571 total)
# Selection: forehand technique, biomechanics, feel-based drills
# Excluded: volley, serve-only, backhand-only, very short clips, paid content
# ---------------------------------------------------------------------------

FEELTENNIS_FOREHAND_VIDEOS: list[dict] = [
    # Core forehand technique and feel
    {"video_id": "m62xjbSvZgc", "title": "Why You Don't Feel a Good Swing On The Forehand", "duration": 109},
    {"video_id": "5Z9etBWK2Kg", "title": "Forehand Contact: More Forward Than You Think", "duration": 224},
    {"video_id": "PZLiR8Mcl8o", "title": "How to Build an Effortless Forehand Through Feel", "duration": 1017},
    {"video_id": "RWWplT5yiRs", "title": "The Inside-Out Swing That Makes Forehands Feel Easy", "duration": 548},
    {"video_id": "FRfXJ9UolL0", "title": "How to Feel the Forearm Stretch in Your Forehand", "duration": 736},
    {"video_id": "zlk-BqvXTLA", "title": "Feel the Natural Forehand Swing: 3 Biomechanical Steps", "duration": 524},
    {"video_id": "5LOKkHpFpFU", "title": "How To Hit A Tennis Forehand - 3 Simple Concepts", "duration": 435},
    {"video_id": "AbI2u36c23o", "title": "Tennis FOREHAND The Simplest Mental Image : Stability - rotation - swing", "duration": 404},
    # Wrist action and lag
    {"video_id": "2D7UlPQHce4", "title": "Tennis Forehand Wrist Action: Slap vs Snap Explained", "duration": 852},
    {"video_id": "htLslcohtDQ", "title": "Forehand Wrist Action - How Does Flexion Happen?", "duration": 906},
    {"video_id": "gxXnTXfurMA", "title": "How Forehand Wrist Stability Depends on Your Aim", "duration": 1126},
    {"video_id": "Av4Rx7YJHzA", "title": "Tennis FOREHAND Wrist Position - Closed Or Not?", "duration": 1242},
    {"video_id": "iaNHD-M3SOw", "title": "FOREHAND Wrist Lag - Why You Need Time To Get Into A Good Lag", "duration": 860},
    {"video_id": "vfsS9JAAdMc", "title": "Tennis Forehand Wrist Lag Comparison - Federer vs Halep", "duration": 1160},
    # Topspin technique
    {"video_id": "DfBF-Xtbo_4", "title": "Flat vs Topspin Forehand - And Why Hitting With Topspin Is Difficult", "duration": 435},
    {"video_id": "5rTG-uxWmsg", "title": "How to Hit Topspin In Tennis with Power (Without Brushing Up)", "duration": 833},
    {"video_id": "cwwKLLw9mII", "title": "Why More Topspin In Tennis Won't Work For You", "duration": 718},
    {"video_id": "xSuDf1aYztY", "title": "How High Level Tennis Players Hit Topspin Effortlessly", "duration": 950},
    {"video_id": "kBDKMAHrzNo", "title": "How To Get More Topspin In Tennis - 7 Topspin Killers That Hold You Back", "duration": 1150},
    {"video_id": "xLCqJkS3A_4", "title": "FOREHAND & NO TOPSPIN - Why Hitting Late Robs You Of Topspin", "duration": 601},
    {"video_id": "UEVH46_mea4", "title": "Tennis Forehand Topspin Drills - Teaching Topspin Implicitly", "duration": 668},
    # Stance and footwork
    {"video_id": "4quGmV2vh7I", "title": "How To Move Backwards In Open Stance Forehand", "duration": 145},
    {"video_id": "ZTHWa_mZJRw", "title": "Open vs. Neutral Stance Forehand: What Skilled Players Really Use", "duration": 1061},
    {"video_id": "tS3t-SoFxOs", "title": "The 3 Key Forehand Footwork Patterns Explained", "duration": 676},
    {"video_id": "DoxgjejOQ-U", "title": "Forehand Open Stance Tips For Easy Power", "duration": 379},
    {"video_id": "tNZJNHYYMDA", "title": "FOREHAND OPEN STANCE - 2 Common Mistakes And How To Correct Them", "duration": 597},
    {"video_id": "Auem1-8t3rE", "title": "Why Every Tennis Forehand Starts With An Open Stance", "duration": 1093},
    {"video_id": "EWrlCzQV4Hk", "title": "Open Or Neutral Stance Tennis Forehand? Pros & Cons", "duration": 658},
    {"video_id": "mnv0O2Z3MWc", "title": "Tennis Forehand Footwork For Attacking Short Balls", "duration": 366},
    # Rotation and power
    {"video_id": "VLoTdbA_l5o", "title": "How To Feel Forehand Shoulder Rotation And Connected Shoulders", "duration": 876},
    {"video_id": "NJvL5WtleNA", "title": "Keys To A Good Tennis Forehand - Shoulder Rotation Power", "duration": 1006},
    {"video_id": "YuP1z9_MmeY", "title": "FOREHAND Shoulder Rotation - The Most Effective Drill That I Know", "duration": 402},
    {"video_id": "xb01L1WhOFY", "title": "FOREHAND - 4 Major Biomechanical Components Of The Forehand", "duration": 682},
    {"video_id": "9msLFmROUVc", "title": "The Main Source Of Power On A Tennis Forehand", "duration": 91},
    {"video_id": "awYl86mkZu8", "title": "Kinetic Chain In Tennis And How To Apply It In Practice", "duration": 616},
    # Weight transfer
    {"video_id": "atD6qRDx_ck", "title": "FOREHAND Analysis - Weight Transfer Misconception", "duration": 848},
    {"video_id": "kGzHlnp4fEE", "title": "Busting the Weight Transfer Myth Once and for All", "duration": 1695},
    {"video_id": "8b96lTo4zKA", "title": "How To Use Weight Transfer In Tennis For More Power", "duration": 901},
    # Backswing, unit turn, preparation
    {"video_id": "BxDjH888MmY", "title": "Tennis Forehand Backswing Tip - Ideal Forehand Loop Size", "duration": 710},
    {"video_id": "vcWAEcF6klU", "title": "Tennis Forehand Unit Turn - It's Not A Backswing", "duration": 551},
    {"video_id": "guUg4hVI1AE", "title": "Deconstructing A Tennis Forehand Backswing", "duration": 734},
    {"video_id": "EJEWsypByQg", "title": "FOREHAND Preparation - The Work Of The Non-Dominant Arm", "duration": 726},
    # Contact, consistency, modern technique
    {"video_id": "xzFTxKMtI_w", "title": "The Forehand Consistency Blueprint: Technique, Targets, and Toughness", "duration": 1333},
    {"video_id": "MO01CaN6lFc", "title": "Tennis Forehand Contact Point And How To Find It", "duration": 638},
    {"video_id": "9KRYA9ZlYmM", "title": "Modern Tennis Forehand Technique In 8 Steps", "duration": 1472},
    {"video_id": "0Mf8SFX_LuI", "title": "Classic Tennis Forehand vs Modern Forehand Technique", "duration": 758},
]


# ---------------------------------------------------------------------------
# State generation functions
# ---------------------------------------------------------------------------

def _generate_channel_state(
    channel_id: str,
    videos: list[dict],
) -> dict:
    """Build a video state dict for any channel.

    Uses the same schema as video_state.py's generate_initial_state().
    """
    entries: dict[str, VideoEntry] = {}
    for v in videos:
        vid_id = v["video_id"]
        entries[vid_id] = VideoEntry(
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
        "channel_id": channel_id,
        "total_videos": len(entries),
        "enumerated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "videos": entries,
    }


def generate_tomallsopp_state() -> dict:
    """Build video state for TomAllsopp channel forehand videos."""
    return _generate_channel_state("@TomAllsopp", TOMALLSOPP_FOREHAND_VIDEOS)


def generate_feeltennis_state() -> dict:
    """Build video state for Feel Tennis channel forehand videos."""
    return _generate_channel_state("@feeltennis", FEELTENNIS_FOREHAND_VIDEOS)


# ---------------------------------------------------------------------------
# CLI entry point for state generation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from pathlib import Path
    from knowledge.pipeline.video_state import save_state

    tom_state = generate_tomallsopp_state()
    tom_path = Path("knowledge/state/tomallsopp_video_state.json")
    save_state(tom_state, tom_path)
    print(f"TomAllsopp: {tom_state['total_videos']} videos -> {tom_path}")

    feel_state = generate_feeltennis_state()
    feel_path = Path("knowledge/state/feeltennis_video_state.json")
    save_state(feel_state, feel_path)
    print(f"Feel Tennis: {feel_state['total_videos']} videos -> {feel_path}")

    print("Done.")
