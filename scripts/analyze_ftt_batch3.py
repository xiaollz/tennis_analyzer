#!/usr/bin/env python3
"""Batch 3: Analyze remaining 14 FTT videos via Gemini API.

Videos cover: movement, vision, drills, serve, backhand, overhead.
Plus 1 skipped video (dnNOOornvek, 48s -- too short).

Uses batch3_state.json as per-plan state slice (NOT shared ftt_video_state.json).
Staggered start: sleeps 60s before first API call to offset from batch1/batch2.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from knowledge.pipeline.video_analyzer import (
    analyze_video,
    create_client,
    load_analysis_prompt,
    load_api_config,
)
from knowledge.pipeline.video_concept_extractor import (
    extract_concepts_from_analysis,
    save_video_extraction,
)
from knowledge.pipeline.video_state import load_state, mark_video, save_state
from knowledge.registry import ConceptRegistry

# ── Video lists ──────────────────────────────────────────────────────

BATCH3_VIDEO_IDS = [
    "V-QkILd4V-w",   # 8 Visual Return Strategies Tested by WTA Pro
    "XXlndjnrA4E",   # Peripheral Vision Lets You Volley Like the Pros
    "FOmz8Wjv3DQ",   # Deeper Than Just "Footwork" - Movement Fundamentals
    "wVa4XQPcaqs",   # Use The Wall to Find Your Perfect Contact
    "E_zmENJIj4g",   # Coiling/Spinal Motion/Eye Dominance/Arm Slot - 15 Serves (43:45 LONG)
    "Fu6DkHvZlGY",   # The Pure-Dextral Pinpoint - Serve Like Mensik
    "dx8aGSIo24w",   # 4 Tips for Two-Handers
    "PUN6qIIYU-4",   # 4 Tips for the Two-Handed Backhand
    "dDYKuNZtdyU",   # Find Your One-Handed Backhand "Pull Slot"
    "pQ793MBQE50",   # How Scap Retraction Powers One-Handed Backhand
    "NEpD7fIM7HI",   # Learning from Federer's Slice - 4 Tips + 3 Drills
    "YbLit9png2U",   # Fix Your Kick Serve by Throwing Sideways
    "QXAtdSEUkfY",   # 3 Tips to Rip the Low Backhand
    "XjJHA91HDbU",   # Don't Swing AT the Ball on the Overhead
]

SKIP_VIDEO_IDS = [
    "dnNOOornvek",   # Retired WTA Pro Still Ripping Backhands (48s -- too short)
]

# Title lookup for metadata
VIDEO_TITLES = {
    "V-QkILd4V-w": "8 Visual Return Strategies Tested by WTA Pro",
    "XXlndjnrA4E": "Peripheral Vision Lets You Volley Like the Pros",
    "FOmz8Wjv3DQ": "Deeper Than Just Footwork - Movement Fundamentals",
    "wVa4XQPcaqs": "Use The Wall to Find Your Perfect Contact",
    "E_zmENJIj4g": "Coiling/Spinal Motion/Eye Dominance/Arm Slot - 15 Serves",
    "Fu6DkHvZlGY": "The Pure-Dextral Pinpoint - Serve Like Mensik",
    "dx8aGSIo24w": "4 Tips for Two-Handers",
    "PUN6qIIYU-4": "4 Tips for the Two-Handed Backhand",
    "dDYKuNZtdyU": "Find Your One-Handed Backhand Pull Slot",
    "pQ793MBQE50": "How Scap Retraction Powers One-Handed Backhand",
    "NEpD7fIM7HI": "Learning from Federer's Slice - 4 Tips + 3 Drills",
    "YbLit9png2U": "Fix Your Kick Serve by Throwing Sideways",
    "QXAtdSEUkfY": "3 Tips to Rip the Low Backhand",
    "XjJHA91HDbU": "Don't Swing AT the Ball on the Overhead",
    "dnNOOornvek": "Retired WTA Pro Still Ripping Backhands",
}

VIDEO_DURATIONS = {
    "V-QkILd4V-w": 1022,
    "XXlndjnrA4E": 499,
    "FOmz8Wjv3DQ": 1066,
    "wVa4XQPcaqs": 276,
    "E_zmENJIj4g": 2625,
    "Fu6DkHvZlGY": 436,
    "dx8aGSIo24w": 283,
    "PUN6qIIYU-4": 887,
    "dDYKuNZtdyU": 666,
    "pQ793MBQE50": 988,
    "NEpD7fIM7HI": 1052,
    "YbLit9png2U": 202,
    "QXAtdSEUkfY": 186,
    "XjJHA91HDbU": 212,
    "dnNOOornvek": 48,
}

# Non-forehand video IDs get lower confidence (0.6 instead of 0.8)
NON_FOREHAND_IDS = {
    "E_zmENJIj4g",   # Serve
    "Fu6DkHvZlGY",   # Serve
    "dx8aGSIo24w",   # Backhand
    "PUN6qIIYU-4",   # Backhand
    "dDYKuNZtdyU",   # Backhand
    "pQ793MBQE50",   # Backhand
    "NEpD7fIM7HI",   # Slice (backhand-family)
    "YbLit9png2U",   # Serve
    "QXAtdSEUkfY",   # Backhand
    "XjJHA91HDbU",   # Overhead
}

# ── Paths ────────────────────────────────────────────────────────────

CONFIG_PATH = PROJECT_ROOT / "config" / "youtube_api_config.json"
PROMPT_PATH = PROJECT_ROOT / "docs" / "knowledge_graph" / "video_analysis_prompt.md"
STATE_PATH = PROJECT_ROOT / "knowledge" / "state" / "batch3_state.json"
ANALYSIS_DIR = PROJECT_ROOT / "docs" / "research" / "ftt_video_analyses"
EXTRACT_DIR = PROJECT_ROOT / "knowledge" / "extracted" / "ftt_videos"
REGISTRY_PATH = PROJECT_ROOT / "knowledge" / "graph" / "_registry_snapshot.json"

DELAY_BETWEEN_CALLS = 20  # seconds between API calls (conservative for parallel batches)


def init_batch3_state() -> dict:
    """Create or load batch3-specific state with all 15 videos."""
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))

    all_ids = BATCH3_VIDEO_IDS + SKIP_VIDEO_IDS
    videos = {}
    for vid in all_ids:
        videos[vid] = {
            "video_id": vid,
            "title": VIDEO_TITLES.get(vid, "Unknown"),
            "url": f"https://www.youtube.com/watch?v={vid}",
            "duration": VIDEO_DURATIONS.get(vid, 0),
            "status": "pending",
            "analysis_file": None,
            "extracted_file": None,
            "analyzed_at": None,
            "error": None,
        }

    state = {
        "batch": 3,
        "channel_id": "@FaultTolerantTennis",
        "total_videos": len(all_ids),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "videos": videos,
    }
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_state(state, STATE_PATH)
    return state


def handle_skipped(state: dict) -> None:
    """Mark skipped videos and create minimal JSON."""
    for vid in SKIP_VIDEO_IDS:
        if state["videos"].get(vid, {}).get("status") == "skipped":
            continue

        # Mark as skipped in state
        state["videos"][vid]["status"] = "skipped"
        state["videos"][vid]["error"] = "skipped: too short for analysis (48s)"
        state["videos"][vid]["analyzed_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        save_state(state, STATE_PATH)

        # Create minimal empty extraction JSON
        EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
        minimal = {
            "video_id": vid,
            "title": VIDEO_TITLES.get(vid, "Unknown"),
            "url": f"https://www.youtube.com/watch?v={vid}",
            "duration": VIDEO_DURATIONS.get(vid, 0),
            "status": "skipped",
            "reason": "Too short for meaningful analysis (48 seconds)",
            "concepts": [],
            "edges": [],
            "diagnostic_chains": [],
        }
        out_path = EXTRACT_DIR / f"{vid}.json"
        out_path.write_text(json.dumps(minimal, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  SKIPPED: {vid} ({VIDEO_TITLES.get(vid, '')}) -- too short")


def main() -> None:
    """Run batch 3 analysis pipeline."""
    print("=" * 60)
    print("FTT Batch 3: Analyzing 14 remaining videos")
    print("=" * 60)

    # 1. Load API config and verify proxy
    config = load_api_config(CONFIG_PATH)
    base_url = config.get("base_url", "DEFAULT (no proxy)")
    print(f"Using API base_url: {base_url}")
    assert config.get("base_url"), "ERROR: base_url not set in config, proxy not configured"

    # 2. Create client
    client = create_client(config)
    model = config.get("model", "gemini-3-flash-preview")
    print(f"Model: {model}")

    # 3. Load prompt
    prompt = load_analysis_prompt(PROMPT_PATH)
    print(f"Prompt loaded: {len(prompt)} chars")

    # 4. Initialize state
    state = init_batch3_state()

    # 5. Handle skipped videos first
    print("\n--- Handling skipped videos ---")
    handle_skipped(state)

    # 6. Load registry for concept extraction
    registry = ConceptRegistry()
    if REGISTRY_PATH.exists():
        registry.load(REGISTRY_PATH)
        print(f"Registry loaded: {len(registry)} concepts")
    else:
        print("Registry not found, starting fresh")

    # 7. Staggered start
    print("\nStaggered start: sleeping 60s to offset from batch1 and batch2...")
    time.sleep(60)

    # 8. Process each video
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    success_count = 0
    fail_count = 0

    for i, vid in enumerate(BATCH3_VIDEO_IDS):
        title = VIDEO_TITLES.get(vid, "Unknown")
        entry = state["videos"].get(vid, {})

        # Skip already processed
        if entry.get("status") in ("analyzed", "extracted"):
            print(f"\n[{i+1}/14] SKIP (already {entry['status']}): {vid} - {title}")
            success_count += 1
            continue

        print(f"\n[{i+1}/14] Analyzing: {vid} - {title}")
        is_long = VIDEO_DURATIONS.get(vid, 0) > 2000
        if is_long:
            print(f"  (LONG VIDEO: {VIDEO_DURATIONS[vid]}s -- extended timeout)")

        # ── Analyze via Gemini ──
        try:
            analysis_text = analyze_video(client, vid, prompt, model=model)

            # Save raw markdown
            md_path = ANALYSIS_DIR / f"{vid}.md"
            md_path.write_text(analysis_text, encoding="utf-8")
            print(f"  Raw analysis saved: {md_path.name} ({len(analysis_text)} chars)")

            # Update state to analyzed
            state["videos"][vid]["status"] = "analyzed"
            state["videos"][vid]["analysis_file"] = str(md_path)
            state["videos"][vid]["analyzed_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            save_state(state, STATE_PATH)

        except Exception as e:
            print(f"  FAILED analysis: {e}")
            state["videos"][vid]["status"] = "failed"
            state["videos"][vid]["error"] = str(e)[:500]
            save_state(state, STATE_PATH)
            fail_count += 1
            # Still delay before next call
            if i < len(BATCH3_VIDEO_IDS) - 1:
                time.sleep(DELAY_BETWEEN_CALLS)
            continue

        # ── Extract concepts ──
        try:
            is_non_forehand = vid in NON_FOREHAND_IDS
            concepts, edges, chains = extract_concepts_from_analysis(
                analysis_text, vid, title, registry
            )

            # Lower confidence for non-forehand videos
            if is_non_forehand:
                for c in concepts:
                    c.confidence = 0.6
                for e in edges:
                    e.confidence = min(e.confidence, 0.6)

            # Save extraction JSON
            out_path = save_video_extraction(vid, concepts, edges, chains, EXTRACT_DIR)
            print(f"  Extraction saved: {out_path.name} ({len(concepts)} concepts, {len(edges)} edges)")

            # Update state to extracted
            state["videos"][vid]["status"] = "extracted"
            state["videos"][vid]["extracted_file"] = str(out_path)
            save_state(state, STATE_PATH)

            success_count += 1

        except Exception as e:
            print(f"  FAILED extraction (analysis saved): {e}")
            # Keep status as "analyzed" since raw markdown was saved
            success_count += 1  # analysis succeeded even if extraction failed

        # Delay between API calls (skip after last)
        if i < len(BATCH3_VIDEO_IDS) - 1:
            print(f"  Waiting {DELAY_BETWEEN_CALLS}s before next call...")
            time.sleep(DELAY_BETWEEN_CALLS)

    # ── Summary ──
    print("\n" + "=" * 60)
    print("BATCH 3 COMPLETE")
    print(f"  Succeeded: {success_count}/14")
    print(f"  Failed:    {fail_count}/14")
    print(f"  Skipped:   {len(SKIP_VIDEO_IDS)} (too short)")
    print(f"  State:     {STATE_PATH}")

    # Final state summary
    status_counts = {}
    for v in state["videos"].values():
        s = v["status"]
        status_counts[s] = status_counts.get(s, 0) + 1
    print(f"  Status breakdown: {status_counts}")
    print("=" * 60)


if __name__ == "__main__":
    main()
