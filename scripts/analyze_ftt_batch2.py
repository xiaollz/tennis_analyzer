#!/usr/bin/env python3
"""Batch 2: Analyze 13 FTT philosophy/tactics/topspin videos via Gemini API.

Videos: topspin technique, FTT philosophy, mental models, tactics.
State: knowledge/state/batch2_state.json (per-plan state slice).
Staggered start: 30-second delay before first API call to offset from batch1.
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

# ---------------------------------------------------------------------------
# Batch 2 video list: 13 philosophy/tactics/topspin videos
# ---------------------------------------------------------------------------

BATCH2_VIDEOS = [
    {"video_id": "wd4YRQW3TOc", "title": "4 Tips For Effortless, Controllable Topspin"},
    {"video_id": "GsHkML2mVEI", "title": "Nishioka's Unique Backswing Timing"},
    {"video_id": "OYf48k-cfNI", "title": "Fault Tolerance In Action"},
    {"video_id": "BbGzWTp5pCM", "title": "Practicing in Slow Motion is Killing Match Play"},
    {"video_id": "Qszz0N4fRb4", "title": "Why You Get Tight, and How to Fix It"},
    {"video_id": "JzcA_ku7Yhk", "title": "The Misunderstanding LOSING You Matches"},
    {"video_id": "FxDmVi3EFnE", "title": "The Truth About the Topspin Pro"},
    {"video_id": "Psidjei5BnI", "title": "4 Tips For Consistently Crushing Slow Balls"},
    {"video_id": "mOFtt9PllI0", "title": "The Geometry of Measured Aggression"},
    {"video_id": "w1FakobNq1Q", "title": "Breaking Down the Greatest Tiebreak Ever Played"},
    {"video_id": "8r09TliP-Ak", "title": "I Picked Draper to Win It All (3 patterns)"},
    {"video_id": "_Qu1LOwklAw", "title": "Patient but Ruthless - Alcaraz Broke Down Sinner"},
    {"video_id": "42BfbKsTGb4", "title": "You Aren't Practicing Half of Tennis - RECEIVING"},
]

BATCH2_IDS = [v["video_id"] for v in BATCH2_VIDEOS]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CONFIG_PATH = PROJECT_ROOT / "config" / "youtube_api_config.json"
PROMPT_PATH = PROJECT_ROOT / "docs" / "knowledge_graph" / "video_analysis_prompt.md"
STATE_PATH = PROJECT_ROOT / "knowledge" / "state" / "batch2_state.json"
ANALYSIS_DIR = PROJECT_ROOT / "docs" / "research" / "ftt_video_analyses"
EXTRACT_DIR = PROJECT_ROOT / "knowledge" / "extracted" / "ftt_videos"
REGISTRY_PATH = PROJECT_ROOT / "knowledge" / "extracted" / "_registry_snapshot.json"

DELAY_BETWEEN = 20  # seconds between API calls (conservative for parallel batches)


def init_batch2_state() -> dict:
    """Create or load batch2-specific state file."""
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))

    # Build initial state from BATCH2_VIDEOS
    videos = {}
    for v in BATCH2_VIDEOS:
        vid = v["video_id"]
        videos[vid] = {
            "video_id": vid,
            "title": v["title"],
            "url": f"https://www.youtube.com/watch?v={vid}",
            "status": "pending",
            "analysis_file": None,
            "extracted_file": None,
            "analyzed_at": None,
            "error": None,
        }

    state = {
        "batch": 2,
        "description": "FTT philosophy/tactics/topspin videos",
        "total_videos": len(BATCH2_VIDEOS),
        "videos": videos,
    }

    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return state


def load_registry() -> ConceptRegistry:
    """Load existing concept registry for dedup during extraction."""
    registry = ConceptRegistry()
    if REGISTRY_PATH.exists():
        data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        # Registry expects list of concept dicts
        if isinstance(data, dict) and "concepts" in data:
            for c in data["concepts"]:
                from knowledge.schemas import Concept
                try:
                    concept = Concept(**c)
                    registry.add(concept)
                except Exception:
                    pass
    return registry


def main():
    # --- Startup verification ---
    print("=" * 60)
    print("FTT Batch 2 Analysis: 13 philosophy/tactics/topspin videos")
    print("=" * 60)

    # Load API config
    config = load_api_config(CONFIG_PATH)
    print(f"Using API base_url: {config.get('base_url', 'DEFAULT (no proxy)')}")
    assert config.get("base_url"), "ERROR: base_url not set in config, proxy not configured"

    # Create client
    client = create_client(config)
    model = config.get("model", "gemini-3-flash-preview")
    print(f"Model: {model}")

    # Load prompt
    prompt = load_analysis_prompt(PROMPT_PATH)
    print(f"Prompt loaded: {len(prompt)} chars")

    # Load/create state
    state = init_batch2_state()
    print(f"State: {STATE_PATH}")

    # Load registry for extraction dedup
    registry = load_registry()
    print(f"Registry loaded: {len(registry._concepts)} existing concepts")

    # Create output directories
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Staggered start ---
    print("\nStaggered start: sleeping 30s to offset from batch1...")
    time.sleep(30)
    print("Starting analysis...\n")

    # --- Process each video ---
    total = len(BATCH2_IDS)
    success_count = 0
    fail_count = 0

    for idx, video_id in enumerate(BATCH2_IDS):
        video_info = next(v for v in BATCH2_VIDEOS if v["video_id"] == video_id)
        title = video_info["title"]

        entry = state["videos"].get(video_id, {})
        current_status = entry.get("status", "pending")

        # Skip already completed
        if current_status == "extracted":
            print(f"[{idx+1}/{total}] SKIP (already extracted): {title}")
            success_count += 1
            continue
        if current_status == "analyzed":
            # Need extraction only
            print(f"[{idx+1}/{total}] EXTRACT ONLY (already analyzed): {title}")
            md_path = ANALYSIS_DIR / f"{video_id}.md"
            if md_path.exists():
                analysis_text = md_path.read_text(encoding="utf-8")
                _extract_and_save(video_id, title, analysis_text, state, registry)
                success_count += 1
                continue

        print(f"[{idx+1}/{total}] Analyzing: {title} ({video_id})")

        # --- Step 1: Analyze via Gemini ---
        try:
            analysis_text = analyze_video(client, video_id, prompt, model=model)

            # Save raw markdown
            md_path = ANALYSIS_DIR / f"{video_id}.md"
            md_path.write_text(analysis_text, encoding="utf-8")
            print(f"  -> Raw analysis saved: {md_path.name} ({len(analysis_text)} chars)")

            # Checkpoint: mark analyzed
            mark_video(state, video_id, "analyzed",
                       analysis_file=str(md_path),
                       analyzed_at=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
            save_state(state, STATE_PATH)

        except Exception as e:
            print(f"  !! FAILED analysis: {e}")
            mark_video(state, video_id, "failed", error=str(e))
            save_state(state, STATE_PATH)
            fail_count += 1

            # Still delay before next call
            if idx < total - 1:
                time.sleep(DELAY_BETWEEN)
            continue

        # --- Step 2: Extract concepts ---
        _extract_and_save(video_id, title, analysis_text, state, registry)
        success_count += 1

        # --- Delay between API calls ---
        if idx < total - 1:
            print(f"  -> Waiting {DELAY_BETWEEN}s before next call...")
            time.sleep(DELAY_BETWEEN)

    # --- Summary ---
    print("\n" + "=" * 60)
    print(f"Batch 2 Complete: {success_count} succeeded, {fail_count} failed out of {total}")
    print(f"State saved: {STATE_PATH}")
    print(f"Analyses dir: {ANALYSIS_DIR}")
    print(f"Extractions dir: {EXTRACT_DIR}")
    print("=" * 60)

    return 0 if success_count >= 11 else 1


def _extract_and_save(
    video_id: str,
    title: str,
    analysis_text: str,
    state: dict,
    registry: ConceptRegistry,
) -> None:
    """Extract structured concepts from analysis text and save JSON."""
    try:
        concepts, edges, chains = extract_concepts_from_analysis(
            analysis_text, video_id, title, registry
        )

        out_path = save_video_extraction(
            video_id, concepts, edges, chains, EXTRACT_DIR
        )
        print(f"  -> Extraction saved: {out_path.name} "
              f"({len(concepts)} concepts, {len(edges)} edges)")

        # Checkpoint: mark extracted
        mark_video(state, video_id, "extracted",
                   extracted_file=str(out_path))
        save_state(state, STATE_PATH)

    except Exception as e:
        print(f"  !! FAILED extraction: {e}")
        # Keep status as analyzed (analysis succeeded, extraction failed)
        # Don't overwrite analyzed status with failed
        save_state(state, STATE_PATH)


if __name__ == "__main__":
    sys.exit(main())
