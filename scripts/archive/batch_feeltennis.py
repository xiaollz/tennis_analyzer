#!/usr/bin/env python3
"""Batch analyze Feel Tennis forehand videos via Gemini API and extract concepts.

Only processes publicly accessible videos (31/46 are members-only).
Uses extended timeout to handle proxy slow responses.
Source tag: feeltennis_video_{video_id}.
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from google import genai
from google.genai import types

from knowledge.pipeline.video_analyzer import load_api_config, load_analysis_prompt
from knowledge.pipeline.video_concept_extractor import (
    extract_concepts_from_analysis,
    save_video_extraction,
)
from knowledge.pipeline.video_state import (
    get_state_summary,
    get_videos_by_status,
    load_state,
    mark_video,
    save_state,
)
from knowledge.registry import ConceptRegistry


def create_extended_client(config: dict) -> genai.Client:
    """Create Gemini client with extended timeout for video analysis."""
    return genai.Client(
        api_key=config["api_key"],
        http_options=types.HttpOptions(
            base_url=config["base_url"],
            timeout=600000,  # 10 minute timeout for video processing
        ),
    )


def analyze_video_extended(
    client: genai.Client,
    video_id: str,
    prompt: str,
    model: str = "gemini-3-flash-preview",
    max_retries: int = 3,
) -> str:
    """Analyze a YouTube video with extended timeout and manual retry."""
    url = f"https://www.youtube.com/watch?v={video_id}"

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=types.Content(
                    parts=[
                        types.Part(file_data=types.FileData(file_uri=url)),
                        types.Part(text=prompt),
                    ]
                ),
                config=types.GenerateContentConfig(temperature=0),
            )
            return response.text
        except Exception as e:
            err_str = str(e)
            # 403 = not accessible, no point retrying
            if "403" in err_str:
                raise
            wait = min(30 * (2 ** attempt), 120)
            print(f"    Attempt {attempt+1}/{max_retries} failed: {type(e).__name__}: {err_str[:150]}", flush=True)
            if attempt < max_retries - 1:
                print(f"    Retrying in {wait}s...", flush=True)
                time.sleep(wait)

    raise Exception(f"All {max_retries} attempts failed for {video_id}")


def main():
    config_path = PROJECT_ROOT / "config" / "youtube_api_config.json"
    state_path = PROJECT_ROOT / "knowledge" / "state" / "feeltennis_video_state.json"
    analysis_dir = PROJECT_ROOT / "docs" / "research" / "feeltennis_video_analyses"
    extract_dir = PROJECT_ROOT / "knowledge" / "extracted" / "feeltennis_videos"
    registry_path = PROJECT_ROOT / "knowledge" / "extracted" / "_registry_snapshot.json"

    # 1. Load config and create client with extended timeout
    print("Loading API config...", flush=True)
    config = load_api_config(config_path)
    client = create_extended_client(config)

    # 2. Load prompt
    print("Loading analysis prompt...", flush=True)
    prompt = load_analysis_prompt()

    # 3. Load state
    print("Loading Feel Tennis video state...", flush=True)
    state = load_state(state_path)
    summary = get_state_summary(state)
    print(f"State: {summary}", flush=True)

    # 4. Get pending video IDs (only accessible ones)
    pending = get_videos_by_status(state, "pending")
    pending_ids = [v["video_id"] for v in pending]
    print(f"Pending accessible videos: {len(pending_ids)}", flush=True)

    if not pending_ids:
        print("No pending videos.", flush=True)
    else:
        # 5. Analyze each video
        analysis_dir.mkdir(parents=True, exist_ok=True)
        analyzed_count = 0
        failed_count = 0

        for i, video_id in enumerate(pending_ids):
            entry = state["videos"].get(video_id, {})
            if entry.get("status") in ("analyzed", "extracted"):
                continue

            title = entry.get("title", video_id)
            print(f"\n[{i+1}/{len(pending_ids)}] Analyzing: {title} ({video_id})", flush=True)

            try:
                analysis_text = analyze_video_extended(client, video_id, prompt)

                # Save raw markdown
                md_path = analysis_dir / f"{video_id}.md"
                md_path.write_text(analysis_text, encoding="utf-8")

                # Update state
                mark_video(
                    state, video_id, "analyzed",
                    analysis_file=str(md_path),
                    analyzed_at=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                )
                save_state(state, state_path)

                analyzed_count += 1
                print(f"  OK - {len(analysis_text)} chars", flush=True)

            except Exception as e:
                mark_video(state, video_id, "failed", error=str(e)[:500])
                save_state(state, state_path)
                failed_count += 1
                print(f"  FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)

            # Delay between calls
            if i < len(pending_ids) - 1:
                time.sleep(20)

        print(f"\nAnalysis complete: {analyzed_count} OK, {failed_count} failed", flush=True)

    # 6. Load registry
    print("\nLoading concept registry...", flush=True)
    registry = ConceptRegistry()
    if registry_path.exists():
        snapshot = json.loads(registry_path.read_text(encoding="utf-8"))
        for concept_data in snapshot.get("concepts", []):
            from knowledge.schemas import Concept
            try:
                c = Concept(**concept_data)
                registry.add(c)
            except Exception:
                pass
    print(f"Registry loaded: {len(registry)} concepts", flush=True)

    # 7. Extract concepts from analyzed videos
    analyzed = get_videos_by_status(state, "analyzed")
    print(f"\nVideos needing extraction: {len(analyzed)}", flush=True)

    total_concepts = 0
    total_edges = 0

    for v in analyzed:
        vid_id = v["video_id"]
        title = v["title"]
        md_path = analysis_dir / f"{vid_id}.md"

        if not md_path.exists():
            print(f"  SKIP {vid_id}: no analysis file", flush=True)
            continue

        try:
            analysis_text = md_path.read_text(encoding="utf-8")

            concepts, edges, chains = extract_concepts_from_analysis(
                analysis_text=analysis_text,
                video_id=vid_id,
                video_title=f"[FeelTennis] {title}",
                registry=registry,
            )

            # Post-process source tags
            for c in concepts:
                if "ftt" in c.sources:
                    c.sources = ["feeltennis"]
            for e in edges:
                if e.source_file.startswith("ftt_video_"):
                    e.source_file = e.source_file.replace("ftt_video_", "feeltennis_video_")

            out_path = save_video_extraction(vid_id, concepts, edges, chains, extract_dir)
            mark_video(state, vid_id, "extracted", extracted_file=str(out_path))
            save_state(state, state_path)

            total_concepts += len(concepts)
            total_edges += len(edges)
            print(f"  OK {vid_id}: {len(concepts)} concepts, {len(edges)} edges", flush=True)

        except Exception as e:
            print(f"  ERR {vid_id}: {e}", flush=True)
            mark_video(state, vid_id, "failed", error=f"extraction: {str(e)}")
            save_state(state, state_path)

    # 8. Final summary
    final_summary = get_state_summary(state)
    print(f"\n=== FINAL STATE ===", flush=True)
    print(f"State: {final_summary}", flush=True)
    print(f"Total new concepts: {total_concepts}", flush=True)
    print(f"Total new edges: {total_edges}", flush=True)
    print(f"Analysis files: {len(list(analysis_dir.glob('*.md'))) if analysis_dir.exists() else 0}", flush=True)
    print(f"Extraction files: {len(list(extract_dir.glob('*.json'))) if extract_dir.exists() else 0}", flush=True)


if __name__ == "__main__":
    main()
