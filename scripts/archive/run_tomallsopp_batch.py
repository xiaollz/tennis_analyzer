"""Batch process TomAllsopp forehand videos via Gemini API.

Analyzes all pending videos and extracts structured concepts.
Uses the existing Phase 3 pipeline with TomAllsopp-specific output dirs.
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from knowledge.pipeline.video_analyzer import (
    analyze_batch,
    create_client,
    load_api_config,
    load_analysis_prompt,
)
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


def main():
    config_path = PROJECT_ROOT / "config" / "youtube_api_config.json"
    state_path = PROJECT_ROOT / "knowledge" / "state" / "tomallsopp_video_state.json"
    analysis_dir = PROJECT_ROOT / "docs" / "research" / "tomallsopp_video_analyses"
    extract_dir = PROJECT_ROOT / "knowledge" / "extracted" / "tomallsopp_videos"
    registry_path = PROJECT_ROOT / "knowledge" / "extracted" / "_registry_snapshot.json"

    # 1. Load config and create client
    print("Loading API config...")
    config = load_api_config(config_path)
    client = create_client(config)

    # 2. Load prompt
    print("Loading analysis prompt...")
    prompt = load_analysis_prompt()

    # 3. Load state
    print("Loading TomAllsopp video state...")
    state = load_state(state_path)
    summary = get_state_summary(state)
    print(f"State: {summary}")

    # 4. Get pending video IDs
    pending = get_videos_by_status(state, "pending")
    pending_ids = [v["video_id"] for v in pending]
    print(f"Pending videos: {len(pending_ids)}")

    if not pending_ids:
        print("No pending videos. Checking for analyzed-but-not-extracted...")
    else:
        # 5. Analyze batch
        print(f"\n--- Starting Gemini API analysis for {len(pending_ids)} videos ---")
        print(f"Estimated time: ~{len(pending_ids) * 25 / 60:.0f} minutes")

        results = analyze_batch(
            client=client,
            video_ids=pending_ids,
            prompt=prompt,
            state=state,
            state_path=state_path,
            delay=20.0,
            output_dir=analysis_dir,
        )
        print(f"\nAnalysis complete: {len(results)} succeeded out of {len(pending_ids)} pending")

    # 6. Load registry for concept extraction
    print("\nLoading concept registry...")
    registry = ConceptRegistry()
    if registry_path.exists():
        snapshot = json.loads(registry_path.read_text(encoding="utf-8"))
        # Load existing concepts into registry for dedup
        for concept_data in snapshot.get("concepts", []):
            from knowledge.schemas import Concept
            try:
                c = Concept(**concept_data)
                registry.add(c)
            except Exception:
                pass
    print(f"Registry loaded: {len(registry)} existing concepts")

    # 7. Extract concepts from all analyzed videos
    analyzed = get_videos_by_status(state, "analyzed")
    print(f"\nVideos needing extraction: {len(analyzed)}")

    total_concepts = 0
    total_edges = 0
    extraction_errors = 0

    for v in analyzed:
        vid_id = v["video_id"]
        title = v["title"]
        md_path = analysis_dir / f"{vid_id}.md"

        if not md_path.exists():
            print(f"  SKIP {vid_id}: analysis file not found at {md_path}")
            extraction_errors += 1
            continue

        try:
            analysis_text = md_path.read_text(encoding="utf-8")

            # Extract with TomAllsopp source tag via title prefix
            concepts, edges, chains = extract_concepts_from_analysis(
                analysis_text=analysis_text,
                video_id=vid_id,
                video_title=f"[TomAllsopp] {title}",
                registry=registry,
            )

            # Post-process: fix source_file from ftt_video_ to tomallsopp_video_
            for c in concepts:
                if "ftt" in c.sources:
                    c.sources = ["tomallsopp"]
            for e in edges:
                if e.source_file.startswith("ftt_video_"):
                    e.source_file = e.source_file.replace("ftt_video_", "tomallsopp_video_")

            # Save extraction
            out_path = save_video_extraction(vid_id, concepts, edges, chains, extract_dir)

            # Mark as extracted
            mark_video(state, vid_id, "extracted", extracted_file=str(out_path))
            save_state(state, state_path)

            total_concepts += len(concepts)
            total_edges += len(edges)
            print(f"  OK {vid_id}: {len(concepts)} concepts, {len(edges)} edges")

        except Exception as e:
            print(f"  ERR {vid_id}: {e}")
            mark_video(state, vid_id, "failed", error=f"extraction: {str(e)}")
            save_state(state, state_path)
            extraction_errors += 1

    # 8. Final summary
    final_summary = get_state_summary(state)
    print(f"\n=== FINAL STATE ===")
    print(f"State: {final_summary}")
    print(f"Total new concepts: {total_concepts}")
    print(f"Total new edges: {total_edges}")
    print(f"Extraction errors: {extraction_errors}")
    print(f"Analysis files: {len(list(analysis_dir.glob('*.md'))) if analysis_dir.exists() else 0}")
    print(f"Extraction files: {len(list(extract_dir.glob('*.json'))) if extract_dir.exists() else 0}")


if __name__ == "__main__":
    main()
