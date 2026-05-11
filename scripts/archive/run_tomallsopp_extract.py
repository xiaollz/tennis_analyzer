"""Extract structured concepts from already-analyzed TomAllsopp videos.

Run after run_tomallsopp_batch.py has completed the Gemini API analysis phase.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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
from knowledge.schemas import Concept


def main():
    state_path = PROJECT_ROOT / "knowledge" / "state" / "tomallsopp_video_state.json"
    analysis_dir = PROJECT_ROOT / "docs" / "research" / "tomallsopp_video_analyses"
    extract_dir = PROJECT_ROOT / "knowledge" / "extracted" / "tomallsopp_videos"
    registry_path = PROJECT_ROOT / "knowledge" / "extracted" / "_registry_snapshot.json"

    # Load state
    state = load_state(state_path)
    summary = get_state_summary(state)
    print(f"Current state: {summary}")

    # Load registry (it's a flat list of concept dicts)
    print("Loading concept registry...")
    registry = ConceptRegistry()
    if registry_path.exists():
        snapshot = json.loads(registry_path.read_text(encoding="utf-8"))
        if isinstance(snapshot, list):
            concept_list = snapshot
        elif isinstance(snapshot, dict):
            concept_list = snapshot.get("concepts", [])
        else:
            concept_list = []

        loaded = 0
        for concept_data in concept_list:
            try:
                c = Concept(**concept_data)
                registry.add(c)
                loaded += 1
            except Exception:
                pass
        print(f"Registry loaded: {loaded} existing concepts")

    # Extract concepts from all analyzed videos
    analyzed = get_videos_by_status(state, "analyzed")
    print(f"Videos needing extraction: {len(analyzed)}")

    total_concepts = 0
    total_edges = 0
    extraction_errors = 0

    for v in analyzed:
        vid_id = v["video_id"]
        title = v["title"]
        md_path = analysis_dir / f"{vid_id}.md"

        if not md_path.exists():
            print(f"  SKIP {vid_id}: analysis file not found")
            extraction_errors += 1
            continue

        try:
            analysis_text = md_path.read_text(encoding="utf-8")

            concepts, edges, chains = extract_concepts_from_analysis(
                analysis_text=analysis_text,
                video_id=vid_id,
                video_title=f"[TomAllsopp] {title}",
                registry=registry,
            )

            # Post-process: fix source tags
            for c in concepts:
                if "ftt" in c.sources:
                    c.sources = ["tomallsopp"]
            for e in edges:
                if e.source_file.startswith("ftt_video_"):
                    e.source_file = e.source_file.replace("ftt_video_", "tomallsopp_video_")

            out_path = save_video_extraction(vid_id, concepts, edges, chains, extract_dir)

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

    # Final summary
    final_summary = get_state_summary(state)
    print(f"\n=== FINAL STATE ===")
    print(f"State: {final_summary}")
    print(f"Total new concepts: {total_concepts}")
    print(f"Total new edges: {total_edges}")
    print(f"Extraction errors: {extraction_errors}")
    print(f"Analysis files: {len(list(analysis_dir.glob('*.md')))}")
    print(f"Extraction files: {len(list(extract_dir.glob('*.json'))) if extract_dir.exists() else 0}")


if __name__ == "__main__":
    main()
