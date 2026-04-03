"""Extraction pipeline with file-type handler dispatch.

Provides handler functions for each source category and an orchestration
function that dispatches files to the appropriate handler based on filename
prefix patterns.

Processing order (per RESEARCH.md):
  synthesis (13, 12, 15, 17) -> FTT book/blog (01, 02, 04-08) ->
  FTT specific (18-23) -> FTT videos (09) -> TPA (14, 16) ->
  biomechanics (24-28) -> misc (03)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from knowledge.registry import ConceptRegistry
from knowledge.schemas import Concept, Edge

from knowledge.pipeline.handlers import (
    extract_biomechanics,
    extract_ftt_blog,
    extract_ftt_book,
    extract_ftt_specific,
    extract_ftt_videos,
    extract_generic,
    extract_synthesis,
    extract_tpa_videos,
    extract_user_journey,
)


@dataclass
class ExtractionResult:
    """Result of extracting knowledge from a single source file."""

    concepts: list[Concept] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    source_file: str = ""


# ---------------------------------------------------------------------------
# Handler dispatch
# ---------------------------------------------------------------------------

# Map filename prefixes to handler functions
FILE_HANDLERS: dict[str, callable] = {
    # Synthesis files
    "13_synthesis": extract_synthesis,
    "12_ftt_videos_synthesis": extract_synthesis,
    "15_tpa_synthesis": extract_synthesis,
    "17_kinetic_chain_synthesis": extract_synthesis,
    # FTT Book
    "01_ftt_book": extract_ftt_book,
    "02_revolutionary_tennis": extract_ftt_book,
    # FTT Blog
    "04_ftt_blog": extract_ftt_blog,
    "05_ftt_blog": extract_ftt_blog,
    "06_ftt_blog": extract_ftt_blog,
    "07_ftt_blog": extract_ftt_blog,
    "08_ftt_blog": extract_ftt_blog,
    # FTT Specific deep-dives
    "18_ftt_": extract_ftt_specific,
    "19_forearm": extract_ftt_specific,
    "20_ftt_": extract_ftt_specific,
    "21_ftt_": extract_ftt_specific,
    "22_ftt_": extract_ftt_specific,
    "23_ftt_": extract_ftt_specific,
    # FTT Videos
    "09_ftt_videos": extract_ftt_videos,
    # TPA Videos + Kinetic Chain
    "14_tpa_": extract_tpa_videos,
    "16_tpa_": extract_tpa_videos,
    # Biomechanics
    "24_bio": extract_biomechanics,
    "25_bio": extract_biomechanics,
    "26_bio": extract_biomechanics,
    "27_bio": extract_biomechanics,
    "28_bio": extract_biomechanics,
    # Generic fallback entries
    "03_youtube": extract_generic,
    # User training journey
    "learning": extract_user_journey,  # noqa: E501
}

# Correct processing order: synthesis first, then primary sources,
# then specific, then videos, then biomechanics, then misc
PROCESSING_ORDER = [
    # Wave 1: Synthesis (establish canonical names)
    "13_synthesis",
    "12_ftt_videos_synthesis",
    "15_tpa_synthesis",
    "17_kinetic_chain_synthesis",
    # Wave 2: FTT primary sources
    "01_ftt_book",
    "02_revolutionary_tennis",
    # Wave 3: FTT blog
    "04_ftt_blog_forehand_1",
    "04_ftt_blog_forehand_2",
    "05_ftt_blog_players",
    "06_ftt_blog_movement",
    "07_ftt_blog_vision_movement",
    "08_ftt_blog_vision_strategy",
    # Wave 4: FTT specific deep-dives
    "18_ftt_build_foundation",
    "19_forearm_compensation_analysis",
    "20_ftt_grip_rotation_axis",
    "21_ftt_chest_engagement",
    "22_ftt_scapular_glide",
    "23_ftt_trunk_sequencing",
    # Wave 5: FTT videos (per-video analysis)
    "09_ftt_videos_1",
    "09_ftt_videos_2",
    "09_ftt_videos_3",
    # Wave 6: TPA
    "14_tpa_videos_1",
    "14_tpa_videos_2",
    "14_tpa_videos_3",
    "16_tpa_kinetic_chain_1",
    "16_tpa_kinetic_chain_2",
    # Wave 7: Biomechanics
    "24_biomechanics_ch1_ch8",
    "25_biomechanics_upper_body",
    "26_biomechanics_core_legs",
    "27_biomechanics_new_insights",
    "28_biomechanics_problem_solutions",
    # Wave 8: Miscellaneous
    "03_youtube_notes",
]


def get_handler(filename: str):
    """Return the appropriate handler function for a given filename.

    Matches against registered filename prefixes. Falls back to
    extract_generic if no prefix matches.
    """
    for prefix, handler in FILE_HANDLERS.items():
        if filename.startswith(prefix):
            return handler
    return extract_generic


def sort_files_by_processing_order(files: list[Path]) -> list[Path]:
    """Sort files according to the correct processing order."""
    order_map = {}
    for i, stem in enumerate(PROCESSING_ORDER):
        order_map[stem] = i

    def sort_key(fp: Path) -> int:
        stem = fp.stem
        if stem in order_map:
            return order_map[stem]
        # Try prefix matching
        for key, idx in order_map.items():
            if stem.startswith(key):
                return idx
        return 9999  # Unknown files go last

    return sorted(files, key=sort_key)


def run_extraction(
    files: list[Path], registry: ConceptRegistry
) -> list[ExtractionResult]:
    """Run extraction pipeline on a list of files.

    Dispatches each file to its appropriate handler based on filename prefix.
    Files should be pre-sorted via sort_files_by_processing_order.

    Args:
        files: List of file paths to process.
        registry: ConceptRegistry for deduplication.

    Returns:
        List of ExtractionResult objects, one per file.
    """
    results: list[ExtractionResult] = []
    for filepath in files:
        handler = get_handler(filepath.name)
        result = handler(filepath, registry)
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Category mapping for output directory structure
# ---------------------------------------------------------------------------

def _get_category_dir(filename: str) -> str:
    """Map a filename to its output category subdirectory."""
    if filename.startswith(("13_", "12_ftt_videos_synth", "15_tpa_synth", "17_kinetic_chain_synth")):
        return "synthesis"
    if filename.startswith(("01_", "02_")):
        return "ftt_book"
    if filename.startswith(("04_", "05_", "06_", "07_", "08_")):
        return "ftt_blog"
    if filename.startswith(("18_", "19_", "20_", "21_", "22_", "23_")):
        return "ftt_specific"
    if filename.startswith("09_"):
        return "ftt_videos"
    if filename.startswith(("14_", "16_")):
        return "tpa"
    if filename.startswith(("24_", "25_", "26_", "27_", "28_")):
        return "biomechanics"
    return "misc"


def save_extraction_results(
    results: list[ExtractionResult],
    output_dir: Path,
) -> None:
    """Save per-file JSON outputs into category subdirectories."""
    for result in results:
        filename = Path(result.source_file).stem
        category = _get_category_dir(Path(result.source_file).name)
        cat_dir = output_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        output = {
            "source_file": result.source_file,
            "concepts": [c.model_dump() for c in result.concepts],
            "edges": [e.model_dump() for e in result.edges],
        }
        out_path = cat_dir / f"{filename}.json"
        out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))


def save_registry_snapshot(registry: ConceptRegistry, output_path: Path) -> None:
    """Save the complete registry state as a JSON array."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    concepts = [c.model_dump() for c in registry.all_concepts()]
    output_path.write_text(json.dumps(concepts, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# Main entry point for running the full pipeline
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from knowledge.pipeline.seed import seed_registry_from_legacy_json

    # 1. Setup
    project_root = Path(__file__).resolve().parent.parent.parent
    research_dir = project_root / "docs" / "research"
    output_dir = project_root / "knowledge" / "extracted"

    # 2. Seed registry
    registry = ConceptRegistry()
    seed_concepts = seed_registry_from_legacy_json(registry)
    print(f"Seeded: {len(registry)} concepts from legacy JSON")

    # 3. Collect all research files
    all_files = sorted(research_dir.glob("*.md"))
    print(f"Found {len(all_files)} research files")

    # 4. Sort by processing order
    sorted_files = sort_files_by_processing_order(all_files)

    # 5. Run extraction
    results = run_extraction(sorted_files, registry)

    # 6. Save results
    save_extraction_results(results, output_dir)

    # 7. Save registry snapshot
    snapshot_path = output_dir / "_registry_snapshot.json"
    save_registry_snapshot(registry, snapshot_path)

    # 8. Print summary
    total_concepts = sum(len(r.concepts) for r in results)
    total_edges = sum(len(r.edges) for r in results)
    print(f"\n--- Extraction Summary ---")
    print(f"Files processed: {len(results)}")
    print(f"New concepts extracted: {total_concepts}")
    print(f"Edges extracted: {total_edges}")
    print(f"Registry total: {len(registry)} concepts")

    # Per-category counts
    from collections import Counter
    cat_counts = Counter()
    for r in results:
        cat = _get_category_dir(Path(r.source_file).name)
        cat_counts[cat] += len(r.concepts)
    print(f"\nPer-category new concepts:")
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat}: {count}")

    # Biomechanics muscle check
    bio_with_muscles = sum(
        1 for c in registry.all_concepts()
        if c.muscles_involved
    )
    print(f"\nConcepts with muscles_involved: {bio_with_muscles}")
