"""Extraction pipeline with file-type handler dispatch.

Provides handler functions for each source category and an orchestration
function that dispatches files to the appropriate handler based on filename
prefix patterns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from knowledge.registry import ConceptRegistry
from knowledge.schemas import Concept, Edge


@dataclass
class ExtractionResult:
    """Result of extracting knowledge from a single source file."""

    concepts: list[Concept] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    source_file: str = ""


# ---------------------------------------------------------------------------
# Handler stubs (implemented in Plans 02 and 03)
# ---------------------------------------------------------------------------


def extract_ftt_book(filepath: Path, registry: ConceptRegistry) -> ExtractionResult:
    """Extract concepts from FTT book Markdown files."""
    return ExtractionResult(source_file=str(filepath))


def extract_ftt_blog(filepath: Path, registry: ConceptRegistry) -> ExtractionResult:
    """Extract concepts from FTT blog article Markdown files."""
    return ExtractionResult(source_file=str(filepath))


def extract_ftt_videos(filepath: Path, registry: ConceptRegistry) -> ExtractionResult:
    """Extract concepts from FTT video synthesis Markdown files."""
    return ExtractionResult(source_file=str(filepath))


def extract_tpa_videos(filepath: Path, registry: ConceptRegistry) -> ExtractionResult:
    """Extract concepts from TPA Tennis video synthesis files."""
    return ExtractionResult(source_file=str(filepath))


def extract_biomechanics(filepath: Path, registry: ConceptRegistry) -> ExtractionResult:
    """Extract concepts from biomechanics textbook Markdown files."""
    return ExtractionResult(source_file=str(filepath))


def extract_user_journey(filepath: Path, registry: ConceptRegistry) -> ExtractionResult:
    """Extract concepts from user training journey files."""
    return ExtractionResult(source_file=str(filepath))


def extract_generic(filepath: Path, registry: ConceptRegistry) -> ExtractionResult:
    """Fallback handler for unrecognized file types."""
    return ExtractionResult(source_file=str(filepath))


# ---------------------------------------------------------------------------
# Handler dispatch
# ---------------------------------------------------------------------------

FILE_HANDLERS: dict[str, callable] = {
    "01_ftt_book": extract_ftt_book,
    "02_revolutionary_tennis": extract_ftt_book,  # same format
    "04_ftt_blog_": extract_ftt_blog,
    "12_ftt_videos": extract_ftt_videos,
    "15_tpa": extract_tpa_videos,
    "17_kinetic": extract_tpa_videos,
    "24_bio": extract_biomechanics,
    "25_bio": extract_biomechanics,
    "13_synthesis": extract_generic,  # synthesis is cross-cutting
    "learning": extract_user_journey,
}


def get_handler(filename: str):
    """Return the appropriate handler function for a given filename.

    Matches against registered filename prefixes. Falls back to
    extract_generic if no prefix matches.
    """
    for prefix, handler in FILE_HANDLERS.items():
        if filename.startswith(prefix):
            return handler
    return extract_generic


def run_extraction(
    files: list[Path], registry: ConceptRegistry
) -> list[ExtractionResult]:
    """Run extraction pipeline on a list of files.

    Dispatches each file to its appropriate handler based on filename prefix.

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
