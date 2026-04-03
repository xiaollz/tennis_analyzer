"""Structured concept extraction from FTT video analysis text.

Parses Gemini-generated analysis markdown into Concept, Edge, and
DiagnosticChain Pydantic objects. Works with both existing multi-video
analysis files (09_ftt_videos_{1,2,3}.md) and new per-video analyses.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from knowledge.registry import ConceptRegistry
from knowledge.schemas import (
    Concept,
    ConceptType,
    DiagnosticChain,
    DiagnosticStep,
    Edge,
    RelationType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_snake_id(english_name: str) -> str:
    """Convert an English name to a valid snake_case concept ID.

    E.g. 'Hitting Through the Ball' -> 'hitting_through_the_ball'
    """
    # Remove non-alphanumeric chars except spaces
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", english_name)
    # Collapse whitespace and convert to snake_case
    parts = cleaned.lower().split()
    # Filter out very short words to keep IDs manageable
    snake = "_".join(p for p in parts if p)
    # Ensure starts with letter
    if snake and not snake[0].isalpha():
        snake = "c_" + snake
    return snake or "unknown_concept"


def _parse_category(cat_str: str) -> ConceptType:
    """Map a category string to ConceptType enum."""
    mapping = {
        "technique": ConceptType.TECHNIQUE,
        "biomechanics": ConceptType.BIOMECHANICS,
        "drill": ConceptType.DRILL,
        "symptom": ConceptType.SYMPTOM,
        "mental_model": ConceptType.MENTAL_MODEL,
        "connection": ConceptType.CONNECTION,
        # Chinese category names
        "技术": ConceptType.TECHNIQUE,
        "生物力学": ConceptType.BIOMECHANICS,
        "训练": ConceptType.DRILL,
        "训练方法": ConceptType.DRILL,
        "症状": ConceptType.SYMPTOM,
        "心理模型": ConceptType.MENTAL_MODEL,
    }
    normalized = cat_str.strip().lower()
    return mapping.get(normalized, ConceptType.TECHNIQUE)


def _parse_relation(rel_str: str) -> RelationType:
    """Map a relation string to RelationType enum."""
    mapping = {
        "supports": RelationType.SUPPORTS,
        "refines": RelationType.SUPPORTS,  # Map refines to supports
        "extends": RelationType.SUPPORTS,  # Map extends to supports
        "contradicts": RelationType.CONTRADICTS,
        "causes": RelationType.CAUSES,
        "prevents": RelationType.PREVENTS,
        "requires": RelationType.REQUIRES,
        "drills_for": RelationType.DRILLS_FOR,
        "visible_as": RelationType.VISIBLE_AS,
        "is_instance_of": RelationType.SUPPORTS,
        "provides_drill_for": RelationType.DRILLS_FOR,
    }
    normalized = rel_str.strip().lower()
    return mapping.get(normalized, RelationType.SUPPORTS)


# ---------------------------------------------------------------------------
# Multi-video markdown parsing
# ---------------------------------------------------------------------------

def extract_from_existing_markdown(md_path: Path) -> list[dict]:
    """Parse multi-video analysis files into per-video sections.

    Splits by '## N.' headers and extracts video_id from YouTube URL.
    Deduplicates by video_id (keeps first occurrence).

    Args:
        md_path: Path to a multi-video analysis markdown file
                 (e.g. 09_ftt_videos_1.md).

    Returns:
        List of dicts with keys: video_id, title, content.
    """
    text = md_path.read_text(encoding="utf-8")

    # Split by ## N. headers (where N is a number)
    # Pattern: ## <number>. <title>
    sections = re.split(r"^## (\d+)\.\s+", text, flags=re.MULTILINE)

    videos: list[dict] = []
    seen_ids: set[str] = set()

    # sections[0] is the preamble before first ## N.
    # Then pairs: sections[1]=num, sections[2]=content, sections[3]=num, ...
    for i in range(1, len(sections), 2):
        if i + 1 >= len(sections):
            break

        num = sections[i]
        content = sections[i + 1]

        # Extract title from first line of content
        title_line = content.split("\n")[0].strip()

        # Extract video ID from YouTube URL
        url_match = re.search(
            r"youtube\.com/watch\?v=([A-Za-z0-9_-]+)", content
        )
        video_id = url_match.group(1) if url_match else f"unknown_{num}"

        # Deduplicate by video_id
        if video_id in seen_ids:
            continue
        seen_ids.add(video_id)

        videos.append({
            "video_id": video_id,
            "title": title_line,
            "content": content,
        })

    return videos


# ---------------------------------------------------------------------------
# Concept extraction from analysis text
# ---------------------------------------------------------------------------

def extract_concepts_from_analysis(
    analysis_text: str,
    video_id: str,
    video_title: str,
    registry: ConceptRegistry,
) -> tuple[list[Concept], list[Edge], list[DiagnosticChain]]:
    """Parse analysis text to extract structured Pydantic objects.

    Extracts concepts from:
    - Bold terms (**English (Chinese)**) in technique sections
    - Teaching points (numbered items with descriptions)
    - Drill/training methods

    Also extracts edges linking related concepts and basic diagnostic info.

    Args:
        analysis_text: Raw analysis markdown from Gemini.
        video_id: YouTube video ID for provenance.
        video_title: Video title for provenance.
        registry: ConceptRegistry for dedup via resolve().

    Returns:
        Tuple of (concepts, edges, diagnostic_chains).
    """
    concepts: list[Concept] = []
    edges: list[Edge] = []
    chains: list[DiagnosticChain] = []

    source_file = f"ftt_video_{video_id}"

    # --- Extract bold terms as concepts ---
    # Pattern: **Chinese (English)：** description  (colon inside bold)
    # Also handles: **English Term (Chinese)：** description
    bold_pattern = re.compile(
        r"\*\*(.+?)(?:：|:)\*\*\s*(.+?)(?:\n|$)"
    )

    for match in bold_pattern.finditer(analysis_text):
        term = match.group(1).strip()
        description = match.group(2).strip()

        # Try to extract English and Chinese names
        en_zh = re.match(r"(.+?)\s*[（(](.+?)[)）]", term)
        if en_zh:
            name_en = en_zh.group(1).strip()
            name_zh = en_zh.group(2).strip()
            # Check if Chinese is first
            if re.match(r"[\u4e00-\u9fff]", name_en):
                name_en, name_zh = name_zh, name_en
        else:
            name_en = term
            name_zh = term

        concept_id = _to_snake_id(name_en)
        if not concept_id or len(concept_id) < 2:
            continue

        # Check registry for existing match
        existing = registry.resolve(name_en, threshold=70)
        if existing:
            # Create edge to existing concept instead
            edges.append(Edge(
                source_id=existing,
                target_id=existing,
                relation=RelationType.SUPPORTS,
                confidence=0.7,
                evidence=f"Referenced in video: {video_title}",
                source_file=source_file,
            ))
            continue

        # Create new concept
        concept = Concept(
            id=concept_id,
            name=name_en,
            name_zh=name_zh,
            category=ConceptType.TECHNIQUE,
            sources=["ftt"],
            description=description[:200],
            confidence=0.8,
        )

        # Try to register (dedup check)
        dup = registry.add(concept)
        if dup is None:
            concepts.append(concept)

    # --- Extract drill/training concepts ---
    drill_pattern = re.compile(
        r"训练(?:方法|名称)?[：:].*?\*\*(.+?)\*\*",
        re.MULTILINE,
    )
    for match in drill_pattern.finditer(analysis_text):
        drill_name = match.group(1).strip()
        en_match = re.search(r"[（(](.+?)[)）]", drill_name)
        name_en = en_match.group(1) if en_match else drill_name
        name_zh = re.sub(r"\s*[（(].+?[)）]", "", drill_name)

        drill_id = _to_snake_id(name_en)
        if not drill_id or len(drill_id) < 2:
            continue

        existing = registry.resolve(name_en, threshold=70)
        if existing:
            continue

        drill_concept = Concept(
            id=drill_id,
            name=name_en,
            name_zh=name_zh or name_en,
            category=ConceptType.DRILL,
            sources=["ftt"],
            description=f"Training drill from FTT video: {video_title}",
            confidence=0.8,
        )
        dup = registry.add(drill_concept)
        if dup is None:
            concepts.append(drill_concept)

    # --- Create edges between extracted concepts ---
    concept_ids = [c.id for c in concepts]
    if len(concept_ids) >= 2:
        # Link concepts that appear in the same video analysis
        for i, src_id in enumerate(concept_ids):
            for tgt_id in concept_ids[i + 1:]:
                edges.append(Edge(
                    source_id=src_id,
                    target_id=tgt_id,
                    relation=RelationType.SUPPORTS,
                    confidence=0.6,
                    evidence=f"Co-occurring in video: {video_title}",
                    source_file=source_file,
                ))

    return concepts, edges, chains


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_video_extraction(
    video_id: str,
    concepts: list[Concept],
    edges: list[Edge],
    chains: list[DiagnosticChain],
    output_dir: Path,
) -> Path:
    """Save per-video extraction results as JSON.

    Args:
        video_id: YouTube video ID.
        concepts: Extracted Concept objects.
        edges: Extracted Edge objects.
        chains: Extracted DiagnosticChain objects.
        output_dir: Directory for output files.

    Returns:
        Path to the created JSON file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{video_id}.json"

    output = {
        "video_id": video_id,
        "concepts": [c.model_dump() for c in concepts],
        "edges": [e.model_dump() for e in edges],
        "diagnostic_chains": [ch.model_dump() for ch in chains],
    }

    out_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_path
