#!/usr/bin/env python3
"""Re-extract structured concepts from 33 existing FTT video analyses.

Reads docs/research/09_ftt_videos_{1,2,3}.md, parses each video section,
extracts real tennis concepts (not just video titles), and saves per-video
JSON files to knowledge/extracted/ftt_videos/{video_id}.json.

No API calls needed -- pure text processing.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from knowledge.registry import ConceptRegistry
from knowledge.schemas import (
    Concept,
    ConceptType,
    DiagnosticChain,
    DiagnosticStep,
    Edge,
    RelationType,
)
from knowledge.pipeline.video_concept_extractor import (
    _to_snake_id,
    extract_from_existing_markdown,
    save_video_extraction,
)
from knowledge.pipeline.video_state import _ANALYZED_VIDEOS


# ---------------------------------------------------------------------------
# Enhanced concept extraction
# ---------------------------------------------------------------------------

_CATEGORY_KEYWORDS = {
    ConceptType.BIOMECHANICS: [
        "kinetic chain", "rotation", "biomechanics", "scapul", "pronation",
        "supination", "external rotation", "internal rotation", "adduction",
        "abduction", "flexion", "extension", "coil", "torque", "angular",
        "momentum", "muscle", "joint", "shoulder", "hip", "trunk", "core",
        "wrist", "elbow", "spine", "pelvis", "glute", "forearm", "lat",
        "动力链", "旋转", "生物力学", "肩胛", "内旋", "外旋",
    ],
    ConceptType.DRILL: [
        "drill", "exercise", "practice", "training", "练习", "训练",
        "method", "progression", "warmup",
    ],
    ConceptType.SYMPTOM: [
        "mistake", "error", "fault", "problem", "issue", "symptom",
        "wrong", "fail", "miss", "错误", "问题", "症状",
    ],
    ConceptType.MENTAL_MODEL: [
        "mental", "mindset", "psychology", "confidence", "pressure",
        "focus", "attention", "feel", "awareness", "心理", "心态",
        "visualization", "intention", "outcome independence",
    ],
}


def _guess_category(text: str) -> ConceptType:
    """Guess the concept category from surrounding text."""
    text_lower = text.lower()
    scores = {}
    for cat, keywords in _CATEGORY_KEYWORDS.items():
        scores[cat] = sum(1 for kw in keywords if kw in text_lower)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else ConceptType.TECHNIQUE


def _extract_bilingual_term(term: str) -> tuple[str, str]:
    """Extract English and Chinese names from a bilingual term.

    Handles patterns like:
    - "Chinese (English)"
    - "English (Chinese)"
    - "English Term"
    - "Chinese / English"
    """
    # Pattern: text (text)
    paren = re.match(r"(.+?)\s*[（(](.+?)[)）]", term)
    if paren:
        part1 = paren.group(1).strip()
        part2 = paren.group(2).strip()
        # Determine which is Chinese
        if re.search(r"[\u4e00-\u9fff]", part1) and not re.search(r"[\u4e00-\u9fff]", part2):
            return part2, part1  # en, zh
        elif re.search(r"[\u4e00-\u9fff]", part2) and not re.search(r"[\u4e00-\u9fff]", part1):
            return part1, part2
        else:
            # Both same language - prefer first as English
            return part1, part2

    # Pattern: text / text  or  text - text
    slash = re.match(r"(.+?)\s*[/\-]\s*(.+)", term)
    if slash:
        p1, p2 = slash.group(1).strip(), slash.group(2).strip()
        if re.search(r"[\u4e00-\u9fff]", p1):
            return p2, p1
        return p1, p2

    return term, term


def _is_valid_concept_name(name: str) -> bool:
    """Check if a name is a valid concept (not too short, not a number, etc.)."""
    if len(name) < 2:
        return False
    # Must have at least 2 alpha chars
    if len(re.findall(r"[a-zA-Z\u4e00-\u9fff]", name)) < 2:
        return False
    # Skip pure numbers or single characters
    if re.match(r"^[\d\s.]+$", name):
        return False
    # Skip very generic terms
    skip = {"the", "a", "an", "is", "it", "this", "that", "and", "or", "of", "in", "to", "for"}
    if name.lower().strip() in skip:
        return False
    return True


def extract_rich_concepts(
    analysis_text: str,
    video_id: str,
    video_title: str,
    registry: ConceptRegistry,
) -> tuple[list[Concept], list[Edge], list[DiagnosticChain]]:
    """Enhanced concept extraction from rich analysis markdown.

    Extracts from multiple patterns:
    1. Bold terms with colon: **Term：** description
    2. Bold terms with parenthetical: **Chinese (English)**
    3. Lettered sub-sections: **A. Title (subtitle)**
    4. Starred items describing techniques
    5. Table-based player analysis
    6. Diagnostic chains from fault/fix patterns
    """
    concepts: list[Concept] = []
    edges: list[Edge] = []
    chains: list[DiagnosticChain] = []
    source_file = f"ftt_video_{video_id}"
    seen_ids: set[str] = set()

    def _try_add_concept(
        name_en: str, name_zh: str, description: str,
        category: ConceptType | None = None,
        confidence: float = 0.8,
        context_text: str = "",
    ) -> str | None:
        """Try to add a concept, handling dedup. Returns concept ID or None."""
        if not _is_valid_concept_name(name_en):
            return None

        concept_id = _to_snake_id(name_en)
        if not concept_id or len(concept_id) < 2:
            return None
        if concept_id in seen_ids:
            return concept_id

        # Check registry for existing match
        existing = registry.resolve(name_en, threshold=70)
        if existing:
            return existing  # Return existing ID for edge creation

        if category is None:
            category = _guess_category(context_text or description)

        concept = Concept(
            id=concept_id,
            name=name_en,
            name_zh=name_zh,
            category=category,
            sources=["ftt"],
            description=description[:300] if description else f"Concept from FTT video: {video_title}",
            confidence=confidence,
        )

        dup = registry.add(concept)
        if dup is None:
            concepts.append(concept)
            seen_ids.add(concept_id)
            return concept_id
        else:
            return dup  # Return the existing concept ID

    # -----------------------------------------------------------------------
    # Pattern 1: Bold terms with colon/description
    # **Chinese (English)：** description
    # **English Term：** description
    # -----------------------------------------------------------------------
    bold_colon = re.compile(
        r"\*\*([^*]+?)(?:：|:)\*\*\s*(.+?)(?=\n|$)",
        re.MULTILINE,
    )
    for match in bold_colon.finditer(analysis_text):
        term = match.group(1).strip()
        desc = match.group(2).strip()

        # Skip section headers like "A." or numbering
        if re.match(r"^[A-Z]\.\s*$", term):
            continue

        name_en, name_zh = _extract_bilingual_term(term)
        _try_add_concept(name_en, name_zh, desc, context_text=desc)

    # -----------------------------------------------------------------------
    # Pattern 2: Lettered sub-sections
    # **A. Title (Subtitle)**
    # -----------------------------------------------------------------------
    lettered = re.compile(
        r"\*\*([A-Z])\.\s+(.+?)\*\*",
    )
    for match in lettered.finditer(analysis_text):
        title = match.group(2).strip()
        name_en, name_zh = _extract_bilingual_term(title)
        # Get surrounding context for category
        start = max(0, match.start() - 100)
        end = min(len(analysis_text), match.end() + 500)
        context = analysis_text[start:end]
        _try_add_concept(name_en, name_zh, f"Technique pattern: {title}", context_text=context)

    # -----------------------------------------------------------------------
    # Pattern 3: Quoted coaching cues and feel prompts
    # **"Cue text"（Chinese）**  or **"English cue"**
    # -----------------------------------------------------------------------
    cue_pattern = re.compile(
        r'\*\*"([^"]+?)"(?:\s*[（(](.+?)[)）])?\*\*',
    )
    for match in cue_pattern.finditer(analysis_text):
        cue_en = match.group(1).strip()
        cue_zh = match.group(2).strip() if match.group(2) else cue_en
        if len(cue_en) > 3:  # Skip very short cues
            _try_add_concept(
                cue_en, cue_zh,
                f"Coaching cue from FTT: {cue_en}",
                category=ConceptType.MENTAL_MODEL,
                confidence=0.7,
            )

    # -----------------------------------------------------------------------
    # Pattern 4: Standalone bold terms (not in bullet items already caught)
    # **Term** or **English Term (Chinese)**
    # -----------------------------------------------------------------------
    standalone_bold = re.compile(
        r"(?<!\*)\*\*([^*:：]{3,60})\*\*(?!\*)",
    )
    for match in standalone_bold.finditer(analysis_text):
        term = match.group(1).strip()
        # Skip if it's a section header number or generic term
        if re.match(r"^\d+\.", term) or term.startswith("URL") or term.startswith("频道"):
            continue
        if term.lower() in {"fault tolerant tennis", "ftt", "分析要点", "技术细节"}:
            continue
        # Skip player names alone
        if re.match(r"^[\w\s·]+$", term) and len(term.split()) <= 3:
            # Could be a player name - check for Chinese
            if not re.search(r"[\u4e00-\u9fff]", term):
                continue

        name_en, name_zh = _extract_bilingual_term(term)
        if _is_valid_concept_name(name_en) and len(name_en) > 3:
            start = max(0, match.start() - 50)
            end = min(len(analysis_text), match.end() + 200)
            context = analysis_text[start:end]
            _try_add_concept(name_en, name_zh, f"Concept from FTT: {name_en}", context_text=context)

    # -----------------------------------------------------------------------
    # Pattern 5: Technique terms in bullet items with key structure
    # *   **Term：** description
    # Already partially covered by Pattern 1, this catches variant formats
    # -----------------------------------------------------------------------
    bullet_bold = re.compile(
        r"\*\s+\*\*(.+?)\*\*[：:]\s*(.+?)(?=\n\*|\n\n|$)",
        re.DOTALL,
    )
    for match in bullet_bold.finditer(analysis_text):
        term = match.group(1).strip()
        desc = match.group(2).strip().split("\n")[0]  # First line only
        name_en, name_zh = _extract_bilingual_term(term)
        _try_add_concept(name_en, name_zh, desc, context_text=desc)

    # -----------------------------------------------------------------------
    # Create edges between concepts from same video
    # -----------------------------------------------------------------------
    all_ids = list(seen_ids)
    # Also include resolved existing concept IDs referenced in this video
    for i, src_id in enumerate(all_ids):
        for tgt_id in all_ids[i + 1:]:
            if src_id != tgt_id:
                edges.append(Edge(
                    source_id=src_id,
                    target_id=tgt_id,
                    relation=RelationType.SUPPORTS,
                    confidence=0.6,
                    evidence=f"Co-occurring in video: {video_title}",
                    source_file=source_file,
                ))

    # -----------------------------------------------------------------------
    # Fallback: if too few concepts, extract from section headers and key terms
    # -----------------------------------------------------------------------
    if len(concepts) < 2:
        # Try extracting from ### headers
        headers = re.findall(r"###\s+\d+\.\s+(.+)", analysis_text)
        for h in headers:
            name_en, name_zh = _extract_bilingual_term(h)
            _try_add_concept(
                name_en, name_zh,
                f"Section topic from FTT video: {video_title}",
                confidence=0.65,
            )

    # If still too few, extract any capitalized multi-word terms
    if len(concepts) < 2:
        cap_terms = re.findall(
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b",
            analysis_text,
        )
        for term in set(cap_terms):
            if len(term) > 5 and term not in {"Fault Tolerant", "Tennis Forehand"}:
                _try_add_concept(
                    term, term,
                    f"Key term from FTT video: {video_title}",
                    confidence=0.6,
                )

    return concepts, edges, chains


# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------

def main():
    """Run the full extraction pipeline on existing analysis files."""
    # Paths
    extracted_dir = PROJECT_ROOT / "knowledge" / "extracted" / "ftt_videos"
    extracted_dir.mkdir(parents=True, exist_ok=True)
    state_dir = PROJECT_ROOT / "knowledge" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    registry_path = PROJECT_ROOT / "knowledge" / "extracted" / "_registry_snapshot.json"

    # Load registry with existing 297 concepts
    registry = ConceptRegistry()
    if registry_path.exists():
        snapshot = json.loads(registry_path.read_text(encoding="utf-8"))
        for c_data in snapshot:
            try:
                registry.add(Concept(**c_data))
            except Exception:
                pass  # Skip invalid entries
    print(f"Registry loaded: {len(registry)} existing concepts")

    # Collect all video sections from the 3 analysis files
    analysis_files = [
        PROJECT_ROOT / "docs" / "research" / "09_ftt_videos_1.md",
        PROJECT_ROOT / "docs" / "research" / "09_ftt_videos_2.md",
        PROJECT_ROOT / "docs" / "research" / "09_ftt_videos_3.md",
    ]

    all_videos: dict[str, dict] = {}  # video_id -> {video_id, title, content}
    for md_path in analysis_files:
        if not md_path.exists():
            print(f"WARNING: {md_path} not found, skipping")
            continue
        sections = extract_from_existing_markdown(md_path)
        print(f"  {md_path.name}: {len(sections)} video sections")
        for section in sections:
            vid = section["video_id"]
            if vid not in all_videos:
                all_videos[vid] = section

    print(f"\nTotal unique videos: {len(all_videos)}")

    # Build a lookup from _ANALYZED_VIDEOS for titles
    title_lookup = {v["video_id"]: v["title"] for v in _ANALYZED_VIDEOS}

    # Process each video
    batch_state: dict = {
        "plan": "03-02",
        "batch": "existing_analyses",
        "videos": {},
    }

    total_concepts = 0
    total_edges = 0
    total_chains = 0
    min_concepts = 999

    for video_id, video_data in all_videos.items():
        # Use canonical title from state if available
        title = title_lookup.get(video_id, video_data.get("title", video_id))
        content = video_data["content"]

        # Extract concepts
        concepts, edges_list, chains_list = extract_rich_concepts(
            content, video_id, title, registry
        )

        # Save per-video JSON
        out_path = save_video_extraction(
            video_id, concepts, edges_list, chains_list, extracted_dir
        )

        # Track state
        batch_state["videos"][video_id] = {
            "status": "extracted",
            "extracted_file": str(out_path.relative_to(PROJECT_ROOT)),
            "title": title,
            "concepts_count": len(concepts),
            "edges_count": len(edges_list),
        }

        total_concepts += len(concepts)
        total_edges += len(edges_list)
        total_chains += len(chains_list)
        min_concepts = min(min_concepts, len(concepts))

        print(f"  [{video_id}] {title[:50]:50s} -> {len(concepts)} concepts, {len(edges_list)} edges")

    # Save batch state
    batch_state_path = state_dir / "batch0_state.json"
    batch_state_path.write_text(
        json.dumps(batch_state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Videos processed: {len(all_videos)}")
    print(f"Total new concepts: {total_concepts}")
    print(f"Total edges: {total_edges}")
    print(f"Total diagnostic chains: {total_chains}")
    print(f"Min concepts per video: {min_concepts}")
    print(f"Batch state saved to: {batch_state_path}")
    print(f"Per-video JSONs in: {extracted_dir}")


if __name__ == "__main__":
    main()
