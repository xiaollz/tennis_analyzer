"""Per-file-type extraction handlers for research Markdown files.

Each handler reads a Markdown file, resolves concepts against the registry,
and returns an ExtractionResult with concepts and edges.

Key principle: Handlers match text against the seeded registry (via
registry.resolve()) and extract RELATIONSHIPS. New concepts are only
created when clearly not matching anything in the registry.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from knowledge.pipeline.seed import to_snake_id
from knowledge.registry import ConceptRegistry
from knowledge.schemas import Concept, ConceptType, Edge, RelationType


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Arrow-chain pattern: splits on ->, =>, →
_ARROW_RE = re.compile(r"\s*(?:→|->|==>)\s*")

# FTT mapping pattern: FTT映射.*?`something`
_FTT_MAPPING_RE = re.compile(r"FTT映射.*?[`「](.+?)[`」]", re.IGNORECASE)

# Muscle table row: | muscle_name | ... |
_TABLE_ROW_RE = re.compile(r"^\|(.+)\|", re.MULTILINE)

# Header pattern (## or ###)
_HEADER_RE = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)

# Concept-like bold terms: **Term**
_BOLD_TERM_RE = re.compile(r"\*\*(.+?)\*\*")

# Chinese concept with English: 中文（English）or 中文/English
_ZH_EN_RE = re.compile(r"[\u4e00-\u9fff]+[\s]*[（(]([A-Za-z][\w\s\-]+)[)）]")


def _extract_arrow_edges(
    line: str,
    registry: ConceptRegistry,
    source_file: str,
    confidence: float,
) -> list[Edge]:
    """Extract CAUSES edges from arrow chains like A -> B -> C."""
    parts = _ARROW_RE.split(line)
    if len(parts) < 2:
        return []
    edges = []
    resolved = []
    for part in parts:
        # Clean markdown
        clean = re.sub(r"[*#`\[\]]", "", part).strip()
        if not clean or len(clean) < 2:
            continue
        cid = registry.resolve(clean)
        if cid:
            resolved.append(cid)
    # Create chain edges
    for i in range(len(resolved) - 1):
        if resolved[i] != resolved[i + 1]:
            edges.append(Edge(
                source_id=resolved[i],
                target_id=resolved[i + 1],
                relation=RelationType.CAUSES,
                confidence=confidence,
                evidence=line.strip()[:200],
                source_file=source_file,
            ))
    return edges


def _extract_ftt_mapping_edges(
    text: str,
    context_concept_id: str | None,
    registry: ConceptRegistry,
    source_file: str,
    confidence: float,
) -> list[Edge]:
    """Extract SUPPORTS edges from FTT mapping references."""
    edges = []
    for match in _FTT_MAPPING_RE.finditer(text):
        ref = match.group(1).strip()
        target_id = registry.resolve(ref)
        if target_id and context_concept_id and target_id != context_concept_id:
            edges.append(Edge(
                source_id=context_concept_id,
                target_id=target_id,
                relation=RelationType.SUPPORTS,
                confidence=confidence,
                evidence=match.group(0)[:200],
                source_file=source_file,
            ))
    return edges


def _resolve_or_create(
    name: str,
    name_zh: str,
    category: ConceptType,
    description: str,
    registry: ConceptRegistry,
    source: str,
    confidence: float = 1.0,
    aliases: list[str] | None = None,
    muscles: list[str] | None = None,
) -> tuple[str, Concept | None]:
    """Resolve a name against the registry; create if not found.

    Returns (concept_id, new_concept_or_None).
    """
    existing_id = registry.resolve(name)
    if existing_id:
        # Merge source if needed
        existing = registry.get(existing_id)
        if existing and source not in existing.sources:
            existing.sources.append(source)
        if existing and muscles:
            for m in muscles:
                if m not in existing.muscles_involved:
                    existing.muscles_involved.append(m)
        return existing_id, None

    snake_id = to_snake_id(name)
    if not snake_id or not re.match(r"^[a-z]", snake_id):
        snake_id = "c_" + (snake_id or "unnamed")

    concept = Concept(
        id=snake_id,
        name=name,
        name_zh=name_zh,
        aliases=aliases or [],
        category=category,
        sources=[source],
        description=description[:500],
        muscles_involved=muscles or [],
        confidence=confidence,
    )

    dup_id = registry.add(concept)
    if dup_id is not None:
        existing = registry.get(dup_id)
        if existing and source not in existing.sources:
            existing.sources.append(source)
        return dup_id, None

    return snake_id, concept


def _extract_concepts_from_headers(
    text: str,
    registry: ConceptRegistry,
    source_file: str,
    source_tag: str,
    confidence: float,
    category: ConceptType = ConceptType.TECHNIQUE,
) -> tuple[list[Concept], list[Edge]]:
    """Extract concepts from Markdown headers and bold terms."""
    concepts = []
    edges = []
    headers = _HEADER_RE.findall(text)
    seen_ids: set[str] = set()

    for _level, title in headers:
        # Strip numbering like "1.2 " or "### 2.1 "
        clean_title = re.sub(r"^[\d.]+\s*", "", title).strip()
        # Try English extraction from Chinese(English) pattern
        en_match = _ZH_EN_RE.search(clean_title)
        if en_match:
            en_name = en_match.group(1).strip()
        else:
            # Use title directly if it contains English
            en_name = re.sub(r"[\u4e00-\u9fff：:—–（）()「」]", " ", clean_title).strip()
            en_name = re.sub(r"\s+", " ", en_name).strip()

        if not en_name or len(en_name) < 3:
            continue

        # Filter out junk: purely numeric, too short after cleanup,
        # or doesn't contain at least 2 alphabetic chars
        alpha_count = sum(1 for ch in en_name if ch.isalpha())
        if alpha_count < 2:
            continue

        cid = registry.resolve(en_name)
        if cid:
            if cid not in seen_ids:
                seen_ids.add(cid)
                existing = registry.get(cid)
                if existing and source_tag not in existing.sources:
                    existing.sources.append(source_tag)
        else:
            # Only create if this looks like a real concept name
            if len(en_name) > 40 or len(en_name) < 4:
                continue  # Too long or too short to be a concept name
            cid, new_c = _resolve_or_create(
                name=en_name,
                name_zh=clean_title,
                category=category,
                description=f"Concept from {source_file}: {clean_title}",
                registry=registry,
                source=source_tag,
                confidence=confidence,
            )
            if new_c:
                concepts.append(new_c)
            if cid:
                seen_ids.add(cid)

    return concepts, edges


# ---------------------------------------------------------------------------
# KNOWN_MUSCLES: Common muscle names for matching in biomechanics tables
# ---------------------------------------------------------------------------

KNOWN_MUSCLES = [
    "腓肠肌", "比目鱼肌", "四头肌", "股四头肌", "臀肌", "臀大肌",
    "腘绳肌", "髋关节", "腹斜肌", "内斜", "外斜", "腹肌",
    "竖脊肌", "背阔肌", "三角肌", "前三角肌", "中三角肌", "后三角肌",
    "棘下肌", "小圆肌", "肩胛下肌", "冈上肌", "冈下肌",
    "二头肌", "三头肌", "肱二头肌", "肱三头肌",
    "胸大肌", "胸肌", "菱形肌", "前锯肌", "斜方肌",
    "旋前圆肌", "旋后肌", "肱桡肌",
    "腕屈肌", "腕伸肌", "指屈肌", "指伸肌",
    "掌长肌", "尺侧腕屈肌", "桡侧腕屈肌",
    "桡侧伸腕长肌", "桡侧伸腕短肌", "尺侧腕伸肌",
    # English muscle names
    "gastrocnemius", "soleus", "quadriceps", "gluteus", "hamstrings",
    "obliques", "rectus abdominis", "erector spinae",
    "latissimus dorsi", "lats", "deltoid", "anterior deltoid",
    "middle deltoid", "posterior deltoid",
    "infraspinatus", "teres minor", "subscapularis", "supraspinatus",
    "biceps", "triceps", "pectoralis", "pectoralis major",
    "rhomboid", "serratus anterior", "trapezius",
    "pronator teres", "supinator", "brachioradialis",
]


def _extract_muscles_from_table(text: str) -> list[str]:
    """Extract muscle names from table rows in text."""
    muscles = []
    for row in _TABLE_ROW_RE.findall(text):
        cells = [c.strip() for c in row.split("|")]
        for cell in cells:
            for muscle in KNOWN_MUSCLES:
                if muscle in cell and muscle not in muscles:
                    muscles.append(muscle)
    return muscles


# ---------------------------------------------------------------------------
# Handler: Synthesis files (12, 13, 15, 17)
# ---------------------------------------------------------------------------

SYNTHESIS_CONFIDENCE = 0.9


def extract_synthesis(filepath: Path, registry: ConceptRegistry):
    """Extract concepts from synthesis files (most structured).

    Parse headers as concept names, resolve against registry,
    extract cross-references as edges.
    """
    from knowledge.pipeline.extractor import ExtractionResult

    text = filepath.read_text(encoding="utf-8")
    source_file = filepath.name
    concepts: list[Concept] = []
    edges: list[Edge] = []

    # Extract concepts from headers
    new_concepts, new_edges = _extract_concepts_from_headers(
        text, registry, source_file, "ftt", SYNTHESIS_CONFIDENCE
    )
    concepts.extend(new_concepts)
    edges.extend(new_edges)

    # Extract arrow-chain edges
    for line in text.splitlines():
        if any(arrow in line for arrow in ["→", "->", "==>"]):
            edges.extend(_extract_arrow_edges(
                line, registry, source_file, SYNTHESIS_CONFIDENCE
            ))

    # Extract bold-term cross-references within sections
    sections = re.split(r"\n#{1,3}\s+", text)
    for section in sections:
        # Find the section's primary concept
        first_line = section.strip().split("\n")[0] if section.strip() else ""
        section_concept = registry.resolve(
            re.sub(r"[\u4e00-\u9fff：:—–（）()「」#*\d.]", " ", first_line).strip()
        )

        bold_terms = _BOLD_TERM_RE.findall(section)
        for term in bold_terms:
            clean = re.sub(r"[\u4e00-\u9fff：:—–]", " ", term).strip()
            if len(clean) < 3:
                continue
            target_id = registry.resolve(clean)
            if target_id and section_concept and target_id != section_concept:
                edges.append(Edge(
                    source_id=section_concept,
                    target_id=target_id,
                    relation=RelationType.SUPPORTS,
                    confidence=SYNTHESIS_CONFIDENCE,
                    evidence=f"Cross-reference in {source_file}: {term}",
                    source_file=source_file,
                ))

    return ExtractionResult(concepts=concepts, edges=edges, source_file=source_file)


# ---------------------------------------------------------------------------
# Handler: FTT Book (01, 02)
# ---------------------------------------------------------------------------

FTT_BOOK_CONFIDENCE = 0.8


def extract_ftt_book(filepath: Path, registry: ConceptRegistry):
    """Extract concepts from FTT book Markdown files."""
    from knowledge.pipeline.extractor import ExtractionResult

    text = filepath.read_text(encoding="utf-8")
    source_file = filepath.name
    concepts: list[Concept] = []
    edges: list[Edge] = []

    new_concepts, _ = _extract_concepts_from_headers(
        text, registry, source_file, "ftt", FTT_BOOK_CONFIDENCE
    )
    concepts.extend(new_concepts)

    # Arrow chain edges
    for line in text.splitlines():
        if any(arrow in line for arrow in ["→", "->", "==>"]):
            edges.extend(_extract_arrow_edges(
                line, registry, source_file, FTT_BOOK_CONFIDENCE
            ))

    # Bold term cross-references
    bold_terms = _BOLD_TERM_RE.findall(text)
    for term in bold_terms:
        clean = re.sub(r"[\u4e00-\u9fff：:—–]", " ", term).strip()
        if len(clean) < 3:
            continue
        registry.resolve(clean)  # Just register the reference

    return ExtractionResult(concepts=concepts, edges=edges, source_file=source_file)


# ---------------------------------------------------------------------------
# Handler: FTT Blog (04-08)
# ---------------------------------------------------------------------------

FTT_BLOG_CONFIDENCE = 0.8


def extract_ftt_blog(filepath: Path, registry: ConceptRegistry):
    """Extract concepts from FTT blog article Markdown files."""
    from knowledge.pipeline.extractor import ExtractionResult

    text = filepath.read_text(encoding="utf-8")
    source_file = filepath.name
    concepts: list[Concept] = []
    edges: list[Edge] = []

    new_concepts, _ = _extract_concepts_from_headers(
        text, registry, source_file, "ftt", FTT_BLOG_CONFIDENCE
    )
    concepts.extend(new_concepts)

    # Look for "key concepts" or "key principles" sections
    key_sections = re.findall(
        r"(?:核心概念|关键概念|核心原则|关键技术|Key (?:Concept|Principle)s?)[\s:：]*\n((?:.*\n)*?)\n#",
        text, re.IGNORECASE
    )
    for section in key_sections:
        for line in section.splitlines():
            clean = re.sub(r"^[\s\-*]+", "", line).strip()
            if not clean:
                continue
            en_match = _ZH_EN_RE.search(clean)
            if en_match:
                name = en_match.group(1).strip()
                cid = registry.resolve(name)
                if not cid:
                    cid, new_c = _resolve_or_create(
                        name=name,
                        name_zh=clean,
                        category=ConceptType.TECHNIQUE,
                        description=clean[:300],
                        registry=registry,
                        source="ftt",
                        confidence=FTT_BLOG_CONFIDENCE,
                    )
                    if new_c:
                        concepts.append(new_c)

    # Arrow chain edges
    for line in text.splitlines():
        if any(arrow in line for arrow in ["→", "->", "==>"]):
            edges.extend(_extract_arrow_edges(
                line, registry, source_file, FTT_BLOG_CONFIDENCE
            ))

    return ExtractionResult(concepts=concepts, edges=edges, source_file=source_file)


# ---------------------------------------------------------------------------
# Handler: FTT Videos (09)
# ---------------------------------------------------------------------------

FTT_VIDEOS_CONFIDENCE = 0.6


def extract_ftt_videos(filepath: Path, registry: ConceptRegistry):
    """Extract concepts from FTT video analysis Markdown files.

    Only extracts UNIQUE concepts not already in registry.
    """
    from knowledge.pipeline.extractor import ExtractionResult

    text = filepath.read_text(encoding="utf-8")
    source_file = filepath.name
    concepts: list[Concept] = []
    edges: list[Edge] = []

    new_concepts, _ = _extract_concepts_from_headers(
        text, registry, source_file, "ftt", FTT_VIDEOS_CONFIDENCE
    )
    concepts.extend(new_concepts)

    # Arrow chain edges
    for line in text.splitlines():
        if any(arrow in line for arrow in ["→", "->", "==>"]):
            edges.extend(_extract_arrow_edges(
                line, registry, source_file, FTT_VIDEOS_CONFIDENCE
            ))

    return ExtractionResult(concepts=concepts, edges=edges, source_file=source_file)


# ---------------------------------------------------------------------------
# Handler: TPA Videos (14, 16)
# ---------------------------------------------------------------------------

TPA_CONFIDENCE = 0.6


def extract_tpa_videos(filepath: Path, registry: ConceptRegistry):
    """Extract concepts from TPA Tennis video synthesis files."""
    from knowledge.pipeline.extractor import ExtractionResult

    text = filepath.read_text(encoding="utf-8")
    source_file = filepath.name
    concepts: list[Concept] = []
    edges: list[Edge] = []

    new_concepts, _ = _extract_concepts_from_headers(
        text, registry, source_file, "tpa", TPA_CONFIDENCE
    )
    concepts.extend(new_concepts)

    # Arrow chain edges
    for line in text.splitlines():
        if any(arrow in line for arrow in ["→", "->", "==>"]):
            edges.extend(_extract_arrow_edges(
                line, registry, source_file, TPA_CONFIDENCE
            ))

    return ExtractionResult(concepts=concepts, edges=edges, source_file=source_file)


# ---------------------------------------------------------------------------
# Handler: FTT Specific deep-dives (18-23)
# ---------------------------------------------------------------------------

FTT_SPECIFIC_CONFIDENCE = 0.8


def extract_ftt_specific(filepath: Path, registry: ConceptRegistry):
    """Extract from FTT specific deep-dive files (18-23).

    The main concept should already be in registry; extract supporting
    details, aliases, and edges.
    """
    from knowledge.pipeline.extractor import ExtractionResult

    text = filepath.read_text(encoding="utf-8")
    source_file = filepath.name
    concepts: list[Concept] = []
    edges: list[Edge] = []

    new_concepts, _ = _extract_concepts_from_headers(
        text, registry, source_file, "ftt", FTT_SPECIFIC_CONFIDENCE
    )
    concepts.extend(new_concepts)

    # Arrow chain edges
    for line in text.splitlines():
        if any(arrow in line for arrow in ["→", "->", "==>"]):
            edges.extend(_extract_arrow_edges(
                line, registry, source_file, FTT_SPECIFIC_CONFIDENCE
            ))

    # FTT mapping edges (common in specific files)
    # Use first header as context concept
    first_header = _HEADER_RE.search(text)
    context_id = None
    if first_header:
        en = re.sub(r"[\u4e00-\u9fff：:—–（）()「」#*\d.]", " ", first_header.group(2)).strip()
        context_id = registry.resolve(en)

    edges.extend(_extract_ftt_mapping_edges(
        text, context_id, registry, source_file, FTT_SPECIFIC_CONFIDENCE
    ))

    return ExtractionResult(concepts=concepts, edges=edges, source_file=source_file)


# ---------------------------------------------------------------------------
# Handler: Biomechanics (24-28)
# ---------------------------------------------------------------------------

BIOMECHANICS_CONFIDENCE = 0.8


def extract_biomechanics(filepath: Path, registry: ConceptRegistry):
    """Extract concepts from biomechanics textbook Markdown files.

    Critical: Parse muscle tables and create concepts with muscles_involved
    populated. Extract FTT mapping cross-references as supports edges.
    File 28 has problem-solution mappings -> DRILLS_FOR edges.
    """
    from knowledge.pipeline.extractor import ExtractionResult

    text = filepath.read_text(encoding="utf-8")
    source_file = filepath.name
    concepts: list[Concept] = []
    edges: list[Edge] = []

    # Split into sections to process each with its muscle context
    sections = re.split(r"\n(?=#{1,3}\s+)", text)

    for section in sections:
        header_match = _HEADER_RE.search(section)
        if not header_match:
            continue

        title = header_match.group(2).strip()
        clean_title = re.sub(r"^[\d.]+\s*", "", title).strip()
        en_match = _ZH_EN_RE.search(clean_title)
        if en_match:
            en_name = en_match.group(1).strip()
        else:
            en_name = re.sub(r"[\u4e00-\u9fff：:—–（）()「」]", " ", clean_title).strip()
            en_name = re.sub(r"\s+", " ", en_name).strip()

        # Extract muscles from this section
        section_muscles = _extract_muscles_from_table(section)

        # Try to resolve concept
        if en_name and len(en_name) >= 3 and len(en_name) <= 40:
            cid = registry.resolve(en_name)
            if cid:
                existing = registry.get(cid)
                if existing:
                    if "biomechanics_book" not in existing.sources:
                        existing.sources.append("biomechanics_book")
                    for m in section_muscles:
                        if m not in existing.muscles_involved:
                            existing.muscles_involved.append(m)
            elif section_muscles:
                # Only create new concept if it has muscle data
                cid, new_c = _resolve_or_create(
                    name=en_name,
                    name_zh=clean_title,
                    category=ConceptType.BIOMECHANICS,
                    description=f"Biomechanics concept from {source_file}: {clean_title}",
                    registry=registry,
                    source="biomechanics_book",
                    confidence=BIOMECHANICS_CONFIDENCE,
                    muscles=section_muscles,
                )
                if new_c:
                    concepts.append(new_c)

        # Also attach muscles to any concepts referenced via bold terms in section
        if section_muscles:
            for bold in _BOLD_TERM_RE.findall(section):
                bold_clean = re.sub(r"[\u4e00-\u9fff：:—–]", " ", bold).strip()
                if len(bold_clean) >= 3:
                    ref_id = registry.resolve(bold_clean)
                    if ref_id:
                        ref_concept = registry.get(ref_id)
                        if ref_concept:
                            for m in section_muscles:
                                if m not in ref_concept.muscles_involved:
                                    ref_concept.muscles_involved.append(m)

        # FTT mapping edges
        context_id = registry.resolve(en_name) if en_name and len(en_name) >= 3 else None
        edges.extend(_extract_ftt_mapping_edges(
            section, context_id, registry, source_file, BIOMECHANICS_CONFIDENCE
        ))

        # Arrow chain edges
        for line in section.splitlines():
            if any(arrow in line for arrow in ["→", "->", "==>"]):
                edges.extend(_extract_arrow_edges(
                    line, registry, source_file, BIOMECHANICS_CONFIDENCE
                ))

    # File 28 special: problem-solution mappings -> DRILLS_FOR edges
    if "28" in filepath.name or "problem" in filepath.name.lower():
        problem_sections = re.split(r"\n(?=## Problem Area)", text)
        for psection in problem_sections:
            problem_header = re.search(r"Problem Area \d+:\s*(.+)", psection)
            if not problem_header:
                continue
            problem_name = problem_header.group(1).strip()
            problem_en = re.sub(r"[\u4e00-\u9fff：:—–（）()「」]", " ", problem_name).strip()
            problem_id = registry.resolve(problem_en)

            # Find training recommendations
            training_section = re.search(r"训练层.*?\n((?:.*\n)*?)(?=\n## |$)", psection)
            if training_section and problem_id:
                for line in training_section.group(1).splitlines():
                    if "|" in line:
                        cells = [c.strip() for c in line.split("|")]
                        for cell in cells:
                            if cell and len(cell) > 2:
                                drill_id = registry.resolve(cell)
                                if drill_id and drill_id != problem_id:
                                    edges.append(Edge(
                                        source_id=drill_id,
                                        target_id=problem_id,
                                        relation=RelationType.DRILLS_FOR,
                                        confidence=BIOMECHANICS_CONFIDENCE,
                                        evidence=f"Problem-solution mapping: {cell} -> {problem_name}",
                                        source_file=source_file,
                                    ))

    return ExtractionResult(concepts=concepts, edges=edges, source_file=source_file)


# ---------------------------------------------------------------------------
# Handler: Generic fallback (03)
# ---------------------------------------------------------------------------

GENERIC_CONFIDENCE = 0.7


def extract_generic(filepath: Path, registry: ConceptRegistry):
    """Fallback handler for unrecognized file types. Simple header + keyword extraction."""
    from knowledge.pipeline.extractor import ExtractionResult

    text = filepath.read_text(encoding="utf-8")
    source_file = filepath.name
    concepts: list[Concept] = []
    edges: list[Edge] = []

    source_tag = "user_experience" if "learning" in source_file else "ftt"
    new_concepts, _ = _extract_concepts_from_headers(
        text, registry, source_file, source_tag, GENERIC_CONFIDENCE
    )
    concepts.extend(new_concepts)

    # Arrow chain edges
    for line in text.splitlines():
        if any(arrow in line for arrow in ["→", "->", "==>"]):
            edges.extend(_extract_arrow_edges(
                line, registry, source_file, GENERIC_CONFIDENCE
            ))

    return ExtractionResult(concepts=concepts, edges=edges, source_file=source_file)


# ---------------------------------------------------------------------------
# Handler: User Training Journey (learning.md)
# ---------------------------------------------------------------------------

_LEGACY_JOURNEY_PATH = (
    Path(__file__).resolve().parent.parent.parent / "docs" / "knowledge_graph" / "user_journey.json"
)

_DATE_RE = re.compile(r"^##\s+(\d{4}-\d{2}-\d{2})")
_FIX_RE = re.compile(r"(?:解决方案|解决|fix|solution)[：:]\s*(.+)", re.IGNORECASE)
_CAUSE_RE = re.compile(r"(?:原因|根因|root.?cause)[：:]\s*(.+)", re.IGNORECASE)


def _parse_sessions(filepath: Path) -> list[dict]:
    """Parse learning.md into per-date session dicts."""
    text = filepath.read_text(encoding="utf-8")
    sessions: list[dict] = []
    current: dict | None = None
    for raw_line in text.splitlines():
        m = _DATE_RE.match(raw_line)
        if m:
            if current is not None:
                sessions.append(current)
            current = {"date": m.group(1), "lines": []}
        elif current is not None:
            current["lines"].append(raw_line)
    if current is not None:
        sessions.append(current)
    return sessions


def _extract_journey_arrow_edges(
    lines: list[str],
    registry: ConceptRegistry,
    date: str,
    concepts_out: list[Concept],
) -> list[Edge]:
    """Extract CAUSES edges from arrow-notation causal chains in learning.md."""
    edges: list[Edge] = []
    for raw_line in lines:
        stripped = raw_line.strip().lstrip("- ").lstrip("> ")
        # Skip non-arrow lines and URLs
        if not _ARROW_RE.search(stripped) or "http" in stripped:
            continue
        parts = _ARROW_RE.split(stripped)
        if len(parts) < 2:
            continue
        prev_id: str | None = None
        for seg in parts:
            clean = re.sub(r"[*`#\[\]]", "", seg).strip()
            clean = re.sub(r"\(.*?\)", "", clean).strip()
            if not clean or len(clean) < 2:
                continue
            cid = registry.resolve(clean)
            if not cid:
                # Create new concept for unresolved chain node
                snake = to_snake_id(clean[:60])
                if not snake or not re.match(r"^[a-z]", snake):
                    snake = "c_" + (snake or "unnamed")
                concept = Concept(
                    id=snake,
                    name=clean[:60],
                    name_zh=clean[:60],
                    aliases=[],
                    category=ConceptType.SYMPTOM,
                    sources=["user_experience"],
                    description=f"From causal chain on {date}: {clean[:120]}",
                    confidence=0.7,
                )
                dup = registry.add(concept)
                if dup is not None:
                    cid = dup
                else:
                    cid = snake
                    concepts_out.append(concept)
            if prev_id is not None and prev_id != cid:
                edges.append(Edge(
                    source_id=prev_id,
                    target_id=cid,
                    relation=RelationType.CAUSES,
                    confidence=0.7,
                    evidence=f"Causal chain from learning.md {date}: {stripped[:120]}",
                    source_file="learning.md",
                ))
            prev_id = cid
    return edges


def _migrate_legacy_journey(
    registry: ConceptRegistry,
) -> tuple[list[Concept], list[Edge]]:
    """Migrate legacy user_journey.json into concepts and edges."""
    concepts: list[Concept] = []
    edges: list[Edge] = []

    if not _LEGACY_JOURNEY_PATH.exists():
        return concepts, edges

    data = json.loads(_LEGACY_JOURNEY_PATH.read_text(encoding="utf-8"))

    # --- Problems -> resolve against registry, create related CAUSES edges ---
    prob_id_map: dict[str, str] = {}  # P01 -> canonical_id
    for prob in data.get("problems", []):
        prob_name = prob.get("name", "")
        parts = prob_name.split(" / ")
        en_name = parts[-1].strip() if len(parts) >= 2 else prob_name
        zh_name = parts[0].strip() if len(parts) >= 2 else prob_name

        cid, new_c = _resolve_or_create(
            en_name, zh_name, ConceptType.SYMPTOM,
            prob.get("root_cause", en_name), registry, "user_experience", 0.8,
        )
        if new_c:
            concepts.append(new_c)
        prob_id_map[prob.get("id", "")] = cid

    # Create edges from related_problems
    for prob in data.get("problems", []):
        pid = prob_id_map.get(prob.get("id", ""))
        if not pid:
            continue
        for rel_id in prob.get("related_problems", []):
            rel_cid = prob_id_map.get(rel_id)
            if rel_cid and pid != rel_cid:
                edges.append(Edge(
                    source_id=pid,
                    target_id=rel_cid,
                    relation=RelationType.CAUSES,
                    confidence=0.8,
                    evidence=f"Related problems: {prob.get('id')}->{rel_id}",
                    source_file="user_journey.json",
                ))

    # --- Breakthroughs -> resolve ---
    for bt in data.get("breakthroughs", []):
        desc = bt.get("description", "")
        cid, new_c = _resolve_or_create(
            desc[:60], desc[:60], ConceptType.TECHNIQUE,
            desc, registry, "user_experience", 0.8,
        )
        if new_c:
            concepts.append(new_c)

    # --- Cue evolutions -> SUPPORTS edges ---
    for cue in data.get("oral_cues_evolution", []):
        old_text = cue.get("cue_text", "")
        new_text = cue.get("replaced_by")
        if not new_text:
            continue
        reason = cue.get("reason_for_change", "") or ""

        old_id, old_new = _resolve_or_create(
            old_text[:60], old_text[:60], ConceptType.MENTAL_MODEL,
            f"Cue: {old_text}", registry, "user_experience", 0.6,
        )
        if old_new:
            concepts.append(old_new)

        new_id, new_new = _resolve_or_create(
            new_text[:60], new_text[:60], ConceptType.MENTAL_MODEL,
            f"Cue: {new_text}", registry, "user_experience", 0.6,
        )
        if new_new:
            concepts.append(new_new)

        if old_id != new_id:
            edges.append(Edge(
                source_id=new_id,
                target_id=old_id,
                relation=RelationType.SUPPORTS,
                confidence=0.6,
                evidence=f"Cue: '{old_text[:40]}' -> '{new_text[:40]}'. {reason[:80]}",
                source_file="user_journey.json",
            ))

    # --- problem_dependency_graph -> direct CAUSES chains ---
    for chain_str in data.get("problem_dependency_graph", {}).get("chains", []):
        pids = re.findall(r"P(\d+)", chain_str)
        prev_cid = None
        for pid_num in pids:
            pid_str = f"P{pid_num.zfill(2)}"
            cid = prob_id_map.get(pid_str)
            if cid and prev_cid and prev_cid != cid:
                edges.append(Edge(
                    source_id=prev_cid,
                    target_id=cid,
                    relation=RelationType.CAUSES,
                    confidence=0.8,
                    evidence=f"Dependency chain: {chain_str[:100]}",
                    source_file="user_journey.json",
                ))
            if cid:
                prev_cid = cid

    return concepts, edges


def extract_user_journey(
    filepath: Path, registry: ConceptRegistry,
) -> "ExtractionResult":
    """Extract concepts and edges from the user training journal (learning.md).

    1. Parses date-based training entries and extracts causal chains
    2. Migrates legacy user_journey.json (problems, breakthroughs, cues, deps)
    3. Links all concepts to canonical registry IDs
    4. Saves extraction result to knowledge/extracted/user_journey/learning.json

    Args:
        filepath: Path to learning.md
        registry: ConceptRegistry for dedup/resolution

    Returns:
        ExtractionResult with concepts and edges, source="user_experience"
    """
    from knowledge.pipeline.extractor import ExtractionResult

    all_concepts: list[Concept] = []
    all_edges: list[Edge] = []

    # --- Parse learning.md ---
    sessions = _parse_sessions(filepath)
    for session in sessions:
        date = session["date"]
        lines = session["lines"]
        arrow_edges = _extract_journey_arrow_edges(lines, registry, date, all_concepts)
        all_edges.extend(arrow_edges)

    # --- Migrate legacy user_journey.json ---
    legacy_concepts, legacy_edges = _migrate_legacy_journey(registry)
    all_concepts.extend(legacy_concepts)
    all_edges.extend(legacy_edges)

    # --- Deduplicate edges ---
    seen: set[tuple[str, str, str]] = set()
    unique_edges: list[Edge] = []
    for edge in all_edges:
        key = (edge.source_id, edge.target_id, edge.relation.value)
        if key not in seen:
            seen.add(key)
            unique_edges.append(edge)

    # --- Save to JSON ---
    output_dir = Path(__file__).resolve().parent.parent / "extracted" / "user_journey"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "learning.json"
    output_data = {
        "source": str(filepath),
        "source_type": "user_experience",
        "concepts": [c.model_dump() for c in all_concepts],
        "edges": [e.model_dump() for e in unique_edges],
        "stats": {
            "total_sessions": len(sessions),
            "total_concepts": len(all_concepts),
            "total_edges": len(unique_edges),
        },
    }
    output_path.write_text(json.dumps(output_data, ensure_ascii=False, indent=2))

    return ExtractionResult(
        concepts=all_concepts,
        edges=unique_edges,
        source_file=str(filepath),
    )
