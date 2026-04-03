#!/usr/bin/env python3
"""Merge all FTT video batch states, re-extract missing concepts from Markdown,
merge concepts into the canonical registry with deduplication, and extract
diagnostic chains.

Usage:
    python scripts/merge_ftt_video_concepts.py
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Ensure project root is on path
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

STATE_DIR = PROJECT_ROOT / "knowledge" / "state"
EXTRACTED_DIR = PROJECT_ROOT / "knowledge" / "extracted"
FTT_VIDEOS_DIR = EXTRACTED_DIR / "ftt_videos"
ANALYSES_DIR = PROJECT_ROOT / "docs" / "research" / "ftt_video_analyses"
SNAPSHOT_PATH = EXTRACTED_DIR / "_registry_snapshot.json"

# ── Helpers ─────────────────────────────────────────────────────────────


def _to_snake_id(english_name: str) -> str:
    """Convert an English name to a valid snake_case concept ID."""
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", english_name)
    parts = cleaned.lower().split()
    snake = "_".join(p for p in parts if p)
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
        # Chinese
        "技术": ConceptType.TECHNIQUE,
        "生物力学": ConceptType.BIOMECHANICS,
        "训练": ConceptType.DRILL,
        "训练方法": ConceptType.DRILL,
        "症状": ConceptType.SYMPTOM,
        "心理模型": ConceptType.MENTAL_MODEL,
        "连接机制": ConceptType.CONNECTION,
        "连接": ConceptType.CONNECTION,
        "视觉": ConceptType.TECHNIQUE,
        "视觉系统": ConceptType.TECHNIQUE,
        "移动": ConceptType.TECHNIQUE,
        "发力模型": ConceptType.BIOMECHANICS,
        "节奏": ConceptType.TECHNIQUE,
    }
    normalized = cat_str.strip().lower()
    # Try direct match
    if normalized in mapping:
        return mapping[normalized]
    # Check if any key is contained in the string
    for key, val in mapping.items():
        if key in normalized:
            return val
    return ConceptType.TECHNIQUE


def _parse_relation(rel_str: str) -> RelationType:
    """Map a relation string to RelationType enum."""
    mapping = {
        "supports": RelationType.SUPPORTS,
        "refines": RelationType.SUPPORTS,
        "extends": RelationType.SUPPORTS,
        "contradicts": RelationType.CONTRADICTS,
        "causes": RelationType.CAUSES,
        "prevents": RelationType.PREVENTS,
        "requires": RelationType.REQUIRES,
        "drills_for": RelationType.DRILLS_FOR,
        "visible_as": RelationType.VISIBLE_AS,
        "is_instance_of": RelationType.SUPPORTS,
        "provides cause for": RelationType.CAUSES,
        "provides drill for": RelationType.DRILLS_FOR,
    }
    normalized = rel_str.strip().lower()
    return mapping.get(normalized, RelationType.SUPPORTS)


# ── Quality filter ──────────────────────────────────────────────────────


def _is_quality_concept_fn():
    """Return a quality filter function for concept dicts.

    Extracted as a factory so tests can import it.
    """
    def _is_quality_concept(c_data: dict) -> bool:
        """Filter out garbage concepts."""
        cid = c_data.get("id", "")
        name = c_data.get("name", "")
        # Too short ID
        if len(cid) < 3:
            return False
        # Numeric-only IDs like c_5
        if re.match(r"^c_\d+$", cid):
            return False
        # ID too long (>40 chars means it's a sentence, not a concept)
        if len(cid) > 40:
            return False
        # unknown_concept placeholder
        if cid == "unknown_concept":
            return False
        # Player names (common pattern in extraction)
        player_names = {
            "giovanni", "mpetshi", "perricard", "shapovalov", "sinner",
            "alcaraz", "federer", "nadal", "djokovic", "draper",
            "aliassime", "branstine", "allsopp", "mensik", "fonseca",
            "nishioka", "de_minaur", "rublev", "rune", "khachanov",
            "griekspoor", "schwartzman", "berrettini", "tsitsipas",
            "medvedev", "zverev", "sampras",
        }
        cid_lower = cid.lower()
        if any(pn in cid_lower for pn in player_names):
            return False
        # Per-video hash IDs (drill_VIDEOID_N, tp_VIDEOID_N patterns)
        if re.match(r"^(drill|tp)_[a-z0-9]{8,}_\d+$", cid, re.IGNORECASE):
            return False
        # Drill with video ID hash (drill_xxxxx format where xxxxx looks like a video ID)
        if re.match(r"^drill_[a-z0-9_]{10,}$", cid) and "_" in cid[6:]:
            return False
        # OTI drill references (specific match session references)
        if cid.startswith("oti_"):
            return False
        # Name is too long (likely a sentence)
        if len(name) > 50:
            return False
        # Name starts with ** (markdown bold not stripped)
        if name.startswith("**") or name.startswith("*"):
            return False
        # Description field used as name (contains colons, newlines, etc)
        if ":" in name and len(name) > 25:
            return False
        # Pure Chinese name with no meaningful English ID
        if re.match(r"^[\u4e00-\u9fff\s]+$", name) and len(cid) < 5:
            return False
        # Generic coaching/cue concepts that don't add knowledge graph value
        generic_terms = {
            "coaching_command", "coaching_cue", "coaching_cues", "cues",
            "feeling_cue", "visual_cue", "sensory_cues", "distal_end_coaching",
            "coaching_cues_body_mechanics",
        }
        if cid in generic_terms:
            return False
        # Teaching point IDs that are just Chinese transliterations
        if re.match(r"^tp_[a-z0-9]+_\d+$", cid):
            return False
        # Single common word concepts that are too generic
        generic_singles = {
            "legs", "timing", "balance", "power", "speed", "rhythm",
            "contact", "follow_through", "grip", "stance", "rotation",
            "extension", "relaxation", "acceleration", "deceleration",
        }
        if cid in generic_singles:
            return False
        # Names with quotation marks (typically pulled from text, not real concepts)
        if name.startswith('"') or name.startswith("'"):
            return False
        # Concept name contains "you" / motivational phrases
        if re.match(r"^you\s", name, re.IGNORECASE):
            return False
        # Non-concept terms
        non_concept_patterns = {
            "amelia", "jacob_degrom", "degrom",
            "metaphors", "metaphor", "analogy", "analogies",
            "concrete_vs_sand", "external_cues", "internal_cues",
            "stable_vs_loose",
        }
        if cid in non_concept_patterns:
            return False
        # Phrases that are instructions, not concepts (>5 words)
        word_count = len(name.split())
        if word_count > 5:
            return False
        # IDs that look like they contain video IDs (mixed case alphanumeric, not pure words)
        # Video IDs have digits mixed in like "pWzyP_xfLfU" -> "pwzyp_xflfu"
        if re.match(r"^[a-z]+_[a-z]*\d[a-z0-9]*$", cid) and len(cid) > 15:
            return False
        return True

    return _is_quality_concept


# ── Markdown concept extraction ─────────────────────────────────────────


def extract_concepts_from_markdown(
    md_path: Path,
    video_id: str,
    video_title: str,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Extract concepts, edges, and chains from a Gemini analysis Markdown file.

    Parses the table in '### Section 2: New Concept Identification' section with
    columns: ID | Name (EN) | Name (ZH) | Definition | Category | Active/Passive

    Also extracts edges from the '### Section 5' connections table and drills
    from '### Section 7' training methods table.

    Returns (concepts_dicts, edges_dicts, chains_dicts) ready for JSON serialization.
    """
    text = md_path.read_text(encoding="utf-8")
    source_file = f"ftt_video_{video_id}"

    concepts: list[dict] = []
    edges: list[dict] = []
    chains: list[dict] = []

    # ── Extract new concepts from table ──
    # Find tables with concept-like columns (ID | Name columns)
    # Pattern: rows like | V01-01 | English Name | Chinese Name | Definition | Category | Active/Passive |
    concept_table_pattern = re.compile(
        r"\|\s*V?\d+[-_]\d+\s*\|(.+?\|){3,5}\s*\n",
        re.MULTILINE,
    )

    # More robust: find all markdown table rows that look like concept entries
    # They have an ID starting with V or a number, followed by pipe-separated fields
    row_pattern = re.compile(
        r"^\|\s*(V?\d+[-_]\d+)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|",
        re.MULTILINE,
    )

    seen_ids = set()
    for match in row_pattern.finditer(text):
        _local_id = match.group(1).strip()
        name_en = match.group(2).strip().strip("*")
        name_zh = match.group(3).strip().strip("*")
        definition = match.group(4).strip()
        category_str = match.group(5).strip()
        active_passive = match.group(6).strip()

        # Skip header/separator rows
        if name_en.startswith("---") or name_en.lower().startswith("name"):
            continue
        if not name_en or len(name_en) < 2:
            continue

        concept_id = _to_snake_id(name_en)
        if not concept_id or concept_id in seen_ids:
            continue
        seen_ids.add(concept_id)

        concept = {
            "id": concept_id,
            "name": name_en,
            "name_zh": name_zh,
            "aliases": [],
            "category": _parse_category(category_str).value,
            "sources": ["ftt"],
            "description": definition[:200],
            "vlm_features": [],
            "muscles_involved": [],
            "active_or_passive": active_passive if active_passive.lower() in ("active", "passive", "active (意图)") else None,
            "confidence": 0.8,
        }
        concepts.append(concept)

    # ── Also extract bold-term concepts (from Section 4 teaching points) ──
    # Only extract from teaching point headers like [Square Contact Zone] or [抛球追踪法]
    teaching_point_pattern = re.compile(
        r"\d+\.\s*\*\*\[(.+?)\]\*\*"
    )
    for match in teaching_point_pattern.finditer(text):
        term = match.group(1).strip()

        # Skip if too long (likely a sentence, not a concept)
        if len(term) > 50:
            continue

        # Try to extract bilingual names
        en_zh = re.match(r"(.+?)\s*[（(](.+?)[)）]", term)
        if en_zh:
            name_en = en_zh.group(1).strip()
            name_zh = en_zh.group(2).strip()
            if re.match(r"[\u4e00-\u9fff]", name_en):
                name_en, name_zh = name_zh, name_en
        else:
            # Pure Chinese term -- try to use as-is
            if re.match(r"^[\u4e00-\u9fff\s]+$", term):
                name_zh = term
                name_en = term  # Will be filtered by _to_snake_id quality check
            else:
                name_en = term
                name_zh = term

        concept_id = _to_snake_id(name_en)
        if not concept_id or len(concept_id) < 3 or concept_id in seen_ids:
            continue
        # Filter out IDs that are just numbers or very generic
        if re.match(r"^c_\d+$", concept_id) or len(concept_id) > 50:
            continue
        seen_ids.add(concept_id)

        concept = {
            "id": concept_id,
            "name": name_en,
            "name_zh": name_zh,
            "aliases": [],
            "category": "technique",
            "sources": ["ftt"],
            "description": f"Teaching point from FTT video: {video_title}",
            "vlm_features": [],
            "muscles_involved": [],
            "active_or_passive": None,
            "confidence": 0.7,
        }
        concepts.append(concept)

    # ── Extract edges from Section 5 (connections to existing concepts) ──
    # Pattern: | what video says | **CXX concept** | relation | meaning |
    edge_pattern = re.compile(
        r"\|\s*.+?\s*\|\s*\*\*([A-Z]\d+)\s+(.+?)\*\*\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|",
        re.MULTILINE,
    )
    for match in edge_pattern.finditer(text):
        existing_id = match.group(1).strip().lower()
        relation_str = match.group(3).strip()
        evidence = match.group(4).strip()

        # Map CXX style IDs to snake_case from concept name
        concept_name = match.group(2).strip()
        target_id = _to_snake_id(concept_name)

        for c in concepts:
            edges.append({
                "source_id": c["id"],
                "target_id": target_id,
                "relation": _parse_relation(relation_str).value,
                "confidence": 0.6,
                "evidence": evidence[:200],
                "source_file": source_file,
                "resolution": None,
            })

    # ── Create co-occurrence edges between extracted concepts ──
    concept_ids = [c["id"] for c in concepts]
    if len(concept_ids) >= 2:
        for i, src_id in enumerate(concept_ids):
            for tgt_id in concept_ids[i + 1:]:
                edges.append({
                    "source_id": src_id,
                    "target_id": tgt_id,
                    "relation": "supports",
                    "confidence": 0.6,
                    "evidence": f"Co-occurring in video: {video_title}",
                    "source_file": source_file,
                    "resolution": None,
                })

    # ── Extract drill concepts from Section 7 ──
    drill_pattern = re.compile(
        r"\|\s*\*\*(.+?)\*\*\s*\|.*?\|.*?\|.*?\|.*?\|",
        re.MULTILINE,
    )
    # Only look in section 7 area
    section7_match = re.search(r"第七部分|训练方法", text)
    if section7_match:
        section7_text = text[section7_match.start():]
        # Find next section boundary
        next_section = re.search(r"### 第[八九]部分|---\n\n###", section7_text[10:])
        if next_section:
            section7_text = section7_text[:next_section.start() + 10]

        for match in drill_pattern.finditer(section7_text):
            drill_name = match.group(1).strip()
            if drill_name.startswith("训练") or drill_name.lower().startswith("name"):
                continue

            # Try to get English name from parentheses
            en_match = re.search(r"[（(](.+?)[)）]", drill_name)
            if en_match:
                name_en = en_match.group(1)
                name_zh = re.sub(r"\s*[（(].+?[)）]", "", drill_name)
            else:
                name_en = drill_name
                name_zh = drill_name

            drill_id = _to_snake_id(name_en)
            if not drill_id or len(drill_id) < 3 or drill_id in seen_ids:
                continue
            seen_ids.add(drill_id)

            drill = {
                "id": drill_id,
                "name": name_en,
                "name_zh": name_zh,
                "aliases": [],
                "category": "drill",
                "sources": ["ftt"],
                "description": f"Training drill from FTT video: {video_title}",
                "vlm_features": [],
                "muscles_involved": [],
                "active_or_passive": None,
                "confidence": 0.8,
            }
            concepts.append(drill)

    return concepts, edges, chains


# ── Step 1: Merge per-plan state slices ──────────────────────────────────


def merge_state_slices() -> dict:
    """Merge batch0-3 state slices into canonical ftt_video_state.json."""
    canonical_path = STATE_DIR / "ftt_video_state.json"
    canonical = json.loads(canonical_path.read_text())

    slice_files = [
        "batch0_state.json",
        "batch1_state.json",
        "batch2_state.json",
        "batch3_state.json",
    ]

    for sf in slice_files:
        sp = STATE_DIR / sf
        if not sp.exists():
            print(f"WARNING: {sf} not found, skipping")
            continue
        slice_data = json.loads(sp.read_text())
        for vid, vdata in slice_data["videos"].items():
            if vid in canonical["videos"]:
                canonical["videos"][vid].update(vdata)
            else:
                print(f"WARNING: {vid} from {sf} not in canonical state, adding")
                canonical["videos"][vid] = vdata

    # Validate
    by_status: dict[str, int] = {}
    for v in canonical["videos"].values():
        s = v.get("status", "unknown")
        by_status[s] = by_status.get(s, 0) + 1

    total = len(canonical["videos"])
    pending = by_status.get("pending", 0)
    analyzed_only = by_status.get("analyzed", 0)

    print(f"\n=== STATE MERGE ===")
    print(f"Total videos: {total}")
    print(f"By status: {by_status}")

    if pending > 0:
        print(f"WARNING: {pending} videos still pending")
    if analyzed_only > 0:
        print(f"NOTE: {analyzed_only} videos have 'analyzed' status (from batch0, will be updated to 'extracted')")

    # Save
    canonical_path.write_text(
        json.dumps(canonical, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved canonical state to {canonical_path}")
    return canonical


# ── Step 2: Re-extract concepts from Markdown for 0-concept videos ───────


def reextract_missing_concepts(canonical_state: dict) -> int:
    """Re-extract concepts from Markdown files for videos with 0 concepts in their JSON."""
    reextracted = 0

    for vf in sorted(FTT_VIDEOS_DIR.glob("*.json")):
        if vf.name.startswith("_") or vf.name.startswith("09_"):
            continue

        video_id = vf.stem
        data = json.loads(vf.read_text())
        existing_concepts = len(data.get("concepts", []))

        if existing_concepts > 0:
            continue  # Already has concepts

        # Check if markdown exists
        md_path = ANALYSES_DIR / f"{video_id}.md"
        if not md_path.exists():
            continue

        # Get video title from state
        video_info = canonical_state.get("videos", {}).get(video_id, {})
        video_title = video_info.get("title", video_id)

        concepts, edges, chains = extract_concepts_from_markdown(
            md_path, video_id, video_title
        )

        if concepts:
            # Update the JSON file
            data["concepts"] = concepts
            data["edges"] = edges
            data["diagnostic_chains"] = chains
            vf.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"  Re-extracted {video_id}: {len(concepts)} concepts, {len(edges)} edges")
            reextracted += 1

            # Update state
            if video_id in canonical_state.get("videos", {}):
                canonical_state["videos"][video_id]["concepts_count"] = len(concepts)
                canonical_state["videos"][video_id]["edges_count"] = len(edges)
        else:
            print(f"  {video_id}: Markdown exists but no concepts extracted")

    print(f"\nRe-extracted concepts for {reextracted} videos")
    return reextracted


# ── Step 3: Merge concepts into registry ─────────────────────────────────


def merge_concepts_into_registry() -> tuple[dict, ConceptRegistry, list[dict]]:
    """Load base registry, merge all per-video concepts with dedup.

    Returns (merge_stats, registry, all_edges).
    """
    # Load base registry (with quality filtering on load)
    registry = ConceptRegistry()
    base_data = json.loads(SNAPSHOT_PATH.read_text())
    base_filtered = 0
    # Filter garbage from base registry too
    garbage_base_ids = {
        "ftt", "youtube", "vlm", "through", "coco", "rafa", "sinner",
        "alcaraz", "scooping", "grip", "stance", "balance", "vision",
        "timing", "control", "power", "unknown_concept",
    }
    for c_data in base_data:
        cid = c_data.get("id", "")
        if cid in garbage_base_ids:
            base_filtered += 1
            continue
        registry.add(Concept(**c_data))
    base_count = len(registry)
    if base_filtered:
        print(f"Filtered {base_filtered} garbage concepts from base registry")
    print(f"\n=== CONCEPT MERGE ===")
    print(f"Base registry: {base_count} concepts")

    # Load all per-video extraction JSONs
    video_files = sorted(FTT_VIDEOS_DIR.glob("*.json"))
    video_files = [f for f in video_files if not f.name.startswith("_")]

    stats = {
        "base_registry_count": base_count,
        "total_raw_concepts_from_videos": 0,
        "new_concepts_added": 0,
        "concepts_deduplicated": 0,
        "per_video": {},
    }
    all_edges: list[dict] = []
    all_chains: list[dict] = []
    new_concepts: list[Concept] = []

    is_quality = _is_quality_concept_fn()

    for vf in video_files:
        data = json.loads(vf.read_text())
        video_id = vf.stem
        video_new = 0
        video_dedup = 0
        video_filtered = 0

        for c_data in data.get("concepts", []):
            stats["total_raw_concepts_from_videos"] += 1

            if not is_quality(c_data):
                video_filtered += 1
                continue

            try:
                concept = Concept(**c_data)
            except Exception as e:
                print(f"  WARNING: Invalid concept in {video_id}: {e}")
                continue

            # Pre-check with lower threshold (75) to catch more near-duplicates
            pre_match = registry.resolve(concept.name, threshold=75)
            if pre_match is not None:
                video_dedup += 1
                stats["concepts_deduplicated"] += 1
                continue

            dup_id = registry.add(concept)
            if dup_id is None:
                video_new += 1
                stats["new_concepts_added"] += 1
                new_concepts.append(concept)
            else:
                video_dedup += 1
                stats["concepts_deduplicated"] += 1

        all_edges.extend(data.get("edges", []))
        all_chains.extend(data.get("diagnostic_chains", []))
        stats["per_video"][video_id] = {
            "raw": len(data.get("concepts", [])),
            "new": video_new,
            "deduped": video_dedup,
        }

    # Deduplicate edges
    edge_key = lambda e: (e.get("source_id"), e.get("target_id"), e.get("relation"))
    edge_groups: dict[tuple, list[dict]] = defaultdict(list)
    for e in all_edges:
        edge_groups[edge_key(e)].append(e)

    unique_edges = []
    for key, group in edge_groups.items():
        # Keep highest confidence
        best = max(group, key=lambda e: e.get("confidence", 0))
        unique_edges.append(best)

    stats["total_edges"] = len(all_edges)
    stats["unique_edges"] = len(unique_edges)
    stats["final_registry_count"] = len(registry)

    print(f"Raw concepts from videos: {stats['total_raw_concepts_from_videos']}")
    print(f"New added: {stats['new_concepts_added']}")
    print(f"Deduplicated: {stats['concepts_deduplicated']}")
    print(f"Final registry: {len(registry)} concepts")
    print(f"Edges: {len(all_edges)} total -> {len(unique_edges)} unique")
    print(f"Videos processed: {len(stats['per_video'])}")

    # Warnings
    if len(registry) > 700:
        print("WARNING: Concept explosion detected!")
    elif len(registry) > 500:
        print(f"NOTE: Registry ({len(registry)}) slightly above 500 target -- 73 videos produce many unique concepts")
    elif len(registry) < 310:
        print("WARNING: Very few new concepts -- extraction quality may be poor")

    # Top 10 most-connected concepts by edge count
    edge_counts: Counter = Counter()
    for e in unique_edges:
        edge_counts[e.get("source_id", "")] += 1
        edge_counts[e.get("target_id", "")] += 1
    print(f"\nTop 10 most-connected concepts:")
    for cid, count in edge_counts.most_common(10):
        print(f"  {cid}: {count} edges")

    # Spot-check: 10 random new concepts
    if new_concepts:
        sample_size = min(10, len(new_concepts))
        sample = random.sample(new_concepts, sample_size)
        print(f"\nSpot-check ({sample_size} random new concepts):")
        for c in sample:
            print(f"  [{c.category.value}] {c.id}: {c.name} / {c.name_zh}")

    # Save updated registry snapshot
    all_concepts_data = [c.model_dump() for c in registry.all_concepts()]
    # Serialize enum values
    for cd in all_concepts_data:
        if hasattr(cd.get("category"), "value"):
            cd["category"] = cd["category"].value
    SNAPSHOT_PATH.write_text(
        json.dumps(all_concepts_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved registry snapshot ({len(registry)} concepts) to {SNAPSHOT_PATH}")

    # Save merge report
    report_path = FTT_VIDEOS_DIR / "_merge_report.json"
    report_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved merge report to {report_path}")

    return stats, registry, all_chains


# ── Step 4: Extract and consolidate diagnostic chains ────────────────────


def consolidate_diagnostic_chains(
    all_chains_from_merge: list[dict],
    registry: ConceptRegistry,
) -> dict:
    """Consolidate diagnostic chains: dedup, validate, supplement with FTT standard patterns."""
    # Dedup by symptom_concept_id (keep the one with more steps)
    chain_by_symptom: dict[str, dict] = {}
    for ch in all_chains_from_merge:
        scid = ch.get("symptom_concept_id", "")
        if scid not in chain_by_symptom:
            chain_by_symptom[scid] = ch
        else:
            existing_steps = len(chain_by_symptom[scid].get("check_sequence", []))
            new_steps = len(ch.get("check_sequence", []))
            if new_steps > existing_steps:
                chain_by_symptom[scid] = ch

    from_videos = len(chain_by_symptom)

    # Validate concept references
    warnings = []
    for ch in chain_by_symptom.values():
        scid = ch.get("symptom_concept_id", "")
        if registry.get(scid) is None and registry.resolve(scid) is None:
            warnings.append(f"symptom_concept_id '{scid}' not in registry")
        for rc in ch.get("root_causes", []):
            if registry.get(rc) is None and registry.resolve(rc) is None:
                warnings.append(f"root_cause '{rc}' not in registry")
        for d in ch.get("drills", []):
            if registry.get(d) is None and registry.resolve(d) is None:
                warnings.append(f"drill '{d}' not in registry")

    # Supplement with FTT standard diagnostic patterns
    standard_chains = [
        {
            "id": "dc_arm_driven_hitting",
            "symptom": "Arm initiating swing instead of body rotation",
            "symptom_zh": "手臂主导挥拍而非身体旋转",
            "symptom_concept_id": "forearm_compensation",
            "check_sequence": [
                {"check": "Is the hip rotating before the shoulder?", "check_zh": "髋部是否在肩部之前旋转？", "if_true": "kinetic_chain_sequence", "if_false": None},
                {"check": "Is the arm starting the swing independently?", "check_zh": "手臂是否独立启动挥拍？", "if_true": "forearm_compensation", "if_false": None},
                {"check": "Is there visible lag between trunk and racket?", "check_zh": "躯干和球拍之间是否有可见的滞后？", "if_true": "wrist_lag", "if_false": "forearm_compensation"},
            ],
            "root_causes": ["forearm_compensation", "unit_turn"],
            "drills": ["drop_feed_rotation_drill", "weighted_racket_swing"],
            "priority": 1,
            "vlm_frame": None,
        },
        {
            "id": "dc_scooping",
            "symptom": "Racket dropping too low and scooping up through contact",
            "symptom_zh": "球拍下沉过低，从下向上舀球",
            "symptom_concept_id": "racket_drop",
            "check_sequence": [
                {"check": "Does the racket drop below the hand before forward swing?", "check_zh": "球拍在前挥前是否低于手部？", "if_true": "racket_drop", "if_false": None},
                {"check": "Is the swing path steeply upward through contact?", "check_zh": "挥拍路径是否在击球点陡峭上升？", "if_true": "racket_drop", "if_false": None},
            ],
            "root_causes": ["racket_drop", "wrist_lag"],
            "drills": ["contact_point_drill"],
            "priority": 2,
            "vlm_frame": None,
        },
        {
            "id": "dc_missing_out_vector",
            "symptom": "No forward extension through the contact zone",
            "symptom_zh": "击球区缺乏向前延伸",
            "symptom_concept_id": "swing_out",
            "check_sequence": [
                {"check": "Does the racket extend forward through contact?", "check_zh": "球拍是否在触球后向前延伸？", "if_true": "swing_out", "if_false": None},
                {"check": "Is the follow-through wrapping around the body too early?", "check_zh": "随挥是否过早缠绕身体？", "if_true": "over_rotation", "if_false": None},
            ],
            "root_causes": ["swing_out", "over_rotation"],
            "drills": ["swing_out_target_drill"],
            "priority": 2,
            "vlm_frame": None,
        },
        {
            "id": "dc_over_rotation",
            "symptom": "Excessive body rotation past the contact point",
            "symptom_zh": "身体过度旋转超过击球点",
            "symptom_concept_id": "over_rotation",
            "check_sequence": [
                {"check": "Does the chest face the net before contact?", "check_zh": "胸部是否在击球前就面向球网？", "if_true": "over_rotation", "if_false": None},
                {"check": "Is the non-dominant arm flying open?", "check_zh": "非持拍手臂是否向外飞开？", "if_true": "non_dominant_arm_role", "if_false": None},
                {"check": "Is the head turning away before contact?", "check_zh": "头部是否在击球前转开？", "if_true": "over_rotation", "if_false": None},
            ],
            "root_causes": ["over_rotation", "non_dominant_arm_role"],
            "drills": ["freeze_at_contact_drill", "non_dominant_arm_catch"],
            "priority": 1,
            "vlm_frame": None,
        },
        {
            "id": "dc_early_release",
            "symptom": "Wrist releasing before reaching the contact zone",
            "symptom_zh": "手腕在到达击球区前就释放",
            "symptom_concept_id": "wrist_lag",
            "check_sequence": [
                {"check": "Is the wrist angle maintained until contact zone?", "check_zh": "手腕角度是否保持到击球区？", "if_true": "wrist_lag", "if_false": None},
                {"check": "Does the racket head pass the hand before contact?", "check_zh": "拍头是否在触球前超过手部？", "if_true": "wrist_lag", "if_false": None},
            ],
            "root_causes": ["wrist_lag", "forearm_compensation"],
            "drills": ["lag_awareness_drill"],
            "priority": 2,
            "vlm_frame": None,
        },
        {
            "id": "dc_trunk_momentum_leak",
            "symptom": "Loss of rotational energy in trunk sequencing",
            "symptom_zh": "躯干序列中旋转能量泄漏",
            "symptom_concept_id": "trunk_sequencing",
            "check_sequence": [
                {"check": "Is there a visible separation between hip and shoulder rotation?", "check_zh": "髋部和肩部旋转之间是否有可见的分离？", "if_true": "trunk_sequencing", "if_false": None},
                {"check": "Does the trunk decelerate smoothly after contact?", "check_zh": "躯干在击球后是否平稳减速？", "if_true": "trunk_sequencing", "if_false": "over_rotation"},
            ],
            "root_causes": ["trunk_sequencing", "kinetic_chain_sequence"],
            "drills": ["trunk_rotation_drill", "abdominal_corkscrew"],
            "priority": 2,
            "vlm_frame": None,
        },
    ]

    # Only add standard chains not already covered
    manually_supplemented = 0
    existing_symptom_ids = set(chain_by_symptom.keys())
    existing_chain_ids = {ch.get("id") for ch in chain_by_symptom.values()}

    for sc in standard_chains:
        if sc["id"] not in existing_chain_ids and sc["symptom_concept_id"] not in existing_symptom_ids:
            chain_by_symptom[sc["symptom_concept_id"]] = sc
            manually_supplemented += 1

    # Validate supplemented chains too
    for ch in standard_chains:
        scid = ch.get("symptom_concept_id", "")
        if registry.get(scid) is None and registry.resolve(scid) is None:
            warnings.append(f"[supplemented] symptom_concept_id '{scid}' not in registry")

    final_chains = list(chain_by_symptom.values())

    result = {
        "chains": final_chains,
        "count": len(final_chains),
        "from_videos": from_videos,
        "manually_supplemented": manually_supplemented,
        "validation_warnings": list(set(warnings)),
    }

    output_path = EXTRACTED_DIR / "ftt_video_diagnostic_chains.json"
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n=== DIAGNOSTIC CHAINS ===")
    print(f"Total chains: {len(final_chains)}")
    print(f"From videos: {from_videos}")
    print(f"Manually supplemented: {manually_supplemented}")
    print(f"Validation warnings: {len(set(warnings))}")
    if warnings:
        for w in sorted(set(warnings))[:10]:
            print(f"  - {w}")
    print(f"Saved to {output_path}")

    return result


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("FTT Video Concept Merge Pipeline")
    print("=" * 60)

    # Step 1: Merge state slices
    canonical_state = merge_state_slices()

    # Step 2: Re-extract concepts from Markdown for 0-concept videos
    print(f"\n=== RE-EXTRACTION ===")
    reextract_missing_concepts(canonical_state)

    # Save updated canonical state (with re-extracted counts)
    canonical_path = STATE_DIR / "ftt_video_state.json"
    canonical_path.write_text(
        json.dumps(canonical_state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Step 3: Merge concepts into registry
    stats, registry, all_chains = merge_concepts_into_registry()

    # Step 4: Consolidate diagnostic chains
    chains_result = consolidate_diagnostic_chains(all_chains, registry)

    print(f"\n{'=' * 60}")
    print(f"MERGE COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
