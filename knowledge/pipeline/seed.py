"""Legacy JSON migration and registry seeding.

Migrates concepts from 3 legacy JSON files (ftt_core_concepts.json,
tpa_kinetic_chain.json, user_journey.json) into Pydantic Concept objects
with snake_case IDs, then seeds them into a ConceptRegistry.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from knowledge.schemas import Concept, ConceptType
from knowledge.registry import ConceptRegistry

LEGACY_JSON_DIR = Path(__file__).resolve().parent.parent.parent / "docs" / "knowledge_graph"

# ---------------------------------------------------------------------------
# Category mapping: Chinese category strings -> ConceptType
# ---------------------------------------------------------------------------

CATEGORY_MAP: dict[str, ConceptType] = {
    # ftt_core_concepts.json categories
    "发力模型": ConceptType.BIOMECHANICS,
    "准备阶段": ConceptType.TECHNIQUE,
    "前挥阶段": ConceptType.TECHNIQUE,
    "连接机制": ConceptType.BIOMECHANICS,
    "心理模型": ConceptType.MENTAL_MODEL,
    "握拍": ConceptType.TECHNIQUE,
    "训练方法": ConceptType.DRILL,
    # user_journey.json implicit categories
    "用户问题": ConceptType.SYMPTOM,
    "用户突破": ConceptType.TECHNIQUE,
    # tpa_kinetic_chain.json - all are technique/biomechanics concepts
    "技术概念": ConceptType.TECHNIQUE,
    "诊断标准": ConceptType.SYMPTOM,
}

# ---------------------------------------------------------------------------
# Cross-source alias mappings for key concepts
# ---------------------------------------------------------------------------

ALIAS_MAP: dict[str, list[str]] = {
    "unit_turn": ["loading phase", "coiling", "backswing rotation", "take back rotation"],
    "rotational_kinetic_chain": ["kinetic chain", "whip chain", "energy chain", "power chain"],
    "windshield_wiper": ["wiper", "racket flip", "forearm pronation finish"],
    "hip_shoulder_separation": ["separation", "X-factor", "torso coil", "hip lead"],
    "forearm_compensation": ["arm driven hitting", "arm dominant", "forearm dominant"],
    "wrist_lag": ["lag", "racket lag", "delayed wrist"],
    "out_vector": ["out direction", "lateral swing", "swing outward"],
    "press_slot": ["slot", "hitting slot", "power slot", "contact slot"],
    "chest_engagement": ["chest press", "pec connection", "chest drive"],
    "scapular_glide": ["scapular retraction", "scapular protraction", "shoulder blade slide"],
    "fault_tolerance": ["fault tolerant", "error margin", "consistency margin"],
    "unit_swing": ["connected swing", "body swing", "whole body swing"],
    "back_muscle_connection": ["lat glue", "back connection", "lat connection"],
    "hand_slot_position": ["hand slot", "hand position", "preparation position"],
    "shoulder_tilt": ["shoulder slant", "tilt", "shoulder drop"],
    "braking_chain": ["reactive braking", "segment braking", "deceleration chain"],
    "over_rotation": ["spinning out", "excessive rotation"],
}


def to_snake_id(name: str) -> str:
    """Convert a display name to a snake_case ID.

    Examples:
        'Unit Turn' -> 'unit_turn'
        'Hip-Shoulder Separation' -> 'hip_shoulder_separation'
        'Out, Up, and Through — Three Vector Model' -> 'out_up_and_through_three_vector_model'
    """
    result = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    # IDs must start with a letter; prefix with 'n' if starts with digit
    if result and result[0].isdigit():
        result = "n" + result
    return result


def _extract_name_zh(raw: dict) -> str:
    """Extract Chinese name from various legacy JSON field formats."""
    # ftt_core_concepts.json: name_zh field
    if "name_zh" in raw:
        return raw["name_zh"]
    # tpa_kinetic_chain.json: name field contains 'English / 中文'
    if "name" in raw:
        parts = raw["name"].split(" / ")
        if len(parts) == 2:
            return parts[1].strip()
        return raw["name"]
    # user_journey.json: name field may contain '中文 / English' or just Chinese
    return raw.get("name", "")


def _extract_name_en(raw: dict) -> str:
    """Extract English name from various legacy JSON field formats."""
    # ftt_core_concepts.json: name_en field
    if "name_en" in raw:
        return raw["name_en"]
    # tpa_kinetic_chain.json: name field contains 'English / 中文'
    if "name" in raw:
        parts = raw["name"].split(" / ")
        if len(parts) == 2:
            return parts[0].strip()
        return raw["name"]
    return raw.get("name", "")


def _process_ftt_concepts(registry: ConceptRegistry) -> list[Concept]:
    """Process ftt_core_concepts.json: 48 concepts with C01-style IDs."""
    path = LEGACY_JSON_DIR / "ftt_core_concepts.json"
    data = json.loads(path.read_text())
    concepts: list[Concept] = []

    for raw in data.get("concepts", []):
        name_en = raw.get("name_en", "")
        snake_id = to_snake_id(name_en)
        category_zh = raw.get("category", "")
        category = CATEGORY_MAP.get(category_zh, ConceptType.TECHNIQUE)
        aliases = ALIAS_MAP.get(snake_id, [])

        concept = Concept(
            id=snake_id,
            name=name_en,
            name_zh=raw.get("name_zh", ""),
            aliases=aliases,
            category=category,
            sources=["ftt"],
            description=raw.get("definition", ""),
            active_or_passive=raw.get("active_or_passive"),
            confidence=1.0,
        )

        existing_id = registry.add(concept)
        if existing_id is not None:
            # Merge aliases and sources into existing concept
            existing = registry.get(existing_id)
            if existing is not None:
                for alias in concept.aliases:
                    if alias not in existing.aliases:
                        existing.aliases.append(alias)
                if "ftt" not in existing.sources:
                    existing.sources.append("ftt")
        else:
            concepts.append(concept)

    return concepts


def _process_tpa_concepts(registry: ConceptRegistry) -> list[Concept]:
    """Process tpa_kinetic_chain.json: 24 concepts with T01-style IDs."""
    path = LEGACY_JSON_DIR / "tpa_kinetic_chain.json"
    data = json.loads(path.read_text())
    concepts: list[Concept] = []

    for raw in data.get("concepts", []):
        name_en = _extract_name_en(raw)
        snake_id = to_snake_id(name_en)
        aliases = ALIAS_MAP.get(snake_id, [])

        concept = Concept(
            id=snake_id,
            name=name_en,
            name_zh=_extract_name_zh(raw),
            aliases=aliases,
            category=ConceptType.TECHNIQUE,
            sources=["tpa"],
            description=raw.get("definition", ""),
            confidence=1.0,
        )

        existing_id = registry.add(concept)
        if existing_id is not None:
            existing = registry.get(existing_id)
            if existing is not None:
                for alias in concept.aliases:
                    if alias not in existing.aliases:
                        existing.aliases.append(alias)
                if "tpa" not in existing.sources:
                    existing.sources.append("tpa")
        else:
            concepts.append(concept)

    return concepts


def _process_user_journey(registry: ConceptRegistry) -> list[Concept]:
    """Process user_journey.json: problems as SYMPTOM, breakthroughs as TECHNIQUE."""
    path = LEGACY_JSON_DIR / "user_journey.json"
    data = json.loads(path.read_text())
    concepts: list[Concept] = []

    # Problems -> SYMPTOM concepts
    for raw in data.get("problems", []):
        raw_name = raw.get("name", "")
        raw_id = raw.get("id", "")
        # Names are mixed Chinese/English: '拍头过度下坠 / Pat the Dog 主动下压'
        # Use the raw ID (P01, P02...) to build a stable snake_case ID
        parts = raw_name.split(" / ")
        name_zh = parts[0].strip() if parts else raw_name
        name_en = parts[1].strip() if len(parts) >= 2 else raw_name

        # Use a descriptive snake_id derived from the problem ID and key English words
        snake_id = f"problem_{raw_id.lower()}" if raw_id else to_snake_id(name_en)
        aliases = ALIAS_MAP.get(snake_id, [])

        # Build description from symptoms + root_cause
        symptoms = raw.get("symptoms", [])
        root_cause = raw.get("root_cause", "")
        description = f"Symptoms: {'; '.join(symptoms)}. Root cause: {root_cause}"

        concept = Concept(
            id=snake_id,
            name=name_en,
            name_zh=name_zh,
            aliases=aliases,
            category=ConceptType.SYMPTOM,
            sources=["user_experience"],
            description=description,
            confidence=1.0,
        )

        existing_id = registry.add(concept)
        if existing_id is not None:
            existing = registry.get(existing_id)
            if existing is not None:
                if "user_experience" not in existing.sources:
                    existing.sources.append("user_experience")
        else:
            concepts.append(concept)

    # Breakthroughs -> TECHNIQUE concepts
    for raw in data.get("breakthroughs", []):
        desc = raw.get("description", "")
        raw_id = raw.get("id", "")
        # Descriptions are Chinese; use B01/B02 IDs for stable snake_case
        snake_id = f"breakthrough_{raw_id.lower()}" if raw_id else to_snake_id(desc)
        what_changed = raw.get("what_changed", {})
        full_desc = (
            f"{desc}. Before: {what_changed.get('before', '')}. "
            f"After: {what_changed.get('after', '')}."
        )

        concept = Concept(
            id=snake_id,
            name=desc,
            name_zh=desc,  # Breakthroughs are already in Chinese
            aliases=[],
            category=ConceptType.TECHNIQUE,
            sources=["user_experience"],
            description=full_desc,
            confidence=0.9,  # User experience, slightly lower confidence
        )

        existing_id = registry.add(concept)
        if existing_id is not None:
            existing = registry.get(existing_id)
            if existing is not None:
                if "user_experience" not in existing.sources:
                    existing.sources.append("user_experience")
        else:
            concepts.append(concept)

    return concepts


def seed_registry_from_legacy_json(registry: ConceptRegistry) -> list[Concept]:
    """Seed the ConceptRegistry from 3 legacy JSON files.

    Processes files in order:
    1. ftt_core_concepts.json (48 concepts, highest priority)
    2. tpa_kinetic_chain.json (24 concepts)
    3. user_journey.json (21 problems + 16 breakthroughs)

    Duplicates are merged (aliases/sources added to existing concept).

    Returns:
        List of successfully added (non-duplicate) Concept objects.
    """
    all_concepts: list[Concept] = []
    all_concepts.extend(_process_ftt_concepts(registry))
    all_concepts.extend(_process_tpa_concepts(registry))
    all_concepts.extend(_process_user_journey(registry))
    return all_concepts


def save_seed_snapshot(concepts: list[Concept], path: Path) -> None:
    """Serialize seeded concepts to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [c.model_dump() for c in concepts]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
