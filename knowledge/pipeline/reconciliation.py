"""Cross-source reconciliation: classify secondary concepts against FTT-primary registry.

Classifies each secondary source concept as:
- agreement: matches existing FTT concept (boost confidence)
- complement: new concept not in FTT (add with medium confidence)
- conflict: contradicts FTT concept (FTT wins, log for review)

Confidence scoring (GRAPH-04 spec):
- FTT only:               0.8
- FTT + 1 secondary:      0.9
- FTT + 2 secondaries:    0.95
- Single secondary only:  0.6
- Two secondaries agree:  0.7
- Secondary contradicts:  FTT keeps 0.8, secondary marked 0.3
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TypedDict

from knowledge.registry import ConceptRegistry
from knowledge.schemas import Concept


# --- Negation words for conflict detection ---
_NEGATION_WORDS = {"not", "don't", "dont", "avoid", "wrong", "incorrect", "never", "shouldn't", "shouldnt"}


class ReconciliationResult(TypedDict):
    agreements: int
    complements: int
    conflicts: int
    total: int
    conflict_log: list[dict]
    complement_concepts: list[dict]
    confidence_boosted: list[dict]


def classify_concept(concept: Concept, registry: ConceptRegistry) -> str:
    """Classify a secondary concept against the FTT registry.

    Returns: "agreement" | "complement" | "conflict"
    """
    matched_id = registry.resolve(concept.name, threshold=70)
    if matched_id is None:
        # Also try aliases
        for alias in concept.aliases:
            matched_id = registry.resolve(alias, threshold=70)
            if matched_id is not None:
                break

    if matched_id is None:
        return "complement"

    # Check for conflict: secondary description contains negation words
    # paired with match to existing concept
    existing = registry.get(matched_id)
    if existing and _descriptions_conflict(existing.description, concept.description):
        return "conflict"

    return "agreement"


def _descriptions_conflict(ftt_desc: str, secondary_desc: str) -> bool:
    """Check if secondary description contradicts FTT via negation heuristic.

    Conflict if secondary contains negation words AND the FTT description
    does NOT contain those same negation words.
    """
    secondary_words = set(re.findall(r"[a-z']+", secondary_desc.lower()))
    ftt_words = set(re.findall(r"[a-z']+", ftt_desc.lower()))

    secondary_negations = secondary_words & _NEGATION_WORDS
    ftt_negations = ftt_words & _NEGATION_WORDS

    # Conflict: secondary has negation words that FTT does not
    if secondary_negations and not ftt_negations:
        return True
    return False


def boost_confidence(
    registry: ConceptRegistry,
    concept_id: str,
    agreement_sources: list[str],
) -> None:
    """Boost an FTT concept's confidence based on secondary source agreement.

    Mutates the concept in the registry in-place.
    """
    concept = registry.get(concept_id)
    if concept is None:
        return

    # Add sources
    for src in agreement_sources:
        if src not in concept.sources:
            concept.sources.append(src)

    # Count secondary sources that agree
    secondary_count = sum(
        1 for s in concept.sources if s not in ("ftt", "biomechanics_book", "user_experience")
    )

    if secondary_count >= 2:
        concept.confidence = 0.95
    elif secondary_count == 1:
        concept.confidence = 0.9


def reconcile_all(
    registry: ConceptRegistry,
    tomallsopp_dir: Path,
    feeltennis_dir: Path,
    ref_id_map: dict[str, str] | None = None,
) -> ReconciliationResult:
    """Run full cross-source reconciliation against the FTT-primary registry.

    1. Load all per-video JSONs from both secondary source directories
    2. Classify each concept as agreement/complement/conflict
    3. Boost confidence for agreements, add complements, log conflicts
    4. Second pass: boost dual-source complements from 0.6 to 0.7

    Returns ReconciliationResult with counts and detail logs.
    """
    agreements = 0
    complements = 0
    conflicts = 0
    conflict_log: list[dict] = []
    complement_concepts: list[dict] = []
    confidence_boosted: list[dict] = []

    # Track which FTT concept IDs have been boosted by which sources
    boost_tracker: dict[str, set[str]] = {}
    # Track complement concept names and their sources for dual-source detection
    complement_tracker: dict[str, list[str]] = {}  # lowered name -> list of source channels
    # Track complement concept IDs for second-pass boosting
    complement_ids: dict[str, str] = {}  # lowered name -> concept_id in registry

    def _load_from_dir(dir_path: Path, source_label: str):
        """Load new concepts and existing concept refs from per-video JSONs."""
        new_concepts = []
        existing_refs = []  # (ref_id, relationship, source_label)
        if not dir_path.exists():
            return new_concepts, existing_refs
        for json_file in sorted(dir_path.glob("*.json")):
            try:
                data = json.loads(json_file.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            for c_data in data.get("concepts", []):
                try:
                    concept = Concept(**c_data)
                    new_concepts.append((concept, source_label))
                except Exception:
                    continue
            # Process existing concept references (from markdown Sections 2 and 5)
            for ref in data.get("existing_concept_refs", []):
                ref_id = ref.get("ref_id", "")
                relationship = ref.get("relationship", "Supports")
                existing_refs.append((ref_id, relationship, source_label))
        return new_concepts, existing_refs

    # Load all secondary data
    ta_concepts, ta_refs = _load_from_dir(tomallsopp_dir, "tomallsopp")
    ft_concepts, ft_refs = _load_from_dir(feeltennis_dir, "feeltennis")
    all_secondary = ta_concepts + ft_concepts
    all_refs = ta_refs + ft_refs

    # Process existing concept references as agreements
    # These are explicit references like "C01 (Supports)" from Gemini analysis
    _ref_map = ref_id_map or {}
    for ref_id, relationship, source_label in all_refs:
        # Resolve short ID (e.g., "C01") to registry concept ID via provided mapping
        matched_id = _ref_map.get(ref_id)
        if matched_id is None:
            # Fallback to fuzzy resolve (works for some IDs)
            matched_id = registry.resolve(ref_id, threshold=70)
        if matched_id is None:
            continue
        matched_concept = registry.get(matched_id)
        if matched_concept and "ftt" in matched_concept.sources:
            rel_lower = relationship.lower()
            if rel_lower in ("contradicts",):
                conflicts += 1
                conflict_log.append({
                    "concept_name": ref_id,
                    "source": source_label,
                    "matched_ftt_id": matched_id,
                    "reason": f"Explicit contradiction relationship from video analysis",
                    "ftt_overrides": True,
                })
            else:
                # Supports, Refines, Extends, Is_instance_of -> agreement
                agreements += 1
                if matched_id not in boost_tracker:
                    boost_tracker[matched_id] = set()
                boost_tracker[matched_id].add(source_label)

    # First pass: classify each concept
    for concept, source_label in all_secondary:
        # Quality filter: skip concepts with invalid/empty names
        if len(concept.name.strip()) < 5 or concept.name.startswith("**"):
            continue
        classification = classify_concept(concept, registry)

        if classification == "agreement":
            matched_id = registry.resolve(concept.name, threshold=70)
            if matched_id is None:
                for alias in concept.aliases:
                    matched_id = registry.resolve(alias, threshold=70)
                    if matched_id:
                        break
            # Check if the matched concept is from FTT or from a previous secondary
            matched_concept = registry.get(matched_id) if matched_id else None
            is_ftt_concept = matched_concept and "ftt" in matched_concept.sources

            if is_ftt_concept:
                agreements += 1
                if matched_id not in boost_tracker:
                    boost_tracker[matched_id] = set()
                old_conf = matched_concept.confidence if matched_concept else 0.8
                boost_tracker[matched_id].add(source_label)
                if matched_concept and source_label not in matched_concept.sources:
                    confidence_boosted.append({
                        "id": matched_id,
                        "old_conf": old_conf,
                        "source_added": source_label,
                    })
            else:
                # Matched a complement from another secondary source -> dual-source complement
                complements += 1
                name_lower = concept.name.lower()
                if name_lower not in complement_tracker:
                    complement_tracker[name_lower] = []
                complement_tracker[name_lower].append(source_label)
                # Also add the original source from the matched concept
                if matched_concept:
                    for s in matched_concept.sources:
                        if s not in complement_tracker[name_lower]:
                            complement_tracker[name_lower].append(s)
                if matched_id:
                    complement_ids[name_lower] = matched_id
                continue

        elif classification == "complement":
            complements += 1
            name_lower = concept.name.lower()

            # Track for dual-source detection
            if name_lower not in complement_tracker:
                complement_tracker[name_lower] = []
            complement_tracker[name_lower].append(source_label)

            # Add to registry if not already added by a previous video
            existing_id = registry.resolve(concept.name, threshold=85)
            if existing_id is None:
                # New complement - add with confidence 0.6
                concept.confidence = 0.6
                if source_label not in concept.sources:
                    concept.sources = [source_label]
                dup_id = registry.add(concept)
                if dup_id is None:
                    # Successfully added
                    complement_ids[name_lower] = concept.id
                    complement_concepts.append({
                        "id": concept.id,
                        "name": concept.name,
                        "source": source_label,
                        "confidence": 0.6,
                    })
                else:
                    complement_ids[name_lower] = dup_id
            else:
                complement_ids[name_lower] = existing_id

        elif classification == "conflict":
            conflicts += 1
            matched_id = registry.resolve(concept.name, threshold=70)
            if matched_id is None:
                for alias in concept.aliases:
                    matched_id = registry.resolve(alias, threshold=70)
                    if matched_id:
                        break
            conflict_log.append({
                "concept_name": concept.name,
                "source": source_label,
                "matched_ftt_id": matched_id,
                "reason": f"Secondary description contains negation: {concept.description[:200]}",
                "ftt_overrides": True,
            })

    # Apply confidence boosts for agreements
    for concept_id, sources in boost_tracker.items():
        boost_confidence(registry, concept_id, list(sources))
        # Update the confidence_boosted entries with new confidence
        new_concept = registry.get(concept_id)
        if new_concept:
            for entry in confidence_boosted:
                if entry["id"] == concept_id:
                    entry["new_conf"] = new_concept.confidence
                    entry["sources"] = list(new_concept.sources)

    # Second pass: boost dual-source complements from 0.6 to 0.7
    for name_lower, sources in complement_tracker.items():
        unique_sources = set(sources)
        if len(unique_sources) >= 2 and name_lower in complement_ids:
            cid = complement_ids[name_lower]
            concept = registry.get(cid)
            if concept and concept.confidence < 0.7:
                concept.confidence = 0.7
                # Add both sources
                for s in unique_sources:
                    if s not in concept.sources:
                        concept.sources.append(s)

    total = agreements + complements + conflicts

    return ReconciliationResult(
        agreements=agreements,
        complements=complements,
        conflicts=conflicts,
        total=total,
        conflict_log=conflict_log,
        complement_concepts=complement_concepts,
        confidence_boosted=confidence_boosted,
    )
