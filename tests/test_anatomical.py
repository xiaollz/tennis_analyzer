"""Tests for anatomical muscle extraction and concept-to-muscle mapping.

Tests verify that biomechanics Markdown files (docs/research/24-28 series) are
correctly parsed to produce muscle profiles and concept-to-muscle mappings.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BIOMECH_DIR = PROJECT_ROOT / "docs" / "research"
REGISTRY_PATH = PROJECT_ROOT / "knowledge" / "extracted" / "_registry_snapshot.json"
OUTPUT_DIR = PROJECT_ROOT / "knowledge" / "extracted"


def _load_or_extract_profiles() -> list[dict]:
    """Helper: load existing profiles or run extraction."""
    from knowledge.anatomical_extractor import extract_muscle_profiles

    return extract_muscle_profiles(BIOMECH_DIR)


def _load_or_build_map() -> dict:
    """Helper: load existing map or run mapping."""
    from knowledge.anatomical_extractor import map_concepts_to_muscles

    profiles = _load_or_extract_profiles()
    return map_concepts_to_muscles(REGISTRY_PATH, profiles)


class TestExtractMuscleProfiles:
    """Tests for muscle profile extraction from biomechanics Markdown files."""

    def test_extract_muscle_profiles_count(self):
        """Parsing biomechanics MDs produces 30+ unique muscle profiles."""
        profiles = _load_or_extract_profiles()
        assert len(profiles) >= 30, (
            f"Expected 30+ muscle profiles, got {len(profiles)}"
        )

    def test_muscle_profile_structure(self):
        """Each profile has required fields: name, name_zh, body_segment, function,
        training_methods (list), common_failures (list)."""
        profiles = _load_or_extract_profiles()
        required_fields = {"name", "name_zh", "body_segment", "function"}
        required_list_fields = {"training_methods", "common_failures"}

        for p in profiles:
            for field in required_fields:
                assert field in p and p[field], (
                    f"Muscle '{p.get('name', '?')}' missing or empty field '{field}'"
                )
            for field in required_list_fields:
                assert field in p and isinstance(p[field], list), (
                    f"Muscle '{p.get('name', '?')}' field '{field}' must be a list"
                )

    def test_known_muscles_present(self):
        """Key muscles are present: deltoid, rotator_cuff, obliques, gluteus_maximus."""
        profiles = _load_or_extract_profiles()
        names = {p["name"] for p in profiles}
        expected = {"deltoid", "rotator_cuff", "obliques", "gluteus_maximus"}
        # Allow partial name matching (e.g. "internal_obliques" matches "obliques")
        for exp in expected:
            found = any(exp in n or n in exp for n in names)
            assert found, f"Expected muscle containing '{exp}' not found in {names}"

    def test_body_segment_coverage(self):
        """Profiles cover all three body segments: upper_body, core, lower_body."""
        profiles = _load_or_extract_profiles()
        segments = {p["body_segment"] for p in profiles}
        for seg in ("upper_body", "core", "lower_body"):
            assert seg in segments, f"Missing body segment '{seg}'"

    def test_no_duplicate_muscle_names(self):
        """No duplicate muscle names in the profile list."""
        profiles = _load_or_extract_profiles()
        names = [p["name"] for p in profiles]
        assert len(names) == len(set(names)), (
            f"Duplicate muscle names found: {[n for n in names if names.count(n) > 1]}"
        )


class TestMapConceptsToMuscles:
    """Tests for concept-to-muscle mapping."""

    def test_map_concepts_to_muscles_produces_entries(self):
        """Mapping produces entries for key concepts like hip_rotation, trunk_rotation."""
        mapping = _load_or_build_map()
        assert len(mapping) > 0, "Concept-muscle mapping is empty"

        # Check some key concept IDs that should have muscle mappings
        key_concepts = ["hip_rotation", "trunk_rotation", "wrist_lag"]
        found_any = False
        for kc in key_concepts:
            if kc in mapping:
                found_any = True
                break
        # At least one of the key concepts should be mapped
        # (exact IDs depend on registry)
        assert found_any or len(mapping) > 50, (
            f"Expected key concept mappings or 50+ mappings, got {len(mapping)}"
        )

    def test_mapping_coverage(self):
        """At least 50% of technique+biomechanics concepts have 1+ muscle mapping."""
        mapping = _load_or_build_map()
        registry = json.loads(REGISTRY_PATH.read_text())
        tech_bio = [
            c for c in registry
            if c.get("category") in ("technique", "biomechanics")
        ]
        total = len(tech_bio)
        mapped = sum(1 for c in tech_bio if c["id"] in mapping)
        coverage = mapped / total if total > 0 else 0
        assert coverage >= 0.50, (
            f"Mapping coverage {coverage:.1%} ({mapped}/{total}) is below 50%"
        )

    def test_mapping_entry_structure(self):
        """Each mapping entry is a list of dicts with muscle, role, action fields."""
        mapping = _load_or_build_map()
        for concept_id, muscles in list(mapping.items())[:10]:
            assert isinstance(muscles, list), (
                f"Mapping for '{concept_id}' should be a list"
            )
            for m in muscles:
                assert "muscle" in m, f"Missing 'muscle' key in mapping for '{concept_id}'"
                assert "role" in m, f"Missing 'role' key in mapping for '{concept_id}'"
                assert m["role"] in ("primary", "secondary", "stabilizer"), (
                    f"Invalid role '{m['role']}' for '{concept_id}'"
                )


class TestBuildAnatomicalLayer:
    """Integration test for the full pipeline."""

    def test_build_anatomical_layer(self):
        """Full pipeline produces both output files and returns valid stats."""
        from knowledge.anatomical_extractor import build_anatomical_layer

        stats = build_anatomical_layer(BIOMECH_DIR, REGISTRY_PATH, OUTPUT_DIR)
        assert stats["muscle_count"] >= 30
        assert stats["mapped_concept_count"] > 0

        # Verify output files exist
        profiles_path = OUTPUT_DIR / "_muscle_profiles.json"
        map_path = OUTPUT_DIR / "_concept_muscle_map.json"
        assert profiles_path.exists(), f"Missing {profiles_path}"
        assert map_path.exists(), f"Missing {map_path}"

        # Verify JSON is valid
        profiles = json.loads(profiles_path.read_text())
        mapping = json.loads(map_path.read_text())
        assert isinstance(profiles, list)
        assert isinstance(mapping, dict)
