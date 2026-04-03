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


# === Plan 04-04: Graph enrichment with muscles + VLM features ===

GRAPH_SNAPSHOT_PATH = OUTPUT_DIR / "_graph_snapshot.json"
MUSCLE_MAP_PATH = OUTPUT_DIR / "_concept_muscle_map.json"
MUSCLE_PROFILES_PATH = OUTPUT_DIR / "_muscle_profiles.json"


def _load_enriched_graph():
    """Helper: load graph and run enrichment pipeline."""
    from knowledge.graph import KnowledgeGraph
    from knowledge.graph_assembler import (
        enrich_with_muscles,
        annotate_vlm_features,
        add_visible_as_edges,
    )

    kg = KnowledgeGraph.from_json(GRAPH_SNAPSHOT_PATH)
    enrich_with_muscles(kg, MUSCLE_MAP_PATH)
    annotate_vlm_features(kg)
    add_visible_as_edges(kg)
    return kg


class TestEnrichGraphWithMuscles:
    """Tests for muscle data integration into graph nodes."""

    def test_enrich_graph_with_muscles(self):
        """After enrichment, 50%+ of technique nodes have non-empty muscles_involved."""
        kg = _load_enriched_graph()
        technique_nodes = [
            (nid, data)
            for nid, data in kg.graph.nodes(data=True)
            if data.get("category") == "technique"
        ]
        total = len(technique_nodes)
        with_muscles = sum(
            1 for _, data in technique_nodes if data.get("muscles_involved")
        )
        coverage = with_muscles / total if total > 0 else 0
        assert coverage >= 0.50, (
            f"Technique nodes with muscles: {with_muscles}/{total} = {coverage:.1%}, need 50%+"
        )

    def test_muscles_are_english_names(self):
        """Muscle names in enriched nodes should be English (from concept_muscle_map)."""
        kg = _load_enriched_graph()
        for nid, data in kg.graph.nodes(data=True):
            muscles = data.get("muscles_involved", [])
            for m in muscles:
                # English muscle names use only ASCII
                assert m.isascii(), (
                    f"Non-ASCII muscle name '{m}' in node '{nid}'"
                )


class TestVlmFeatureAnnotation:
    """Tests for VLM feature population on graph nodes."""

    def test_symptom_vlm_features(self):
        """Symptom nodes have vlm_features populated (all symptom nodes should have 1+ feature)."""
        kg = _load_enriched_graph()
        symptom_nodes = [
            (nid, data)
            for nid, data in kg.graph.nodes(data=True)
            if data.get("category") == "symptom"
        ]
        total = len(symptom_nodes)
        assert total > 0, "No symptom nodes found"
        with_vlm = sum(
            1 for _, data in symptom_nodes if data.get("vlm_features")
        )
        coverage = with_vlm / total if total > 0 else 0
        assert coverage >= 0.75, (
            f"Symptom nodes with VLM features: {with_vlm}/{total} = {coverage:.1%}, need 75%+"
        )

    def test_technique_vlm_features(self):
        """Technique nodes get observable body position features based on keywords."""
        kg = _load_enriched_graph()
        technique_nodes = [
            (nid, data)
            for nid, data in kg.graph.nodes(data=True)
            if data.get("category") == "technique"
        ]
        with_vlm = sum(
            1 for _, data in technique_nodes if data.get("vlm_features")
        )
        # At least some technique nodes should have VLM features
        assert with_vlm >= 10, (
            f"Only {with_vlm} technique nodes have VLM features, need at least 10"
        )


class TestVisibleAsEdges:
    """Tests for visible_as edges connecting techniques to symptoms."""

    def test_visible_as_edges_exist(self):
        """New visible_as edges connect technique concepts to their visual manifestation."""
        kg = _load_enriched_graph()
        visible_as_edges = [
            (u, v, d)
            for u, v, k, d in kg.graph.edges(data=True, keys=True)
            if k == "visible_as" or d.get("relation") == "visible_as"
        ]
        assert len(visible_as_edges) >= 5, (
            f"Only {len(visible_as_edges)} visible_as edges found, need at least 5"
        )


class TestWhyChainTraversal:
    """Tests for 'why' explanation chain building."""

    def test_why_chain_traversal(self):
        """build_why_explanation returns valid chains for at least 10 concepts."""
        from knowledge.graph import KnowledgeGraph
        from knowledge.graph_assembler import (
            enrich_with_muscles,
            annotate_vlm_features,
            add_visible_as_edges,
            build_why_explanation,
        )

        kg = KnowledgeGraph.from_json(GRAPH_SNAPSHOT_PATH)
        enrich_with_muscles(kg, MUSCLE_MAP_PATH)
        annotate_vlm_features(kg)
        add_visible_as_edges(kg)

        valid_chains = 0
        for nid, data in kg.graph.nodes(data=True):
            if data.get("category") in ("technique", "biomechanics"):
                chain = build_why_explanation(kg, nid, MUSCLE_PROFILES_PATH)
                if chain and chain.get("muscles") and chain.get("visible_symptoms"):
                    valid_chains += 1
        assert valid_chains >= 10, (
            f"Only {valid_chains} valid why-chains, need at least 10"
        )

    def test_why_chain_structure(self):
        """Why chain has required keys: concept, muscles, physics, visible_symptoms."""
        from knowledge.graph import KnowledgeGraph
        from knowledge.graph_assembler import (
            enrich_with_muscles,
            annotate_vlm_features,
            add_visible_as_edges,
            build_why_explanation,
        )

        kg = KnowledgeGraph.from_json(GRAPH_SNAPSHOT_PATH)
        enrich_with_muscles(kg, MUSCLE_MAP_PATH)
        annotate_vlm_features(kg)
        add_visible_as_edges(kg)

        # Find a technique node with muscles
        for nid, data in kg.graph.nodes(data=True):
            if data.get("muscles_involved"):
                chain = build_why_explanation(kg, nid, MUSCLE_PROFILES_PATH)
                if chain:
                    assert "concept" in chain
                    assert "muscles" in chain
                    assert "physics" in chain
                    assert "visible_symptoms" in chain
                    assert isinstance(chain["muscles"], list)
                    assert isinstance(chain["visible_symptoms"], list)
                    return
        pytest.fail("No node with muscles found for why-chain test")


class TestRegistrySnapshotUpdated:
    """Tests for registry snapshot update with enrichment data."""

    def test_registry_snapshot_updated(self):
        """After update, registry snapshot has nodes with muscles_involved and vlm_features."""
        from knowledge.graph import KnowledgeGraph
        from knowledge.graph_assembler import (
            enrich_with_muscles,
            annotate_vlm_features,
            add_visible_as_edges,
            update_registry_snapshot,
        )

        kg = KnowledgeGraph.from_json(GRAPH_SNAPSHOT_PATH)
        enrich_with_muscles(kg, MUSCLE_MAP_PATH)
        annotate_vlm_features(kg)
        add_visible_as_edges(kg)

        # Write to a temp path to avoid corrupting actual data
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            import shutil
            shutil.copy2(REGISTRY_PATH, f.name)
            tmp_path = Path(f.name)

        try:
            update_registry_snapshot(kg, tmp_path)
            updated = json.loads(tmp_path.read_text())
            with_muscles = sum(1 for c in updated if c.get("muscles_involved"))
            with_vlm = sum(1 for c in updated if c.get("vlm_features"))
            assert with_muscles > 50, (
                f"Only {with_muscles} concepts with muscles in registry"
            )
            assert with_vlm > 20, (
                f"Only {with_vlm} concepts with VLM features in registry"
            )
        finally:
            tmp_path.unlink(missing_ok=True)
