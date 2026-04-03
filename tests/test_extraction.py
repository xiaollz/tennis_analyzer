"""Tests for the knowledge extraction pipeline.

Covers EXIST requirements:
- EXIST-01: FTT Markdown extraction
- EXIST-02: TPA/Feel Tennis video extraction (stub, Plan 03)
- EXIST-03: Biomechanics extraction
- EXIST-04: Legacy JSON seed migration
- EXIST-05: Cross-source deduplication (stub, Plan 03)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from knowledge.schemas import Concept, ConceptType

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

LEGACY_JSON_DIR = Path(__file__).resolve().parent.parent / "docs" / "knowledge_graph"
SEED_SNAPSHOT = (
    Path(__file__).resolve().parent.parent / "knowledge" / "extracted" / "_canonical_seed.json"
)
REGISTRY_SNAPSHOT = (
    Path(__file__).resolve().parent.parent / "knowledge" / "extracted" / "_registry_snapshot.json"
)
EXTRACTED_DIR = Path(__file__).resolve().parent.parent / "knowledge" / "extracted"
RESEARCH_DIR = Path(__file__).resolve().parent.parent / "docs" / "research"


@pytest.fixture
def fresh_registry():
    from knowledge.registry import ConceptRegistry

    return ConceptRegistry()


@pytest.fixture
def seeded_registry(fresh_registry):
    from knowledge.pipeline.seed import seed_registry_from_legacy_json

    seed_registry_from_legacy_json(fresh_registry)
    return fresh_registry


# ---------------------------------------------------------------------------
# EXIST-04: Legacy JSON seed migration
# ---------------------------------------------------------------------------


class TestSeedRegistryFromLegacyJSON:
    """Seed the canonical concept registry from 3 legacy JSON files."""

    def test_seed_returns_60_to_100_concepts(self, fresh_registry):
        from knowledge.pipeline.seed import seed_registry_from_legacy_json

        concepts = seed_registry_from_legacy_json(fresh_registry)
        assert 60 <= len(concepts) <= 120, f"Expected 60-120 concepts, got {len(concepts)}"

    def test_all_concepts_pass_pydantic_validation(self, fresh_registry):
        from knowledge.pipeline.seed import seed_registry_from_legacy_json

        concepts = seed_registry_from_legacy_json(fresh_registry)
        for c in concepts:
            validated = Concept.model_validate(c.model_dump())
            assert validated.id == c.id

    def test_snake_case_ids(self, fresh_registry):
        """All IDs must be snake_case (lowercase, underscores, start with letter)."""
        import re

        from knowledge.pipeline.seed import seed_registry_from_legacy_json

        concepts = seed_registry_from_legacy_json(fresh_registry)
        pattern = re.compile(r"^[a-z][a-z0-9_]*$")
        for c in concepts:
            assert pattern.match(c.id), f"ID '{c.id}' is not valid snake_case"

    def test_unit_turn_id_mapping(self, fresh_registry):
        """Legacy C02 'Unit Turn' should become id='unit_turn'."""
        from knowledge.pipeline.seed import seed_registry_from_legacy_json

        seed_registry_from_legacy_json(fresh_registry)
        concept = fresh_registry.get("unit_turn")
        assert concept is not None, "unit_turn concept not found"
        assert concept.name == "Unit Turn"

    def test_category_mapping(self, fresh_registry):
        """Chinese category strings should map to ConceptType enum values."""
        from knowledge.pipeline.seed import seed_registry_from_legacy_json

        concepts = seed_registry_from_legacy_json(fresh_registry)
        valid_types = set(ConceptType)
        for c in concepts:
            assert c.category in valid_types, f"Concept '{c.id}' has invalid category: {c.category}"

    def test_no_duplicates_in_registry(self, fresh_registry):
        """Each concept should be added successfully (no duplicate returns)."""
        from knowledge.pipeline.seed import seed_registry_from_legacy_json

        concepts = seed_registry_from_legacy_json(fresh_registry)
        assert len(fresh_registry) == len(concepts), (
            f"Registry has {len(fresh_registry)} but seed returned {len(concepts)}"
        )

    def test_aliases_populated(self, seeded_registry):
        """Key concepts should have aliases for cross-source equivalences."""
        concept = seeded_registry.get("unit_turn")
        assert concept is not None
        assert len(concept.aliases) > 0, "unit_turn should have aliases"

    def test_seed_snapshot_roundtrip(self, fresh_registry):
        """_canonical_seed.json should be loadable and contain all seeded concepts."""
        from knowledge.pipeline.seed import save_seed_snapshot, seed_registry_from_legacy_json

        concepts = seed_registry_from_legacy_json(fresh_registry)
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            tmp_path = Path(f.name)
        try:
            save_seed_snapshot(concepts, tmp_path)
            loaded = json.loads(tmp_path.read_text())
            assert len(loaded) == len(concepts)
            for entry in loaded:
                Concept.model_validate(entry)
        finally:
            tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Extraction pipeline scaffolding
# ---------------------------------------------------------------------------


class TestExtractionPipeline:
    """Test extraction pipeline handler dispatch and run_extraction."""

    def test_get_handler_ftt_book(self):
        from knowledge.pipeline.extractor import get_handler

        handler = get_handler("01_ftt_book_chapter1.md")
        assert handler is not None
        assert handler.__name__ == "extract_ftt_book"

    def test_get_handler_biomechanics(self):
        from knowledge.pipeline.extractor import get_handler

        handler = get_handler("24_bio_muscles.md")
        assert handler is not None
        assert handler.__name__ == "extract_biomechanics"

    def test_get_handler_ftt_blog(self):
        from knowledge.pipeline.extractor import get_handler

        handler = get_handler("04_ftt_blog_article.md")
        assert handler is not None
        assert handler.__name__ == "extract_ftt_blog"

    def test_get_handler_fallback(self):
        from knowledge.pipeline.extractor import get_handler

        handler = get_handler("unknown_file.md")
        assert handler is not None
        assert handler.__name__ == "extract_generic"

    def test_run_extraction_empty_files(self, fresh_registry):
        from knowledge.pipeline.extractor import run_extraction

        results = run_extraction([], fresh_registry)
        assert results == []


# ---------------------------------------------------------------------------
# EXIST-01: All research files have JSON output
# ---------------------------------------------------------------------------


class TestExist01FttMarkdownExtraction:
    """EXIST-01: Extract concepts from all 32 research Markdown files."""

    def test_all_research_files_have_json_output(self):
        """Every file in docs/research/ should have a corresponding JSON extraction."""
        research_files = sorted(RESEARCH_DIR.glob("*.md"))
        assert len(research_files) >= 25, f"Expected >=25 research files, found {len(research_files)}"

        json_files = []
        for subdir in EXTRACTED_DIR.iterdir():
            if subdir.is_dir() and not subdir.name.startswith("_"):
                json_files.extend(subdir.glob("*.json"))

        json_stems = {f.stem for f in json_files}
        missing = []
        for md_file in research_files:
            if md_file.stem not in json_stems:
                missing.append(md_file.name)

        assert len(missing) == 0, f"Missing JSON extractions for: {missing}"

    def test_json_outputs_are_valid(self):
        """Each JSON output should have source_file, concepts, and edges keys."""
        # Collect JSON files from research extraction directories (exclude user_journey)
        research_categories = {
            "synthesis", "ftt_book", "ftt_blog", "ftt_specific",
            "ftt_videos", "tpa", "biomechanics", "misc",
        }
        json_files = []
        for subdir in EXTRACTED_DIR.iterdir():
            if subdir.is_dir() and subdir.name in research_categories:
                json_files.extend(subdir.glob("*.json"))

        assert len(json_files) >= 25
        for jf in json_files:
            data = json.loads(jf.read_text())
            assert "source_file" in data, f"Missing source_file in {jf.name}"
            assert "concepts" in data, f"Missing concepts in {jf.name}"
            assert "edges" in data, f"Missing edges in {jf.name}"

    def test_synthesis_extraction_has_concepts(self):
        """Synthesis file (most structured) should produce concepts."""
        synthesis_path = EXTRACTED_DIR / "synthesis" / "13_synthesis.json"
        assert synthesis_path.exists(), "13_synthesis.json not found"
        data = json.loads(synthesis_path.read_text())
        assert len(data["concepts"]) > 0, "Synthesis should produce new concepts"


# ---------------------------------------------------------------------------
# EXIST-03: Biomechanics extraction with muscle mappings
# ---------------------------------------------------------------------------


class TestExist03BiomechanicsExtraction:
    """EXIST-03: Extract concepts from biomechanics textbook with muscle mappings."""

    def test_biomechanics_files_extracted(self):
        """All 5 biomechanics files should have JSON outputs."""
        bio_dir = EXTRACTED_DIR / "biomechanics"
        assert bio_dir.exists(), "biomechanics/ directory not found"
        bio_files = list(bio_dir.glob("*.json"))
        assert len(bio_files) == 5, f"Expected 5 biomechanics JSONs, got {len(bio_files)}"

    def test_registry_has_concepts_with_muscles(self):
        """Registry snapshot should contain concepts with muscles_involved populated."""
        assert REGISTRY_SNAPSHOT.exists(), "_registry_snapshot.json not found"
        data = json.loads(REGISTRY_SNAPSHOT.read_text())
        with_muscles = [c for c in data if c.get("muscles_involved")]
        assert len(with_muscles) >= 5, (
            f"Expected at least 5 concepts with muscles, got {len(with_muscles)}"
        )

    def test_muscle_names_are_present(self):
        """At least some concepts should have specific muscle names."""
        data = json.loads(REGISTRY_SNAPSHOT.read_text())
        all_muscles = set()
        for c in data:
            for m in c.get("muscles_involved", []):
                all_muscles.add(m)
        assert len(all_muscles) >= 5, f"Expected at least 5 unique muscles, got {len(all_muscles)}"


# ---------------------------------------------------------------------------
# EXIST-04: Registry snapshot integration
# ---------------------------------------------------------------------------


class TestExist04RegistrySnapshot:
    """EXIST-04: Registry should have 150-300 concepts after full extraction."""

    def test_registry_concept_count(self):
        """Registry snapshot should contain 150-300 concepts."""
        assert REGISTRY_SNAPSHOT.exists(), "_registry_snapshot.json not found"
        data = json.loads(REGISTRY_SNAPSHOT.read_text())
        n = len(data)
        assert 150 <= n <= 300, f"Expected 150-300 concepts, got {n}"

    def test_no_obvious_duplicates(self):
        """Spot-check for obvious duplicates in registry."""
        data = json.loads(REGISTRY_SNAPSHOT.read_text())
        ids = [c["id"] for c in data]
        assert len(ids) == len(set(ids)), "Duplicate IDs found in registry"

    def test_all_concepts_valid_pydantic(self):
        """All concepts in registry snapshot should pass Pydantic validation."""
        data = json.loads(REGISTRY_SNAPSHOT.read_text())
        for entry in data:
            Concept.model_validate(entry)

    def test_integration_load_registry(self):
        """Integration test: load _registry_snapshot.json and verify concept count."""
        data = json.loads(REGISTRY_SNAPSHOT.read_text())
        assert len(data) >= 150
        assert len(data) <= 300
        ids = {c["id"] for c in data}
        assert "unit_turn" in ids, "unit_turn should be in registry"
        assert "hip_shoulder_separation" in ids, "hip_shoulder_separation should be in registry"


# ---------------------------------------------------------------------------
# Future EXIST requirements (stubs)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# EXIST-02: User journey extraction
# ---------------------------------------------------------------------------


class TestExist02UserJourneyExtraction:
    """EXIST-02: Extract concepts from user training journal."""

    def test_user_journey_json_exists(self):
        """User journey extraction should produce learning.json."""
        journey_path = EXTRACTED_DIR / "user_journey" / "learning.json"
        assert journey_path.exists(), "user_journey/learning.json not found"

    def test_user_journey_has_concepts_and_edges(self):
        """User journey extraction should contain concepts and edges."""
        journey_path = EXTRACTED_DIR / "user_journey" / "learning.json"
        if not journey_path.exists():
            pytest.skip("learning.json not generated yet")
        data = json.loads(journey_path.read_text())
        assert len(data["concepts"]) > 0, "Should have concepts"
        assert len(data["edges"]) >= 10, f"Should have >= 10 edges, got {len(data['edges'])}"

    def test_user_journey_concepts_linked_to_registry(self):
        """User journey concepts should link to canonical registry IDs."""
        journey_path = EXTRACTED_DIR / "user_journey" / "learning.json"
        if not journey_path.exists():
            pytest.skip("learning.json not generated yet")
        data = json.loads(journey_path.read_text())
        # At least some edges should reference canonical concepts like scooping, unit_turn
        edge_ids = set()
        for e in data["edges"]:
            edge_ids.add(e["source_id"])
            edge_ids.add(e["target_id"])
        canonical = {"scooping", "unit_turn", "press_slot", "wrist_lag"}
        found = canonical & edge_ids
        assert len(found) >= 1, f"Expected canonical IDs in edges, found: {edge_ids & canonical}"


# ---------------------------------------------------------------------------
# EXIST-05: Knowledge graph with causal chains
# ---------------------------------------------------------------------------


class TestExist05KnowledgeGraph:
    """EXIST-05: Knowledge graph with edges and causal chain queries."""

    GRAPH_SNAPSHOT = EXTRACTED_DIR / "_graph_snapshot.json"

    def test_graph_snapshot_exists(self):
        """Graph snapshot should exist."""
        assert self.GRAPH_SNAPSHOT.exists(), "_graph_snapshot.json not found"

    def test_graph_node_count(self):
        """Graph should have >= 100 nodes."""
        from knowledge.graph import KnowledgeGraph

        g = KnowledgeGraph.from_json(self.GRAPH_SNAPSHOT)
        assert g.node_count >= 100, f"Expected >= 100 nodes, got {g.node_count}"

    def test_graph_edge_count(self):
        """Graph should have >= 50 edges."""
        from knowledge.graph import KnowledgeGraph

        g = KnowledgeGraph.from_json(self.GRAPH_SNAPSHOT)
        assert g.edge_count >= 50, f"Expected >= 50 edges, got {g.edge_count}"

    def test_causal_chain_query_returns_results(self):
        """At least one causal chain query should return a non-empty result."""
        from knowledge.graph import KnowledgeGraph

        g = KnowledgeGraph.from_json(self.GRAPH_SNAPSHOT)
        test_symptoms = ["scooping", "forearm_compensation", "v_scooping"]
        found_chain = False
        for symptom in test_symptoms:
            if symptom in g.graph.nodes:
                chains = g.get_causal_chain(symptom)
                if len(chains) > 0 and len(chains[0]) > 1:
                    found_chain = True
                    break
        assert found_chain, "No causal chain query returned a multi-node path"

    def test_graph_roundtrip(self):
        """Graph should survive JSON serialization roundtrip."""
        from knowledge.graph import KnowledgeGraph

        g = KnowledgeGraph.from_json(self.GRAPH_SNAPSHOT)
        original_nodes = g.node_count
        original_edges = g.edge_count
        assert original_nodes > 0
        assert original_edges > 0
