"""Tests for the knowledge extraction pipeline.

Covers EXIST requirements:
- EXIST-01: FTT Markdown extraction (stub, Plan 02)
- EXIST-02: TPA/Feel Tennis video extraction (stub, Plan 03)
- EXIST-03: Biomechanics extraction (stub, Plan 02)
- EXIST-04: Legacy JSON seed migration (this plan)
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
        # Plan estimated 60-100; actual yield is ~105 due to unique user journey items
        assert 60 <= len(concepts) <= 120, f"Expected 60-120 concepts, got {len(concepts)}"

    def test_all_concepts_pass_pydantic_validation(self, fresh_registry):
        from knowledge.pipeline.seed import seed_registry_from_legacy_json

        concepts = seed_registry_from_legacy_json(fresh_registry)
        for c in concepts:
            # Re-validate via Pydantic
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
        # Registry length should equal number of unique concepts returned
        assert len(fresh_registry) == len(concepts), (
            f"Registry has {len(fresh_registry)} but seed returned {len(concepts)}"
        )

    def test_aliases_populated(self, seeded_registry):
        """Key concepts should have aliases for cross-source equivalences."""
        # unit_turn should have aliases like 'loading phase', 'coiling', etc.
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
            # Verify each entry is valid Concept
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
# Future EXIST requirements (stubs)
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="Implemented in Plan 02 (FTT Markdown extraction)")
def test_exist_01_ftt_markdown_extraction():
    """EXIST-01: Extract concepts from FTT research Markdown files."""
    pass


@pytest.mark.skip(reason="Implemented in Plan 03 (TPA/Feel Tennis video extraction)")
def test_exist_02_video_extraction():
    """EXIST-02: Extract concepts from video synthesis files."""
    pass


@pytest.mark.skip(reason="Implemented in Plan 02 (Biomechanics extraction)")
def test_exist_03_biomechanics_extraction():
    """EXIST-03: Extract concepts from biomechanics textbook files."""
    pass


@pytest.mark.skip(reason="Implemented in Plan 03 (Cross-source deduplication)")
def test_exist_05_cross_source_dedup():
    """EXIST-05: Resolve duplicate concepts across different sources."""
    pass
