"""Tests for knowledge/graph_assembler.py -- edge loading, resolution, confidence scoring."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from knowledge.graph_assembler import (
    assemble_graph,
    compute_confidence_scores,
    filter_and_deduplicate,
    load_edges_from_extractions,
    resolve_dangling_edges,
    sync_registry_to_graph,
)
from knowledge.graph import KnowledgeGraph
from knowledge.registry import ConceptRegistry
from knowledge.schemas import Concept, ConceptType, Edge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_concept(cid: str, name: str = "", sources: list[str] | None = None,
                  aliases: list[str] | None = None) -> Concept:
    return Concept(
        id=cid,
        name=name or cid.replace("_", " ").title(),
        name_zh=cid,
        aliases=aliases or [],
        category=ConceptType.TECHNIQUE,
        sources=sources or ["ftt"],
        description=f"Test concept {cid}",
    )


def _make_registry(*concepts: Concept) -> ConceptRegistry:
    reg = ConceptRegistry()
    for c in concepts:
        reg.add(c)
    return reg


def _write_extraction(tmpdir: Path, subdir: str, filename: str, edges: list[dict]):
    d = tmpdir / subdir
    d.mkdir(parents=True, exist_ok=True)
    (d / filename).write_text(json.dumps({"edges": edges}))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSyncRegistryToGraph:
    def test_sync_registry_to_graph(self):
        """Loading registry snapshot and syncing adds all concepts as nodes."""
        c1 = _make_concept("hip_rotation")
        c2 = _make_concept("unit_turn")
        c3 = _make_concept("wrist_lag")
        reg = _make_registry(c1, c2, c3)
        kg = KnowledgeGraph()

        sync_registry_to_graph(reg, kg)

        assert kg.node_count == 3
        assert "hip_rotation" in kg.graph.nodes
        assert "unit_turn" in kg.graph.nodes
        assert "wrist_lag" in kg.graph.nodes


class TestLoadEdges:
    def test_load_edges_from_extractions(self):
        """Collecting edges from all extraction JSONs returns them with source_file tagged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            edges1 = [
                {"source_id": "a", "target_id": "b", "relation": "causes",
                 "confidence": 0.8, "evidence": "test"},
                {"source_id": "c", "target_id": "d", "relation": "supports",
                 "confidence": 0.5, "evidence": "co-occur"},
            ]
            edges2 = [
                {"source_id": "e", "target_id": "f", "relation": "causes",
                 "confidence": 0.9, "evidence": "test2"},
            ]
            _write_extraction(tmpdir, "ftt_videos", "v1.json", edges1)
            _write_extraction(tmpdir, "ftt_blog", "b1.json", edges2)
            # underscore-prefixed files should be skipped
            _write_extraction(tmpdir, "", "_registry_snapshot.json", [
                {"source_id": "x", "target_id": "y", "relation": "causes",
                 "confidence": 1.0, "evidence": "skip"},
            ])

            result = load_edges_from_extractions(tmpdir)

            assert len(result) == 3
            assert all("source_file" in e for e in result)


class TestResolveDanglingEdges:
    def test_resolve_dangling_edge(self):
        """An edge with endpoint 'hip_rotation_timing' resolves to 'hip_rotation' via fuzzy match."""
        reg = _make_registry(
            _make_concept("hip_rotation", "Hip Rotation"),
            _make_concept("wrist_lag", "Wrist Lag"),
        )
        edges = [
            {"source_id": "hip_rotation_timing", "target_id": "wrist_lag",
             "relation": "causes", "confidence": 0.8, "evidence": "test"},
        ]

        resolved, unresolvable = resolve_dangling_edges(edges, reg)

        assert len(resolved) == 1
        assert resolved[0]["source_id"] == "hip_rotation"
        assert len(unresolvable) == 0


class TestFilterAndDeduplicate:
    def test_filter_supports_edges(self):
        """Supports edges with confidence < 0.6 or empty evidence are filtered; self-loops removed."""
        edges = [
            # Should be kept: causes edge (always kept)
            {"source_id": "a", "target_id": "b", "relation": "causes",
             "confidence": 0.3, "evidence": "low conf causes"},
            # Should be filtered: supports with low confidence
            {"source_id": "c", "target_id": "d", "relation": "supports",
             "confidence": 0.4, "evidence": "real evidence"},
            # Should be kept: supports with co-occurrence evidence but good confidence
            {"source_id": "e", "target_id": "f", "relation": "supports",
             "confidence": 0.8, "evidence": "Co-occurring in video: something"},
            # Should be kept: supports with good confidence and real evidence
            {"source_id": "g", "target_id": "h", "relation": "supports",
             "confidence": 0.7, "evidence": "Specific evidence here"},
            # Should be filtered: self-loop
            {"source_id": "x", "target_id": "x", "relation": "supports",
             "confidence": 0.9, "evidence": "self-loop"},
            # Should be filtered: empty evidence
            {"source_id": "m", "target_id": "n", "relation": "supports",
             "confidence": 0.8, "evidence": ""},
        ]

        result = filter_and_deduplicate(edges)

        relation_map = {(e["source_id"], e["target_id"]): e["relation"] for e in result}
        assert ("a", "b") in relation_map  # causes kept
        assert ("c", "d") not in relation_map  # low conf supports filtered
        assert ("e", "f") in relation_map  # co-occurrence with good conf kept
        assert ("g", "h") in relation_map  # good supports kept
        assert ("x", "x") not in relation_map  # self-loop removed
        assert ("m", "n") not in relation_map  # empty evidence filtered

    def test_deduplicate_edges(self):
        """Same src+tgt+relation from multiple files merges into one with best confidence."""
        edges = [
            {"source_id": "a", "target_id": "b", "relation": "causes",
             "confidence": 0.5, "evidence": "evidence1", "source_file": "f1"},
            {"source_id": "a", "target_id": "b", "relation": "causes",
             "confidence": 0.9, "evidence": "evidence2", "source_file": "f2"},
        ]

        result = filter_and_deduplicate(edges)

        assert len(result) == 1
        assert result[0]["confidence"] == 0.9
        # Evidence should be merged
        assert "evidence1" in result[0]["evidence"]
        assert "evidence2" in result[0]["evidence"]


class TestConfidenceScoring:
    def test_confidence_scoring(self):
        """FTT-only concept gets 0.8, multi-source gets 1.0, single secondary gets 0.5."""
        c_ftt_only = _make_concept("c_ftt", sources=["ftt"])
        c_multi = _make_concept("c_multi", sources=["ftt", "tpa", "biomechanics_book"])
        c_single = _make_concept("c_single", sources=["tpa"])
        c_ftt_plus = _make_concept("c_ftt_plus", sources=["ftt", "tpa"])
        c_two_nonftt = _make_concept("c_two_nonftt", sources=["tpa", "biomechanics_book"])

        reg = _make_registry(c_ftt_only, c_multi, c_single, c_ftt_plus, c_two_nonftt)
        kg = KnowledgeGraph()
        sync_registry_to_graph(reg, kg)

        compute_confidence_scores(kg, reg)

        assert kg.graph.nodes["c_ftt"]["confidence"] == 0.8
        assert kg.graph.nodes["c_multi"]["confidence"] == 1.0
        assert kg.graph.nodes["c_single"]["confidence"] == 0.5
        assert kg.graph.nodes["c_ftt_plus"]["confidence"] == 0.9
        assert kg.graph.nodes["c_two_nonftt"]["confidence"] == 0.7


class TestAssembleGraphIntegration:
    def test_assemble_graph_integration(self):
        """Full pipeline produces graph with 582 nodes and 500+ edges."""
        extracted_dir = Path("knowledge/extracted")
        registry_path = Path("knowledge/extracted/_registry_snapshot.json")

        # Only run if real data exists
        if not registry_path.exists():
            pytest.skip("No real data available for integration test")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "_graph_snapshot.json"
            stats = assemble_graph(extracted_dir, registry_path, output_path)

            assert stats["node_count"] == 582
            assert stats["edge_count"] >= 200, f"Only {stats['edge_count']} edges"
            # Verify snapshot was written
            assert output_path.exists()
            data = json.loads(output_path.read_text())
            assert len(data["nodes"]) == 582
