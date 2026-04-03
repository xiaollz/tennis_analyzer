"""Tests for knowledge graph JSON and Markdown export (OUT-01, OUT-02, OUT-03).

Uses a small fixture graph (5 nodes, 3 edges, 1 chain) built from Pydantic models.
Does NOT load the full 526KB graph.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from knowledge.schemas import (
    Concept,
    ConceptType,
    DiagnosticChain,
    DiagnosticStep,
    Edge,
    RelationType,
)
from knowledge.graph import KnowledgeGraph


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_fixture_graph() -> tuple[KnowledgeGraph, list[DiagnosticChain]]:
    """Build a small test graph with 5 nodes, 3 edges, 1 chain."""
    kg = KnowledgeGraph()

    concepts = [
        Concept(
            id="unit_turn",
            name="Unit Turn",
            name_zh="整体转体",
            category=ConceptType.TECHNIQUE,
            sources=["ftt"],
            description="The initial rotation of shoulders and hips together.",
            vlm_features=["shoulder rotation visible in frame 1"],
            muscles_involved=["obliques"],
            confidence=0.95,
        ),
        Concept(
            id="hip_rotation",
            name="Hip Rotation",
            name_zh="髋部旋转",
            category=ConceptType.BIOMECHANICS,
            sources=["ftt", "biomechanics_book"],
            description="Ground-up rotation starting from the hips.",
            vlm_features=["hip angle change between frames"],
            muscles_involved=["gluteus_maximus"],
            confidence=0.9,
        ),
        Concept(
            id="arm_compensation",
            name="Arm-Driven Compensation",
            name_zh="手臂代偿",
            category=ConceptType.SYMPTOM,
            sources=["ftt"],
            description="Using arm muscles instead of body rotation.",
            vlm_features=["elbow angle contracting"],
            confidence=0.85,
        ),
        Concept(
            id="shadow_swing_drill",
            name="Shadow Swing Drill",
            name_zh="挥空拍练习",
            category=ConceptType.DRILL,
            sources=["ftt"],
            description="Practice swings without a ball to focus on form.",
            confidence=0.8,
        ),
        Concept(
            id="relaxed_arm",
            name="Relaxed Arm",
            name_zh="放松手臂",
            category=ConceptType.MENTAL_MODEL,
            sources=["ftt"],
            description="Keep the arm relaxed and let rotation drive the swing.",
            confidence=0.88,
        ),
    ]
    for c in concepts:
        kg.add_concept(c)

    edges = [
        Edge(
            source_id="unit_turn",
            target_id="hip_rotation",
            relation=RelationType.REQUIRES,
            confidence=0.9,
            evidence="FTT Chapter 3: unit turn requires hip rotation",
            source_file="ftt_book_ch3.md",
        ),
        Edge(
            source_id="arm_compensation",
            target_id="unit_turn",
            relation=RelationType.CAUSES,
            confidence=0.85,
            evidence="Missing unit turn causes arm compensation",
            source_file="ftt_blog_post_12.md",
        ),
        Edge(
            source_id="shadow_swing_drill",
            target_id="arm_compensation",
            relation=RelationType.DRILLS_FOR,
            confidence=0.8,
            evidence="Shadow swings help fix arm compensation",
            source_file="ftt_video_01.md",
        ),
    ]
    for e in edges:
        kg.add_edge(e)

    chains = [
        DiagnosticChain(
            id="dc_arm_compensation",
            symptom="Arm-driven hitting with no body rotation",
            symptom_zh="手臂发力、没有身体旋转",
            symptom_concept_id="arm_compensation",
            check_sequence=[
                DiagnosticStep(
                    check="Is elbow angle contracting in frames 3-4?",
                    check_zh="在第3-4帧中肘关节角度是否在减小？",
                    if_true="arm_compensation",
                    if_false=None,
                ),
            ],
            root_causes=["unit_turn"],
            drills=["shadow_swing_drill"],
            priority=1,
            vlm_frame="frame_4",
        ),
    ]

    return kg, chains


@pytest.fixture
def fixture_graph():
    return _build_fixture_graph()


# ---------------------------------------------------------------------------
# OUT-01: JSON Export
# ---------------------------------------------------------------------------


class TestJsonExport:
    """Tests for export_full_graph JSON export."""

    def test_json_export_roundtrip(self, fixture_graph, tmp_path):
        """JSON export produces valid file that can be loaded back with matching counts."""
        from knowledge.output.json_export import export_full_graph

        kg, chains = fixture_graph
        out = tmp_path / "export.json"
        result = export_full_graph(kg, chains, out)

        # File exists and is valid JSON
        assert out.exists()
        loaded = json.loads(out.read_text())

        # Metadata present
        assert "metadata" in loaded
        meta = loaded["metadata"]
        assert meta["version"] == "1.0"
        assert "exported" in meta
        assert meta["node_count"] == 5
        assert meta["edge_count"] == 3
        assert meta["chain_count"] == 1

        # Lists present
        assert len(loaded["nodes"]) == 5
        assert len(loaded["edges"]) == 3
        assert len(loaded["diagnostic_chains"]) == 1

        # Return value matches
        assert result["metadata"]["node_count"] == 5

    def test_json_export_node_fields(self, fixture_graph, tmp_path):
        """Each node has at minimum: id, name, name_zh, category, description, confidence."""
        from knowledge.output.json_export import export_full_graph

        kg, chains = fixture_graph
        out = tmp_path / "export.json"
        result = export_full_graph(kg, chains, out)

        required_fields = {"id", "name", "name_zh", "category", "description", "confidence"}
        for node in result["nodes"]:
            assert required_fields.issubset(set(node.keys())), f"Missing fields in node {node.get('id')}"

    def test_json_export_edge_fields(self, fixture_graph, tmp_path):
        """Each edge has: source, target, relation, confidence, evidence, source_file."""
        from knowledge.output.json_export import export_full_graph

        kg, chains = fixture_graph
        out = tmp_path / "export.json"
        result = export_full_graph(kg, chains, out)

        required_fields = {"source", "target", "relation", "confidence", "evidence", "source_file"}
        for edge in result["edges"]:
            assert required_fields.issubset(set(edge.keys())), f"Missing fields in edge {edge}"

    def test_json_export_chains(self, fixture_graph, tmp_path):
        """diagnostic_chains list matches input count, each has dc_ prefixed id."""
        from knowledge.output.json_export import export_full_graph

        kg, chains = fixture_graph
        out = tmp_path / "export.json"
        result = export_full_graph(kg, chains, out)

        assert len(result["diagnostic_chains"]) == len(chains)
        for chain in result["diagnostic_chains"]:
            assert chain["id"].startswith("dc_")


# ---------------------------------------------------------------------------
# OUT-02/03: Markdown Export
# ---------------------------------------------------------------------------


class TestMarkdownExport:
    """Tests for export_markdown_knowledge_base Markdown generation."""

    def test_markdown_structure(self, fixture_graph, tmp_path):
        """Creates index.md and subdirectories for each ConceptType that has concepts."""
        from knowledge.output.markdown_export import export_markdown_knowledge_base

        kg, chains = fixture_graph
        out = tmp_path / "knowledge"
        export_markdown_knowledge_base(kg, chains, out)

        # index.md exists
        assert (out / "index.md").exists()

        # Category directories exist for our fixture types
        assert (out / "technique").is_dir()
        assert (out / "biomechanics").is_dir()
        assert (out / "symptom").is_dir()
        assert (out / "drill").is_dir()
        assert (out / "mental_model").is_dir()

        # diagnostic_chains directory
        assert (out / "diagnostic_chains").is_dir()

        # Empty categories should NOT have directories
        assert not (out / "connection").exists()

    def test_markdown_concept_page(self, fixture_graph, tmp_path):
        """A concept page includes name (EN+ZH), category, confidence %, description, edge tables."""
        from knowledge.output.markdown_export import export_markdown_knowledge_base

        kg, chains = fixture_graph
        out = tmp_path / "knowledge"
        export_markdown_knowledge_base(kg, chains, out)

        concept_page = (out / "technique" / "unit_turn.md").read_text()

        # Name in both languages
        assert "Unit Turn" in concept_page
        assert "整体转体" in concept_page

        # Category
        assert "technique" in concept_page.lower()

        # Confidence as percentage
        assert "95%" in concept_page

        # Description
        assert "initial rotation" in concept_page

        # Edge tables present
        assert "Hip Rotation" in concept_page  # outgoing: requires hip_rotation
        assert "Arm-Driven Compensation" in concept_page  # incoming: arm_compensation causes unit_turn

    def test_markdown_crossrefs(self, fixture_graph, tmp_path):
        """Cross-references use relative Markdown links like ../category/concept_id.md."""
        from knowledge.output.markdown_export import export_markdown_knowledge_base

        kg, chains = fixture_graph
        out = tmp_path / "knowledge"
        export_markdown_knowledge_base(kg, chains, out)

        concept_page = (out / "technique" / "unit_turn.md").read_text()

        # Outgoing edge to hip_rotation should have relative link
        assert "../biomechanics/hip_rotation.md" in concept_page

        # Incoming edge from arm_compensation
        assert "../symptom/arm_compensation.md" in concept_page

    def test_markdown_diagnostic_chain_page(self, fixture_graph, tmp_path):
        """Chain page includes symptom, check sequence steps, root causes, drills."""
        from knowledge.output.markdown_export import export_markdown_knowledge_base

        kg, chains = fixture_graph
        out = tmp_path / "knowledge"
        export_markdown_knowledge_base(kg, chains, out)

        chain_page = (out / "diagnostic_chains" / "dc_arm_compensation.md").read_text()

        # Symptom
        assert "Arm-driven hitting" in chain_page
        assert "手臂发力" in chain_page

        # Check sequence
        assert "elbow angle contracting" in chain_page

        # Root causes
        assert "unit_turn" in chain_page

        # Drills
        assert "shadow_swing_drill" in chain_page

    def test_markdown_index(self, fixture_graph, tmp_path):
        """Index page includes total stats and links to topic groups."""
        from knowledge.output.markdown_export import export_markdown_knowledge_base

        kg, chains = fixture_graph
        out = tmp_path / "knowledge"
        export_markdown_knowledge_base(kg, chains, out)

        index = (out / "index.md").read_text()

        # Stats
        assert "5" in index  # node count
        assert "3" in index  # edge count
        assert "1" in index  # chain count

        # Links to topic groups
        assert "technique" in index.lower()
        assert "biomechanics" in index.lower()
