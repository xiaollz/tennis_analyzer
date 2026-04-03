"""Tests for VLMPromptCompiler — VLM prompt generation from knowledge graph subgraphs.

Covers: VLM-01, VLM-03, VLM-04, VLM-05, OUT-04.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from knowledge.graph import KnowledgeGraph
from knowledge.schemas import (
    Concept,
    ConceptType,
    DiagnosticChain,
    DiagnosticStep,
    Edge,
    RelationType,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_fixture_graph() -> KnowledgeGraph:
    """Build a small graph with symptom -> cause -> drill edges."""
    kg = KnowledgeGraph()

    # Symptom nodes
    kg.add_concept(Concept(
        id="forearm_compensation",
        name="Forearm Compensation",
        name_zh="小臂代偿",
        category=ConceptType.SYMPTOM,
        description="Arm drives swing instead of body rotation",
        vlm_features=["elbow angle contracting in frames 4-5"],
        confidence=0.95,
    ))
    kg.add_concept(Concept(
        id="racket_drop",
        name="Racket Drop",
        name_zh="拍头过度下坠",
        category=ConceptType.SYMPTOM,
        description="Racket head drops excessively before forward swing",
        vlm_features=["racket below hand level in frame 3"],
        confidence=0.90,
    ))

    # Technique / cause nodes
    kg.add_concept(Concept(
        id="unit_turn",
        name="Unit Turn",
        name_zh="整体转身",
        category=ConceptType.TECHNIQUE,
        description="Full body rotation as preparation",
        confidence=0.95,
    ))
    kg.add_concept(Concept(
        id="wrist_lag",
        name="Wrist Lag",
        name_zh="手腕滞后",
        category=ConceptType.BIOMECHANICS,
        description="Natural lag of racket behind hand",
        confidence=0.90,
    ))

    # Drill node
    kg.add_concept(Concept(
        id="drop_feed_rotation_drill",
        name="Drop Feed Rotation Drill",
        name_zh="喂球旋转练习",
        category=ConceptType.DRILL,
        description="Practice body rotation with drop feeds",
        confidence=1.0,
    ))

    # Edges
    kg.add_edge(Edge(
        source_id="unit_turn",
        target_id="forearm_compensation",
        relation=RelationType.CAUSES,
        confidence=0.85,
        evidence="Missing unit turn causes arm-driven hitting",
        source_file="ftt_book",
    ))
    kg.add_edge(Edge(
        source_id="forearm_compensation",
        target_id="racket_drop",
        relation=RelationType.VISIBLE_AS,
        confidence=0.80,
        evidence="Forearm compensation often manifests as racket drop",
        source_file="ftt_video_01",
    ))
    kg.add_edge(Edge(
        source_id="drop_feed_rotation_drill",
        target_id="forearm_compensation",
        relation=RelationType.DRILLS_FOR,
        confidence=0.90,
        evidence="Drop feed drill addresses arm compensation",
        source_file="ftt_video_02",
    ))

    return kg


def _make_fixture_chains() -> list[DiagnosticChain]:
    """Build 2 fixture diagnostic chains."""
    return [
        DiagnosticChain(
            id="dc_arm_driven_hitting",
            symptom="Arm initiating swing instead of body rotation",
            symptom_zh="手臂主导挥拍而非身体旋转",
            symptom_concept_id="forearm_compensation",
            check_sequence=[
                DiagnosticStep(
                    check="Is the hip rotating before the shoulder?",
                    check_zh="髋部是否在肩部之前旋转？",
                    if_true="kinetic_chain_sequence",
                    if_false=None,
                ),
                DiagnosticStep(
                    check="Is the arm starting the swing independently?",
                    check_zh="手臂是否独立启动挥拍？",
                    if_true="forearm_compensation",
                    if_false=None,
                ),
            ],
            root_causes=["forearm_compensation", "unit_turn"],
            drills=["drop_feed_rotation_drill"],
            priority=1,
            vlm_frame=None,
        ),
        DiagnosticChain(
            id="dc_scooping",
            symptom="Racket dropping too low and scooping up through contact",
            symptom_zh="球拍下沉过低，从下向上舀球",
            symptom_concept_id="racket_drop",
            check_sequence=[
                DiagnosticStep(
                    check="Does the racket drop below the hand before forward swing?",
                    check_zh="球拍在前挥前是否低于手部？",
                    if_true="racket_drop",
                    if_false=None,
                ),
            ],
            root_causes=["racket_drop", "wrist_lag"],
            drills=["contact_point_drill"],
            priority=2,
            vlm_frame=None,
        ),
    ]


@pytest.fixture
def graph():
    return _make_fixture_graph()


@pytest.fixture
def chains():
    return _make_fixture_chains()


@pytest.fixture
def compiler(graph, chains):
    from knowledge.output.vlm_prompt import VLMPromptCompiler
    return VLMPromptCompiler(graph, chains)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSubgraphExtraction:
    """Test KnowledgeGraph.get_symptom_subgraph."""

    def test_subgraph_returns_connected_nodes(self, graph: KnowledgeGraph):
        result = graph.get_symptom_subgraph("forearm_compensation", max_depth=2)
        assert "nodes" in result
        assert "edges" in result
        # Should contain the symptom itself + connected nodes
        assert "forearm_compensation" in result["nodes"]
        # unit_turn causes forearm_compensation (1 hop via causes)
        assert "unit_turn" in result["nodes"]
        # drop_feed_rotation_drill drills_for forearm_compensation (1 hop)
        assert "drop_feed_rotation_drill" in result["nodes"]
        # racket_drop via visible_as (1 hop)
        assert "racket_drop" in result["nodes"]

    def test_subgraph_edges_filtered(self, graph: KnowledgeGraph):
        result = graph.get_symptom_subgraph("forearm_compensation", max_depth=1)
        # Should have edges between relevant nodes
        assert len(result["edges"]) >= 1
        relations = {e["relation"] for e in result["edges"]}
        # All edges should be diagnostic types
        assert relations <= {"causes", "visible_as", "drills_for"}

    def test_subgraph_missing_node(self, graph: KnowledgeGraph):
        result = graph.get_symptom_subgraph("nonexistent_node")
        assert result["nodes"] == {}
        assert result["edges"] == []


class TestVLMPromptCompiler:
    """Test VLMPromptCompiler methods."""

    def test_compile_pass1(self, compiler):
        prompt = compiler.compile_pass1_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) < 3000, f"Pass 1 prompt too long: {len(prompt)} chars"
        # Should contain numbered symptom categories
        assert "1." in prompt
        assert "2." in prompt
        # Should contain Chinese symptom descriptions
        assert "手臂主导挥拍" in prompt
        assert "球拍下沉过低" in prompt
        # Should contain instruction to reply with numbers
        assert "数字" in prompt or "编号" in prompt

    def test_compile_pass2(self, compiler):
        prompt = compiler.compile_pass2_prompt(["dc_arm_driven_hitting"])
        assert isinstance(prompt, str)
        # Should contain chain symptom description
        assert "手臂主导挥拍" in prompt
        # Should contain check sequence content
        assert "髋部是否在肩部之前旋转" in prompt
        # Should contain root causes
        assert "forearm_compensation" in prompt
        # Should contain drill
        assert "drop_feed_rotation_drill" in prompt

    def test_prompt_budget_single_chain(self, compiler):
        prompt = compiler.compile_pass2_prompt(["dc_arm_driven_hitting"])
        # Extract dynamic section (after static prompt)
        static = compiler.compile_system_prompt()
        dynamic = prompt[len(static):]
        assert len(dynamic) < 10000, f"Dynamic section too long: {len(dynamic)} chars"

    def test_prompt_budget_multiple_chains(self, compiler):
        prompt = compiler.compile_pass2_prompt(["dc_arm_driven_hitting", "dc_scooping"])
        static = compiler.compile_system_prompt()
        dynamic = prompt[len(static):]
        assert len(dynamic) < 10000, f"Dynamic section too long: {len(dynamic)} chars"

    def test_diagnostic_coverage(self, compiler):
        """All chains are included in pass1 checklist."""
        prompt = compiler.compile_pass1_prompt()
        for chain in compiler.chains:
            assert chain.symptom_zh in prompt, f"Missing chain: {chain.symptom_zh}"

    def test_output_schema_preserved(self, compiler):
        """Pass 2 prompt contains the JSON output format specification."""
        prompt = compiler.compile_pass2_prompt(["dc_arm_driven_hitting"])
        assert "issues" in prompt
        assert "strengths" in prompt
        assert "overall_assessment" in prompt
        assert "score" in prompt
        assert "drills" in prompt

    def test_static_prompt_content(self, compiler):
        """Static system prompt contains key coaching content."""
        prompt = compiler.compile_system_prompt()
        assert "逐帧分析指南" in prompt
        assert "核心原则" in prompt
        assert "Drill 知识库" in prompt
        assert "输出格式" in prompt

    def test_prompt_template_replaces_hardcoded(self, compiler):
        """VLMPromptCompiler can produce a complete prompt covering all sections."""
        prompt = compiler.compile_pass2_prompt(["dc_arm_driven_hitting"])
        # Frame guide
        assert "图1 准备完成" in prompt
        assert "图6 随挥结束" in prompt
        # Principles
        assert "正手是旋转驱动的鞭打系统" in prompt
        # Drills
        assert "Drill 知识库" in prompt
        # Diagnostic chains (dynamic)
        assert "诊断路径" in prompt or "根因" in prompt
        # Output format
        assert "严格JSON" in prompt


class TestWithRealChains:
    """Test with real diagnostic chains if available."""

    @pytest.fixture
    def real_chains(self):
        chains_path = Path("knowledge/extracted/ftt_video_diagnostic_chains.json")
        if not chains_path.exists():
            pytest.skip("Real chains file not available")
        data = json.loads(chains_path.read_text())
        return [DiagnosticChain(**c) for c in data["chains"]]

    @pytest.fixture
    def real_compiler(self, graph, real_chains):
        from knowledge.output.vlm_prompt import VLMPromptCompiler
        return VLMPromptCompiler(graph, real_chains)

    def test_all_18_chains_in_pass1(self, real_compiler, real_chains):
        prompt = real_compiler.compile_pass1_prompt()
        for chain in real_chains:
            assert chain.symptom_zh in prompt, f"Missing: {chain.symptom_zh}"
        assert len(real_chains) == 18

    def test_pass1_budget_with_real_chains(self, real_compiler):
        prompt = real_compiler.compile_pass1_prompt()
        assert len(prompt) < 3000, f"Pass 1 with 18 chains: {len(prompt)} chars"

    def test_pass2_handles_every_chain(self, real_compiler, real_chains):
        for chain in real_chains:
            prompt = real_compiler.compile_pass2_prompt([chain.id])
            assert chain.symptom_zh in prompt
