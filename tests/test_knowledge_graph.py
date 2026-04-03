"""Tests for KnowledgeGraph wrapper with causal chain queries and JSON roundtrip."""

import pytest

from knowledge.schemas import Concept, ConceptType, Edge, RelationType
from knowledge.graph import KnowledgeGraph


def _concept(id: str, name: str, name_zh: str = "测试") -> Concept:
    return Concept(
        id=id,
        name=name,
        name_zh=name_zh,
        category=ConceptType.TECHNIQUE,
        description=f"Test: {name}",
    )


def _edge(
    source_id: str,
    target_id: str,
    relation: RelationType = RelationType.CAUSES,
) -> Edge:
    return Edge(
        source_id=source_id,
        target_id=target_id,
        relation=relation,
        evidence="test evidence",
        source_file="test.md",
    )


@pytest.fixture
def kg() -> KnowledgeGraph:
    return KnowledgeGraph()


@pytest.fixture
def three_concepts() -> list[Concept]:
    return [
        _concept("active_pat_dog", "Active Pat the Dog", "主动拍狗"),
        _concept("racket_head_drop", "Racket Head Drop", "拍头下垂"),
        _concept("scooping", "Scooping", "兜球"),
    ]


class TestGraphAddConcept:
    def test_graph_add_concept(self, kg: KnowledgeGraph):
        """Add concept node, verify node exists with attributes."""
        c = _concept("hip_rotation", "Hip Rotation", "转髋")
        kg.add_concept(c)
        assert "hip_rotation" in kg.graph.nodes
        assert kg.graph.nodes["hip_rotation"]["name"] == "Hip Rotation"
        assert kg.graph.nodes["hip_rotation"]["name_zh"] == "转髋"

    def test_graph_node_count(self, kg: KnowledgeGraph, three_concepts: list[Concept]):
        """After adding 3 concepts, node_count == 3."""
        for c in three_concepts:
            kg.add_concept(c)
        assert kg.node_count == 3


class TestGraphAddEdge:
    def test_graph_add_edge(self, kg: KnowledgeGraph, three_concepts: list[Concept]):
        """Add edge between two concepts, verify edge exists with relation type."""
        for c in three_concepts:
            kg.add_concept(c)
        edge = _edge("active_pat_dog", "racket_head_drop", RelationType.CAUSES)
        kg.add_edge(edge)
        edges = list(kg.graph.edges("active_pat_dog", data=True, keys=True))
        assert len(edges) == 1
        assert edges[0][1] == "racket_head_drop"
        assert edges[0][3]["relation"] == "causes"

    def test_graph_multi_edge(self, kg: KnowledgeGraph, three_concepts: list[Concept]):
        """Add two edges (CAUSES + REQUIRES) between same pair, both exist."""
        for c in three_concepts:
            kg.add_concept(c)
        kg.add_edge(_edge("active_pat_dog", "racket_head_drop", RelationType.CAUSES))
        kg.add_edge(_edge("active_pat_dog", "racket_head_drop", RelationType.REQUIRES))
        edges = list(kg.graph.edges("active_pat_dog", data=True, keys=True))
        assert len(edges) == 2
        relations = {e[2] for e in edges}  # key is the relation value
        assert relations == {"causes", "requires"}

    def test_graph_edge_count(self, kg: KnowledgeGraph, three_concepts: list[Concept]):
        """After adding 2 edges, edge_count == 2."""
        for c in three_concepts:
            kg.add_concept(c)
        kg.add_edge(_edge("active_pat_dog", "racket_head_drop"))
        kg.add_edge(_edge("racket_head_drop", "scooping"))
        assert kg.edge_count == 2


class TestCausalChain:
    def test_causal_chain(self, kg: KnowledgeGraph, three_concepts: list[Concept]):
        """A->causes->B->causes->C, get_causal_chain('scooping') returns path with 'active_pat_dog'."""
        for c in three_concepts:
            kg.add_concept(c)
        # active_pat_dog -> causes -> racket_head_drop -> causes -> scooping
        kg.add_edge(_edge("active_pat_dog", "racket_head_drop", RelationType.CAUSES))
        kg.add_edge(_edge("racket_head_drop", "scooping", RelationType.CAUSES))
        paths = kg.get_causal_chain("scooping")
        assert len(paths) >= 1
        # The path from scooping backwards should contain active_pat_dog
        all_nodes = set()
        for path in paths:
            all_nodes.update(path)
        assert "active_pat_dog" in all_nodes
        assert "racket_head_drop" in all_nodes
        assert "scooping" in all_nodes

    def test_causal_chain_no_cycle(self, kg: KnowledgeGraph):
        """Graph with potential cycle terminates without infinite loop."""
        concepts = [
            _concept("a", "Concept A"),
            _concept("b", "Concept B"),
            _concept("c", "Concept C"),
        ]
        for c in concepts:
            kg.add_concept(c)
        # A -> B -> C, and C -> A would be a cycle if not guarded
        kg.add_edge(_edge("a", "b", RelationType.CAUSES))
        kg.add_edge(_edge("b", "c", RelationType.CAUSES))
        kg.add_edge(_edge("c", "a", RelationType.CAUSES))
        # Should terminate without error
        paths = kg.get_causal_chain("c")
        assert isinstance(paths, list)
        # Path should not revisit nodes
        for path in paths:
            assert len(path) == len(set(path)), "Path contains cycle"


class TestGraphJsonRoundtrip:
    def test_graph_json_roundtrip(self, kg: KnowledgeGraph, three_concepts: list[Concept], tmp_path):
        """Serialize to JSON, load back, same nodes/edges/attributes."""
        for c in three_concepts:
            kg.add_concept(c)
        kg.add_edge(_edge("active_pat_dog", "racket_head_drop", RelationType.CAUSES))
        kg.add_edge(_edge("racket_head_drop", "scooping", RelationType.CAUSES))

        json_path = tmp_path / "test_graph.json"
        kg.to_json(json_path)

        loaded = KnowledgeGraph.from_json(json_path)
        assert loaded.node_count == kg.node_count
        assert loaded.edge_count == kg.edge_count
        # Verify node attributes survived
        assert loaded.graph.nodes["scooping"]["name"] == "Scooping"
        assert loaded.graph.nodes["scooping"]["name_zh"] == "兜球"
