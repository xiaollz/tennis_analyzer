"""KnowledgeGraph wrapper around NetworkX MultiDiGraph.

Provides typed methods for adding concepts/edges, causal chain traversal
for the VLM diagnostic engine, and JSON serialization for persistence.
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from knowledge.schemas import Concept, Edge


class KnowledgeGraph:
    """Directed multigraph of tennis forehand knowledge concepts and relationships."""

    def __init__(self) -> None:
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()

    def add_concept(self, concept: Concept) -> None:
        """Add a concept as a node with all its attributes."""
        self.graph.add_node(concept.id, **concept.model_dump())

    def add_edge(self, edge: Edge) -> None:
        """Add a typed directed edge between two concepts.

        Uses relation.value as the edge key, allowing multiple distinct
        relationship types between the same node pair.
        """
        self.graph.add_edge(
            edge.source_id,
            edge.target_id,
            key=edge.relation.value,
            **edge.model_dump(),
        )

    def get_causal_chain(
        self, symptom_id: str, cause_type: str = "causes"
    ) -> list[list[str]]:
        """Find all causal paths from root causes leading to the given symptom.

        Walks backwards through edges with matching relation type.
        Returns list of paths, each path is [symptom, ..., root_cause].
        Cycle-safe via visited-node tracking.
        """
        paths: list[list[str]] = []

        def _walk(node: str, path: list[str]) -> None:
            predecessors = [
                (u, data)
                for u, _, data in self.graph.in_edges(node, data=True)
                if data.get("relation") == cause_type
            ]
            if not predecessors:
                # Reached a root cause — record path
                paths.append(list(path))
                return
            for pred, _ in predecessors:
                if pred not in path:  # Avoid cycles
                    _walk(pred, path + [pred])

        _walk(symptom_id, [symptom_id])
        return paths

    def to_json(self, path: Path) -> None:
        """Serialize the graph to a JSON file using node_link_data format."""
        data = nx.node_link_data(self.graph)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    @classmethod
    def from_json(cls, path: Path) -> KnowledgeGraph:
        """Load a KnowledgeGraph from a JSON file."""
        data = json.loads(path.read_text())
        kg = cls()
        kg.graph = nx.node_link_graph(data, multigraph=True, directed=True)
        return kg

    @property
    def node_count(self) -> int:
        """Number of concept nodes in the graph."""
        return len(self.graph.nodes)

    @property
    def edge_count(self) -> int:
        """Number of edges in the graph."""
        return self.graph.number_of_edges()
