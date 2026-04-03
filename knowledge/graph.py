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

    def get_symptom_subgraph(self, symptom_id: str, max_depth: int = 2) -> dict:
        """Extract subgraph relevant to a symptom for VLM prompt injection.

        Returns nodes and edges within max_depth hops of the symptom,
        following causes/visible_as/drills_for edge types.
        """
        diagnostic_types = {"causes", "visible_as", "drills_for"}
        relevant: set[str] = set()
        frontier: set[str] = {symptom_id}

        for _ in range(max_depth):
            next_frontier: set[str] = set()
            for node in frontier:
                if node not in self.graph:
                    continue
                for pred, _, data in self.graph.in_edges(node, data=True):
                    if data.get("relation") in diagnostic_types:
                        next_frontier.add(pred)
                for _, succ, data in self.graph.out_edges(node, data=True):
                    if data.get("relation") in diagnostic_types:
                        next_frontier.add(succ)
            relevant.update(frontier)
            frontier = next_frontier - relevant

        relevant.update(frontier)
        return {
            "nodes": {
                nid: dict(self.graph.nodes[nid])
                for nid in relevant
                if nid in self.graph
            },
            "edges": [
                {"source": u, "target": v, **d}
                for u, v, d in self.graph.edges(data=True)
                if u in relevant
                and v in relevant
                and d.get("relation") in diagnostic_types
            ],
        }

    @property
    def node_count(self) -> int:
        """Number of concept nodes in the graph."""
        return len(self.graph.nodes)

    @property
    def edge_count(self) -> int:
        """Number of edges in the graph."""
        return self.graph.number_of_edges()
