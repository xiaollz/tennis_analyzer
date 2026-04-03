"""Full knowledge graph JSON export (OUT-01).

Exports all nodes, edges, and diagnostic chains into a single JSON file
with metadata for version tracking and integrity checking.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from knowledge.graph import KnowledgeGraph
from knowledge.schemas import DiagnosticChain


def export_full_graph(
    graph: KnowledgeGraph,
    chains: list[DiagnosticChain],
    output_path: Path,
) -> dict:
    """Export the complete knowledge graph as a self-contained JSON file.

    Args:
        graph: The KnowledgeGraph to export.
        chains: List of DiagnosticChain objects.
        output_path: Where to write the JSON file.

    Returns:
        The exported dict (same structure as what's written to disk).
    """
    nodes = []
    for node_id in graph.graph.nodes:
        node_data = dict(graph.graph.nodes[node_id])
        # Ensure 'id' is present (it's stored as node key, may also be in attrs)
        node_data["id"] = node_id
        nodes.append(node_data)

    edges = []
    for u, v, data in graph.graph.edges(data=True):
        edge_dict = dict(data)
        edge_dict["source"] = u
        edge_dict["target"] = v
        edges.append(edge_dict)

    data = {
        "metadata": {
            "version": "1.0",
            "exported": datetime.now(timezone.utc).isoformat(),
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
            "chain_count": len(chains),
        },
        "nodes": nodes,
        "edges": edges,
        "diagnostic_chains": [c.model_dump() for c in chains],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    return data
