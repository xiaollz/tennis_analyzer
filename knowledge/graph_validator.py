"""Graph validation, cycle breaking, orphan analysis, diagnostic chain generation, and visualization.

Validates the assembled KnowledgeGraph for:
- Causal cycle detection and removal
- Orphan (isolated) node identification
- Connected component statistics
- Edge type distribution and degree stats

Also provides:
- Diagnostic chain generation from causal path traversal
- Subgraph visualization for debugging
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

import networkx as nx

from knowledge.graph import KnowledgeGraph
from knowledge.schemas import DiagnosticChain, DiagnosticStep


def validate_graph(kg: KnowledgeGraph) -> dict:
    """Validate the knowledge graph and return a comprehensive report.

    Returns dict with: orphan_count, orphan_sample, cycle_count, cycle_sample,
    component_count, component_sizes, edge_type_distribution, degree_stats.
    """
    g = kg.graph

    # Orphan detection
    orphans = list(nx.isolates(g))
    orphan_sample = orphans[:20]

    # Connected components (weakly connected for directed graph)
    components = list(nx.weakly_connected_components(g))
    component_sizes = sorted([len(c) for c in components], reverse=True)

    # Causal cycle detection
    causal = nx.DiGraph()
    for u, v, data in g.edges(data=True):
        if data.get("relation") == "causes":
            causal.add_edge(u, v)
    cycles = list(nx.simple_cycles(causal))
    cycle_sample = cycles[:10]

    # Edge type distribution
    edge_types = Counter(data.get("relation") for _, _, data in g.edges(data=True))

    # Degree statistics
    degrees = [d for _, d in g.degree()]
    degree_stats = {}
    if degrees:
        degree_stats = {
            "min": min(degrees),
            "max": max(degrees),
            "mean": round(statistics.mean(degrees), 2),
            "median": round(statistics.median(degrees), 2),
        }

    return {
        "node_count": g.number_of_nodes(),
        "edge_count": g.number_of_edges(),
        "orphan_count": len(orphans),
        "orphan_sample": orphan_sample,
        "cycle_count": len(cycles),
        "cycle_sample": cycle_sample,
        "component_count": len(components),
        "component_sizes": component_sizes,
        "edge_type_distribution": dict(edge_types),
        "degree_stats": degree_stats,
    }


def break_cycles(kg: KnowledgeGraph) -> list[tuple]:
    """Remove the lowest-confidence edge in each causal cycle.

    Returns list of (source, target, confidence) tuples for removed edges.
    """
    removed = []

    while True:
        # Build causal subgraph fresh each iteration
        causal = nx.DiGraph()
        edge_confidence: dict[tuple[str, str], float] = {}
        for u, v, data in kg.graph.edges(data=True):
            if data.get("relation") == "causes":
                conf = data.get("confidence", 1.0)
                causal.add_edge(u, v)
                edge_confidence[(u, v)] = conf

        try:
            cycle = nx.find_cycle(causal)
        except nx.NetworkXNoCycle:
            break

        # Find the lowest-confidence edge in this cycle
        cycle_edges = [(u, v) for u, v, _ in cycle] if len(cycle[0]) == 3 else [(u, v) for u, v in cycle]
        min_edge = min(cycle_edges, key=lambda e: edge_confidence.get(e, 1.0))
        min_conf = edge_confidence.get(min_edge, 1.0)

        # Remove from the actual graph (all edges with relation=causes between these nodes)
        u, v = min_edge
        keys_to_remove = [
            k for k, data in kg.graph[u][v].items()
            if data.get("relation") == "causes"
        ]
        for k in keys_to_remove:
            kg.graph.remove_edge(u, v, key=k)

        removed.append((u, v, min_conf))

    return removed


def visualize_subgraph(
    kg: KnowledgeGraph,
    center_id: str,
    depth: int = 2,
    output_path: str | None = None,
) -> None:
    """Visualize a neighborhood subgraph around a center node.

    Colors edges by relation type:
    - causes: red
    - supports: blue
    - drills_for: purple
    - visible_as: orange
    - others: gray
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Extract neighborhood via BFS
    neighborhood = set()
    frontier = {center_id}
    for _ in range(depth):
        next_frontier = set()
        for node in frontier:
            neighborhood.add(node)
            next_frontier.update(kg.graph.successors(node))
            next_frontier.update(kg.graph.predecessors(node))
        frontier = next_frontier - neighborhood
    neighborhood.update(frontier)

    sub = kg.graph.subgraph(neighborhood).copy()

    if sub.number_of_nodes() == 0:
        # Create minimal plot for empty subgraph
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"No neighbors for {center_id}", ha="center", va="center")
        if output_path:
            fig.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return

    # Color map for edge relations
    color_map = {
        "causes": "red",
        "supports": "blue",
        "drills_for": "purple",
        "visible_as": "orange",
    }

    fig, ax = plt.subplots(figsize=(16, 12))
    pos = nx.spring_layout(sub, k=2, seed=42)

    # Draw nodes
    nx.draw_networkx_nodes(sub, pos, ax=ax, node_size=300, node_color="lightblue")

    # Draw labels (truncated)
    labels = {}
    for n in sub.nodes():
        name = sub.nodes[n].get("name", n)
        labels[n] = name[:20] if len(name) > 20 else name
    nx.draw_networkx_labels(sub, pos, labels, ax=ax, font_size=7)

    # Draw edges colored by type
    for u, v, data in sub.edges(data=True):
        rel = data.get("relation", "other")
        color = color_map.get(rel, "gray")
        nx.draw_networkx_edges(
            sub, pos, edgelist=[(u, v)], ax=ax,
            edge_color=color, alpha=0.6, arrows=True,
            connectionstyle="arc3,rad=0.1",
        )

    ax.set_title(f"Subgraph around '{center_id}' (depth={depth})")

    if output_path:
        fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def generate_diagnostic_chains(kg: KnowledgeGraph) -> list[DiagnosticChain]:
    """Generate diagnostic chains by traversing causal paths from symptom nodes.

    For each symptom node:
    1. Walk causal predecessors to find root causes
    2. Look for drills_for edges from root causes
    3. Build check_sequence from intermediate nodes
    4. Create DiagnosticChain with dc_ prefix ID

    Returns list of DiagnosticChain objects.
    """
    chains: list[DiagnosticChain] = []
    symptoms_without_causes: list[str] = []

    # Find all symptom nodes
    symptom_nodes = [
        (nid, data) for nid, data in kg.graph.nodes(data=True)
        if data.get("category") == "symptom"
    ]

    for symptom_id, symptom_data in symptom_nodes:
        # Get causal paths (walks backwards through causes edges)
        causal_paths = kg.get_causal_chain(symptom_id, cause_type="causes")

        if not causal_paths:
            symptoms_without_causes.append(symptom_id)
            continue

        # Collect unique root causes (terminal nodes in paths)
        root_causes = set()
        for path in causal_paths:
            if len(path) > 1:
                root_causes.add(path[-1])

        if not root_causes:
            symptoms_without_causes.append(symptom_id)
            continue

        # Find drills for root causes
        drills = set()
        for rc in root_causes:
            for _, target, data in kg.graph.out_edges(rc, data=True):
                if data.get("relation") == "drills_for":
                    drills.add(target)

        # Also check if any drill nodes have edges TO root causes
        for rc in root_causes:
            for source, _, data in kg.graph.in_edges(rc, data=True):
                if data.get("relation") == "drills_for":
                    drills.add(source)
                node_data = kg.graph.nodes.get(source, {})
                if node_data.get("category") == "drill":
                    drills.add(source)

        # Build check_sequence from the shortest causal path
        shortest = min(causal_paths, key=len) if causal_paths else []
        check_sequence = []
        for i, node_id in enumerate(shortest[1:], 1):  # Skip symptom itself
            node_data = kg.graph.nodes.get(node_id, {})
            check = DiagnosticStep(
                check=f"Check: {node_data.get('name', node_id)} - {node_data.get('description', 'N/A')[:80]}",
                check_zh=f"检查: {node_data.get('name_zh', node_id)}",
                if_true=node_id,
                if_false=shortest[i + 1] if i + 1 < len(shortest) else None,
            )
            check_sequence.append(check)

        # If no check_sequence built, create a basic one
        if not check_sequence and root_causes:
            rc = list(root_causes)[0]
            rc_data = kg.graph.nodes.get(rc, {})
            check_sequence = [DiagnosticStep(
                check=f"Check: {rc_data.get('name', rc)}",
                check_zh=f"检查: {rc_data.get('name_zh', rc)}",
                if_true=rc,
                if_false=None,
            )]

        # Calculate priority based on in-degree of symptom
        in_degree = kg.graph.in_degree(symptom_id)
        priority = max(1, min(5, 5 - in_degree))  # Higher in-degree = lower priority number

        # Create chain ID
        chain_id = f"dc_{symptom_id}"
        # Ensure ID matches pattern
        if not chain_id.replace("dc_", "", 1).replace("_", "").replace("-", "").isalnum():
            chain_id = f"dc_gen_{symptom_id.replace('-', '_')}"

        # Ensure the chain_id matches the regex pattern ^dc_[a-z][a-z0-9_]*$
        import re
        clean_id = re.sub(r"[^a-z0-9_]", "_", chain_id.lower())
        if not re.match(r"^dc_[a-z][a-z0-9_]*$", clean_id):
            clean_id = f"dc_gen_{re.sub(r'[^a-z0-9_]', '_', symptom_id.lower())}"

        symptom_name = symptom_data.get("name", symptom_id)
        symptom_zh = symptom_data.get("name_zh", symptom_id)

        chain = DiagnosticChain(
            id=clean_id,
            symptom=symptom_name,
            symptom_zh=symptom_zh,
            symptom_concept_id=symptom_id,
            check_sequence=check_sequence,
            root_causes=list(root_causes)[:10],  # Cap at 10
            drills=list(drills)[:5],
            priority=priority,
            vlm_frame=None,
        )
        chains.append(chain)

    return chains


def save_validation_report(report: dict, path: Path) -> None:
    """Save validation report to JSON."""
    # Make serializable
    serializable = {
        k: v for k, v in report.items()
    }
    path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2))
