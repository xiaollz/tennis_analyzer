"""Jinja2-based Markdown knowledge base generation (OUT-02, OUT-03).

Exports the knowledge graph as a browsable directory of Markdown files,
organized by ConceptType with cross-references, source citations,
and confidence levels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jinja2

from knowledge.graph import KnowledgeGraph
from knowledge.schemas import ConceptType, DiagnosticChain


# Display names for concept categories
_CATEGORY_DISPLAY = {
    "technique": "Techniques",
    "biomechanics": "Biomechanics",
    "drill": "Drills",
    "symptom": "Symptoms",
    "mental_model": "Mental Models",
    "connection": "Connections",
}


@dataclass
class EdgeView:
    """Flattened edge data for template rendering."""

    source_id: str
    source_name: str
    source_category: str
    target_id: str
    target_name: str
    target_category: str
    relation: str
    confidence: float
    evidence: str
    source_file: str


@dataclass
class ConceptView:
    """Flattened concept data for template rendering."""

    id: str
    name: str
    name_zh: str
    category: str
    sources: list[str]
    description: str
    vlm_features: list[str]
    muscles_involved: list[str]
    confidence: float


def _to_str(val) -> str:
    """Convert a value to string, handling enums."""
    if hasattr(val, "value"):
        return val.value
    return str(val) if val is not None else ""


def _get_node_attr(graph: KnowledgeGraph, node_id: str, attr: str, default=""):
    """Safely get a node attribute, converting enums to strings."""
    val = graph.graph.nodes.get(node_id, {}).get(attr, default)
    if attr == "category":
        return _to_str(val) if val != default else default
    return val


def _build_edge_view(graph: KnowledgeGraph, u: str, v: str, data: dict) -> EdgeView:
    """Build an EdgeView from raw edge data."""
    raw_relation = data.get("relation", "unknown")
    return EdgeView(
        source_id=u,
        source_name=_get_node_attr(graph, u, "name", u),
        source_category=_get_node_attr(graph, u, "category", "unknown"),
        target_id=v,
        target_name=_get_node_attr(graph, v, "name", v),
        target_category=_get_node_attr(graph, v, "category", "unknown"),
        relation=_to_str(raw_relation) if raw_relation != "unknown" else "unknown",
        confidence=data.get("confidence", 0.0),
        evidence=data.get("evidence", ""),
        source_file=data.get("source_file", ""),
    )


def _build_concept_view(node_id: str, attrs: dict) -> ConceptView:
    """Build a ConceptView from node attributes."""
    raw_cat = attrs.get("category", "unknown")
    category = _to_str(raw_cat) if raw_cat != "unknown" else "unknown"
    return ConceptView(
        id=node_id,
        name=attrs.get("name", node_id),
        name_zh=attrs.get("name_zh", ""),
        category=category,
        sources=attrs.get("sources", []),
        description=attrs.get("description", ""),
        vlm_features=attrs.get("vlm_features", []),
        muscles_involved=attrs.get("muscles_involved", []),
        confidence=attrs.get("confidence", 0.0),
    )


def export_markdown_knowledge_base(
    graph: KnowledgeGraph,
    chains: list[DiagnosticChain],
    output_dir: Path,
) -> None:
    """Export the knowledge graph as a Markdown knowledge base.

    Creates a directory structure organized by ConceptType:
      output_dir/
        index.md
        technique/
          index.md
          concept_id.md
        biomechanics/
          ...
        diagnostic_chains/
          chain_id.md

    Args:
        graph: The KnowledgeGraph to export.
        chains: List of DiagnosticChain objects.
        output_dir: Root directory for the Markdown output.
    """
    # Locate templates relative to this package
    template_dir = Path(__file__).resolve().parent.parent / "templates" / "knowledge_base"
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(template_dir)),
        trim_blocks=True,
        lstrip_blocks=True,
    )

    # Group nodes by category
    groups: dict[str, list[ConceptView]] = {}
    for node_id in graph.graph.nodes:
        attrs = dict(graph.graph.nodes[node_id])
        raw_cat = attrs.get("category", "unknown")
        category = _to_str(raw_cat) if raw_cat != "unknown" else "unknown"
        cv = _build_concept_view(node_id, attrs)
        groups.setdefault(category, []).append(cv)

    # Build outgoing/incoming edge maps
    outgoing: dict[str, list[EdgeView]] = {}
    incoming: dict[str, list[EdgeView]] = {}
    for u, v, data in graph.graph.edges(data=True):
        ev = _build_edge_view(graph, u, v, data)
        outgoing.setdefault(u, []).append(ev)
        incoming.setdefault(v, []).append(ev)

    # Render concept pages
    concept_tpl = env.get_template("concept.md.j2")
    for category, concepts in groups.items():
        cat_dir = output_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        for cv in concepts:
            content = concept_tpl.render(
                concept=cv,
                outgoing_edges=outgoing.get(cv.id, []),
                incoming_edges=incoming.get(cv.id, []),
            )
            (cat_dir / f"{cv.id}.md").write_text(content, encoding="utf-8")

        # Render topic group index
        group_tpl = env.get_template("topic_group.md.j2")
        group_content = group_tpl.render(
            display_name=_CATEGORY_DISPLAY.get(category, category.title()),
            count=len(concepts),
            concepts=sorted(concepts, key=lambda c: c.name),
        )
        (cat_dir / "index.md").write_text(group_content, encoding="utf-8")

    # Render diagnostic chain pages
    if chains:
        chain_dir = output_dir / "diagnostic_chains"
        chain_dir.mkdir(parents=True, exist_ok=True)
        chain_tpl = env.get_template("diagnostic_chain.md.j2")
        for chain in chains:
            content = chain_tpl.render(chain=chain)
            (chain_dir / f"{chain.id}.md").write_text(content, encoding="utf-8")

    # Render top-level index
    group_info = []
    for category, concepts in sorted(groups.items()):
        group_info.append({
            "category": category,
            "display_name": _CATEGORY_DISPLAY.get(category, category.title()),
            "count": len(concepts),
        })

    index_tpl = env.get_template("index.md.j2")
    index_content = index_tpl.render(
        node_count=graph.node_count,
        edge_count=graph.edge_count,
        chain_count=len(chains),
        groups=group_info,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "index.md").write_text(index_content, encoding="utf-8")
