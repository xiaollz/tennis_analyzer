"""Training plan generator — prioritized drill recommendations from UserProfile + KnowledgeGraph.

Walks the knowledge graph to find drills targeting the user's current struggle
concepts, scores them by coverage, and outputs a ranked training plan.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from pathlib import Path

from pydantic import BaseModel, Field

from knowledge.graph import KnowledgeGraph
from knowledge.schemas import DiagnosticChain, RelationType
from knowledge.user_profile import ConceptStatus, UserProfile


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class DrillRecommendation(BaseModel):
    """A single drill recommendation with scoring rationale."""

    drill_id: str
    drill_name: str
    drill_name_zh: str
    addresses: list[str] = Field(description="Concept IDs this drill targets")
    priority: int = Field(description="Priority score (higher = more important)")
    rationale: str = Field(description="Why this drill is recommended")


class TrainingPlan(BaseModel):
    """A generated training plan with prioritized drills."""

    generated_date: str
    focus_areas: list[str] = Field(description="Concept names for current focus")
    drills: list[DrillRecommendation] = Field(default_factory=list)
    summary: str = Field(default="")
    summary_zh: str = Field(default="")

    def to_markdown(self) -> str:
        """Render as readable Markdown with Chinese headers."""
        lines = [
            f"# 个人训练计划",
            f"",
            f"生成日期: {self.generated_date}",
            f"",
        ]

        if not self.drills:
            lines.append(f"## 总结")
            lines.append(f"")
            lines.append(self.summary_zh or self.summary)
            return "\n".join(lines)

        lines.append(f"## 重点关注")
        lines.append(f"")
        for area in self.focus_areas:
            lines.append(f"- {area}")
        lines.append(f"")

        lines.append(f"## 推荐练习")
        lines.append(f"")
        for i, drill in enumerate(self.drills, 1):
            lines.append(f"### {i}. {drill.drill_name_zh}（{drill.drill_name}）")
            lines.append(f"")
            lines.append(f"- **优先级**: {drill.priority}")
            lines.append(f"- **针对问题**: {', '.join(drill.addresses)}")
            lines.append(f"- **理由**: {drill.rationale}")
            lines.append(f"")

        lines.append(f"## 总结")
        lines.append(f"")
        lines.append(self.summary_zh or self.summary)

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def generate_training_plan(
    profile: UserProfile,
    graph: KnowledgeGraph,
    chains: list[DiagnosticChain],
) -> TrainingPlan:
    """Generate a training plan with prioritized drill recommendations.

    Parameters
    ----------
    profile : UserProfile
        User's training profile with concept statuses.
    graph : KnowledgeGraph
        The full knowledge graph.
    chains : list[DiagnosticChain]
        Available diagnostic chains.

    Returns
    -------
    TrainingPlan
        Prioritized drill recommendations.
    """
    today = date.today().isoformat()

    # Get active issues (struggling + regressed)
    active = profile.active_issues()
    if not active:
        return TrainingPlan(
            generated_date=today,
            focus_areas=[],
            drills=[],
            summary="All tracked concepts are mastered or improving. Continue current training.",
            summary_zh="所有跟踪的技术概念均已掌握或在改善中。继续当前训练即可。",
        )

    # Build issue concept set with weights
    issue_weights: dict[str, int] = {}
    for link in active:
        if link.status == ConceptStatus.REGRESSED:
            issue_weights[link.concept_id] = 3  # higher priority
        else:
            issue_weights[link.concept_id] = 2

    # Build chain lookup: concept_id -> chain drills
    chain_drills: dict[str, list[str]] = {}
    for chain in chains:
        for rc in chain.root_causes:
            if rc in issue_weights:
                chain_drills.setdefault(rc, []).extend(chain.drills)

    # Find drills from graph via drills_for edges
    drill_scores: dict[str, int] = defaultdict(int)
    drill_addresses: dict[str, set[str]] = defaultdict(set)

    for issue_id, weight in issue_weights.items():
        # Direct drills_for edges targeting this concept
        if issue_id in graph.graph:
            for pred, _, data in graph.graph.in_edges(issue_id, data=True):
                if data.get("relation") == RelationType.DRILLS_FOR.value or data.get("relation") == RelationType.DRILLS_FOR:
                    drill_scores[pred] += weight
                    drill_addresses[pred].add(issue_id)

        # Walk causal chain to find ancestor concepts with drills_for
        causal_paths = graph.get_causal_chain(issue_id, cause_type="causes")
        for path in causal_paths:
            for ancestor in path[1:]:  # skip symptom itself
                if ancestor in graph.graph:
                    for pred, _, data in graph.graph.in_edges(ancestor, data=True):
                        rel = data.get("relation")
                        if rel == RelationType.DRILLS_FOR.value or rel == RelationType.DRILLS_FOR:
                            drill_scores[pred] += max(1, weight - 1)
                            drill_addresses[pred].add(issue_id)

        # Chain drills
        for drill_id in chain_drills.get(issue_id, []):
            if drill_id not in drill_scores:
                drill_scores[drill_id] += weight
                drill_addresses[drill_id].add(issue_id)

    # Build drill recommendations
    drills: list[DrillRecommendation] = []
    for drill_id, score in sorted(drill_scores.items(), key=lambda x: -x[1]):
        node_data = graph.graph.nodes.get(drill_id, {})
        drill_name = node_data.get("name", drill_id)
        drill_name_zh = node_data.get("name_zh", drill_name)
        addresses = sorted(drill_addresses[drill_id])

        # Build rationale
        addr_names = []
        for cid in addresses:
            cn = graph.graph.nodes.get(cid, {})
            addr_names.append(cn.get("name_zh") or cn.get("name") or cid)
        rationale = f"针对: {', '.join(addr_names)}"

        drills.append(DrillRecommendation(
            drill_id=drill_id,
            drill_name=drill_name,
            drill_name_zh=drill_name_zh,
            addresses=addresses,
            priority=score,
            rationale=rationale,
        ))

    # Focus areas
    focus_names = []
    for link in active:
        node_data = graph.graph.nodes.get(link.concept_id, {})
        name = node_data.get("name_zh") or node_data.get("name") or link.concept_id
        focus_names.append(name)

    n_drills = len(drills)
    n_issues = len(active)
    summary = f"Found {n_drills} drills addressing {n_issues} active issues."
    summary_zh = f"找到 {n_drills} 个练习方法，针对 {n_issues} 个当前问题。"

    return TrainingPlan(
        generated_date=today,
        focus_areas=focus_names,
        drills=drills,
        summary=summary,
        summary_zh=summary_zh,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    profile_path = Path("knowledge/extracted/user_journey/user_profile.json")
    graph_path = Path("knowledge/extracted/_graph_snapshot.json")
    chains_path = Path("knowledge/extracted/ftt_video_diagnostic_chains.json")

    if not profile_path.exists():
        print(f"Profile not found: {profile_path}")
        raise SystemExit(1)
    if not graph_path.exists():
        print(f"Graph not found: {graph_path}")
        raise SystemExit(1)

    profile = UserProfile.from_json(profile_path)
    graph = KnowledgeGraph.from_json(graph_path)

    chain_list: list[DiagnosticChain] = []
    if chains_path.exists():
        data = json.loads(chains_path.read_text())
        chain_list = [DiagnosticChain(**c) for c in data["chains"]]

    plan = generate_training_plan(profile, graph, chain_list)
    md = plan.to_markdown()
    print(md)

    output_path = Path("knowledge/output/training_plan_latest.md")
    output_path.write_text(md, encoding="utf-8")
    print(f"\nSaved to {output_path}")
