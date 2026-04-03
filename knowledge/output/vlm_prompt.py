"""VLM Prompt Compiler — converts knowledge graph subgraphs into VLM prompts.

Two-pass architecture:
  Pass 1: Compact symptom checklist (~2K chars) for quick visual scan.
  Pass 2: Static coaching prompt + targeted diagnostic context (~8K dynamic chars).

The static base prompt preserves the hand-crafted coaching voice (frame guide,
16 core principles, drill KB, output format) extracted from the original
650-line _FTT_SYSTEM_PROMPT in evaluation/vlm_analyzer.py.

Dynamic sections inject graph-backed diagnostic chains and concept details
for only the detected symptoms, staying within the ~10K char budget.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import jinja2

from knowledge.graph import KnowledgeGraph
from knowledge.schemas import DiagnosticChain

if TYPE_CHECKING:
    from knowledge.user_profile import UserProfile


_TEMPLATE_DIR = Path(__file__).parent.parent / "templates" / "vlm"

# Maximum character budget for the dynamic diagnostic section in Pass 2.
_DYNAMIC_BUDGET = 10_000

# Maximum character budget for the user context section in Pass 2.
_USER_CONTEXT_BUDGET = 1_500


class VLMPromptCompiler:
    """Compile VLM prompts from knowledge graph subgraphs.

    Parameters
    ----------
    graph : KnowledgeGraph
        The full knowledge graph (used for subgraph extraction).
    chains : list[DiagnosticChain]
        All available diagnostic chains (18 in current system).
    user_profile : UserProfile | None
        Optional user training profile for personalized diagnostics.
    """

    def __init__(
        self,
        graph: KnowledgeGraph,
        chains: list[DiagnosticChain],
        user_profile: "UserProfile | None" = None,
    ) -> None:
        self.graph = graph
        self.chains = chains
        self.user_profile = user_profile
        self.chain_map: dict[str, DiagnosticChain] = {c.id: c for c in chains}
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(_TEMPLATE_DIR)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compile_system_prompt(self) -> str:
        """Render the static base prompt (frame guide, principles, drills, output format)."""
        template = self.env.get_template("system_prompt.md.j2")
        return template.render()

    def compile_pass1_prompt(self) -> str:
        """Pass 1: Compact symptom checklist for quick scan (~2K chars).

        Lists all symptom categories sorted by priority with brief visual cues.
        VLM returns which category numbers it detects.
        """
        template = self.env.get_template("symptom_checklist.j2")
        sorted_chains = sorted(self.chains, key=lambda c: c.priority)
        return template.render(chains=sorted_chains)

    def compile_user_context(self, detected_chain_ids: list[str]) -> str:
        """Render personalized user context section for Pass 2.

        Returns empty string if no user_profile is set.
        Budget-enforced: truncated to ``_USER_CONTEXT_BUDGET`` chars.

        Parameters
        ----------
        detected_chain_ids : list[str]
            Chain IDs detected in Pass 1 — used to cross-reference with
            user's known issues and mark overlapping concepts.
        """
        if self.user_profile is None:
            return ""

        # Collect symptom concept IDs from detected chains
        detected_symptom_ids: set[str] = set()
        for cid in detected_chain_ids:
            chain = self.chain_map.get(cid)
            if chain:
                detected_symptom_ids.add(chain.symptom_concept_id)
                detected_symptom_ids.update(chain.root_causes)

        # Build active issues with resolved concept names
        active = self.user_profile.active_issues()
        issue_items = []
        for link in active[:5]:
            node_data = self.graph.graph.nodes.get(link.concept_id, {})
            concept_name = node_data.get("name_zh") or node_data.get("name") or link.concept_id
            is_recurring = link.concept_id in detected_symptom_ids
            issue_items.append({
                "concept_name": concept_name,
                "status": link.status.value,
                "last_seen": link.last_seen,
                "cues": link.cues,
                "is_recurring": is_recurring,
            })

        # Build recent breakthroughs
        bt_sessions = self.user_profile.recent_breakthroughs(n=3)
        bt_items = []
        for session in bt_sessions:
            bt_items.append({
                "summary": "; ".join(session.breakthroughs[:2]),
                "date": session.date,
            })

        template = self.env.get_template("user_context.j2")
        rendered = template.render(
            active_issues=issue_items,
            breakthroughs=bt_items,
        )

        # Budget enforcement: truncate at line boundary
        if len(rendered) > _USER_CONTEXT_BUDGET:
            cut = rendered.rfind("\n", 0, _USER_CONTEXT_BUDGET)
            if cut == -1:
                cut = _USER_CONTEXT_BUDGET
            rendered = rendered[:cut]

        return rendered

    def compile_pass2_prompt(self, detected_chain_ids: list[str]) -> str:
        """Pass 2: Static prompt + user context + targeted diagnostic context.

        Combines the full static coaching prompt with optional user context
        and dynamic diagnostic sections for only the detected symptom chains.
        Budget-enforced: user context gets up to 1.5K chars, diagnostics
        get up to 8.5K chars (total ~10K dynamic budget).

        Parameters
        ----------
        detected_chain_ids : list[str]
            Chain IDs detected in Pass 1 (e.g. ``["dc_arm_driven_hitting"]``).
        """
        static = self.compile_system_prompt()

        # User context (personalized, optional)
        user_ctx = self.compile_user_context(detected_chain_ids)

        # Diagnostic context
        subgraph = self._extract_relevant_subgraph(detected_chain_ids)
        matched_chains = [
            self.chain_map[cid]
            for cid in detected_chain_ids
            if cid in self.chain_map
        ]

        template = self.env.get_template("diagnostic_deep.j2")
        dynamic = template.render(subgraph=subgraph, chains=matched_chains)

        # Budget enforcement: diagnostics get remaining budget after user context.
        diag_budget = _DYNAMIC_BUDGET - len(user_ctx)
        if len(dynamic) > diag_budget:
            dynamic = self._truncate_by_confidence(dynamic, subgraph, diag_budget)

        parts = [static]
        if user_ctx:
            parts.append(user_ctx)
        parts.append(dynamic)
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_relevant_subgraph(self, chain_ids: list[str]) -> dict:
        """Aggregate subgraphs for all detected symptom chains."""
        all_nodes: dict = {}
        all_edges: list[dict] = []

        for cid in chain_ids:
            chain = self.chain_map.get(cid)
            if not chain:
                continue
            sub = self.graph.get_symptom_subgraph(
                chain.symptom_concept_id, max_depth=2
            )
            all_nodes.update(sub["nodes"])
            all_edges.extend(sub["edges"])

        # Deduplicate edges by (source, target, relation) tuple.
        seen: set[tuple] = set()
        unique_edges: list[dict] = []
        for edge in all_edges:
            key = (edge.get("source"), edge.get("target"), edge.get("relation"))
            if key not in seen:
                seen.add(key)
                unique_edges.append(edge)

        return {"nodes": all_nodes, "edges": unique_edges}

    @staticmethod
    def _truncate_by_confidence(
        dynamic: str, subgraph: dict, budget: int
    ) -> str:
        """Truncate dynamic content to fit within character budget.

        Strategy: if the rendered dynamic text exceeds budget, simply
        truncate to the budget limit at a line boundary to avoid breaking
        mid-sentence.  (In practice with 18 chains and shallow subgraphs
        this is rarely needed.)
        """
        if len(dynamic) <= budget:
            return dynamic
        # Find last newline before budget limit.
        cut = dynamic.rfind("\n", 0, budget)
        if cut == -1:
            cut = budget
        return dynamic[:cut] + "\n\n[...truncated to fit budget...]"
