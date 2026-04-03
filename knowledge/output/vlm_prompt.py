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

import jinja2

from knowledge.graph import KnowledgeGraph
from knowledge.schemas import DiagnosticChain


_TEMPLATE_DIR = Path(__file__).parent.parent / "templates" / "vlm"

# Maximum character budget for the dynamic diagnostic section in Pass 2.
_DYNAMIC_BUDGET = 10_000


class VLMPromptCompiler:
    """Compile VLM prompts from knowledge graph subgraphs.

    Parameters
    ----------
    graph : KnowledgeGraph
        The full knowledge graph (used for subgraph extraction).
    chains : list[DiagnosticChain]
        All available diagnostic chains (18 in current system).
    """

    def __init__(self, graph: KnowledgeGraph, chains: list[DiagnosticChain]) -> None:
        self.graph = graph
        self.chains = chains
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

    def compile_pass2_prompt(self, detected_chain_ids: list[str]) -> str:
        """Pass 2: Static prompt + targeted diagnostic context.

        Combines the full static coaching prompt with dynamic diagnostic
        sections for only the detected symptom chains.  Budget-enforced:
        the dynamic section is truncated if it exceeds 10K chars.

        Parameters
        ----------
        detected_chain_ids : list[str]
            Chain IDs detected in Pass 1 (e.g. ``["dc_arm_driven_hitting"]``).
        """
        static = self.compile_system_prompt()
        subgraph = self._extract_relevant_subgraph(detected_chain_ids)
        matched_chains = [
            self.chain_map[cid]
            for cid in detected_chain_ids
            if cid in self.chain_map
        ]

        template = self.env.get_template("diagnostic_deep.j2")
        dynamic = template.render(subgraph=subgraph, chains=matched_chains)

        # Budget enforcement: truncate if dynamic section exceeds budget.
        if len(dynamic) > _DYNAMIC_BUDGET:
            dynamic = self._truncate_by_confidence(dynamic, subgraph, _DYNAMIC_BUDGET)

        return static + "\n\n" + dynamic

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
