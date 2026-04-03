# Phase 5: Output Generation & VLM Engine - Research

**Researched:** 2026-04-03
**Domain:** Knowledge graph export (JSON/Markdown) + VLM diagnostic prompt engineering
**Confidence:** HIGH

## Summary

Phase 5 converts the 582-node, 869-edge knowledge graph into three output formats: (1) machine-readable JSON export, (2) human-readable Markdown knowledge base, and (3) an upgraded VLM diagnostic prompt that replaces the current ~650-line hardcoded prompt in `evaluation/vlm_analyzer.py` with graph-backed, query-driven subgraph injection.

The critical constraint is the VLM prompt character budget. The full graph is 526KB -- far too large to inject. One node averages ~1000 chars. With a ~10K char budget for diagnostic context, only 8-10 nodes fit per VLM call. This confirms the two-pass VLM architecture: Pass 1 (quick scan) identifies the symptom category from a compact checklist (~2K chars), then Pass 2 (deep analysis) injects only the relevant diagnostic subgraph (~5-8K chars).

The current VLM prompt in `vlm_analyzer.py` contains extremely high-quality coaching knowledge (symptom groups A-G, causal chains, drill prescriptions, frame-by-frame analysis guides). This content must NOT be lost during the migration. The strategy is to decompose the hardcoded prompt into structured templates that can be populated from the knowledge graph, while preserving the coaching voice and diagnostic logic.

**Primary recommendation:** Build a `knowledge/output/` package with three modules (`json_export.py`, `markdown_export.py`, `vlm_prompt.py`) plus Jinja2 templates. The VLM prompt generator queries the graph for relevant subgraphs based on detected symptoms and renders them into the existing prompt structure.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| VLM-01 | Build prompt generator that compiles VLM prompt from knowledge graph subgraph | Subgraph extraction methods on KnowledgeGraph + Jinja2 template rendering |
| VLM-02 | Implement two-pass VLM analysis (quick scan -> targeted deep analysis) | Pass 1: compact symptom checklist (~2K chars); Pass 2: targeted subgraph injection (~8K chars) |
| VLM-03 | Complete diagnostic coverage -- every known symptom has a diagnostic chain | 18 chains exist, 27 symptom nodes; 15 symptoms lack chains (generation_log in diagnostic_chains.json) |
| VLM-04 | Each diagnostic output includes: what's wrong, why, how to fix, how to check | Map to existing output JSON schema (action/body/feel/drill fields) |
| VLM-05 | VLM prompt stays within ~10K char budget via query-based subgraph injection | One node ~1000 chars; budget allows 8-10 nodes per query; two-pass essential |
| OUT-01 | JSON knowledge graph export (nodes + edges + diagnostic chains) | NetworkX node_link_data format already used; add diagnostic chains overlay |
| OUT-02 | Markdown knowledge base export (organized by topic hierarchy) | Jinja2 templates, ConceptType-based grouping |
| OUT-03 | Markdown includes cross-references, source citations, confidence levels | Edge data has evidence/source_file/confidence fields |
| OUT-04 | VLM prompt template file (generated from graph, replaces hardcoded prompt) | Template-based prompt with graph-injected sections |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Jinja2 | 3.1.6 | Template rendering for Markdown + VLM prompts | Already installed; industry standard for text templating |
| NetworkX | 3.6.1 | Graph traversal and subgraph extraction | Already the graph backend; no alternatives needed |
| Pydantic | 2.12.5 | Export schema validation | Already used for all schemas |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| json (stdlib) | - | JSON serialization | All export operations |
| pathlib (stdlib) | - | File path management | Output directory creation |

No new packages needed. Everything required is already installed.

## Architecture Patterns

### Recommended Project Structure
```
knowledge/
  output/
    __init__.py
    json_export.py          # OUT-01: Full graph JSON export
    markdown_export.py      # OUT-02/03: Jinja2-based Markdown generation
    vlm_prompt.py           # VLM-01/04/05: Subgraph -> prompt compiler
  templates/
    knowledge_base/
      index.md.j2           # Top-level index
      concept.md.j2         # Per-concept page
      diagnostic_chain.md.j2 # Per-chain page
      topic_group.md.j2     # Per-category group page
    vlm/
      system_prompt.md.j2   # Base VLM system prompt (frame analysis guide + core principles)
      symptom_checklist.j2  # Pass 1: compact symptom scanning list
      diagnostic_deep.j2    # Pass 2: targeted subgraph with causal chains
evaluation/
  vlm_analyzer.py           # Modified: loads prompt from vlm_prompt.py instead of _FTT_SYSTEM_PROMPT
```

### Pattern 1: Two-Pass VLM Architecture
**What:** Split the VLM analysis into two API calls with different prompt sizes.
**When:** Every VLM analysis call.

```python
# knowledge/output/vlm_prompt.py

class VLMPromptCompiler:
    """Compile VLM prompts from knowledge graph subgraphs."""

    def __init__(self, graph: KnowledgeGraph, chains: list[DiagnosticChain]):
        self.graph = graph
        self.chains = chains
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader("knowledge/templates/vlm"),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def compile_pass1_prompt(self) -> str:
        """Pass 1: Compact symptom checklist for quick scan (~2K chars).

        Lists all symptom categories with brief visual markers.
        VLM returns: which symptom categories are detected.
        """
        template = self.env.get_template("symptom_checklist.j2")
        return template.render(chains=self.chains)

    def compile_pass2_prompt(self, detected_symptoms: list[str]) -> str:
        """Pass 2: Deep analysis with relevant subgraph (~8K chars).

        Given detected symptom IDs, extract the relevant subgraph
        (causal chains, root causes, drills, VLM features) and inject.
        """
        subgraph = self._extract_relevant_subgraph(detected_symptoms)
        template = self.env.get_template("diagnostic_deep.j2")
        return template.render(subgraph=subgraph)

    def _extract_relevant_subgraph(self, symptom_ids: list[str]) -> dict:
        """Extract nodes/edges relevant to detected symptoms."""
        relevant_nodes = set()
        relevant_edges = []

        for sid in symptom_ids:
            # Get causal chains backward from symptom
            chains = self.graph.get_causal_chain(sid, "causes")
            for chain in chains:
                relevant_nodes.update(chain)

            # Get matching diagnostic chains
            # Get drill nodes connected to root causes
            # Get VLM feature descriptions

        return {
            "nodes": [self.graph.graph.nodes[n] for n in relevant_nodes if n in self.graph.graph],
            "edges": relevant_edges,
            "chains": [c for c in self.chains if c.symptom_concept_id in symptom_ids],
        }
```

### Pattern 2: Preserve Existing Prompt Quality via Hybrid Approach
**What:** The current hardcoded prompt contains extremely valuable coaching knowledge that is NOT fully represented in the graph (e.g., the 16 core principles, the 6-frame analysis guide, the drill knowledge base). Rather than trying to regenerate all of this from the graph, use a hybrid approach.
**When:** VLM prompt generation.

Strategy:
1. **Static sections** (kept as Jinja2 templates, not generated from graph):
   - Frame-by-frame analysis guide (lines 383-433 of current prompt)
   - Core principles 1-16 (lines 434-454)
   - Drill knowledge base (lines 456-463)
   - Output format specification (lines 603-646)
   - Scoring criteria (lines 640-646)

2. **Dynamic sections** (generated from graph at runtime):
   - Symptom group diagnostic logic chains (lines 465-583) -- replaced by graph-backed chains
   - VLM-observable features -- pulled from node vlm_features fields
   - Root cause lookup -- traversed from graph causal edges

3. **Budget allocation** (~10K char total for dynamic injection):
   - Static base prompt: loaded from template files (NOT counted against 10K -- sent as system prompt)
   - Dynamic diagnostic context: ~8K chars injected per-query based on detected symptoms

### Pattern 3: JSON Export Schema
**What:** Export the full graph in a self-contained JSON format.

```python
# knowledge/output/json_export.py
def export_full_graph(graph: KnowledgeGraph, chains: list[DiagnosticChain], output_path: Path):
    """Export complete knowledge graph as JSON."""
    data = {
        "metadata": {
            "version": "1.0",
            "exported": datetime.now().isoformat(),
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
            "chain_count": len(chains),
        },
        "nodes": [graph.graph.nodes[n] for n in graph.graph.nodes],
        "edges": [
            {**data, "source": u, "target": v}
            for u, v, data in graph.graph.edges(data=True)
        ],
        "diagnostic_chains": [c.model_dump() for c in chains],
    }
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
```

### Pattern 4: Markdown Topic Hierarchy
**What:** Organize Markdown export by ConceptType categories.

```
docs/knowledge/
  index.md                    # Overview with stats + links
  techniques/
    index.md                  # All technique concepts
    unit_turn.md              # Per-concept: description, edges, muscles, VLM features
    ...
  biomechanics/
    index.md
    rotational_kinetic_chain.md
    ...
  symptoms/
    index.md
    forearm_compensation.md   # Includes diagnostic chain if exists
    ...
  drills/
    index.md
    ...
  diagnostic_chains/
    index.md
    dc_arm_driven_hitting.md  # Full chain visualization
    ...
```

### Anti-Patterns to Avoid

- **Regenerating the frame analysis guide from graph:** The current 6-frame analysis guide (lines 383-433) is hand-crafted coaching expertise with precise frame-by-frame instructions. The graph does not contain this level of detail. Keep it as a static template.

- **Dumping full graph into VLM prompt:** 526KB graph will destroy VLM quality. Always use subgraph selection.

- **Breaking the existing VLM output schema:** The current output JSON format (issues/strengths/overall_assessment/score/drills) is consumed by downstream report generation. Any changes to the prompt must maintain backward compatibility with this schema.

- **Single-pass with massive prompt:** Trying to fit everything in one VLM call. Two passes with focused context will produce better results.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Template rendering | String concatenation / f-strings for Markdown | Jinja2 templates | Maintainable, testable, separates content from structure |
| Graph serialization | Custom JSON format | NetworkX node_link_data | Already used; roundtrip-safe |
| Subgraph extraction | Manual node/edge filtering | NetworkX subgraph() + ego_graph() | Handles edge cases (missing nodes, parallel edges) |
| Markdown cross-references | Manual link construction | Jinja2 macro that generates `[concept_name](../category/id.md)` | DRY, consistent paths |

## Common Pitfalls

### Pitfall 1: Losing Coaching Quality in Graph-Backed Prompts
**What goes wrong:** The graph stores structured data (concept ID, description, edges) but loses the nuanced coaching voice present in the hardcoded prompt. E.g., graph says "forearm_compensation causes scooping" but the current prompt says "look at frames 4-5: is the elbow angle contracting (<90 deg getting smaller) or expanding?"
**Why it happens:** Structured extraction loses prosody and instructional context.
**How to avoid:** Hybrid approach -- keep the static coaching voice in templates, inject only the diagnostic routing from the graph.
**Warning signs:** VLM output becomes generic/robotic compared to current quality.

### Pitfall 2: VLM Pass 1 Returns Unparseable Results
**What goes wrong:** Pass 1 asks VLM to categorize symptoms, but the free-text response doesn't map cleanly to symptom IDs.
**Why it happens:** VLM output is fuzzy text, not structured enums.
**How to avoid:** Pass 1 prompt should use numbered categories with clear labels. Parse response with fuzzy matching against known symptom names, not exact string matching.
**Warning signs:** Pass 2 receives empty or wrong symptom list.

### Pitfall 3: Prompt Budget Overflow in Pass 2
**What goes wrong:** Subgraph extraction pulls too many connected nodes, exceeding the 8K char budget for dynamic content.
**Why it happens:** Graph traversal without depth limits follows long causal chains and pulls in tangentially related nodes.
**How to avoid:** Limit traversal depth to 2 hops. Prioritize by edge confidence. Count characters during assembly and stop at budget.
**Warning signs:** Generated prompt exceeds 10K chars total.

### Pitfall 4: Diagnostic Chain Gaps
**What goes wrong:** 15 symptom nodes lack diagnostic chains (documented in generation_log). VLM encounters one of these symptoms but has no chain to follow.
**Why it happens:** Some symptoms are root causes themselves (no upstream causes in the graph).
**How to avoid:** For symptoms without chains, generate minimal chains that link directly to their VLM features and any available drills. Accept that not all symptoms need multi-step chains -- leaf symptoms can have single-step "check and confirm" chains.
**Warning signs:** VLM reports a symptom but gives no root cause analysis.

### Pitfall 5: Markdown Export Circular References
**What goes wrong:** Concept A links to Concept B which links back to A, creating circular cross-reference chains in Markdown.
**Why it happens:** The graph has bidirectional relationships (A supports B, B supports A).
**How to avoid:** Render edges as directional lists (outgoing vs incoming). Don't recursively expand cross-references.

## Code Examples

### Subgraph Extraction for VLM (needed addition to KnowledgeGraph)
```python
# knowledge/graph.py -- new method needed
def get_symptom_subgraph(self, symptom_id: str, max_depth: int = 2) -> dict:
    """Extract subgraph relevant to a symptom for VLM prompt injection.

    Returns nodes and edges within max_depth hops of the symptom,
    following causes/visible_as/drills_for edge types.
    """
    relevant = set()
    frontier = {symptom_id}
    diagnostic_types = {"causes", "visible_as", "drills_for"}

    for _ in range(max_depth):
        next_frontier = set()
        for node in frontier:
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
        "nodes": {nid: dict(self.graph.nodes[nid]) for nid in relevant if nid in self.graph},
        "edges": [
            {"source": u, "target": v, **d}
            for u, v, d in self.graph.edges(data=True)
            if u in relevant and v in relevant and d.get("relation") in diagnostic_types
        ],
    }
```

### Jinja2 Template for Concept Markdown Page
```jinja2
{# knowledge/templates/knowledge_base/concept.md.j2 #}
# {{ concept.name }} ({{ concept.name_zh }})

**Category:** {{ concept.category }}
**Confidence:** {{ "%.0f"|format(concept.confidence * 100) }}%
**Sources:** {{ concept.sources | join(", ") }}

## Description

{{ concept.description }}

{% if concept.vlm_features %}
## VLM Observable Features

{% for f in concept.vlm_features %}
- {{ f }}
{% endfor %}
{% endif %}

{% if concept.muscles_involved %}
## Muscles Involved

{% for m in concept.muscles_involved %}
- {{ m | replace("_", " ") | title }}
{% endfor %}
{% endif %}

{% if outgoing_edges %}
## Relationships (outgoing)

| Target | Type | Confidence | Evidence |
|--------|------|------------|----------|
{% for e in outgoing_edges %}
| [{{ e.target_name }}](../{{ e.target_category }}/{{ e.target_id }}.md) | {{ e.relation }} | {{ "%.0f"|format(e.confidence * 100) }}% | {{ e.evidence[:80] }}... |
{% endfor %}
{% endif %}

{% if incoming_edges %}
## Referenced By (incoming)

| Source | Type | Confidence |
|--------|------|------------|
{% for e in incoming_edges %}
| [{{ e.source_name }}](../{{ e.source_category }}/{{ e.source_id }}.md) | {{ e.relation }} | {{ "%.0f"|format(e.confidence * 100) }}% |
{% endfor %}
{% endif %}
```

### Two-Pass VLM Integration Point
```python
# evaluation/vlm_analyzer.py -- modified analyze_swing
def analyze_swing(self, keyframe_grid, kpi_summary="", ...):
    compiler = VLMPromptCompiler(self.knowledge_graph, self.diagnostic_chains)

    # Pass 1: Quick symptom scan
    pass1_prompt = compiler.compile_pass1_prompt()  # ~2K chars
    pass1_result = self._call_vlm(keyframe_grid, pass1_prompt)
    detected = parse_symptom_categories(pass1_result)

    # Pass 2: Deep analysis with targeted subgraph
    pass2_prompt = compiler.compile_pass2_prompt(detected)  # ~8K chars
    pass2_result = self._call_vlm(keyframe_grid, pass2_prompt)

    return _parse_json_response(pass2_result)
```

## Graph Data Inventory

Key measurements from the actual graph snapshot:

| Metric | Value | Implication |
|--------|-------|-------------|
| Full graph JSON | 526KB / 538K chars | Cannot fit in VLM prompt |
| Node count | 582 | Manageable for JSON/Markdown export |
| Edge count | 884 (790 supports, 55 causes, 39 visible_as) | Sparse causal edges -- most are co-occurrence |
| Symptom nodes | 27 | Finite set for Pass 1 checklist |
| Diagnostic chains | 18 (12 auto + 6 manual) | 15 symptoms still lack chains |
| Nodes with VLM features | 207 | Rich feature set for VLM prompts |
| One node avg size | ~1000 chars | Budget: ~8-10 nodes per VLM query |
| Categories | technique:396, biomechanics:88, mental_model:32, connection:21, drill:18, symptom:27 | Technique-heavy; drills sparse |

**Critical observation:** Only 55 causal edges exist (vs 790 supports). The diagnostic chains rely heavily on the 18 pre-built DiagnosticChain objects rather than graph traversal. The graph's causal subgraph is shallow. This means the two-pass VLM approach should primarily use the DiagnosticChain objects for Pass 2, supplemented by graph node descriptions and VLM features, rather than attempting deep graph traversal.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (via pyproject.toml) |
| Config file | pyproject.toml `[tool.pytest.ini_options]` |
| Quick run command | `pytest tests/ -x -q` |
| Full suite command | `pytest tests/ -v` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| VLM-01 | Prompt compiler produces valid prompt from graph subgraph | unit | `pytest tests/test_vlm_prompt.py::test_compile_pass1 -x` | Wave 0 |
| VLM-02 | Two-pass analysis returns structured result | unit | `pytest tests/test_vlm_prompt.py::test_two_pass_flow -x` | Wave 0 |
| VLM-03 | All 27 symptoms have diagnostic coverage | unit | `pytest tests/test_vlm_prompt.py::test_diagnostic_coverage -x` | Wave 0 |
| VLM-04 | Output includes action/body/feel/drill fields | unit | `pytest tests/test_vlm_prompt.py::test_output_schema -x` | Wave 0 |
| VLM-05 | Generated prompt stays within 10K char budget | unit | `pytest tests/test_vlm_prompt.py::test_prompt_budget -x` | Wave 0 |
| OUT-01 | JSON export contains all nodes/edges/chains | unit | `pytest tests/test_output_export.py::test_json_export -x` | Wave 0 |
| OUT-02 | Markdown export creates correct directory structure | unit | `pytest tests/test_output_export.py::test_markdown_structure -x` | Wave 0 |
| OUT-03 | Markdown includes cross-refs and confidence | unit | `pytest tests/test_output_export.py::test_markdown_crossrefs -x` | Wave 0 |
| OUT-04 | VLM prompt template replaces hardcoded prompt | integration | `pytest tests/test_output_export.py::test_prompt_template_replaces_hardcoded -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/test_vlm_prompt.py tests/test_output_export.py -x -q`
- **Per wave merge:** `pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_vlm_prompt.py` -- covers VLM-01 through VLM-05
- [ ] `tests/test_output_export.py` -- covers OUT-01 through OUT-04
- [ ] `knowledge/output/__init__.py` -- package init
- [ ] `knowledge/templates/` -- template directory structure

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Jinja2 | Markdown/prompt templates | Yes | 3.1.6 | -- |
| NetworkX | Graph operations | Yes | 3.6.1 | -- |
| Pydantic | Schema validation | Yes | 2.12.5 | -- |
| pytest | Testing | Yes | (installed) | -- |

No missing dependencies. All required packages are already installed.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Hardcoded VLM prompt (~650 lines) | Graph-backed prompt generation | This phase | Maintainable, extensible, testable |
| No export format | JSON + Markdown dual export | This phase | Machine + human readable outputs |
| Single VLM pass | Two-pass (scan + deep) | This phase | Better diagnostic accuracy within token budget |

## Open Questions

1. **Pass 1 response parsing reliability**
   - What we know: VLM free-text responses need to be mapped to symptom categories
   - What's unclear: How reliably can VLM output be parsed into structured symptom IDs?
   - Recommendation: Use numbered categories in Pass 1 prompt (e.g., "Reply with numbers: 1=arm compensation, 2=scooping..."). Parse by looking for numbers in response. Fallback: if parsing fails, skip Pass 1 and use the full static prompt (current behavior).

2. **Double API cost for two-pass**
   - What we know: Two API calls per analysis doubles the cost
   - What's unclear: Whether the quality improvement justifies the cost
   - Recommendation: Make two-pass optional via config flag. Default to two-pass, but allow single-pass fallback that uses the static prompt (current behavior preserved).

3. **15 symptoms without diagnostic chains**
   - What we know: generation_log shows 15 symptoms had no upstream causes in graph
   - What's unclear: Whether these symptoms need chains or are themselves root causes
   - Recommendation: Review the 15 symptoms. Root-cause symptoms (e.g., problem_p06, problem_p09) should be marked as terminal -- they are the answer, not the question. Only create new chains for symptoms that are genuinely observable but lack diagnostic paths.

## Sources

### Primary (HIGH confidence)
- `evaluation/vlm_analyzer.py` -- Current VLM prompt structure (920 lines, prompt lines 370-646)
- `knowledge/extracted/_graph_snapshot.json` -- 582 nodes, 884 edges, node_link_data format
- `knowledge/extracted/ftt_video_diagnostic_chains.json` -- 18 chains with generation log
- `knowledge/schemas.py` -- Pydantic models for Concept, Edge, DiagnosticChain
- `knowledge/graph.py` -- KnowledgeGraph class with get_causal_chain method

### Secondary (MEDIUM confidence)
- `.planning/research/ARCHITECTURE.md` -- Planned output/ directory structure
- `.planning/research/PITFALLS.md` -- VLM prompt budget overflow (Pitfall 3)
- Jinja2 3.1 documentation (installed version verified)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all packages already installed and used in project
- Architecture: HIGH -- based on direct code reading of current VLM analyzer and graph structure
- Pitfalls: HIGH -- derived from actual measurements (526KB graph, 1000 chars/node, 10K budget)
- VLM two-pass: MEDIUM -- design is sound but Pass 1 parsing reliability is unverified

**Research date:** 2026-04-03
**Valid until:** 2026-05-03 (stable domain, no external dependency changes expected)
