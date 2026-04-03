# Domain Pitfalls

**Domain:** Tennis Knowledge Engineering & VLM Diagnostic System
**Researched:** 2026-04-03

## Critical Pitfalls

Mistakes that cause rewrites or major issues.

### Pitfall 1: Concept Explosion Without Deduplication Strategy

**What goes wrong:** Extracting concepts per-video produces 10-30 concepts each. Across 115 FTT videos, that is 1000-3000 raw concepts. Without deduplication, "hip rotation", "hip turn", "hip drive", "hip coil" all become separate nodes, making the graph unusable.

**Why it happens:** LLMs use different terminology for the same concept depending on prompt context and source phrasing. No two extractions will use identical concept IDs.

**Consequences:** Graph becomes a disconnected mess. Diagnostic chains break because the "hip rotation" node from video A has no edges to the "hip turn" node from video B. The entire graph's value depends on merging these.

**Prevention:** Design the deduplication strategy BEFORE starting extraction. Use a canonical concept registry: first pass extracts raw concepts, second pass maps them to canonical IDs using LLM-assisted fuzzy matching. Build the registry incrementally -- start with the 28 existing research files, establish canonical IDs, then use that registry when processing new videos.

**Detection:** If the graph has more than ~300 unique concept nodes after processing FTT alone, deduplication is failing.

### Pitfall 2: Gemini API Rate Limits and Cost Blowup

**What goes wrong:** Running 115 video analyses + 4 extraction passes each = 460+ API calls. With Gemini 2.5 Pro, each video analysis can consume significant token budget. Rate limits cause pipeline stalls; costs exceed expectations.

**Why it happens:** YouTube URL video analysis processes the entire video. Long videos (20+ minutes) consume more tokens. The free tier has aggressive rate limits.

**Consequences:** Pipeline takes days instead of hours. API costs surprise. Partial failures require restart logic.

**Prevention:**
1. Start with Gemini 2.5 Flash (not Pro) for initial extraction -- cheaper, faster, sufficient for structured extraction.
2. Use Pro only for complex reconciliation passes.
3. Implement checkpointing (processed_videos.json) so failures resume, not restart.
4. Set daily budget caps in the pipeline config.
5. Process existing research files FIRST (text-only, much cheaper than video analysis).

**Detection:** Monitor cost per video. If averaging >$0.10/video on Flash, something is wrong.

### Pitfall 3: VLM Prompt Token Budget Overflow

**What goes wrong:** The knowledge graph produces comprehensive diagnostic data, but the VLM prompt has a practical limit (~10K chars per PROJECT.md). Trying to inject the full graph into every VLM analysis makes prompts too long, degrading analysis quality.

**Why it happens:** The graph grows as knowledge is added. Without pruning, the diagnostic prompt keeps growing.

**Consequences:** VLM analysis quality degrades (prompt too long, model loses focus). Or, arbitrary truncation drops important diagnostic chains.

**Prevention:** Design the prompt injection as a QUERY, not a dump. At VLM analysis time: (1) run initial visual scan, (2) identify likely symptom category, (3) query graph for relevant subgraph only, (4) inject relevant diagnostic chains into follow-up prompt. Two-pass VLM analysis.

**Detection:** If the generated diagnostic prompt exceeds 8K chars, the subgraph selection is too broad.

## Moderate Pitfalls

### Pitfall 4: YouTube Transcript Quality Varies Wildly

**What goes wrong:** Auto-generated YouTube subtitles for tennis content are often wrong. Technical terms like "pronation", "supination", "kinetic chain" get misheard.

**Prevention:** Use transcripts as supplementary context only, never as the primary extraction source. Primary extraction should come from Gemini video understanding (which processes the actual video frames + audio). Transcripts help with timestamps and searchability.

### Pitfall 5: Cross-Source Reconciliation Is Harder Than Expected

**What goes wrong:** FTT, TomAllsopp, and Feel Tennis use different terminology, different frameworks, and sometimes genuinely disagree on technique. Identifying what constitutes a conflict vs. a complement is nuanced.

**Prevention:** Build reconciliation as a separate, explicit phase -- not embedded in extraction. Extract each source independently with source attribution. Then run a reconciliation pass that specifically asks: "Do these two concepts agree, complement, or contradict?"

### Pitfall 6: Graph Structure That Cannot Express Diagnostic Chains

**What goes wrong:** Building the graph with only "related_to" edges (undirected, untyped). Cannot answer "what CAUSES this symptom?" because causation direction is lost.

**Prevention:** Use a directed multigraph from day one. Every edge must have a type (causes, prevents, requires, etc.) and a direction. Test with a concrete example: "elbow flying out" -> caused_by -> "insufficient trunk rotation" -> caused_by -> "no hip pre-loading". If the graph cannot express this chain, the schema is wrong.

### Pitfall 7: Losing Raw Analysis When Re-Extracting

**What goes wrong:** Changing the Pydantic schema requires re-extraction. If raw Gemini analysis text was not saved, re-extraction requires re-analyzing all videos (expensive, slow).

**Prevention:** Always save the raw Gemini analysis as a Markdown file in docs/research/ (the project already does this). Structured extraction is a separate step from analysis. Raw text is cheap to store, expensive to regenerate.

## Minor Pitfalls

### Pitfall 8: Non-Deterministic LLM Extraction

**What goes wrong:** Running the same extraction twice produces different concept lists.

**Prevention:** Set temperature=0 for extraction calls. Accept that some variation is inevitable and design deduplication to handle it. Pin model versions in config.

### Pitfall 9: Chinese-English Terminology Mismatch

**What goes wrong:** Knowledge base is in Chinese but source videos are in English. Concept names need both languages. Inconsistent translation creates duplicate concepts.

**Prevention:** Every Concept has both `name` (English) and `name_zh` (Chinese) fields. Canonical ID uses English (lowercase, underscored). Chinese names are for display only, never for matching.

### Pitfall 10: Graph Visualization Ignored

**What goes wrong:** Building a 1000-node graph with no way to visualize it. Cannot debug relationship quality without seeing the graph.

**Prevention:** Add a simple visualization utility early (NetworkX + matplotlib, or export to Graphviz DOT format). Does not need to be pretty -- just needs to show node clusters and edge types for debugging.

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| Existing file extraction | Concept explosion (P1) | Build canonical registry from existing files first |
| FTT video extraction | Rate limits (P2) | Use Flash model, checkpointing, daily caps |
| Graph assembly | Undirected edges (P6) | Enforce typed directed edges from schema level |
| Diagnostic chains | Prompt overflow (P3) | Two-pass VLM with subgraph selection |
| Cross-source | Reconciliation complexity (P5) | Separate reconciliation phase, not inline |
| VLM integration | Prompt size (P3) | Query-based injection, not full dump |

## Sources

- Analysis of existing codebase (docs/research/ patterns, vlm_analyzer.py prompt structure)
- [Gemini API rate limits and pricing](https://ai.google.dev/gemini-api/docs/rate-limits)
- [Instructor retry/validation patterns](https://python.useinstructor.com/)
