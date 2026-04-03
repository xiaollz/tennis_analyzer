# Research Summary: Tennis Forehand Knowledge Engineering & VLM Diagnostic System

**Domain:** Knowledge engineering pipeline for tennis technique analysis
**Researched:** 2026-04-03
**Overall confidence:** HIGH

## Executive Summary

The project needs to transform ~200KB of existing research text and ~115 YouTube videos into a structured, queryable knowledge graph that powers a VLM diagnostic engine. The core technical challenge is not any single component but the integration: extracting structured knowledge from unstructured sources, merging duplicate concepts across sources, and producing a graph that can answer "what causes this symptom and how do I fix it?"

The recommended stack is deliberately lightweight: yt-dlp for video discovery, google-genai SDK for Gemini video analysis (YouTube URL pass-through -- no downloads needed), instructor+pydantic for structured extraction, NetworkX+JSON for the knowledge graph, and Jinja2 for Markdown output. No databases, no heavy frameworks, no infrastructure beyond Python and API keys. This matches the project's reality: single user, personal machine, batch processing.

The highest-risk area is concept deduplication. Across 115+ videos, the same biomechanical concept will appear with different names hundreds of times. Without a robust deduplication strategy (canonical registry + LLM-assisted merge), the graph becomes unusable. This must be designed before extraction begins, not bolted on after.

The second major risk is VLM prompt budget. The knowledge graph will produce far more diagnostic knowledge than fits in a single prompt (~10K char limit). The architecture must support query-based subgraph injection rather than full graph dumping -- a two-pass VLM analysis where the first pass identifies symptom categories and the second pass injects only relevant diagnostic chains.

## Key Findings

**Stack:** Python-native pipeline: yt-dlp + youtube-transcript-api -> google-genai SDK (Gemini 2.5) -> instructor + pydantic -> NetworkX + JSON -> Jinja2 Markdown. No databases, no heavy frameworks.

**Architecture:** Six-layer pipeline with clear data flow. Raw analysis saved as Markdown (human artifacts), structured extraction saved as JSON (machine artifacts), both feeding a NetworkX directed multigraph.

**Critical pitfall:** Concept deduplication. Without a canonical concept registry built upfront, 115 videos produce 1000+ duplicate concept nodes that fragment the graph.

## Implications for Roadmap

Based on research, suggested phase structure:

1. **Schema & Infrastructure** - Define Pydantic models, set up pipeline scaffolding, configure API access
   - Addresses: KE-04 foundation, KE-08 schema design
   - Avoids: Building extraction before having a target schema (Pitfall 6)

2. **Existing File Extraction** - Extract structured knowledge from 28 existing research files (no new API calls)
   - Addresses: KE-04 (initial graph), builds canonical concept registry
   - Avoids: Concept explosion (Pitfall 1) by establishing canonicals before video extraction

3. **FTT Video Extraction** - Complete FTT channel coverage (~85 remaining videos)
   - Addresses: KE-01
   - Avoids: Rate limit issues (Pitfall 2) via checkpointing and Flash model

4. **Graph Assembly & Diagnostic Chains** - Build full knowledge graph, assemble diagnostic chains
   - Addresses: KE-04, KE-07, KE-08
   - Avoids: Undirected graph issues (Pitfall 6) by using typed directed edges

5. **Output Generation** - JSON + Markdown dual output, VLM prompt upgrade
   - Addresses: KE-08, KE-09, KE-11
   - Avoids: Prompt overflow (Pitfall 3) via query-based subgraph injection

6. **Secondary Sources** - TomAllsopp, Feel Tennis extraction + cross-source reconciliation
   - Addresses: KE-02, KE-03, KE-06
   - Avoids: Premature reconciliation (Pitfall 5) by having stable FTT graph first

7. **Deep Layers** - Anatomical mapping, personal journey integration
   - Addresses: KE-05, KE-10, KE-12

**Phase ordering rationale:**
- Phase 2 before Phase 3: Existing files are free to process and establish the canonical concept registry needed for video extraction
- Phase 3 before Phase 6: FTT must be 100% complete before secondary sources (per PROJECT.md constraint)
- Phase 4 before Phase 5: Graph must exist before outputs can be generated from it
- Phase 6 before Phase 7: Secondary sources add breadth; deep layers add depth. Breadth first validates the graph architecture.

**Research flags for phases:**
- Phase 3: May need deeper research on Gemini API rate limits and proxy compatibility with native SDK
- Phase 4: Deduplication algorithm design needs prototyping -- LLM-assisted merge is the riskiest technical component
- Phase 5: VLM prompt budget requires experimentation to find optimal subgraph size

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All libraries verified via official docs and PyPI. google-genai SDK YouTube URL support confirmed in official docs. |
| Features | HIGH | Directly derived from PROJECT.md requirements (KE-01 through KE-12). |
| Architecture | MEDIUM | Pipeline structure is standard, but deduplication approach (canonical registry + LLM merge) needs prototyping. |
| Pitfalls | HIGH | Based on analysis of existing codebase patterns and known LLM extraction challenges. |

## Gaps to Address

- **Gemini API proxy compatibility:** The project uses openclaudecode.cn as a proxy. Need to verify whether the native google-genai SDK can work through this proxy or if a separate API key for direct Gemini access is needed.
- **Deduplication algorithm specifics:** The canonical registry concept is clear, but the exact fuzzy matching / LLM-assisted merge implementation needs prototyping in Phase 2.
- **Instructor + Gemini integration:** instructor supports Gemini, but compatibility with the proxy endpoint needs testing.
- **Video analysis cost estimation:** Need to run a small batch (5-10 videos) to establish actual per-video cost and time benchmarks before committing to full channel extraction.
