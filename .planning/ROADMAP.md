# Roadmap: Tennis Forehand Knowledge Engineering & VLM Diagnostic System

## Overview

This roadmap transforms ~200KB of existing research text and ~115 YouTube videos into a structured knowledge graph powering a VLM diagnostic engine. The journey starts with schema design (Phase 1), bootstraps the canonical concept registry from existing files (Phase 2), achieves complete FTT channel coverage (Phase 3), assembles the full knowledge graph with anatomical depth (Phase 4), generates machine and human outputs including the upgraded VLM engine (Phase 5), expands to secondary sources (Phase 6), and finally integrates the user's personal training journey (Phase 7).

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Schema & Infrastructure** - Define Pydantic models, pipeline scaffolding, and knowledge graph backend
- [ ] **Phase 2: Existing Knowledge Extraction** - Extract structured concepts from all existing research files into canonical registry
- [ ] **Phase 3: FTT Video Extraction** - Complete FTT YouTube channel coverage (~85 remaining videos)
- [ ] **Phase 4: Graph Assembly & Anatomical Layer** - Build full knowledge graph with diagnostic chains and muscle-level depth
- [ ] **Phase 5: Output Generation & VLM Engine** - Dual-format export and upgraded VLM diagnostic system
- [ ] **Phase 6: Secondary Sources** - TomAllsopp and Feel Tennis extraction with cross-source reconciliation
- [ ] **Phase 7: User Integration** - Personal training journey connected to knowledge graph for personalized diagnostics

## Phase Details

### Phase 1: Schema & Infrastructure
**Goal**: All data models and pipeline scaffolding exist so extraction phases have a target schema to write into
**Depends on**: Nothing (first phase)
**Requirements**: INFRA-01, INFRA-02, INFRA-03, INFRA-04, INFRA-05
**Success Criteria** (what must be TRUE):
  1. A Python script can instantiate a Concept, an Edge, and a DiagnosticChain from Pydantic models and serialize them to JSON
  2. The canonical concept registry can add a concept, detect a near-duplicate via fuzzy match, and return the canonical ID
  3. A NetworkX directed multigraph can be created, populated with test nodes/edges, and queried for a causal chain (A causes B causes C)
  4. Running `pytest` on the schema module passes with at least one test per model and one test for dedup matching
**Plans:** 2 plans
Plans:
- [x] 01-01-PLAN.md — Pydantic schemas (Concept, Edge, DiagnosticChain) + package scaffolding
- [x] 01-02-PLAN.md — ConceptRegistry with fuzzy dedup + KnowledgeGraph wrapper
**Risk flags**: Pitfall 1 (concept explosion) -- dedup strategy must be designed here, not bolted on later. Pitfall 6 (undirected edges) -- enforce typed directed edges at schema level.

### Phase 2: Existing Knowledge Extraction
**Goal**: All existing research files (28+ docs) are extracted into structured concepts, populating the canonical registry with ~200-300 concepts before any new API calls
**Depends on**: Phase 1
**Requirements**: EXIST-01, EXIST-02, EXIST-03, EXIST-04, EXIST-05
**Success Criteria** (what must be TRUE):
  1. Every file in docs/research/ has been processed and its concepts exist as JSON in the extraction output directory
  2. The canonical concept registry contains 150-300 unique concepts with no obvious duplicates (manual spot-check of 20 random pairs)
  3. The initial knowledge graph has nodes and edges from existing cross-references, and a simple query like "what causes elbow flying out" returns at least one causal chain
  4. Biomechanics book anatomy files (24 files) are extracted with muscle-to-concept mappings
**Plans:** 3 plans
Plans:
- [x] 02-01-PLAN.md — Seed registry from legacy JSON + extraction pipeline scaffolding
- [x] 02-02-PLAN.md — Extract concepts from all 31 research files
- [x] 02-03-PLAN.md — User journey extraction + knowledge graph assembly
**Risk flags**: Pitfall 1 (concept explosion) -- this phase establishes the canonical registry that all future extraction depends on. Pitfall 9 (Chinese-English mismatch) -- ensure bilingual concept names from the start.

### Phase 3: FTT Video Extraction
**Goal**: 100% FTT YouTube channel coverage -- every video analyzed, concepts extracted, and merged into canonical registry
**Depends on**: Phase 2
**Requirements**: FTT-01, FTT-02, FTT-03, FTT-04, FTT-05
**Success Criteria** (what must be TRUE):
  1. A complete FTT video inventory exists (title, URL, duration, processed status) and shows 100% coverage
  2. Already-analyzed videos (~30) have structured concepts extracted from existing Markdown analyses (no re-analysis needed)
  3. All remaining ~85 videos have been analyzed via Gemini API with raw analysis saved as Markdown and structured concepts as JSON
  4. FTT video concepts are merged into the canonical registry with deduplication (registry stays under ~500 concepts total)
  5. Diagnostic chains (symptom-to-cause-to-fix patterns) are extracted from FTT content
**Plans:** 2/6 plans executed
Plans:
- [x] 03-01-PLAN.md — Video inventory + pipeline infrastructure (state manager, analyzer, extractor)
- [x] 03-02-PLAN.md — Re-extract concepts from 33 existing video analyses (no API)
- [x] 03-03-PLAN.md — Gemini API batch 1: 12 forehand-priority videos
- [x] 03-04-PLAN.md — Gemini API batch 2: 13 philosophy/tactics/topspin videos
- [x] 03-05-PLAN.md — Gemini API batch 3: 14 remaining videos (movement/serve/backhand)
- [x] 03-06-PLAN.md — Merge all into registry + extract diagnostic chains
**Risk flags**: Pitfall 2 (API rate limits/cost) -- use Flash model, implement checkpointing, set daily budget caps. Pitfall 7 (losing raw analysis) -- always save raw Markdown before structured extraction. Pitfall 8 (non-deterministic extraction) -- temperature=0, pin model versions.

### Phase 4: Graph Assembly & Anatomical Layer
**Goal**: A complete, validated knowledge graph with diagnostic chains and deep anatomical mappings -- the foundation for all outputs
**Depends on**: Phase 3
**Requirements**: GRAPH-01, GRAPH-02, GRAPH-03, GRAPH-04, ANAT-01, ANAT-02, ANAT-03, ANAT-04
**Success Criteria** (what must be TRUE):
  1. The knowledge graph has all sources merged and deduplicated, with zero orphan nodes (every node has at least one edge)
  2. Diagnostic paths have no cycles -- traversing any symptom-to-root-cause chain terminates
  3. Every known forehand symptom group (current 7 + any new ones discovered) has a complete diagnostic chain: symptom -> branching logic -> root cause(s) -> drill(s) -> check criteria
  4. Cross-source confidence scores are assigned (FTT-only=high, multi-source=very high, single-secondary=medium)
  5. Each canonical concept has muscle mappings with function, training method, common failure, and VLM-detectable feature where applicable
**Plans:** 4 plans
Plans:
- [x] 04-01-PLAN.md — Edge assembly: sync 582 registry nodes + load 1665 edges with fuzzy resolution + confidence scoring
- [x] 04-02-PLAN.md — Anatomical extraction: muscle profiles from biomechanics Markdown files
- [x] 04-03-PLAN.md — Graph validation (cycles, orphans) + diagnostic chain generation (15-25 chains)
- [x] 04-04-PLAN.md — VLM feature annotation + muscle integration + "why" explanation chains
**Risk flags**: Pitfall 6 (undirected edges) -- validate all edges have type and direction. Pitfall 10 (no visualization) -- add simple graph visualization utility for debugging.

### Phase 5: Output Generation & VLM Engine
**Goal**: The knowledge graph produces both machine-readable (JSON) and human-readable (Markdown) outputs, and the VLM diagnostic engine is upgraded to use graph-backed diagnostics
**Depends on**: Phase 4
**Requirements**: VLM-01, VLM-02, VLM-03, VLM-04, VLM-05, OUT-01, OUT-02, OUT-03, OUT-04
**Success Criteria** (what must be TRUE):
  1. JSON export contains the full knowledge graph (nodes, edges, diagnostic chains) and can be loaded back into NetworkX
  2. Markdown export is organized by topic hierarchy with cross-references, source citations, and confidence levels
  3. VLM prompt generator compiles a prompt from a graph subgraph query, staying within ~10K char budget
  4. Two-pass VLM analysis works: quick scan identifies symptom category, then targeted analysis with relevant diagnostic chains produces a report with what/why/how-to-fix/check-criteria
  5. The generated VLM prompt template replaces the hardcoded prompt in vlm_analyzer.py
**Plans:** 3 plans
Plans:
- [x] 05-01-PLAN.md — JSON + Markdown export generators (Jinja2 templates, topic-grouped structure)
- [x] 05-02-PLAN.md — VLM prompt compiler (graph-to-prompt, static templates + dynamic subgraph injection)
- [x] 05-03-PLAN.md — Two-pass VLM integration into vlm_analyzer.py
**Risk flags**: Pitfall 3 (prompt overflow) -- two-pass VLM with subgraph selection is critical. Test prompt size early.

### Phase 6: Secondary Sources
**Goal**: TomAllsopp and Feel Tennis content extracted and reconciled against the FTT-primary knowledge graph, adding breadth without degrading coherence
**Depends on**: Phase 3, Phase 4
**Requirements**: SEC-01, SEC-02, SEC-03, SEC-04, SEC-05
**Success Criteria** (what must be TRUE):
  1. TomAllsopp technique-relevant videos are identified, analyzed, and concepts extracted into the canonical registry
  2. Feel Tennis free videos are identified, analyzed, and concepts extracted into the canonical registry
  3. Cross-source reconciliation is complete: conflicts resolved (FTT wins), agreements marked as reinforced, complements integrated
  4. Knowledge graph confidence scores updated to reflect multi-source validation
**Plans:** TBD
**Risk flags**: Pitfall 5 (reconciliation complexity) -- reconciliation is a separate explicit pass, not embedded in extraction. Pitfall 4 (transcript quality) -- use Gemini video understanding as primary, transcripts as supplementary.

### Phase 7: User Integration
**Goal**: The knowledge system connects to the user's personal training journey, enabling personalized diagnostics and training plan generation
**Depends on**: Phase 4, Phase 5
**Requirements**: USER-01, USER-02, USER-03
**Success Criteria** (what must be TRUE):
  1. User training records (learning.md) are linked to knowledge graph concepts -- each training entry maps to relevant concepts and progress status
  2. VLM diagnostic output considers the user's known issues and recent breakthroughs (personalized analysis, not generic)
  3. A training plan can be generated based on the user's current knowledge gaps and technique weaknesses, recommending specific drills from the knowledge graph
**Plans:** TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7

Note: Phase 6 depends on Phases 3+4; Phase 7 depends on Phases 4+5. Phases 5 and 6 could theoretically run in parallel after Phase 4, but sequential execution is simpler for a solo developer.

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Schema & Infrastructure | 2/2 | Complete | 2026-04-03 |
| 2. Existing Knowledge Extraction | 0/3 | Planning complete | - |
| 3. FTT Video Extraction | 2/6 | In Progress|  |
| 4. Graph Assembly & Anatomical Layer | 0/TBD | Not started | - |
| 5. Output Generation & VLM Engine | 0/3 | Planning complete | - |
| 6. Secondary Sources | 0/TBD | Not started | - |
| 7. User Integration | 0/TBD | Not started | - |
