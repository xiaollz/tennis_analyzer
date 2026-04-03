# Requirements: Tennis Forehand Knowledge Engineering & VLM Diagnostic System

**Defined:** 2026-04-03
**Core Value:** VLM diagnostic engine must trace any visible forehand flaw back to its biomechanical root cause and prescribe the correct drill.

## v1 Requirements

### Knowledge Infrastructure

- [x] **INFRA-01**: Define Pydantic schema for concepts (id, name, aliases, source, type, description, muscles, VLM features)
- [x] **INFRA-02**: Define Pydantic schema for edges (source→target, type: causes/fixes/requires/contradicts, confidence, evidence)
- [x] **INFRA-03**: Define Pydantic schema for diagnostic chains (symptom→root causes→drills, with branching logic)
- [x] **INFRA-04**: Build canonical concept registry with deduplication (fuzzy match + LLM-assisted merge)
- [x] **INFRA-05**: Set up NetworkX directed multigraph as knowledge graph backend

### Existing Knowledge Extraction

- [x] **EXIST-01**: Extract structured concepts from all 21+ existing research files in docs/research/
- [x] **EXIST-02**: Extract structured concepts from docs/record/learning.md training journey
- [x] **EXIST-03**: Extract structured concepts from 24 biomechanics book anatomy files
- [x] **EXIST-04**: Populate canonical concept registry (~200-300 concepts) from existing files
- [x] **EXIST-05**: Build initial knowledge graph edges from existing cross-references

### FTT Video Extraction

- [x] **FTT-01**: Enumerate all FTT YouTube channel videos (complete list with titles, URLs, durations)
- [ ] **FTT-02**: Identify already-analyzed videos (~30) and extract structured concepts from existing analyses
- [x] **FTT-03**: Analyze remaining ~85 FTT videos via Gemini API with structured extraction
- [x] **FTT-04**: Merge FTT video concepts into canonical registry (deduplication)
- [x] **FTT-05**: Extract diagnostic chains from FTT content (symptom→cause→fix patterns)

### Secondary Source Extraction

- [x] **SEC-01**: Enumerate TomAllsopp channel videos, select technique-relevant subset
- [x] **SEC-02**: Analyze selected TomAllsopp videos via Gemini API with structured extraction
- [x] **SEC-03**: Enumerate Feel Tennis channel free videos, select technique-relevant subset
- [x] **SEC-04**: Analyze selected Feel Tennis videos via Gemini API with structured extraction
- [x] **SEC-05**: Cross-source reconciliation — resolve conflicts (FTT wins), mark agreements as reinforced

### Knowledge Graph Assembly

- [x] **GRAPH-01**: Complete concept graph with all sources merged, deduplicated, and connected
- [x] **GRAPH-02**: Causal edge chains validated (no orphan nodes, no cycles in diagnostic paths)
- [x] **GRAPH-03**: Every diagnostic chain has: entry symptom → branching logic → root cause(s) → drill(s) → check criteria
- [x] **GRAPH-04**: Cross-source confidence scoring (FTT-only=high, multi-source agreement=very high, single secondary=medium)

### Anatomical Layer

- [x] **ANAT-01**: Map each canonical concept to involved muscles (from biomechanics book)
- [x] **ANAT-02**: Map each muscle to: function, training methods, common failures
- [x] **ANAT-03**: Connect anatomical data to VLM-detectable features (what you can see in video)
- [x] **ANAT-04**: Build "why" explanations: concept → muscle → physics → visible symptom

### VLM Diagnostic Engine

- [x] **VLM-01**: Build prompt generator that compiles VLM prompt from knowledge graph subgraph
- [x] **VLM-02**: Implement two-pass VLM analysis (quick scan → targeted deep analysis with relevant subgraph)
- [x] **VLM-03**: Complete diagnostic coverage — every known forehand symptom has a diagnostic chain
- [x] **VLM-04**: Each diagnostic output includes: what's wrong, why (biomechanics), how to fix (drill), how to check (criteria)
- [x] **VLM-05**: VLM prompt stays within ~10K char budget via query-based subgraph injection

### Output Generation

- [x] **OUT-01**: JSON knowledge graph export (nodes + edges + diagnostic chains, machine-readable)
- [x] **OUT-02**: Markdown knowledge base export (organized by topic hierarchy, human-readable)
- [x] **OUT-03**: Markdown includes cross-references, source citations, confidence levels
- [x] **OUT-04**: VLM prompt template file (generated from graph, replaces hardcoded prompt in vlm_analyzer.py)

### User Integration

- [x] **USER-01**: Connect user training records (learning.md) to knowledge graph concepts
- [ ] **USER-02**: Personalized diagnostics — VLM considers user's known issues and progress
- [ ] **USER-03**: Training plan generation based on current knowledge gaps and breakthroughs

## v2 Requirements

### Product Expansion

- **PROD-01**: Multi-user support with individual training profiles
- **PROD-02**: Web interface for video upload and report viewing
- **PROD-03**: Multi-stroke analysis (serve, backhand, volley)
- **PROD-04**: Real-time feedback during practice sessions
- **PROD-05**: Social features — share progress, compare with peers

### Knowledge Maintenance

- **MAINT-01**: Automated new video detection and extraction pipeline
- **MAINT-02**: Knowledge graph versioning and change tracking
- **MAINT-03**: A/B testing of diagnostic prompt variants

## Out of Scope

| Feature | Reason |
|---------|--------|
| Native mobile app | Web/CLI sufficient for personal use; future productization |
| Neo4j/database backend | NetworkX+JSON sufficient at current scale (~2000 nodes) |
| Feel Tennis paid content | Respect paywall; free content only |
| VLM model fine-tuning | Prompt engineering sufficient; fine-tuning is premature |
| Real-time video analysis | Batch processing meets current needs |
| Multi-stroke coverage | Forehand-only until knowledge system is validated |
| Commercial content redistribution | Legal/ethical boundary |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| INFRA-01 | Phase 1 | Complete |
| INFRA-02 | Phase 1 | Complete |
| INFRA-03 | Phase 1 | Complete |
| INFRA-04 | Phase 1 | Complete |
| INFRA-05 | Phase 1 | Complete |
| EXIST-01 | Phase 2 | Complete |
| EXIST-02 | Phase 2 | Complete |
| EXIST-03 | Phase 2 | Complete |
| EXIST-04 | Phase 2 | Complete |
| EXIST-05 | Phase 2 | Complete |
| FTT-01 | Phase 3 | Complete |
| FTT-02 | Phase 3 | Pending |
| FTT-03 | Phase 3 | Complete |
| FTT-04 | Phase 3 | Complete |
| FTT-05 | Phase 3 | Complete |
| GRAPH-01 | Phase 4 | Complete |
| GRAPH-02 | Phase 4 | Complete |
| GRAPH-03 | Phase 4 | Complete |
| GRAPH-04 | Phase 4 | Complete |
| ANAT-01 | Phase 4 | Complete |
| ANAT-02 | Phase 4 | Complete |
| ANAT-03 | Phase 4 | Complete |
| ANAT-04 | Phase 4 | Complete |
| VLM-01 | Phase 5 | Complete |
| VLM-02 | Phase 5 | Complete |
| VLM-03 | Phase 5 | Complete |
| VLM-04 | Phase 5 | Complete |
| VLM-05 | Phase 5 | Complete |
| OUT-01 | Phase 5 | Complete |
| OUT-02 | Phase 5 | Complete |
| OUT-03 | Phase 5 | Complete |
| OUT-04 | Phase 5 | Complete |
| SEC-01 | Phase 6 | Complete |
| SEC-02 | Phase 6 | Complete |
| SEC-03 | Phase 6 | Complete |
| SEC-04 | Phase 6 | Complete |
| SEC-05 | Phase 6 | Complete |
| USER-01 | Phase 7 | Complete |
| USER-02 | Phase 7 | Pending |
| USER-03 | Phase 7 | Pending |

**Coverage:**
- v1 requirements: 38 total
- Mapped to phases: 38
- Unmapped: 0 ✓

---
*Requirements defined: 2026-04-03*
*Last updated: 2026-04-03 after initial definition*
