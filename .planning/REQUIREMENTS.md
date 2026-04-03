# Requirements: Tennis Forehand Knowledge Engineering & VLM Diagnostic System

**Defined:** 2026-04-03
**Core Value:** VLM diagnostic engine must trace any visible forehand flaw back to its biomechanical root cause through multi-round iterative analysis -- observe, think, re-observe, correct, confirm -- producing an "Aha moment" where the user realizes the single root cause behind all their surface symptoms.

## v1 Requirements

### Knowledge Infrastructure

- [x] **INFRA-01**: Define Pydantic schema for concepts (id, name, aliases, source, type, description, muscles, VLM features)
- [x] **INFRA-02**: Define Pydantic schema for edges (source->target, type: causes/fixes/requires/contradicts, confidence, evidence)
- [x] **INFRA-03**: Define Pydantic schema for diagnostic chains (symptom->root causes->drills, with branching logic)
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
- [x] **FTT-02**: Identify already-analyzed videos (~30) and extract structured concepts from existing analyses
- [x] **FTT-03**: Analyze remaining ~85 FTT videos via Gemini API with structured extraction
- [x] **FTT-04**: Merge FTT video concepts into canonical registry (deduplication)
- [x] **FTT-05**: Extract diagnostic chains from FTT content (symptom->cause->fix patterns)

### Secondary Source Extraction

- [x] **SEC-01**: Enumerate TomAllsopp channel videos, select technique-relevant subset
- [x] **SEC-02**: Analyze selected TomAllsopp videos via Gemini API with structured extraction
- [x] **SEC-03**: Enumerate Feel Tennis channel free videos, select technique-relevant subset
- [x] **SEC-04**: Analyze selected Feel Tennis videos via Gemini API with structured extraction
- [x] **SEC-05**: Cross-source reconciliation -- resolve conflicts (FTT wins), mark agreements as reinforced

### Knowledge Graph Assembly

- [x] **GRAPH-01**: Complete concept graph with all sources merged, deduplicated, and connected
- [x] **GRAPH-02**: Causal edge chains validated (no orphan nodes, no cycles in diagnostic paths)
- [x] **GRAPH-03**: Every diagnostic chain has: entry symptom -> branching logic -> root cause(s) -> drill(s) -> check criteria
- [x] **GRAPH-04**: Cross-source confidence scoring (FTT-only=high, multi-source agreement=very high, single secondary=medium)

### Anatomical Layer

- [x] **ANAT-01**: Map each canonical concept to involved muscles (from biomechanics book)
- [x] **ANAT-02**: Map each muscle to: function, training methods, common failures
- [x] **ANAT-03**: Connect anatomical data to VLM-detectable features (what you can see in video)
- [x] **ANAT-04**: Build "why" explanations: concept -> muscle -> physics -> visible symptom

### VLM Diagnostic Engine (v1)

- [x] **VLM-01**: Build prompt generator that compiles VLM prompt from knowledge graph subgraph
- [x] **VLM-02**: Implement two-pass VLM analysis (quick scan -> targeted deep analysis with relevant subgraph)
- [x] **VLM-03**: Complete diagnostic coverage -- every known forehand symptom has a diagnostic chain
- [x] **VLM-04**: Each diagnostic output includes: what's wrong, why (biomechanics), how to fix (drill), how to check (criteria)
- [x] **VLM-05**: VLM prompt stays within ~10K char budget via query-based subgraph injection

### Output Generation

- [x] **OUT-01**: JSON knowledge graph export (nodes + edges + diagnostic chains, machine-readable)
- [x] **OUT-02**: Markdown knowledge base export (organized by topic hierarchy, human-readable)
- [x] **OUT-03**: Markdown includes cross-references, source citations, confidence levels
- [x] **OUT-04**: VLM prompt template file (generated from graph, replaces hardcoded prompt in vlm_analyzer.py)

### User Integration

- [x] **USER-01**: Connect user training records (learning.md) to knowledge graph concepts
- [x] **USER-02**: Personalized diagnostics -- VLM considers user's known issues and progress
- [x] **USER-03**: Training plan generation based on current knowledge gaps and breakthroughs

## v2 Requirements: Multi-Round VLM Diagnostic System

### Multi-Round Loop Infrastructure

- [ ] **MR-01**: Define Pydantic models for Hypothesis, Observation, DiagnosticSession, and RoundResult
- [ ] **MR-02**: Implement `MultiRoundAnalyzer` class that orchestrates the observe-reason-re-observe loop
- [ ] **MR-03**: Implement convergence detection: exit loop when dominant hypothesis reaches confidence >= 0.8 or max rounds (4) reached
- [ ] **MR-04**: Implement `analyze_swing_iterative()` in VLMForehandAnalyzer as the v2.0 entry point, with fallback to v1.0 `analyze_swing()`
- [ ] **MR-05**: Round history persistence: save full DiagnosticSession as JSON for debugging and replay

### Knowledge-Driven Observation Directives

- [ ] **KD-01**: New VLMPromptCompiler method `compile_observation_directive()` that generates targeted observation prompts from active hypotheses + unchecked diagnostic steps
- [ ] **KD-02**: New Jinja2 template `observation_directive.j2` for per-round targeted observation prompts (hypothesis context + specific frame/feature questions)
- [ ] **KD-03**: New Jinja2 template `confirmation.j2` for the final confirmation round (all evidence summary + root_cause_tree generation)
- [ ] **KD-04**: Directive generation maps `DiagnosticStep.check` + `Concept.vlm_features` + `DiagnosticChain.vlm_frame` into specific VLM questions

### Hypothesis Tracking

- [ ] **HT-01**: Hypothesis lifecycle: created (from Pass 1 detected chains) -> active -> confirmed/eliminated (via VLM observations)
- [ ] **HT-02**: Confidence scoring: each observation that supports/contradicts a hypothesis adjusts its confidence
- [ ] **HT-03**: Cross-hypothesis reasoning via KnowledgeGraph.get_causal_chain(): if confirmed hypothesis A causes hypothesis B in the graph, auto-eliminate B as independent root cause and mark as downstream symptom
- [ ] **HT-04**: Progressive narrowing guarantee: each round must either confirm or eliminate at least one hypothesis (enforced by the observation directive design, not by hard constraint)

### Hallucination Mitigation

- [ ] **HM-01**: Observation anchoring: every VLM observation must reference a specific frame number and visual feature; unanchored observations are flagged
- [ ] **HM-02**: Contradiction detection: if VLM contradicts itself across rounds on the same visual feature, trigger re-observation with more specific prompt
- [ ] **HM-03**: Quantitative cross-validation: compare VLM observations against YOLO-based kinematic data (elbow angle, wrist trajectory) where available; override VLM on clear conflict
- [ ] **HM-04**: Confidence-weighted observations: low-confidence (< 0.5) observations trigger re-observation in next round

### Report Generation

- [ ] **RG-01**: New report section "diagnostic journey" showing the iterative reasoning process (hypotheses tested, evidence found, eliminations)
- [ ] **RG-02**: Diagnostic journey renders as a narrative, not a log dump (readable by a non-technical tennis player)
- [ ] **RG-03**: Existing root_cause_tree format preserved in final output -- multi-round produces the same JSON schema as v1.0 for backward compatibility
- [ ] **RG-04**: Optional: diagnostic journey section can be hidden/collapsed for users who only want the result

### Cost and Performance

- [ ] **CP-01**: Multi-round analysis stays under $0.01 per swing on Gemini Flash
- [ ] **CP-02**: Total latency under 20 seconds for typical 3-round analysis
- [ ] **CP-03**: Focused per-round prompts stay under 4K chars each (except final confirmation at ~6K)

## Future Requirements (v3+)

### Product Expansion

- **PROD-01**: Multi-user support with individual training profiles
- **PROD-02**: Web interface for video upload and report viewing
- **PROD-03**: Multi-stroke analysis (serve, backhand, volley)
- **PROD-04**: Real-time feedback during practice sessions
- **PROD-05**: Social features -- share progress, compare with peers

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
| LiveAPI multi-turn sessions | Stateless API calls are simpler and avoid token accumulation costs |
| Video re-cropping between rounds | Same keyframe grid reused; prompt focus changes, not image |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| INFRA-01 through INFRA-05 | Phase 1 | Complete |
| EXIST-01 through EXIST-05 | Phase 2 | Complete |
| FTT-01 through FTT-05 | Phase 3 | Complete |
| SEC-01 through SEC-05 | Phase 6 | Complete |
| GRAPH-01 through GRAPH-04 | Phase 4 | Complete |
| ANAT-01 through ANAT-04 | Phase 4 | Complete |
| VLM-01 through VLM-05 | Phase 5 | Complete |
| OUT-01 through OUT-04 | Phase 5 | Complete |
| USER-01 through USER-03 | Phase 7 | Complete |
| MR-01 through MR-05 | Phase 8 | Pending |
| KD-01 through KD-04 | Phase 9 | Pending |
| HT-01 through HT-04 | Phase 9 | Pending |
| HM-01 through HM-04 | Phase 10 | Pending |
| RG-01 through RG-04 | Phase 10 | Pending |
| CP-01 through CP-03 | Phase 8-10 | Pending |

**Coverage:**
- v1 requirements: 38 total, all complete
- v2 requirements: 21 total, all pending
- Mapped to phases: 59
- Unmapped: 0

---
*Requirements defined: 2026-04-03*
*Last updated: 2026-04-03 after v2.0 milestone definition*
