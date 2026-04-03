# Tennis Forehand Knowledge Engineering & VLM Diagnostic System

## What This Is

A comprehensive knowledge engineering system that extracts, structures, and operationalizes tennis forehand technique knowledge from three expert YouTube channels (FTT, TomAllsopp, Feel Tennis) and a biomechanics textbook. The system produces a VLM diagnostic engine (symptom→root cause→fix) backed by a networked knowledge graph, serving both personal training improvement and future productization.

## Core Value

**The VLM diagnostic engine must trace any visible forehand flaw back to its biomechanical root cause through multi-round iterative analysis — observe, think, re-observe, correct, confirm — producing an "Aha moment" where the user realizes the single root cause behind all their surface symptoms.**

## Current Milestone: v2.0 Multi-Round VLM Diagnostic System

**Goal:** Upgrade from single-pass VLM analysis to a multi-round iterative diagnostic loop that uses the knowledge graph to guide what to observe, reasons about observations, and re-observes to verify root causes.

**Target features:**
- Knowledge-driven observation directives (VLM prompts generated from knowledge graph specify WHAT to look for)
- Multi-round iterative analysis (observe → think with knowledge graph → re-observe → correct → confirm)
- Progressive root cause narrowing (each round eliminates hypotheses, not just describes symptoms)
- Observation-specific frame requests (after first pass, system requests specific frames/angles to verify hypotheses)

## Requirements

### Validated

- ✓ FTT book (152 pages) fully extracted and synthesized — existing (`docs/research/`)
- ✓ FTT blog (~80 articles) extracted — existing (`docs/research/`)
- ✓ ~30 FTT YouTube videos analyzed via Gemini API — existing (`docs/research/`)
- ✓ Revolutionary Tennis 9 PDFs extracted — existing
- ✓ YouTube learning notes organized — existing
- ✓ Basic VLM prompt with 16 principles + 7 symptom groups — existing (`evaluation/vlm_analyzer.py`)
- ✓ Biomechanics book: 24 anatomy files extracted — existing
- ✓ Comprehensive synthesis document — existing (`docs/research/synthesis.md`)

### Active

- [ ] **KE-01**: Extract ALL remaining FTT YouTube videos (complete channel coverage, ~115 videos total)
- [ ] **KE-02**: Extract TomAllsopp channel videos (selective, technique-relevant content)
- [ ] **KE-03**: Extract Feel Tennis channel videos (selective, free content only)
- [ ] **KE-04**: Build networked knowledge graph (concepts as nodes, causal/supporting/contradicting edges)
- [ ] **KE-05**: Deep anatomical layer — each technique concept mapped to: muscles involved → training methods → common failures → VLM-detectable features
- [ ] **KE-06**: Cross-source reconciliation — resolve conflicts between FTT/TomAllsopp/Feel Tennis (FTT wins on conflicts)
- [ ] **KE-07**: VLM diagnostic decision tree — complete symptom→root cause→drill chains for all known forehand issues
- [ ] **KE-08**: JSON knowledge graph output (machine-readable, for VLM prompt injection)
- [ ] **KE-09**: Markdown knowledge base output (human-readable, for learning and coaching reference)
- [ ] **KE-10**: Integrate biomechanics book anatomy layer into knowledge graph (muscle→concept→VLM feature mapping)
- [ ] **KE-11**: Upgrade VLM prompt system with full diagnostic engine (replace current 7 symptom groups with complete graph-backed diagnostics)
- [ ] **KE-12**: User training journey integration — connect knowledge to personal training records for personalized diagnostics

### Out of Scope

- Native mobile app — future productization milestone
- Multi-stroke analysis (serve, backhand, volley) — forehand only for now
- Real-time video analysis — batch processing only
- Feel Tennis paid/premium content — free videos only
- Commercial licensing or distribution of extracted content

## Context

### Technical Environment
- Python codebase with YOLO pose detection + KPI scoring
- Gemini API for video analysis and VLM diagnostics
- 60fps video input (Blackmagic Cam → 960×540 compression)
- Existing `docs/research/` with 21+ research files (~200KB)
- Existing `evaluation/vlm_analyzer.py` with diagnostic prompt

### Knowledge Sources (Priority Order)
1. **FTT (Fault Tolerant Tennis)** — Primary authority. Book + blog + ALL videos. On conflicts, FTT wins.
2. **TomAllsopp (TPA Tennis)** — Secondary. Kinetic chain emphasis, pronation/separation concepts.
3. **Feel Tennis** — Secondary (same priority as TomAllsopp). Practical drills and feel-based teaching.
4. **《网球运动系统训练》** — Anatomical/biomechanical layer. 211 pages, 24 files already extracted.

### Current Knowledge Architecture
- Existing concept network: `docs/research/concept_network.json`
- Existing diagnostic engine: `docs/research/diagnostic_engine.json`
- User training journey: `docs/record/learning.md` (3/15 to 4/2, extensive)
- Conversation recovery: `docs/record/conversation_recovery_0402.md`

### Product Vision
- **Phase 1 (current)**: Personal training tool — knowledge system serves user's own improvement
- **Phase 2 (future)**: Productization — VLM analysis as a service for other tennis learners

## Constraints

- **API**: Gemini API via proxy (packyapi.com primary, openclaudecode.cn backup). Subject to rate limits and occasional downtime.
- **Content**: Feel Tennis paid content excluded. Only freely available videos.
- **Priority**: FTT content must be 100% complete before moving to other channels.
- **Language**: Knowledge base in Chinese (user's preference), with English technical terms preserved.
- **Token Budget**: VLM prompt has practical size limit (~10K chars). Diagnostic engine must be efficient.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| VLM diagnostic engine as primary output, knowledge graph as support | User needs actionable diagnostics, not just reference material | — Pending |
| FTT > TomAllsopp = Feel Tennis priority | FTT is most comprehensive and trusted; others supplement | — Pending |
| Deep anatomical layer (muscle→concept→VLM) | Enables "why" explanations and targeted training prescriptions | — Pending |
| Dual output format (JSON + Markdown) | JSON for machine consumption (VLM), Markdown for human learning | — Pending |
| Personal-first, productize later | Validate on single user before scaling | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd:transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd:complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-03 after initialization*
