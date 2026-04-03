# Feature Landscape

**Domain:** Tennis Forehand Knowledge Engineering & VLM Diagnostic System
**Researched:** 2026-04-03

## Table Stakes

Features that must exist for the system to deliver its core value.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Full FTT channel video extraction | FTT is primary authority; incomplete coverage = incomplete knowledge | Med | ~115 videos. Gemini YouTube URL pass-through makes this tractable. |
| Structured concept extraction per video | Raw text analysis already exists; structured schemas are the upgrade | High | Pydantic models + instructor. Most complex new code. |
| Concept deduplication and merging | Same concept appears across 50+ videos | Med | Fuzzy matching + LLM-assisted merge. Critical for graph quality. |
| Causal relationship extraction | "Why" is the core value prop -- symptom causes, biomechanical chains | High | Must extract directional relationships, not just co-occurrence. |
| Diagnostic chain assembly | Symptom to root cause to drill is the product | High | Graph traversal over extracted relationships. |
| JSON knowledge graph output | Machine-readable for VLM prompt injection (KE-08) | Low | NetworkX serialization to JSON. Straightforward. |
| Markdown knowledge base output | Human-readable reference (KE-09) | Med | Jinja2 templates. Multiple page types (concept, diagnostic, drill). |
| VLM prompt upgrade | Replace 7 symptom groups with graph-backed diagnostics (KE-11) | Med | Query graph for relevant diagnostic chains, format into prompt. |

## Differentiators

Features that make this system uniquely valuable vs. generic tennis analysis.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Cross-source reconciliation | FTT/TomAllsopp/Feel Tennis teach differently; reconciliation surfaces the truth | High | "contradicts" edges in graph. FTT wins on conflicts. |
| Anatomical layer mapping | Each concept linked to muscles, training methods, failure modes | High | Requires biomechanics book integration (KE-10). |
| VLM-detectable feature tagging | Each concept tagged with what a camera can actually see | Med | Bridges knowledge graph to practical VLM analysis. |
| Personal training journey integration | Connect knowledge to user's learning.md for personalized diagnostics | Med | KE-12. Query graph filtered by user's current issues. |
| Drill recommendation engine | Not just "what's wrong" but "how to fix it" with specific drills | Med | Graph traversal: symptom to root cause to drills_for edges. |
| Confidence-weighted relationships | Not all knowledge is equally certain | Low | Float confidence on each edge. Enables "high confidence" vs "speculative" paths. |

## Anti-Features

Features to explicitly NOT build.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Real-time video analysis pipeline | Batch is sufficient; real-time adds latency/infrastructure complexity | Process recorded videos, cache results |
| Multi-stroke knowledge (serve, backhand, volley) | Scope creep. Forehand knowledge alone is 200+ concepts | Keep architecture extensible but build forehand only |
| Custom LLM fine-tuning | Knowledge graph + prompt injection achieves same goal without training cost | Inject relevant subgraph into VLM prompt at query time |
| Web UI for knowledge browsing | Markdown files + JSON are sufficient for Phase 1 | Productization milestone can add UI later |
| Automated video discovery (new uploads) | Manual curation is fine for 3 channels | Batch re-run periodically |
| Natural language graph queries | Overkill for single-user system | Direct Python API for graph queries |

## Feature Dependencies

```
Channel enumeration (yt-dlp)
  -> Video analysis (Gemini)
    -> Structured extraction (instructor/pydantic)
      -> Concept deduplication and merging
        -> Relationship extraction
          -> Knowledge graph assembly (NetworkX)
            -> Diagnostic chain assembly
              -> JSON output (KE-08)
              -> Markdown output (KE-09)
              -> VLM prompt upgrade (KE-11)

Existing research files (docs/research/*.md)
  -> Structured extraction (same pipeline, text input instead of video)
    -> Same downstream flow

Biomechanics book (24 extracted files)
  -> Anatomical layer extraction
    -> Merge into knowledge graph (KE-10)

User training journey (learning.md)
  -> Personal diagnostic filtering (KE-12)
```

## MVP Recommendation

Prioritize:
1. **Structured extraction from existing research files** -- 28 files already analyzed, extract concepts/relationships without new API calls
2. **FTT remaining video extraction** -- complete channel coverage (KE-01)
3. **Knowledge graph assembly** -- merge extracted concepts, build graph (KE-04)
4. **Diagnostic chain assembly** -- the core value output (KE-07)
5. **VLM prompt upgrade** -- deliver the value to the existing analysis tool (KE-11)

Defer:
- **TomAllsopp/Feel Tennis extraction** (KE-02, KE-03): Until FTT is 100% complete and graph architecture is validated
- **Anatomical layer** (KE-10): Valuable but not blocking core diagnostic chains
- **Personal journey integration** (KE-12): Requires stable graph first

## Sources

- Project requirements from PROJECT.md
- Existing codebase analysis (evaluation/vlm_analyzer.py, docs/research/)
- [Gemini video understanding capabilities](https://ai.google.dev/gemini-api/docs/video-understanding)
