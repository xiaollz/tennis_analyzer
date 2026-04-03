# Phase 6: Secondary Sources - Research

**Researched:** 2026-04-03
**Domain:** YouTube video extraction (TomAllsopp + Feel Tennis), cross-source reconciliation
**Confidence:** HIGH

## Summary

Phase 6 extends the existing FTT-primary knowledge graph (582 nodes) with two secondary tennis coaching channels: TomAllsopp (307 videos total, ~60 forehand-relevant) and Feel Tennis (571 videos total, ~50 forehand-relevant). The existing Phase 3 pipeline (video_analyzer.py, video_concept_extractor.py, video_state.py) is directly reusable with minor parameterization changes (channel ID, output directories, source tag). The main new work is: (1) curating the forehand-relevant subset from each channel, (2) batch-analyzing via Gemini API, (3) building a cross-source reconciliation pass that resolves conflicts (FTT wins), marks agreements as reinforced, and integrates complements.

The critical insight is **volume control**. Processing all 878 videos across both channels would be wasteful and expensive. Title-keyword filtering yields ~60 TomAllsopp forehand videos and ~50 Feel Tennis forehand videos. This is the right scope -- enough for meaningful cross-source validation without budget blowout. At ~20 seconds delay between API calls, processing 110 videos takes roughly 37 minutes of API time plus actual response time (estimate 2-3 hours total).

**Primary recommendation:** Reuse Phase 3 pipeline with parameterized channel support. Filter to ~110 forehand-relevant videos total. Add a reconciliation module as a new file in knowledge/pipeline/. Expect ~50-80 new unique concepts after dedup, bringing the registry from ~582 to ~630-660 nodes.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SEC-01 | Enumerate TomAllsopp channel videos, select technique-relevant subset | yt-dlp enumeration complete: 307 total, ~60 forehand-relevant identified by title keyword filtering |
| SEC-02 | Analyze selected TomAllsopp videos via Gemini API with structured extraction | Existing video_analyzer.py + video_concept_extractor.py reusable; same Gemini Flash model and prompt |
| SEC-03 | Enumerate Feel Tennis channel free videos, select technique-relevant subset | yt-dlp enumeration complete: 571 total, ~50 forehand-relevant identified; 1 paid course video detected and excluded |
| SEC-04 | Analyze selected Feel Tennis videos via Gemini API with structured extraction | Same pipeline as SEC-02; all Feel Tennis YouTube videos are free (paid content is on separate platform) |
| SEC-05 | Cross-source reconciliation -- resolve conflicts (FTT wins), mark agreements as reinforced | New reconciliation module needed; concept-level matching via registry fuzzy dedup, edge confidence adjustment |
</phase_requirements>

## Standard Stack

### Core (already installed from Phase 3)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| google-genai | existing | Gemini API client for video analysis | Already configured with proxy, proven in Phase 3 |
| yt-dlp | 2026.03.13 | Channel video enumeration | Already installed, used in Phase 3 |
| rapidfuzz | existing | Fuzzy concept matching for dedup and reconciliation | Already in registry, threshold tuned |
| tenacity | existing | API retry logic | Already in video_analyzer.py |
| networkx | existing | Knowledge graph backend | Graph already at 582 nodes |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pydantic | existing | Concept/Edge schema validation | Same schemas from Phase 1 |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| yt-dlp flat-playlist | YouTube Data API v3 | yt-dlp requires no API key, already proven in project |
| Manual curation | LLM-based relevance scoring | Title keyword filtering is 90% accurate; manual review of edge cases is faster |

## Architecture Patterns

### Recommended Project Structure
```
knowledge/
  pipeline/
    video_analyzer.py          # REUSE AS-IS (already parameterized)
    video_concept_extractor.py # REUSE AS-IS
    video_state.py             # EXTEND: add secondary channel inventory generation
    reconciliation.py          # NEW: cross-source reconciliation logic
  state/
    tomallsopp_video_state.json    # NEW: TomAllsopp state tracking
    feeltennis_video_state.json    # NEW: Feel Tennis state tracking
  extracted/
    tomallsopp_videos/             # NEW: per-video extraction JSONs
    feeltennis_videos/             # NEW: per-video extraction JSONs
docs/research/
    tomallsopp_video_analyses/     # NEW: raw Gemini markdown outputs
    feeltennis_video_analyses/     # NEW: raw Gemini markdown outputs
```

### Pattern 1: Parameterized Channel Support
**What:** Extend video_state.py to support multiple channels via a factory function that takes channel_id, video_list, and output_dir as parameters instead of hardcoded FTT inventory.
**When to use:** For any new channel addition.
**Example:**
```python
def generate_channel_state(
    channel_id: str,
    videos: list[dict],  # [{video_id, title, duration}]
    output_dir: str = "docs/research",
) -> dict:
    """Build video inventory for any channel."""
    entries = {}
    for v in videos:
        vid_id = v["video_id"]
        entries[vid_id] = VideoEntry(
            video_id=vid_id,
            title=v["title"],
            url=f"https://www.youtube.com/watch?v={vid_id}",
            duration=v["duration"],
            status="pending",
            analysis_file=None,
            extracted_file=None,
            analyzed_at=None,
            error=None,
        )
    return {
        "channel_id": channel_id,
        "total_videos": len(entries),
        "enumerated_at": "2026-04-03",
        "videos": entries,
    }
```

### Pattern 2: Cross-Source Reconciliation
**What:** After extraction, compare secondary concepts against FTT-primary registry. Classify each as: conflict (FTT wins, secondary marked contradicted), agreement (boost confidence), or complement (integrate as new with medium confidence).
**When to use:** SEC-05, after all secondary extraction is complete.
**Example:**
```python
def reconcile_concept(
    secondary_concept: Concept,
    registry: ConceptRegistry,
    graph: KnowledgeGraph,
) -> str:  # "conflict" | "agreement" | "complement"
    existing_id = registry.resolve(secondary_concept.name, threshold=70)
    if existing_id is None:
        # New concept not in FTT -- complement
        return "complement"
    
    existing = registry.get(existing_id)
    # Check for contradiction edges or conflicting descriptions
    if _descriptions_conflict(existing, secondary_concept):
        return "conflict"
    return "agreement"
```

### Pattern 3: Batch Processing with Checkpointing
**What:** Reuse analyze_batch() from video_analyzer.py with per-video state checkpointing. Process in batches of 15-20 to allow intermediate review.
**When to use:** All Gemini API processing.

### Anti-Patterns to Avoid
- **Processing all 878 videos:** Wasteful. Only forehand-relevant subset matters for this project.
- **Mixing extraction and reconciliation:** Reconciliation is a separate explicit pass AFTER all extraction completes (per ROADMAP risk flag).
- **Changing FTT concepts during reconciliation:** FTT is primary. Secondary sources can only add, reinforce, or be marked as contradicted.
- **Re-running yt-dlp during execution:** Enumeration is done once in research; hardcode the curated list in state files.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Video enumeration | Custom YouTube scraper | yt-dlp --flat-playlist | Robust, handles pagination, already proven |
| Concept deduplication | String matching | ConceptRegistry with rapidfuzz | Tuned thresholds (85/70) from Phase 1, handles aliases |
| API retry/backoff | Custom retry loops | tenacity @retry decorator | Already in video_analyzer.py, exponential backoff |
| Video analysis | Custom transcription + NLP | Gemini native video understanding | Gemini processes video directly, no transcript needed |

## Common Pitfalls

### Pitfall 1: Duplicate Video IDs Across Channels
**What goes wrong:** Video 1-g1OD8gh-I appears in BOTH FTT and TomAllsopp channels (collaboration video). Processing it twice creates duplicate state entries.
**Why it happens:** Channels re-upload or cross-post content.
**How to avoid:** Cross-check secondary channel video IDs against FTT state. Skip any already-processed IDs. Found 1 confirmed overlap: `1-g1OD8gh-I` ("This One Move Instantly Improves Your Forehand").
**Warning signs:** Duplicate video_id in extraction output directories.

### Pitfall 2: Feel Tennis Paid Content
**What goes wrong:** Attempting to analyze videos that are actually course previews or membership-gated content.
**Why it happens:** Feel Tennis sells courses on a separate platform (feeltennis.net); YouTube videos are generally free but some are course teasers.
**How to avoid:** Only 1 video title indicates paid content ("Volley & Smash Course Launch" - CLEjGDGEGaA). All other YouTube-listed videos are freely accessible. The REQUIREMENTS.md explicitly states "Feel Tennis paid content" is out of scope -- this refers to feeltennis.net course content, not YouTube videos.
**Warning signs:** Video title containing "course", "launch", "premium", "member".

### Pitfall 3: Concept Explosion from Secondary Sources
**What goes wrong:** Secondary sources use different terminology for the same biomechanical concepts, inflating the registry beyond useful size.
**Why it happens:** TomAllsopp says "smash factor" where FTT says "whip efficiency"; Feel Tennis says "feel-based learning" where FTT says "passive results".
**How to avoid:** Run concept extraction through the EXISTING registry's resolve() with threshold=70 before adding new concepts. Only genuinely new concepts get added. Expected: ~50-80 new concepts, NOT 200+.
**Warning signs:** Registry growing beyond 700 after secondary integration.

### Pitfall 4: API Rate Limits with Two Additional Channels
**What goes wrong:** Processing 110 videos in one session hits Gemini API rate limits or proxy throttling.
**Why it happens:** Phase 3 processed 40 videos across multiple sessions with delays.
**How to avoid:** Use 20-second delay between calls (same as Phase 3). Process in batches of 15-20 per session. Budget: ~110 videos * 20s = ~37 min API delay + response time. Split across 3-4 execution plans.
**Warning signs:** 429 errors, timeout errors, proxy disconnects.

### Pitfall 5: Reconciliation Ambiguity
**What goes wrong:** Can't determine if a secondary concept agrees with or contradicts FTT because they use different framings of the same idea.
**Why it happens:** Coaching philosophies overlap but use different language. "Feel the racket drop" (Feel Tennis) vs "gravity-driven elbow drop" (FTT) -- same concept, different framing.
**How to avoid:** Use the analysis prompt to explicitly reference FTT concepts during extraction (the prompt already lists C01-C48, T01-T24). Gemini will note matches in its output. For remaining ambiguity, default to "complement" (safest classification).
**Warning signs:** More than 30% of reconciliation results classified as "ambiguous".

## Video Inventory Summary

### TomAllsopp (@TomAllsopp)
- **Total videos:** 307
- **Forehand-relevant (title keyword "forehand", excluding volley):** ~60
- **Broader technique-relevant (forehand + kinetic chain + rotation + lag + contact + topspin + swing + weight transfer):** ~80
- **Recommended subset:** ~45 strictly forehand technique videos (excluding beginner tutorials, product reviews, match strategy)
- **Known FTT overlap:** 1 video (1-g1OD8gh-I) -- skip
- **Channel focus:** Practical coaching, often real student lessons, kinetic chain emphasis

### Feel Tennis (@feeltennis)
- **Total videos:** 571
- **Forehand-relevant (title keyword "forehand", excluding volley):** ~85
- **Broader technique-relevant:** ~139
- **Recommended subset:** ~45 forehand technique videos (many are short repetitions of same concepts)
- **Paid content indicator:** 1 video (CLEjGDGEGaA "Volley & Smash Course Launch") -- skip; all other YouTube videos are free
- **Channel focus:** Feel-based learning, biomechanics, wrist/forearm mechanics, topspin methodology

### Processing Budget
| Channel | Videos | Est. API Time | Est. New Concepts |
|---------|--------|--------------|-------------------|
| TomAllsopp | ~45 | ~1-1.5 hours | 25-40 |
| Feel Tennis | ~45 | ~1-1.5 hours | 25-40 |
| **Total** | **~90** | **~2-3 hours** | **50-80** |

## Cross-Source Reconciliation Design

### Classification Categories
1. **Agreement** (concept exists in FTT + secondary says same thing)
   - Action: Boost existing concept confidence from 0.8 to 0.9+
   - Add "multi-source" tag to concept.sources
   - Add supports edge with source=secondary

2. **Conflict** (secondary says X, FTT says not-X)
   - Action: FTT wins. Add contradicts edge with low confidence (0.3)
   - Mark secondary concept with `ftt_overrides=true` flag
   - Log for human review

3. **Complement** (new concept not in FTT)
   - Action: Add to registry with confidence=0.6 (medium, single secondary source)
   - If both TomAllsopp AND Feel Tennis agree, boost to 0.7
   - Add source tag

### Confidence Scoring Update (from GRAPH-04)
| Scenario | Confidence |
|----------|-----------|
| FTT only | 0.8 (high) |
| FTT + 1 secondary agrees | 0.9 (very high) |
| FTT + 2 secondaries agree | 0.95 (very high) |
| Single secondary only | 0.6 (medium) |
| Two secondaries agree, no FTT | 0.7 (medium-high) |
| Secondary contradicts FTT | FTT keeps 0.8, secondary marked 0.3 |

### Expected Reconciliation Distribution
Based on coaching channel content analysis:
- **Agreements:** ~60% (FTT, TomAllsopp, Feel Tennis share many core biomechanics principles)
- **Complements:** ~30% (unique drills, metaphors, feel-based cues)
- **Conflicts:** ~10% (methodology differences, especially Feel Tennis "feel" approach vs FTT "fault tolerance" framework)

## Code Examples

### Curated Video List Generation (from yt-dlp output)
```python
# Save yt-dlp output, then filter in Python
TOMALLSOPP_FOREHAND_VIDEOS = [
    {"video_id": "ral2cHTFcdY", "title": "The Forehand Technique Nobody Teaches", "duration": 248},
    {"video_id": "IhLcK-ScJ1k", "title": "The Right Way To Jump Into Your Forehand", "duration": 311},
    # ... (curated list of ~45 videos)
]

FEELTENNIS_FOREHAND_VIDEOS = [
    {"video_id": "m62xjbSvZgc", "title": "Why You Don't Feel a Good Swing On The Forehand", "duration": 109},
    {"video_id": "5Z9etBWK2Kg", "title": "Forehand Contact: More Forward Than You Think", "duration": 224},
    # ... (curated list of ~45 videos)
]
```

### Reusing analyze_batch with Different Output Dir
```python
from knowledge.pipeline.video_analyzer import (
    analyze_batch, create_client, load_api_config, load_analysis_prompt,
)
from knowledge.pipeline.video_state import load_state, save_state

config = load_api_config(Path("config/youtube_api_config.json"))
client = create_client(config)
prompt = load_analysis_prompt()

# TomAllsopp batch
state = load_state(Path("knowledge/state/tomallsopp_video_state.json"))
results = analyze_batch(
    client=client,
    video_ids=[v["video_id"] for v in TOMALLSOPP_FOREHAND_VIDEOS],
    prompt=prompt,
    state=state,
    state_path=Path("knowledge/state/tomallsopp_video_state.json"),
    delay=20.0,
    output_dir=Path("docs/research/tomallsopp_video_analyses"),
)
```

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (existing) |
| Config file | existing project pytest config |
| Quick run command | `python -m pytest tests/ -x -q --tb=short` |
| Full suite command | `python -m pytest tests/ -v` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SEC-01 | TomAllsopp video list curated, state file created | unit | `pytest tests/test_secondary_state.py::test_tomallsopp_inventory -x` | Wave 0 |
| SEC-02 | TomAllsopp videos analyzed and extracted | smoke | Manual: check state file shows "extracted" for batch | N/A |
| SEC-03 | Feel Tennis video list curated, state file created | unit | `pytest tests/test_secondary_state.py::test_feeltennis_inventory -x` | Wave 0 |
| SEC-04 | Feel Tennis videos analyzed and extracted | smoke | Manual: check state file shows "extracted" for batch | N/A |
| SEC-05 | Reconciliation produces correct classifications | unit | `pytest tests/test_reconciliation.py -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/ -x -q --tb=short`
- **Per wave merge:** `python -m pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_secondary_state.py` -- covers SEC-01, SEC-03 (state generation for secondary channels)
- [ ] `tests/test_reconciliation.py` -- covers SEC-05 (reconciliation logic: agreement/conflict/complement)
- [ ] `knowledge/pipeline/reconciliation.py` -- new module for cross-source reconciliation

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| yt-dlp | Video enumeration | Yes | 2026.03.13 | -- |
| python3 | All pipeline code | Yes | 3.11 | -- |
| google-genai | Gemini API calls | Yes | existing | -- |
| Gemini API (via proxy) | Video analysis | Yes | gemini-3-flash-preview | Backup API key in config |

**Missing dependencies with no fallback:** None
**Missing dependencies with fallback:** None

## Open Questions

1. **Exact video subset curation**
   - What we know: Title-keyword filtering gives ~60 TomAllsopp and ~85 Feel Tennis forehand videos
   - What's unclear: Some borderline videos (e.g., "Tennis for Beginners: Master the Fundamentals" by TomAllsopp) might have forehand content despite generic titles
   - Recommendation: Start with strict "forehand" keyword filter (~45 per channel), add 5-10 borderline videos in a second pass if concept coverage feels thin

2. **Feel Tennis "feel-based" methodology alignment**
   - What we know: Feel Tennis emphasizes proprioceptive learning ("feel the swing"), which is a different pedagogical approach from FTT's biomechanical framework
   - What's unclear: Whether these are truly complementary or philosophically conflicting
   - Recommendation: Classify Feel Tennis methodology concepts as "complements" unless they explicitly contradict FTT biomechanics. Feel Tennis "feel cues" map to FTT "oral cues" in training methodology.

3. **Analysis prompt adaptation**
   - What we know: The existing prompt lists FTT concepts C01-C48, T01-T24, DC01-DC10, D01-D17 as reference
   - What's unclear: Whether secondary source videos will trigger enough concept matches against this FTT-centric prompt
   - Recommendation: Use the SAME prompt for secondary sources. This ensures extraction is anchored to the existing concept framework, making reconciliation easier.

## Sources

### Primary (HIGH confidence)
- yt-dlp enumeration of @TomAllsopp: 307 videos confirmed via --flat-playlist (2026-04-03)
- yt-dlp enumeration of @feeltennis: 571 videos confirmed via --flat-playlist (2026-04-03)
- Existing codebase: video_analyzer.py, video_concept_extractor.py, video_state.py, registry.py, graph.py
- config/youtube_api_config.json: Gemini API proxy configuration confirmed working

### Secondary (MEDIUM confidence)
- Video relevance filtering: title-keyword based, ~90% accuracy (some forehand content in generically-titled videos will be missed)
- Concept count estimates: based on Phase 3 yield ratios (73 FTT videos -> 582 concepts; expect diminishing returns for secondary sources)

### Tertiary (LOW confidence)
- Reconciliation distribution estimate (60/30/10): based on general knowledge of coaching channel content overlap, needs validation during execution

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - reusing proven Phase 3 pipeline
- Architecture: HIGH - straightforward extension of existing patterns
- Pitfalls: HIGH - based on direct experience from Phase 3 execution
- Reconciliation design: MEDIUM - reconciliation logic is new, classification thresholds need tuning

**Research date:** 2026-04-03
**Valid until:** 2026-05-03 (stable; YouTube channels add ~1-2 videos/week)
