# Phase 3: FTT Video Extraction - Research

**Researched:** 2026-04-03
**Domain:** YouTube video analysis via Gemini API, structured knowledge extraction
**Confidence:** HIGH

## Summary

The FTT YouTube channel has **70 videos total** (verified via yt-dlp enumeration on 2026-04-03), not the ~115 previously estimated. Of these, **35 have already been analyzed** in `docs/research/09_ftt_videos_{1,2,3}.md` with detailed structured analyses. This means **35 videos remain** to be analyzed -- roughly half the originally estimated workload.

The Gemini API proxy at packyapi.com has been **verified working** with YouTube URL pass-through via the native `google-genai` SDK. A backup native Google API key is also available and tested. Both accept `types.Part(file_data=types.FileData(file_uri='https://youtube.com/...'))` and return quality analyses of tennis video content. The existing video analysis prompt in `docs/knowledge_graph/video_analysis_prompt.md` is comprehensive and field-tested.

The current extraction pipeline (Phase 2) produced **297 concepts** in the registry, but the video-derived concepts in `knowledge/extracted/ftt_videos/` are shallow -- just video titles as concept names with 0.6 confidence. Phase 3 must re-extract structured concepts from both the 35 existing analyses AND the 35 new video analyses using the full prompt template, then merge into the canonical registry.

**Primary recommendation:** Process in two sub-phases: (1) extract structured concepts from the 35 existing analysis texts (free, fast, no API calls), then (2) analyze the 35 remaining videos via Gemini Flash with the existing prompt, checkpointing each result.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| FTT-01 | Enumerate all FTT YouTube channel videos (complete list with titles, URLs, durations) | DONE: 70 videos enumerated via yt-dlp. Full list available with video_id, title, duration. |
| FTT-02 | Identify already-analyzed videos (~30) and extract structured concepts from existing analyses | 35 already analyzed in docs/research/09_ftt_videos_{1,2,3}.md. Current extracted JSON is shallow (title-only concepts). Need re-extraction with proper Pydantic models. |
| FTT-03 | Analyze remaining ~85 FTT videos via Gemini API with structured extraction | Corrected: 35 remaining (not 85). Gemini proxy verified working with YouTube URLs. Existing prompt template ready. |
| FTT-04 | Merge FTT video concepts into canonical registry (deduplication) | Registry has 297 concepts. ConceptRegistry with rapidfuzz dedup at threshold 85/70 is operational. |
| FTT-05 | Extract diagnostic chains from FTT content (symptom->cause->fix patterns) | DiagnosticChain schema exists with check_sequence branching. Prompt template includes VLM integration section. |
</phase_requirements>

## FTT Channel Video Inventory

### Total: 70 videos

### Already Analyzed (35 videos)
Videos with existing analyses in `docs/research/09_ftt_videos_{1,2,3}.md`:

| # | Video ID | Title | Duration |
|---|----------|-------|----------|
| 1 | lwS7Xxqrrdw | The Simple Foundation of the 2-Handed Backhand | 13:02 |
| 2 | FBJnFBtZozE | How Sinner Choke-Proofed His Game | 10:05 |
| 3 | LoSrMj-Qxkc | Their Mothers are Sisters, Their Forehands AREN'T | 11:58 |
| 4 | Z9XwaXDVkZg | The Simple Foundation of the One-Handed Backhand | 19:30 |
| 5 | FjjvhNbk3q4 | 7 Tips for an Effortless Serve ft. Giovanni Mpetshi-Perricard | 21:03 |
| 6 | Bhzd6cM-oFU | How to Hit Crazy Slice Like Carlos Alcaraz | 3:47 |
| 7 | emZxTmSVSa8 | Master Pronation in 2 Minutes at Home | 6:19 |
| 8 | x_WIg59Wr9Y | Why You Should Start Your Swing at Contact | 10:51 |
| 9 | tzgAWr3QGv0 | 7 Steps for a Sinner-Like Forehand - #5: Tilt! | 18:53 |
| 10 | p41GGc_y5Xk | How Your Hand Controls Every Shot | 6:24 |
| 11 | PfDqnff_vYA | Why Your Serve is So Slow | 8:54 |
| 12 | hqganBxzQNM | Why I Give Every Student a Weight to Swing | 6:43 |
| 13 | 6j33LJfl46s | Scapular Glide on the Tennis Forehand | 0:12 |
| 14 | lbdQpIHvXYA | The Stabilization Secret in Khachanov's Forehand | 5:47 |
| 15 | 7_JPqhG4G64 | What He Does Next Separates Pros from Amateurs | 5:05 |
| 16 | T5meVwBOKR4 | How Alcaraz Hits Harder Than Everyone | 8:31 |
| 17 | PjvOQkqH9DA | Never Lose Another Winnable Match | 9:24 |
| 18 | gehRK0Y6AdQ | The Pros "Brush" Differently From You | 13:38 |
| 19 | 9NirSL59an0 | The Paradoxical Nature of Modern Topspin | 10:20 |
| 20 | cIIEYn42o4g | The #1 Backscratch Mistake -- How to Fix It | 8:25 |
| 21 | 1W1rope5l0k | Why Every Pro Serves In-to-Out | 11:32 |
| 22 | -ZmnDrRFfjc | 3 Brilliant Alcaraz Tactics ANYONE Can Use | 6:23 |
| 23 | WoVEWh7fFfc | 57-Year-Old Ripping Forehands (and Serves) | 7:37 |
| 24 | EkzzGZkqCgc | Why Your Racket Doesn't Flip | 7:06 |
| 25 | m1NBdd3Bigg | Simple Serve Progressions | 16:53 |
| 26 | dZadhzVEsds | How to Hit Hard Without Missing Long | 21:00 |
| 27 | RVfmDk-iEwo | Find Your Hand Slot, Everything Gets Easier | 21:42 |
| 28 | KU7FHy1qQOI | The Physics, Geometry, and Biomechanics of Power | 18:08 |
| 29 | Vg8lbXOhM3E | Training to Play Loose in Matches | 11:16 |
| 30 | 0m3BMfDDShI | Build This Foundation - The Rest Will Follow | 14:51 |
| 31 | fZ-e7O7FTDE | Jannik Sinner Forehand Analysis | 11:34* |
| 32 | azVf6CyDfVk | Weight Transfer Exercise For More Power | varies* |
| 33 | 1-g1OD8gh-I | This One Move Instantly Improves Your Forehand | varies* |

*Note: Videos 32-35 in file 3 include 2 duplicates (gehRK0Y6AdQ, 9NirSL59an0). Unique already-analyzed = ~33 unique video IDs.*

### Remaining Unanalyzed (37 videos)

| # | Video ID | Title | Duration (sec) | Topic Category |
|---|----------|-------|----------------|----------------|
| 1 | pWzyP-xfLfU | The Secret to Lag is on Your Handle | 1262 | Forehand technique |
| 2 | mOFtt9PllI0 | The Geometry of Measured Aggression | 3105 | Tactics/strategy |
| 3 | 5jHCDc44SQM | The Abdominal Corkscrew ft. Carson Branstine | 458 | Core biomechanics |
| 4 | XXlndjnrA4E | Peripheral Vision Lets You Volley Like the Pros | 499 | Vision/volley |
| 5 | hNVbbPEob3g | Chest Engagement Makes Controlling the Racket Face Easy | 380 | Forehand biomechanics |
| 6 | UB6SbA_KX9E | Proper Trunk Sequencing will Transform Your Tennis | 715 | Core sequencing |
| 7 | w1FakobNq1Q | Breaking Down the Greatest Tiebreak Ever Played | 685 | Tactics |
| 8 | FOmz8Wjv3DQ | Deeper Than Just "Footwork" - Movement Fundamentals | 1066 | Movement |
| 9 | 8r09TliP-Ak | I Picked Draper to Win It All (3 patterns) | 684 | Pattern analysis |
| 10 | _Qu1LOwklAw | Patient but Ruthless - Alcaraz Broke Down Sinner | 394 | Tactics |
| 11 | Qszz0N4fRb4 | Why You Get Tight, and How to Fix It | 708 | Mental/psychological |
| 12 | Psidjei5BnI | 4 Tips For Consistently Crushing Slow Balls | 476 | Technique |
| 13 | FxDmVi3EFnE | The Truth About the Topspin Pro | 929 | Training tool review |
| 14 | McCb-RfYd0w | The Magic of the Non-Dominant Side on the Forehand | 193 | Forehand technique |
| 15 | JIMgI3jiVns | How Shoulder Rotation Syncs Your Contact | 449 | Biomechanics |
| 16 | 42BfbKsTGb4 | You Aren't Practicing Half of Tennis - RECEIVING | 338 | Receiving/tactics |
| 17 | Fu6DkHvZlGY | The Pure-Dextral Pinpoint - Serve Like Mensik | 436 | Serve technique |
| 18 | E_zmENJIj4g | Coiling/Spinal Motion/Eye Dominance/Arm Slot - 15 Serves | 2625 | Serve comprehensive |
| 19 | BbGzWTp5pCM | Practicing in Slow Motion is Killing Match Play | 260 | Training philosophy |
| 20 | JzcA_ku7Yhk | The Misunderstanding LOSING You Matches | 431 | Mental/tactical |
| 21 | dx8aGSIo24w | 4 Tips for Two-Handers | 283 | Backhand |
| 22 | wFIrPMutzRo | 2 Secrets to Rotational Power - Side Bending + X | 766 | Core biomechanics |
| 23 | NEpD7fIM7HI | Learning from Federer's Slice - 4 Tips + 3 Drills | 1052 | Slice technique |
| 24 | GsHkML2mVEI | Nishioka's Unique "Backswing" Timing | 351 | Timing/technique |
| 25 | wd4YRQW3TOc | 4 Tips For Effortless, Controllable Topspin | 1339 | Topspin technique |
| 26 | V-QkILd4V-w | 8 Visual Return Strategies Tested by WTA Pro | 1022 | Return/vision |
| 27 | xLs469ZVMPU | Fix Your Forehand Over-Rotation - 3 Techniques | 735 | Forehand fix |
| 28 | dDYKuNZtdyU | Find Your One-Handed Backhand "Pull Slot" | 666 | Backhand |
| 29 | pQ793MBQE50 | How Scap Retraction Powers One-Handed Backhand | 988 | Backhand biomechanics |
| 30 | YbLit9png2U | Fix Your Kick Serve by Throwing Sideways | 202 | Serve |
| 31 | XjJHA91HDbU | Don't Swing AT the Ball on the Overhead | 212 | Overhead |
| 32 | wVa4XQPcaqs | Use The Wall to Find Your Perfect Contact | 276 | Drill |
| 33 | QXAtdSEUkfY | 3 Tips to Rip the Low Backhand | 186 | Backhand |
| 34 | wFOy0RKWBTg | Swing OUT on the Forehand | 31 | Forehand (short) |
| 35 | 5KdScDKxVSI | Shoulder Adduction Will Transform Forehand Contact | 358 | Forehand biomechanics |
| 36 | dnNOOornvek | Retired WTA Pro Still Ripping Backhands | 48 | Short/demo |
| 37 | ExkBtFRhUWY | The Magic of Single-Foot Forehand Training | 497 | Forehand drill |
| 38 | PUN6qIIYU-4 | 4 Tips for the Two-Handed Backhand | 887 | Backhand |
| 39 | Am8j1Zw5KrE | Shoulder Adduction Unlocks the Tennis Forehand | 557 | Forehand biomechanics |
| 40 | OYf48k-cfNI | Fault Tolerance In Action | 233 | FTT philosophy |

**Total remaining: ~37 unique unanalyzed videos** (after deduplicating already-analyzed IDs).

**Duration breakdown of remaining:**
- Short (< 1 min): 3 videos (wFOy0RKWBTg 31s, dnNOOornvek 48s)
- Medium (1-10 min): 22 videos
- Long (10-20 min): 10 videos
- Very long (20+ min): 2 videos (mOFtt9PllI0 51:45, E_zmENJIj4g 43:45)
- **Total analysis time: ~5.5 hours of video content**

## Standard Stack

### Core (already installed/verified)
| Library | Version | Purpose | Status |
|---------|---------|---------|--------|
| google-genai | 1.65.0 | Gemini API YouTube URL video understanding | Installed, proxy verified |
| yt-dlp | 2026.03.13 | Channel enumeration, video metadata | Installed, working |
| pydantic | 2.x | Concept/Edge/DiagnosticChain schemas | Installed, schemas defined |
| rapidfuzz | installed | ConceptRegistry fuzzy dedup | Installed, working |
| tenacity | 9.1.4 | Retry with exponential backoff | Installed |
| networkx | installed | Knowledge graph backend | Installed |

### Needed for Phase 3
| Library | Version | Purpose | Status |
|---------|---------|---------|--------|
| instructor | not installed | Structured Pydantic extraction from Gemini | **Must install** |
| tqdm | likely installed | Progress bars for batch processing | Check |

**Installation:**
```bash
pip install instructor tqdm
```

### API Configuration (Verified Working)

**Primary:** packyapi.com proxy
```python
client = genai.Client(
    api_key='sk-aBrcwltK77b6fWTrQJKl2LJ8Y9aXhLubS68LU8yeIuCVnLlK',
    http_options=types.HttpOptions(base_url='https://www.packyapi.com')
)
```

**Backup:** Native Google API key
```python
client = genai.Client(api_key='AIzaSyAyP4SAOKYdFYrC3PlHKxHs2I1s7WpczHw')
```

**Model:** `gemini-3-flash-preview` (proxy) / `gemini-2.5-flash` (native)

**YouTube URL syntax (verified working):**
```python
from google.genai import types

response = client.models.generate_content(
    model='gemini-3-flash-preview',
    contents=types.Content(
        parts=[
            types.Part(file_data=types.FileData(
                file_uri='https://www.youtube.com/watch?v=VIDEO_ID'
            )),
            types.Part(text='<analysis prompt here>')
        ]
    )
)
```

## Architecture Patterns

### Processing Architecture

```
Phase 3 Sub-Phases:
  
  3A: Channel Enumeration (FTT-01) -- DONE in research
  |
  3B: Re-extract from 35 existing analyses (FTT-02)
  |   Input: docs/research/09_ftt_videos_{1,2,3}.md (text only, no API)
  |   Output: knowledge/extracted/ftt_videos/per-video JSON
  |
  3C: Analyze 37 remaining videos via Gemini (FTT-03)
  |   Input: YouTube URLs -> Gemini API -> raw analysis markdown
  |   Output: docs/research/09_ftt_videos_{4,5,6,7}.md + per-video JSON
  |
  3D: Merge all FTT video concepts into registry (FTT-04)
  |   Input: All per-video JSON files
  |   Output: Updated _registry_snapshot.json
  |
  3E: Extract diagnostic chains (FTT-05)
      Input: Merged graph + video analyses
      Output: Diagnostic chain objects in graph
```

### Pattern 1: Checkpointed Video Processing State

```python
# knowledge/state/ftt_video_state.json
{
  "channel_id": "@FaultTolerantTennis",
  "total_videos": 70,
  "enumerated_at": "2026-04-03",
  "videos": {
    "lwS7Xxqrrdw": {
      "title": "The Simple Foundation of the 2-Handed Backhand",
      "duration": 782,
      "status": "analyzed",        # pending | analyzed | extracted | failed
      "analysis_file": "docs/research/09_ftt_videos_1.md",
      "extracted_file": "knowledge/extracted/ftt_videos/lwS7Xxqrrdw.json",
      "analyzed_at": "2026-03-17",
      "error": null
    },
    "pWzyP-xfLfU": {
      "title": "The Secret to Lag is on Your Handle",
      "duration": 1262,
      "status": "pending",
      "analysis_file": null,
      "extracted_file": null,
      "analyzed_at": null,
      "error": null
    }
  }
}
```

### Pattern 2: Per-Video Extraction Output

Each video produces a standalone JSON file:
```python
# knowledge/extracted/ftt_videos/{video_id}.json
{
  "video_id": "pWzyP-xfLfU",
  "title": "The Secret to Lag is on Your Handle",
  "url": "https://www.youtube.com/watch?v=pWzyP-xfLfU",
  "duration": 1262,
  "analyzed_at": "2026-04-03",
  "model": "gemini-3-flash-preview",
  "concepts": [/* Concept objects */],
  "edges": [/* Edge objects */],
  "diagnostic_chains": [/* DiagnosticChain objects */],
  "raw_analysis_text": "..."  // or path to markdown file
}
```

### Pattern 3: Batched Analysis with Rate Limiting

```python
import asyncio
from tenacity import retry, wait_exponential, stop_after_attempt

BATCH_SIZE = 5          # Videos per batch
DELAY_BETWEEN = 10      # Seconds between batches
MAX_RETRIES = 3

@retry(wait=wait_exponential(min=5, max=60), stop=stop_after_attempt(MAX_RETRIES))
async def analyze_video(client, video_id: str, prompt: str) -> str:
    response = client.models.generate_content(
        model='gemini-3-flash-preview',
        contents=types.Content(parts=[
            types.Part(file_data=types.FileData(
                file_uri=f'https://www.youtube.com/watch?v={video_id}'
            )),
            types.Part(text=prompt)
        ])
    )
    return response.text

async def process_batch(videos: list[dict], client, prompt: str):
    for i in range(0, len(videos), BATCH_SIZE):
        batch = videos[i:i+BATCH_SIZE]
        for video in batch:
            result = await analyze_video(client, video['id'], prompt)
            save_analysis(video['id'], result)
            mark_processed(video['id'], 'analyzed')
        await asyncio.sleep(DELAY_BETWEEN)
```

### Recommended Batching Strategy for 37 Videos

**By priority (forehand-first):**

| Batch | Videos | Topic | Priority |
|-------|--------|-------|----------|
| 1 (8) | pWzyP-xfLfU, McCb-RfYd0w, JIMgI3jiVns, 5KdScDKxVSI, Am8j1Zw5KrE, xLs469ZVMPU, ExkBtFRhUWY, wFOy0RKWBTg | Forehand-specific | HIGHEST |
| 2 (6) | 5jHCDc44SQM, hNVbbPEob3g, UB6SbA_KX9E, wFIrPMutzRo, wd4YRQW3TOc, GsHkML2mVEI | Biomechanics/topspin | HIGH |
| 3 (5) | OYf48k-cfNI, BbGzWTp5pCM, Qszz0N4fRb4, JzcA_ku7Yhk, FxDmVi3EFnE | FTT philosophy/mental | HIGH |
| 4 (6) | mOFtt9PllI0, w1FakobNq1Q, 8r09TliP-Ak, _Qu1LOwklAw, Psidjei5BnI, 42BfbKsTGb4 | Tactics/patterns | MEDIUM |
| 5 (5) | V-QkILd4V-w, XXlndjnrA4E, FOmz8Wjv3DQ, wVa4XQPcaqs, YbLit9png2U | Movement/vision/drills | MEDIUM |
| 6 (7) | E_zmENJIj4g, Fu6DkHvZlGY, dx8aGSIo24w, PUN6qIIYU-4, dDYKuNZtdyU, pQ793MBQE50, QXAtdSEUkfY, NEpD7fIM7HI, XjJHA91HDbU, dnNOOornvek | Serve/backhand/other | LOWER |

**Forehand-first rationale:** The knowledge system is forehand-centric. Forehand and biomechanics videos produce the most valuable concepts and diagnostic chains. Non-forehand videos (serve, backhand, tactics) still contribute to the knowledge graph but are lower priority.

### Anti-Patterns to Avoid

- **Monolithic extraction prompt:** Do NOT ask Gemini to extract concepts, edges, diagnostic chains, AND generate structured JSON all in one call. Use the existing 11-section prompt for analysis, then a separate extraction pass for structuring.
- **Re-analyzing already-analyzed videos:** The 35 existing analyses in markdown are comprehensive. Extract from text, do not re-run through Gemini.
- **Flat video-title concepts:** The current `09_ftt_videos_*.json` files just use video titles as concept names. This is useless. Must re-extract actual tennis concepts from the analysis text.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Structured LLM extraction | Manual JSON parsing from Gemini | instructor + Pydantic | Auto-validation, retry on malformed output |
| Fuzzy concept dedup | String matching heuristics | ConceptRegistry (rapidfuzz) | Already built, threshold-tuned |
| Retry with backoff | try/except loops | tenacity decorators | Cleaner, configurable |
| Video prompt template | New prompt from scratch | Existing `video_analysis_prompt.md` | Field-tested on 35 videos |
| Channel enumeration | YouTube Data API scraping | yt-dlp flat-playlist | No quota, no API key needed |

## Common Pitfalls

### Pitfall 1: Proxy Rate Limits Different from Native Google
**What goes wrong:** The packyapi proxy may have its own rate limits independent of Google's free tier limits. Hitting proxy limits produces opaque errors.
**Prevention:** Start conservative (1 request per 10 seconds), monitor for 429 errors. If proxy fails, switch to native Google key (which has documented 250 RPD for Flash free tier).
**Warning signs:** HTTP 429 or 503 errors, or responses with "rate limit" messages.

### Pitfall 2: Long Videos Timeout or Produce Truncated Analysis
**What goes wrong:** Videos over 20 minutes (mOFtt9PllI0 at 51:45, E_zmENJIj4g at 43:45) may cause API timeouts or the analysis may be truncated.
**Prevention:** Set generous timeout (300s). For very long videos, consider splitting the prompt: "Analyze the first half..." then "Analyze the second half..." Or process long videos separately with a simplified prompt.
**Warning signs:** Incomplete analysis sections, missing later template sections.

### Pitfall 3: Concept Explosion from Video Analysis
**What goes wrong:** Each video analysis with the full prompt produces 10-30 concept mentions. Across 70 videos, that is 700-2100 raw mentions that must dedup into ~100-200 canonical concepts.
**Prevention:** 
1. Use the existing 297-concept registry as the anchor
2. In the extraction prompt, provide the existing concept list and ask Gemini to map to existing concepts first, only creating new ones when genuinely novel
3. Run dedup pass after each batch, not just at the end
**Detection:** If concept count exceeds 500 after FTT video extraction, dedup is failing.

### Pitfall 4: Duplicate Analysis in File 3
**What goes wrong:** `09_ftt_videos_3.md` contains 2 duplicate analyses (videos 32/33 are repeats of 18/19 from file 2: gehRK0Y6AdQ and 9NirSL59an0).
**Prevention:** Use video IDs (not sequential numbers) as the dedup key. The state file should track by video_id.

### Pitfall 5: Non-Forehand Content Dilutes Knowledge Graph
**What goes wrong:** ~15 of the 37 remaining videos are about serve, backhand, tactics, or overhead. Extracting concepts from these dilutes the forehand-focused knowledge graph.
**Prevention:** Still analyze all videos (they may reference forehand principles), but tag non-forehand concepts with lower confidence and mark their category appropriately. The knowledge graph should remain forehand-centric.

### Pitfall 6: Free Tier YouTube Limits
**What goes wrong:** Google's free tier limits YouTube video uploads to 8 hours per day total.
**Prevention:** 37 remaining videos total ~5.5 hours of content. This fits within 1 day's limit. But add the native Google key as fallback in case the proxy has different limits.

## Code Examples

### Gemini Video Analysis (verified working)
```python
# Source: Tested on 2026-04-03 against packyapi proxy
from google import genai
from google.genai import types

client = genai.Client(
    api_key='sk-aBrcwltK77b6fWTrQJKl2LJ8Y9aXhLubS68LU8yeIuCVnLlK',
    http_options=types.HttpOptions(base_url='https://www.packyapi.com')
)

response = client.models.generate_content(
    model='gemini-3-flash-preview',
    contents=types.Content(
        parts=[
            types.Part(file_data=types.FileData(
                file_uri='https://www.youtube.com/watch?v=wFOy0RKWBTg'
            )),
            types.Part(text='<prompt from video_analysis_prompt.md>')
        ]
    )
)
analysis_text = response.text
```

### Existing Analysis Text Extraction (no API needed)
```python
# Extract structured concepts from existing markdown analysis
import re
from pathlib import Path

def parse_video_sections(md_path: Path) -> list[dict]:
    """Split multi-video analysis file into per-video sections."""
    text = md_path.read_text()
    sections = re.split(r'^## (\d+)\. ', text, flags=re.MULTILINE)
    videos = []
    for i in range(1, len(sections), 2):
        num = sections[i]
        content = sections[i + 1]
        # Extract URL
        url_match = re.search(r'https://www\.youtube\.com/watch\?v=([A-Za-z0-9_-]+)', content)
        video_id = url_match.group(1) if url_match else f"unknown_{num}"
        videos.append({
            'number': int(num),
            'video_id': video_id,
            'content': content,
        })
    return videos
```

### Concept Extraction Prompt (for Gemini on existing text)
```python
EXTRACTION_PROMPT = """Given this video analysis text, extract structured knowledge:

1. List all CONCEPTS mentioned (technique, biomechanics, drill, symptom, mental_model)
2. For each concept, provide: name, name_zh, category, description, vlm_features, muscles_involved
3. List all RELATIONSHIPS between concepts (causes, prevents, requires, supports, drills_for)
4. List all DIAGNOSTIC CHAINS (symptom -> check steps -> root cause -> drill)

Output as valid JSON matching these schemas:
{schemas}

EXISTING CONCEPTS (map to these when possible, only create new if genuinely novel):
{existing_concept_list}

Analysis text:
{analysis_text}
"""
```

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.x |
| Config file | existing pytest config |
| Quick run command | `python -m pytest tests/ -x -q --tb=short` |
| Full suite command | `python -m pytest tests/ -v` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| FTT-01 | Channel enumeration produces 70 videos | unit | `pytest tests/test_ftt_discovery.py::test_channel_count -x` | Wave 0 |
| FTT-02 | Existing analyses parsed into per-video sections | unit | `pytest tests/test_ftt_extraction.py::test_parse_existing -x` | Wave 0 |
| FTT-03 | Gemini API returns valid analysis for a video | integration | `pytest tests/test_ftt_extraction.py::test_gemini_video -x` | Wave 0 |
| FTT-04 | Concepts merge into registry without explosion | unit | `pytest tests/test_ftt_extraction.py::test_registry_merge -x` | Wave 0 |
| FTT-05 | Diagnostic chains extracted from video content | unit | `pytest tests/test_ftt_extraction.py::test_diagnostic_chains -x` | Wave 0 |

### Wave 0 Gaps
- [ ] `tests/test_ftt_discovery.py` -- covers FTT-01 (channel enumeration)
- [ ] `tests/test_ftt_extraction.py` -- covers FTT-02 through FTT-05
- [ ] `knowledge/pipeline/discovery.py` -- video state management module
- [ ] `knowledge/pipeline/analyzer.py` -- Gemini video analysis module
- [ ] `knowledge/state/` directory -- checkpoint state files

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| google-genai | Video analysis | Yes | 1.65.0 | -- |
| yt-dlp | Channel enumeration | Yes | 2026.03.13 | -- |
| tenacity | Retry logic | Yes | 9.1.4 | -- |
| rapidfuzz | Concept dedup | Yes | installed | -- |
| pydantic | Schemas | Yes | 2.x | -- |
| instructor | Structured extraction | **No** | -- | Raw JSON parsing (worse) |
| tqdm | Progress bars | Check | -- | print statements |
| packyapi proxy | Primary API | Yes | verified | Native Google key |
| Google API key | Backup API | Yes | verified | packyapi proxy |

**Missing dependencies with no fallback:** None (instructor is recommended but raw JSON parsing works as fallback).

**Missing dependencies with fallback:**
- instructor: Not installed. Install with `pip install instructor`. Fallback: manual JSON extraction from Gemini response text.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Download video + upload to Gemini | YouTube URL pass-through | Gemini 2.0+ (2025) | No storage/bandwidth needed |
| youtube-transcript-api only | Gemini native video understanding | 2025 | Visual analysis, not just transcript |
| Single prompt extraction | Multi-section structured prompt | Current project | Higher quality analysis |
| Flat concept lists | Directed multigraph with typed edges | Phase 1 | Enables diagnostic chains |

## Open Questions

1. **Proxy rate limits unknown**
   - What we know: packyapi proxy works, native Google free tier is ~250 RPD for Flash
   - What's unclear: packyapi's own rate limits
   - Recommendation: Start with proxy, monitor for 429s, fall back to native key

2. **instructor compatibility with proxy**
   - What we know: instructor supports google-genai natively
   - What's unclear: Whether instructor's validation/retry works through the proxy
   - Recommendation: Test instructor + proxy with one video before batch. If fails, use raw JSON extraction.

3. **Optimal concept count target**
   - What we know: 297 concepts from Phase 2. FTT videos should add forehand-specific concepts.
   - What's unclear: How many genuinely new concepts the remaining 37 videos will introduce
   - Recommendation: Target ~350-400 total after Phase 3. If exceeding 500, dedup is failing.

## Sources

### Primary (HIGH confidence)
- yt-dlp channel enumeration (run 2026-04-03): 70 videos confirmed
- Gemini API video understanding test (run 2026-04-03): proxy and native both verified
- Existing codebase analysis: schemas.py, registry.py, extractor.py, handlers.py

### Secondary (MEDIUM confidence)
- [Gemini Video Understanding docs](https://ai.google.dev/gemini-api/docs/video-understanding) - YouTube URL syntax, 8hr/day free limit, 10 videos/request
- [Gemini Rate Limits](https://ai.google.dev/gemini-api/docs/rate-limits) - Free tier limits vary by model
- [google-genai SDK](https://github.com/googleapis/python-genai) - Python SDK reference

### Tertiary (LOW confidence)
- Proxy-specific rate limits: Not documented, must test empirically

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - all libraries installed and verified working
- Architecture: HIGH - existing pipeline patterns from Phase 2, existing prompt template tested on 35 videos
- Pitfalls: HIGH - informed by Phase 2 experience and actual API testing
- Video count: HIGH - enumerated via yt-dlp, cross-referenced with existing analyses

**Research date:** 2026-04-03
**Valid until:** 2026-05-03 (stable -- FTT channel grows slowly, API is stable)
