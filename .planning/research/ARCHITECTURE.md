# Architecture Patterns

**Domain:** Tennis Forehand Knowledge Engineering & VLM Diagnostic System
**Researched:** 2026-04-03

## Recommended Architecture

```
knowledge/                    # New top-level package
  __init__.py
  config.py                   # Pipeline configuration (models, rate limits, paths)
  schemas.py                  # Pydantic models (Concept, Relationship, DiagnosticChain)
  pipeline/
    __init__.py
    discovery.py              # yt-dlp channel enumeration, video list management
    transcript.py             # youtube-transcript-api wrapper
    analyzer.py               # Gemini video analysis (YouTube URL pass-through)
    extractor.py              # instructor + pydantic structured extraction
    deduplicator.py           # Concept merging and deduplication
  graph/
    __init__.py
    builder.py                # NetworkX graph construction from extracted data
    query.py                  # Graph traversal (diagnostic chains, concept lookup)
    serializer.py             # JSON import/export, NetworkX <-> JSON roundtrip
  output/
    __init__.py
    json_writer.py            # concept_network.json, diagnostic_engine.json
    markdown_writer.py        # Jinja2-based Markdown generation
    vlm_prompt.py             # Generate VLM diagnostic prompt from graph
  templates/                  # Jinja2 templates
    concept.md.j2
    diagnostic_chain.md.j2
    drill.md.j2
    index.md.j2
  run_pipeline.py             # CLI entry point
```

### Component Boundaries

| Component | Responsibility | Communicates With |
|-----------|---------------|-------------------|
| discovery | Enumerate channel videos, manage processing state (done/pending) | analyzer, transcript |
| transcript | Extract text transcripts with timestamps | extractor (provides supplementary text) |
| analyzer | Send YouTube URL to Gemini, get raw analysis | extractor (provides raw LLM output) |
| extractor | Parse raw analysis into Pydantic models via instructor | deduplicator, graph/builder |
| deduplicator | Merge duplicate concepts across sources | graph/builder |
| graph/builder | Construct NetworkX DiGraph from concepts + relationships | graph/query, graph/serializer |
| graph/query | Traverse graph for diagnostic chains, concept neighborhoods | output/* |
| graph/serializer | JSON <-> NetworkX roundtrip | graph/builder, output/json_writer |
| output/json_writer | Write concept_network.json, diagnostic_engine.json | filesystem |
| output/markdown_writer | Render Jinja2 templates to Markdown files | filesystem |
| output/vlm_prompt | Generate optimized VLM prompt from graph subsets | evaluation/vlm_analyzer.py |

### Data Flow

```
1. DISCOVERY PHASE
   yt-dlp -> list of {video_id, title, url, duration}
   Check against processed_videos.json (idempotency)

2. ANALYSIS PHASE (per video, async with rate limiting)
   YouTube URL -> Gemini API -> raw analysis text
   + youtube-transcript-api -> timestamped transcript
   Both saved to docs/research/ as Markdown (human-readable artifact)

3. EXTRACTION PHASE (per analysis)
   Raw text -> instructor/pydantic -> list[Concept], list[Relationship]
   Saved to knowledge/extracted/{video_id}.json (intermediate artifact)

4. GRAPH ASSEMBLY PHASE (batch)
   All extracted/*.json -> deduplicator -> merged concepts
   Merged concepts + relationships -> NetworkX DiGraph
   Graph saved to docs/research/concept_network.json

5. DIAGNOSTIC CHAIN PHASE (batch)
   Graph traversal: find all "symptom" nodes
   For each symptom: trace causes backward, trace drills forward
   Assemble DiagnosticChain objects
   Save to docs/research/diagnostic_engine.json

6. OUTPUT PHASE
   Graph -> Jinja2 -> docs/knowledge/*.md (Markdown knowledge base)
   Graph -> vlm_prompt.py -> updated VLM diagnostic prompt
```

## Patterns to Follow

### Pattern 1: Idempotent Processing with State File

**What:** Track which videos have been processed in a JSON state file. Skip already-processed videos on re-run.

**When:** Every pipeline run. Critical for resumability after API failures.

**Example:**
```python
# knowledge/pipeline/discovery.py
import json
from pathlib import Path

STATE_FILE = Path("knowledge/state/processed_videos.json")

def get_pending_videos(channel_videos: list[dict]) -> list[dict]:
    """Return only videos not yet processed."""
    if STATE_FILE.exists():
        processed = set(json.loads(STATE_FILE.read_text()).keys())
    else:
        processed = set()
    return [v for v in channel_videos if v["video_id"] not in processed]

def mark_processed(video_id: str, status: str, output_path: str):
    """Mark a video as processed with its output location."""
    state = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
    state[video_id] = {"status": status, "output": output_path}
    STATE_FILE.write_text(json.dumps(state, indent=2))
```

### Pattern 2: Two-Pass Extraction (Coarse then Fine)

**What:** First pass extracts concepts and relationships from raw text. Second pass uses the accumulated concept list to improve deduplication and relationship quality.

**When:** After all videos in a source are analyzed. The second pass has global context.

**Why:** First-pass extraction produces duplicates ("hip rotation", "hip turn", "hip drive" are the same concept). Second pass with the full concept list can merge intelligently.

```python
# Pass 1: Extract per-video (no global context)
for video in videos:
    concepts, relationships = extract_from_text(video.analysis)
    save_intermediate(video.id, concepts, relationships)

# Pass 2: Merge with global context
all_concepts = load_all_intermediates()
merged = deduplicate_concepts(all_concepts)  # LLM-assisted merge
rebuild_graph(merged)
```

### Pattern 3: Graph-Backed VLM Prompt Generation

**What:** Instead of hardcoded symptom groups in the VLM prompt, query the knowledge graph at analysis time to generate the relevant diagnostic context.

**When:** KE-11 -- upgrading vlm_analyzer.py.

**Example:**
```python
# knowledge/output/vlm_prompt.py
def generate_diagnostic_context(graph, max_chars=8000) -> str:
    """Generate VLM diagnostic prompt section from knowledge graph."""
    chains = get_all_diagnostic_chains(graph)
    # Sort by priority (most common symptoms first)
    chains.sort(key=lambda c: c.priority)
    
    sections = []
    char_count = 0
    for chain in chains:
        section = format_chain(chain)
        if char_count + len(section) > max_chars:
            break
        sections.append(section)
        char_count += len(section)
    
    return "\n".join(sections)
```

### Pattern 4: Source Priority Conflict Resolution

**What:** When FTT and another source contradict, the graph stores both with a "contradicts" edge and marks which one to prefer.

**When:** Cross-source reconciliation (KE-06).

```python
class Relationship(BaseModel):
    # ... other fields
    relation: RelationType
    resolution: str | None = None  # "ftt_wins", "tpa_wins", "unresolved"
```

## Anti-Patterns to Avoid

### Anti-Pattern 1: Monolithic Extraction Prompt

**What:** Sending one massive prompt asking Gemini to extract concepts, relationships, drills, anatomical mappings, and VLM features all at once.

**Why bad:** LLMs produce worse output when asked to do too many things simultaneously. Extraction quality degrades.

**Instead:** Sequential focused prompts: (1) extract concepts and descriptions, (2) extract relationships between concepts, (3) extract VLM-observable features, (4) extract drill associations.

### Anti-Pattern 2: Storing the Graph Only in NetworkX

**What:** Keeping the authoritative graph only as a NetworkX object serialized with pickle.

**Why bad:** Pickle is not human-readable, not version-controllable, and breaks across Python versions.

**Instead:** JSON is the authoritative format. NetworkX is the in-memory runtime representation. Always roundtrip through JSON.

### Anti-Pattern 3: Re-analyzing Videos on Schema Change

**What:** If the Pydantic schema changes, re-running all 115 video analyses through Gemini.

**Why bad:** Expensive (API costs), slow (hours), and unnecessary.

**Instead:** Save raw Gemini analysis text as Markdown files (already done in docs/research/). Re-extraction from saved text is free and fast. Only re-analyze videos when the analysis PROMPT changes significantly.

### Anti-Pattern 4: Flat Concept List Instead of Graph

**What:** Storing concepts as a list with tags, without explicit directional relationships.

**Why bad:** Cannot traverse "why" chains. "Elbow flying out" needs to connect to "insufficient hip rotation" via "causes" edge. A flat list cannot answer "what causes this symptom?"

**Instead:** Always model as directed graph with typed edges.

## Scalability Considerations

| Concern | Current (1 user) | Future (productization) |
|---------|-------------------|------------------------|
| Graph size | ~500-2000 nodes, in-memory | Same graph, still fits in memory |
| Video analysis | Batch, hours to run | Cache results, only analyze new videos |
| VLM prompt size | ~10K chars limit | Subgraph selection based on detected issues |
| API costs | Personal budget | Per-user analysis costs, caching critical |
| Knowledge updates | Re-run pipeline | Incremental graph updates |

## Sources

- Existing codebase architecture analysis
- [NetworkX DiGraph documentation](https://networkx.org/documentation/stable/reference/classes/digraph.html)
- [Instructor structured extraction patterns](https://python.useinstructor.com/)
- [Gemini video understanding API](https://ai.google.dev/gemini-api/docs/video-understanding)
