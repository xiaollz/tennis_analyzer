# Technology Stack

**Project:** Tennis Forehand Knowledge Engineering & VLM Diagnostic System
**Researched:** 2026-04-03
**Overall confidence:** HIGH

## Recommended Stack

### Layer 1: YouTube Video Discovery & Metadata

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| yt-dlp | >=2024.12.0 | Channel enumeration, video metadata, subtitle download | De facto standard. Returns structured dicts with title, description, upload_date, duration, tags. Enumerate entire channels without YouTube Data API quota. |
| youtube-transcript-api | >=1.2.0 | Transcript/subtitle extraction with timestamps | 1.2M+ monthly downloads. No API key needed. Supports auto-generated and manual subtitles, multiple languages, timestamp segments. Lighter than yt-dlp for transcript-only tasks. |

**Why not YouTube Data API v3:** Quota limits (10,000 units/day) make full-channel extraction painful. yt-dlp has no quota. The project needs to sweep ~115 FTT + selective TomAllsopp + Feel Tennis videos -- quota would be a bottleneck.

**Why not the existing MCP YouTube transcript server:** It is a Node.js MCP server -- fine for interactive Claude use, but the extraction pipeline needs a Python-native solution for batch processing, retry logic, and integration with downstream Gemini calls.

### Layer 2: Video Content Analysis (Gemini API)

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| google-genai (Python SDK) | >=1.0.0 | Video understanding via YouTube URLs or file upload | Project already uses Gemini through OpenAI-compatible proxy. The native SDK supports `Part.from_uri()` with YouTube URLs directly -- no download needed. Gemini 2.5 Pro is SOTA for video understanding. Supports dynamic FPS (critical for tennis motion analysis at 60fps). |
| OpenAI-compatible proxy | existing | Fallback/alternative Gemini access | Already configured at openclaudecode.cn. Keep as fallback when native SDK has issues. |

**Architecture decision:** Use `google-genai` SDK natively for video analysis (YouTube URL pass-through), fall back to proxy for image-based VLM analysis (existing `vlm_analyzer.py` workflow). Do NOT mix -- keep two clear paths.

**Why not download-then-upload:** Gemini accepts YouTube URLs directly via `Part.from_uri("https://www.youtube.com/watch?v=VIDEO_ID", mime_type="video/mp4")`. Eliminates storage, bandwidth, and copyright concerns. Only works for public videos (all target channels are public).

**Rate limiting:** Gemini API has per-minute request limits. Build with exponential backoff + configurable concurrency (start with 2 concurrent, tune up).

### Layer 3: Structured Knowledge Extraction

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| pydantic | >=2.5.0 | Schema definitions for knowledge entities | Standard for Python data validation. Define Concept, Relationship, DiagnosticChain, Drill as Pydantic models. Get JSON serialization for free. |
| instructor | >=1.7.0 | Structured extraction from Gemini responses | 3M+ monthly downloads. Wraps LLM calls to enforce Pydantic schema output with automatic validation and retries. Supports Gemini provider. Eliminates manual JSON parsing from LLM responses. |

**Why not raw JSON parsing from LLM output:** LLMs produce inconsistent JSON. Instructor handles validation, retries on malformed output, and type coercion. For 100+ videos producing structured knowledge, reliability matters more than simplicity.

**Why not LangChain:** Massive dependency, abstracts away control the project needs. Instructor is 1 dependency (plus pydantic) and does exactly one thing well.

### Layer 4: Knowledge Graph Storage

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| NetworkX | >=3.2 | In-memory graph operations, traversal, querying | Already a Python dependency (numpy ecosystem). Perfect for graphs under 10K nodes. Supports directed multigraph (concepts can have multiple relationship types). Built-in shortest path, connected components, subgraph extraction -- all needed for diagnostic chain traversal. |
| JSON files | - | Persistent storage format | Project already uses JSON (concept_network.json, diagnostic_engine.json). No database server needed. Version-controllable in git. Dual output requirement (JSON + Markdown) makes JSON the natural primary format. |

**Why not Neo4j:** Overkill. The knowledge graph will have ~500-2000 concept nodes and ~2000-5000 edges. Neo4j adds operational complexity (server process, Cypher query language, Docker) for zero benefit at this scale. NetworkX loads the entire graph in <100ms.

**Why not RDF/kglab:** RDF's triple-store model is designed for open-world ontologies (DBpedia, Wikidata). This is a closed-domain expert knowledge graph. RDF adds complexity (SPARQL, namespaces, serialization formats) without benefit.

**Why not SQLite:** Graphs stored in relational tables require complex JOINs for traversal. NetworkX's native graph traversal is simpler for the diagnostic chain use case (symptom -> intermediate causes -> root cause -> drill).

### Layer 5: Markdown Generation

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Jinja2 | >=3.1.0 | Template-based Markdown generation | Standard Python templating. Define templates for concept pages, diagnostic trees, drill references. Separates content from presentation. |

**Why not string concatenation:** With 500+ concept pages and multiple output formats, templates prevent bugs and enable format iteration without touching generation logic.

### Layer 6: Pipeline Orchestration

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Python asyncio | 3.11+ | Async pipeline orchestration | Already using Python 3.11. asyncio handles concurrent Gemini API calls with rate limiting. No framework needed for a linear pipeline with fan-out at the API call stage. |
| tqdm | >=4.66.0 | Progress tracking for batch operations | 115+ videos to process. Progress bars are essential for long-running extraction. |
| tenacity | >=8.2.0 | Retry logic with exponential backoff | API calls fail. tenacity provides decorator-based retry with configurable backoff, stop conditions, and logging. Cleaner than hand-rolled retry loops. |

**Why not Celery/Airflow/Prefect:** Single-machine batch pipeline run by one person. Task queue and orchestration infrastructure is unnecessary overhead.

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| httpx | >=0.27.0 | Async HTTP client for API calls | When google-genai SDK doesn't cover a use case (e.g., proxy routing) |
| rich | >=13.7.0 | Console output, tables, progress | Pipeline status display, debugging output |
| pytest | >=8.0.0 | Testing extraction schemas and graph operations | Development and CI |
| python-dotenv | >=1.0.0 | API key management | Load keys from .env instead of hardcoding in config JSON |

## Architecture: How Layers Connect

```
Layer 1: yt-dlp + youtube-transcript-api
  |
  v  (video URLs, metadata, transcripts)
Layer 2: google-genai SDK (Gemini 2.5)
  |
  v  (raw LLM analysis text)
Layer 3: instructor + pydantic
  |
  v  (structured Concept, Relationship, DiagnosticChain objects)
Layer 4: NetworkX graph + JSON persistence
  |
  v  (queryable knowledge graph)
Layer 5: Jinja2 templates
  |
  v  (Markdown output files)
Layer 6: asyncio + tenacity orchestrate the whole pipeline
```

## Pydantic Schema Design (Core Models)

```python
from pydantic import BaseModel, Field
from enum import Enum

class RelationType(str, Enum):
    CAUSES = "causes"           # A causes B (biomechanical)
    PREVENTS = "prevents"       # A prevents B
    REQUIRES = "requires"       # A requires B as prerequisite
    CONTRADICTS = "contradicts" # A contradicts B (cross-source)
    SUPPORTS = "supports"       # A supports/reinforces B
    DRILLS_FOR = "drills_for"   # Drill A trains Concept B
    VISIBLE_AS = "visible_as"   # Concept A is visible as VLM Feature B

class Concept(BaseModel):
    id: str                     # e.g., "hip_rotation"
    name: str                   # Human-readable name
    name_zh: str                # Chinese name
    category: str               # "technique", "biomechanics", "drill", "symptom"
    source: str                 # "ftt", "tpa", "feel_tennis", "biomechanics_book"
    description: str
    vlm_features: list[str] = []  # What camera can see
    muscles_involved: list[str] = []

class Relationship(BaseModel):
    source_id: str
    target_id: str
    relation: RelationType
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str               # Quote or reference
    source_file: str            # Which research file

class DiagnosticChain(BaseModel):
    symptom: str                # VLM-observable symptom
    symptom_zh: str
    root_causes: list[str]      # Concept IDs
    check_sequence: list[str]   # Ordered investigation steps
    drills: list[str]           # Concept IDs of type "drill"
    priority: int = Field(ge=1, le=5)  # 1=most common
```

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Video metadata | yt-dlp | YouTube Data API v3 | Quota limits block full-channel sweeps |
| Transcripts | youtube-transcript-api | Whisper (local) | Unnecessary -- YouTube already has subtitles/auto-captions |
| Video analysis | google-genai SDK | Download + local model | Gemini accepts YouTube URLs directly; local models cannot match Gemini 2.5 Pro for video |
| Structured extraction | instructor + pydantic | Raw JSON.loads() | Unreliable for 100+ extractions; no validation/retry |
| Knowledge graph | NetworkX + JSON | Neo4j | Operational overhead for <2K nodes |
| Knowledge graph | NetworkX + JSON | SQLite | Poor graph traversal ergonomics |
| Templating | Jinja2 | f-strings | Unmaintainable at scale |
| Orchestration | asyncio | Celery/Airflow | Single-user batch pipeline, no distributed workers needed |
| LLM framework | instructor (thin) | LangChain | Massive dependency, unnecessary abstraction |

## Installation

```bash
# Core pipeline
pip install yt-dlp youtube-transcript-api google-genai instructor pydantic networkx

# Pipeline utilities
pip install jinja2 tqdm tenacity httpx rich python-dotenv

# Dev
pip install pytest
```

## Version Pinning (pyproject.toml additions)

```toml
[project.optional-dependencies]
knowledge = [
    "yt-dlp>=2024.12.0",
    "youtube-transcript-api>=1.2.0",
    "google-genai>=1.0.0",
    "instructor>=1.7.0",
    "pydantic>=2.5.0",
    "networkx>=3.2",
    "jinja2>=3.1.0",
    "tqdm>=4.66.0",
    "tenacity>=8.2.0",
    "httpx>=0.27.0",
    "rich>=13.7.0",
    "python-dotenv>=1.0.0",
]
```

## Key Integration Points with Existing Codebase

1. **VLM Analyzer** (`evaluation/vlm_analyzer.py`): Currently uses OpenAI-compatible proxy for Gemini. The knowledge graph's diagnostic chains will be injected into the VLM prompt to replace the current 7 symptom groups (KE-11).

2. **Config** (`config/vlm_config.json`): API keys already managed here. Add a `knowledge_config.json` for pipeline-specific settings (rate limits, model selection, output paths).

3. **Existing Research Files** (`docs/research/*.md`): These are the INPUTS to the knowledge graph. The pipeline extracts structured data FROM these files (already-analyzed content) as well as from new video analysis.

4. **concept_network.json / diagnostic_engine.json**: Currently empty. These become the PRIMARY OUTPUTS of the knowledge graph pipeline.

## Sources

- [youtube-transcript-api on PyPI](https://pypi.org/project/youtube-transcript-api/)
- [yt-dlp GitHub](https://github.com/yt-dlp/yt-dlp)
- [Gemini Video Understanding docs](https://ai.google.dev/gemini-api/docs/video-understanding)
- [google-genai Python SDK](https://github.com/googleapis/python-genai)
- [Instructor library](https://python.useinstructor.com/)
- [Gemini 2.5 video understanding announcement](https://developers.googleblog.com/en/gemini-2-5-video-understanding/)
- [Google Cloud Gemini YouTube video sample](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/samples/googlegenaisdk-textgen-with-youtube-video)
- [NetworkX documentation](https://networkx.org/)
