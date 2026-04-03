# Phase 2: Existing Knowledge Extraction - Research

**Researched:** 2026-04-03
**Domain:** Text-to-structured-knowledge extraction from existing Markdown/JSON research files
**Confidence:** HIGH

## Summary

Phase 2 transforms 31 existing Markdown research files (~960KB, ~17,000 lines) and 3 pre-existing JSON knowledge files into structured Concept/Edge objects conforming to the Phase 1 Pydantic schemas. No API calls are needed -- this is purely local text processing by Claude (reading files, extracting structured JSON). The existing JSON files in `docs/knowledge_graph/` already contain ~72 concepts, ~15 causal links, ~17 drills, ~21 user problems, and ~16 breakthroughs in a legacy schema that must be migrated to the new Pydantic schema.

The key technical challenge is NOT extraction (Claude can read Markdown and output JSON) but rather **deduplication and canonical ID assignment**. The same concept appears across multiple files with different names (e.g., "hip rotation" / "hip turn" / "hip coil" / "hip drive"). The ConceptRegistry's fuzzy matching (rapidfuzz token_sort_ratio, threshold 85) must be seeded with a canonical concept list, then extraction scripts must resolve raw concepts against it.

**Primary recommendation:** Build extraction as a single Python script with per-file-type handler functions. Seed the registry with ~80-100 canonical concepts first (derived from the existing JSON files + synthesis documents), then process each research file against that registry. Output one JSON file per source file into `knowledge/extracted/`.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| EXIST-01 | Extract structured concepts from all 21+ existing research files in docs/research/ | 31 files identified, categorized by source (FTT book/blog: 9, FTT videos: 10, TPA: 7, Biomechanics: 5). All are Markdown prose with embedded tables, causal chains, and cross-references. |
| EXIST-02 | Extract structured concepts from docs/record/learning.md training journey | 1,332-line file. Pre-existing JSON extraction exists at docs/knowledge_graph/user_journey.json (21 problems, 16 breakthroughs, 57 cue evolutions). Needs schema migration + concept linkage. |
| EXIST-03 | Extract structured concepts from 24 biomechanics book anatomy files | Actually 5 files (24-28), not 24 separate files. Files 24-26 contain detailed muscle tables with columns for muscle name, action type, and forehand role. File 27 has new biomechanical insights. File 28 has problem-solution mappings. |
| EXIST-04 | Populate canonical concept registry (~200-300 concepts) from existing files | Existing JSON files contain ~72 pre-extracted concepts. Research files should yield ~150-200 additional unique concepts after dedup. Target: 200-300 total canonical concepts. |
| EXIST-05 | Build initial knowledge graph edges from existing cross-references | Existing JSON has 15 explicit causal_links. Research files contain extensive "FTT mapping" cross-references, causal chains (e.g., learning.md has explicit cause-effect chains), and "related_concepts" fields. Target: 300-500 edges. |
</phase_requirements>

## Project Constraints (from CLAUDE.md)

- Pydantic v2 API exclusively, snake-case IDs via regex, dc_ prefix for diagnostic chains
- English-only fuzzy matching for dedup; token_sort_ratio with threshold 85/70
- Chinese names are display-only, never used for matching
- JSON is the authoritative format; NetworkX is runtime representation
- All code in `knowledge/` package

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pydantic | 2.12.5 | Concept/Edge/DiagnosticChain validation | Already installed, Phase 1 schema |
| rapidfuzz | 3.14.3 | Fuzzy string matching for dedup | Already installed, Phase 1 registry |
| networkx | 3.6.1 | Knowledge graph construction | Already installed, Phase 1 graph |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest | 9.0.2 | Test extraction correctness | Validate extraction output |
| json (stdlib) | - | Read/write extracted JSON | All extraction I/O |
| pathlib (stdlib) | - | File path management | All file operations |
| re (stdlib) | - | Parse Markdown structure (headers, tables) | Biomechanics table extraction |

### No Additional Installs Needed

All required libraries are already available. No new dependencies for this phase.

## Architecture Patterns

### Recommended Project Structure
```
knowledge/
  extracted/                    # NEW: extraction output directory
    ftt_book/                   # Per-source subdirectories
      01_ftt_book.json
    ftt_blog/
      04_ftt_blog_forehand_1.json
      ...
    ftt_videos/
      09_ftt_videos_1.json
      ...
    tpa/
      14_tpa_videos_1.json
      ...
    biomechanics/
      24_biomechanics_ch1_ch8.json
      ...
    user_journey/
      learning.json
    _canonical_seed.json        # Initial canonical concept list (registry seed)
    _registry_snapshot.json     # Registry state after all extraction
    _graph_snapshot.json        # Graph state after all extraction
  pipeline/
    extractor.py                # NEW: Main extraction script
  schemas.py                    # Existing Phase 1
  registry.py                   # Existing Phase 1
  graph.py                      # Existing Phase 1
```

### Pattern 1: Canonical Registry Seeding
**What:** Before processing any raw Markdown, pre-populate the ConceptRegistry with ~80-100 canonical concepts derived from the 3 existing JSON files (ftt_core_concepts.json, tpa_kinetic_chain.json, user_journey.json).

**When:** First step of extraction pipeline, before any file processing.

**Why:** Without pre-seeding, the first few files will create concept IDs that later files won't match against. Pre-seeding establishes the canonical vocabulary.

**Example:**
```python
def seed_registry_from_legacy_json(registry: ConceptRegistry) -> list[Concept]:
    """Load existing JSON knowledge files and convert to Pydantic Concepts."""
    concepts = []
    
    # ftt_core_concepts.json has 48 concepts with id/name_en/name_zh/definition
    ftt_data = json.loads(FTT_JSON_PATH.read_text())
    for raw in ftt_data["concepts"]:
        concept = Concept(
            id=_to_snake_case(raw["name_en"]),  # "Unit Turn" -> "unit_turn"
            name=raw["name_en"],
            name_zh=raw["name_zh"],
            aliases=[],
            category=_map_category(raw["category"]),
            sources=["ftt"],
            description=raw["definition"],
            active_or_passive=raw.get("active_or_passive"),
        )
        registry.add(concept)
        concepts.append(concept)
    
    return concepts
```

### Pattern 2: File-Type Handler Functions
**What:** Each source category (FTT book, FTT blog, FTT videos, TPA, biomechanics, user journey) gets a dedicated handler function because the Markdown structure differs.

**When:** Processing each research file.

**Why:** FTT book files have narrative prose. Biomechanics files have muscle tables. Video analysis files have per-video sections. User journey has date-based entries with causal chains. Different structures require different extraction logic.

**Example:**
```python
FILE_HANDLERS = {
    "01_ftt_book": extract_ftt_book,
    "02_revolutionary_tennis": extract_ftt_book,  # same format
    "04_ftt_blog_": extract_ftt_blog,
    "09_ftt_videos_": extract_ftt_videos,
    "24_biomechanics_": extract_biomechanics,
    "25_biomechanics_": extract_biomechanics,
    # ...
}

def get_handler(filename: str):
    for prefix, handler in FILE_HANDLERS.items():
        if filename.startswith(prefix):
            return handler
    return extract_generic  # fallback
```

### Pattern 3: Extract-Then-Resolve (Two-Pass)
**What:** Pass 1 extracts raw concepts per file with provisional IDs. Pass 2 resolves all provisional IDs against the canonical registry, merging duplicates.

**When:** After all files are processed individually.

**Why:** A single-pass approach would either (a) miss cross-file duplicates or (b) require processing files in a specific order. Two-pass is order-independent.

```python
# Pass 1: Extract raw concepts per file
for filepath in research_files:
    handler = get_handler(filepath.name)
    raw_concepts, raw_edges = handler(filepath)
    save_intermediate(filepath.stem, raw_concepts, raw_edges)

# Pass 2: Resolve against registry
for extracted_file in intermediate_files:
    raw = load_intermediate(extracted_file)
    for concept in raw["concepts"]:
        existing_id = registry.resolve(concept["name"])
        if existing_id:
            concept["canonical_id"] = existing_id
            # Merge aliases, sources, etc. into existing concept
        else:
            registry.add(Concept(**concept))
```

### Pattern 4: Edge Extraction from Prose
**What:** Extract causal/relationship edges from Markdown text patterns.

**When:** Processing any file that contains cross-references or causal chains.

**How to detect edges in the existing files:**
1. **Explicit causal chains** (learning.md format):
   ```
   原因A → 导致B → 导致C → 导致D
   ```
   Parse arrows to create `causes` edges: A->B, B->C, C->D

2. **"FTT mapping" cross-references** (biomechanics files):
   ```
   > **FTT映射：** 对应 `22_ftt_scapular_glide.md` 中"后缩→前伸"
   ```
   Parse to create `supports` edges between biomechanics concept and FTT concept

3. **"related_concepts" fields** (existing JSON):
   ```json
   "related_concepts": ["C01", "C04", "C12"]
   ```
   Create `supports` edges

4. **Table-based relationships** (problem_dependency_graph in user_journey.json):
   Direct edge definitions

### Anti-Patterns to Avoid
- **Over-engineering extraction prompts:** This is NOT an LLM API extraction task. Claude reads the files directly and writes Python code to parse them. No Gemini/OpenAI calls needed.
- **One giant extraction script:** Use handler functions per file type. A monolithic parser will be unmaintainable.
- **Ignoring existing JSON:** The 3 JSON files in docs/knowledge_graph/ contain ~72 pre-extracted concepts that are the BEST starting point for registry seeding. Don't re-extract from Markdown what's already structured.
- **Processing files randomly:** Process in this order: (1) Seed from existing JSON, (2) Synthesis files (13, 12, 15, 17), (3) Primary source files, (4) Biomechanics, (5) User journey. Synthesis files establish canonical names first.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Fuzzy string matching | Custom Levenshtein | rapidfuzz (already in ConceptRegistry) | Edge cases in CJK, Unicode, performance |
| Markdown parsing | Regex-only parser | Simple regex + line-by-line state machine | Markdown is regular enough; a full parser (mistune, etc.) is overkill for this structured content |
| Snake-case ID generation | Manual string manipulation | `re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')` | Handles all edge cases consistently |
| JSON schema validation | Manual field checks | Pydantic model_validate() | Already have the schemas from Phase 1 |

## Common Pitfalls

### Pitfall 1: Legacy JSON Schema Mismatch
**What goes wrong:** The existing JSON files use a different schema (id: "C01", "T01", "P01") than the Phase 1 Pydantic schema (id: snake_case like "unit_turn"). Direct loading fails validation.
**Why it happens:** Existing JSONs were created before Phase 1 schema was defined.
**How to avoid:** Write explicit migration functions per JSON file. Map legacy IDs to snake_case: `C02 "Unit Turn"` -> `unit_turn`. Map legacy category strings ("准备阶段", "前挥阶段") to ConceptType enum values.
**Warning signs:** Pydantic ValidationError on id field pattern `^[a-z][a-z0-9_]*$`.

### Pitfall 2: Concept Count Confusion (24 vs 5 Biomechanics Files)
**What goes wrong:** EXIST-03 mentions "24 biomechanics book anatomy files" but there are only 5 files (24-28). The "24" likely refers to the number of muscle/anatomy sections across those files, not 24 separate files.
**Why it happens:** Requirements were drafted before detailed file inventory.
**How to avoid:** The 5 biomechanics files contain approximately 24 distinct muscle group sections. Extract per-section, not per-file. Each muscle table row becomes a biomechanics concept with muscles_involved populated.
**Warning signs:** Looking for 24 files that don't exist.

### Pitfall 3: Duplicate Concepts Across Sources
**What goes wrong:** FTT, TPA, and biomechanics all discuss the same concepts with different names. "Unit Turn" (FTT) = "Loading Phase" (biomechanics) = "Coiling" (TPA). Without careful alias management, these become 3 separate nodes.
**Why it happens:** Different pedagogical traditions use different terminology.
**How to avoid:** When seeding the registry, include comprehensive aliases. E.g., `unit_turn` should have aliases: ["unit turn", "loading phase", "coiling", "backswing rotation", "preparation turn"]. The synthesis files (13_synthesis.md, 15_tpa_synthesis.md, 17_kinetic_chain_synthesis.md) explicitly map cross-source equivalences -- extract aliases from these first.
**Warning signs:** Registry size > 350 after all extraction = likely duplicates.

### Pitfall 4: Losing Edge Direction
**What goes wrong:** Extracting "related to" relationships without direction. "Scooping" and "arm-driven hitting" are related, but the useful edge is `arm_driven_hitting --causes--> scooping`, not the reverse.
**Why it happens:** Prose says "scooping is related to arm-driven hitting" without explicit direction.
**How to avoid:** Use the explicit causal chains in learning.md and the causal_links in ftt_core_concepts.json as ground truth for edge direction. When direction is ambiguous in prose, use the `supports` relation type instead of `causes`.
**Warning signs:** Causal chain queries returning empty or nonsensical paths.

### Pitfall 5: Over-Extracting from Video Analysis Files
**What goes wrong:** Files 09_ftt_videos_1/2/3 contain per-video analysis (~20 videos each) with repetitive concepts. Extracting every mention creates massive duplication.
**Why it happens:** The same FTT concept is explained in multiple videos.
**How to avoid:** For video analysis files, extract UNIQUE concepts only. Use the synthesis files (12_ftt_videos_synthesis.md, 15_tpa_synthesis.md) as the primary extraction source -- they already deduplicate across videos. Use the per-video files only to find concepts NOT in the synthesis.
**Warning signs:** 500+ raw concepts from video files alone.

## Code Examples

### Converting Legacy JSON Concept to Pydantic Schema
```python
import re
from knowledge.schemas import Concept, ConceptType

# Map legacy categories to ConceptType
CATEGORY_MAP = {
    "发力模型": ConceptType.BIOMECHANICS,
    "准备阶段": ConceptType.TECHNIQUE,
    "前挥阶段": ConceptType.TECHNIQUE,
    "连接机制": ConceptType.CONNECTION,
    "心理模型": ConceptType.MENTAL_MODEL,
    "症状/错误": ConceptType.SYMPTOM,
}

def to_snake_id(name: str) -> str:
    """Convert English name to snake_case ID."""
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')

def migrate_ftt_concept(raw: dict) -> Concept:
    return Concept(
        id=to_snake_id(raw["name_en"]),
        name=raw["name_en"],
        name_zh=raw["name_zh"],
        aliases=[],
        category=CATEGORY_MAP.get(raw["category"], ConceptType.TECHNIQUE),
        sources=["ftt"],
        description=raw["definition"],
        active_or_passive=raw.get("active_or_passive"),
    )
```

### Extracting Causal Chains from Arrow Notation
```python
import re
from knowledge.schemas import Edge, RelationType

def extract_causal_chain(text: str, source_file: str) -> list[Edge]:
    """Extract edges from '原因A → 导致B → 导致C' patterns."""
    edges = []
    # Match lines with arrow chains (both → and ->)
    arrow_pattern = re.compile(r'(.+?)(?:→|->|==>)')
    
    for line in text.split('\n'):
        parts = re.split(r'\s*(?:→|->|==>)\s*', line.strip())
        if len(parts) >= 2:
            for i in range(len(parts) - 1):
                cause = parts[i].strip().lstrip('- ')
                effect = parts[i + 1].strip()
                if cause and effect:
                    edges.append(Edge(
                        source_id=to_snake_id(cause),
                        target_id=to_snake_id(effect),
                        relation=RelationType.CAUSES,
                        confidence=0.8,
                        evidence=line.strip(),
                        source_file=source_file,
                    ))
    return edges
```

### Extracting Muscle Tables from Biomechanics Files
```python
def extract_muscle_table(text: str) -> list[dict]:
    """Extract muscle data from Markdown tables in biomechanics files."""
    muscles = []
    in_table = False
    headers = []
    
    for line in text.split('\n'):
        if '|' in line and ('肌群' in line or '肌肉' in line or 'muscle' in line.lower()):
            headers = [h.strip() for h in line.split('|')[1:-1]]
            in_table = True
            continue
        if in_table and line.strip().startswith('|---'):
            continue
        if in_table and '|' in line:
            cells = [c.strip() for c in line.split('|')[1:-1]]
            if len(cells) >= len(headers):
                muscles.append(dict(zip(headers, cells)))
        elif in_table and not line.strip():
            in_table = False
    
    return muscles
```

## Detailed File Inventory

### docs/research/ (31 files, ~960KB total)

| File | Lines | Source | Content Type | Key Extraction Targets |
|------|-------|--------|-------------|----------------------|
| 01_ftt_book.md | 644 | FTT | Book notes | Core FTT concepts, Fundamental Theorem, grip, contact point |
| 02_revolutionary_tennis.md | 654 | RT | Book notes | Alternative perspective, complementary concepts |
| 03_youtube_notes.md | 320 | Various | Video notes | Mixed source concepts |
| 04_ftt_blog_forehand_1.md | 915 | FTT Blog | Blog analysis | Forehand technique details |
| 04_ftt_blog_forehand_2.md | 857 | FTT Blog | Blog analysis | Forehand technique details |
| 05_ftt_blog_players.md | 683 | FTT Blog | Player analysis | Pro player technique examples |
| 06_ftt_blog_movement.md | 806 | FTT Blog | Blog analysis | Footwork, movement patterns |
| 07_ftt_blog_vision_movement.md | 771 | FTT Blog | Blog analysis | Vision + movement integration |
| 08_ftt_blog_vision_strategy.md | 668 | FTT Blog | Blog analysis | Vision strategy concepts |
| 09_ftt_videos_1.md | 686 | FTT Videos | Per-video analysis | ~10 video breakdowns |
| 09_ftt_videos_2.md | 629 | FTT Videos | Per-video analysis | ~10 video breakdowns |
| 09_ftt_videos_3.md | 1081 | FTT Videos | Per-video analysis | ~10 video breakdowns |
| 12_ftt_videos_synthesis.md | 477 | FTT Videos | Synthesis | **Deduplicated FTT video concepts** |
| 13_synthesis.md | 945 | All | Master synthesis | **Most comprehensive single file** |
| 14_tpa_videos_1.md | 413 | TPA Videos | Per-video analysis | ~10 video breakdowns |
| 14_tpa_videos_2.md | 427 | TPA Videos | Per-video analysis | ~10 video breakdowns |
| 14_tpa_videos_3.md | 487 | TPA Videos | Per-video analysis | ~10 video breakdowns |
| 15_tpa_synthesis.md | 283 | TPA | Synthesis | **Deduplicated TPA concepts** |
| 16_tpa_kinetic_chain_1.md | 332 | TPA | Deep dive | Kinetic chain details |
| 16_tpa_kinetic_chain_2.md | 312 | TPA | Deep dive | Kinetic chain details |
| 17_kinetic_chain_synthesis.md | 215 | TPA/FTT | Synthesis | **Cross-source kinetic chain** |
| 18_ftt_build_foundation.md | 188 | FTT | Core teaching | Foundation concepts |
| 19_forearm_compensation_analysis.md | 1304 | FTT/User | Deep analysis | Compensation patterns, causal chains |
| 20_ftt_grip_rotation_axis.md | 153 | FTT | Specific concept | Grip rotation details |
| 21_ftt_chest_engagement.md | 99 | FTT | Specific concept | Chest engagement / Press Slot |
| 22_ftt_scapular_glide.md | 86 | FTT | Specific concept | Scapular mechanics |
| 23_ftt_trunk_sequencing.md | 125 | FTT | Specific concept | Trunk sequencing order |
| 24_biomechanics_ch1_ch8.md | 405 | Book | Anatomy + physics | Muscle tables, kinetic chain physics |
| 25_biomechanics_upper_body.md | 502 | Book | Anatomy | Shoulder, chest, back muscle tables |
| 26_biomechanics_core_legs.md | 490 | Book | Anatomy | Core and leg muscle tables |
| 27_biomechanics_new_insights.md | 677 | Book | Analysis | Biomechanical insights mapped to FTT |
| 28_biomechanics_problem_solutions.md | 794 | Book | Problem-solution | Training prescriptions |

### docs/knowledge_graph/ (3 JSON files, pre-extracted)

| File | Size | Content | Concept Count |
|------|------|---------|--------------|
| ftt_core_concepts.json | 53KB | 48 concepts, 15 misconceptions, 15 causal_links, 17 drills | 48 |
| tpa_kinetic_chain.json | 55KB | 24 concepts, diagnostic criteria, training progressions, cross-references | 24 |
| user_journey.json | 43KB | 21 problems, 16 breakthroughs, 57 cue evolutions, problem dependency graph | 37+ |

### docs/record/learning.md (1 file)

| File | Lines | Content |
|------|-------|---------|
| learning.md | 1332 | Date-by-date training journal with explicit causal chains, problem discoveries, drills tried |

## Extraction Processing Order

This order minimizes deduplication issues:

1. **Seed from existing JSON** (ftt_core_concepts.json, tpa_kinetic_chain.json, user_journey.json) -> ~80-100 canonical concepts established
2. **Synthesis files** (13, 12, 15, 17) -> Add any concepts not in JSON, establish cross-source aliases
3. **FTT book + blog** (01, 02, 04x2, 05, 06, 07, 08) -> Fill in details, add aliases
4. **FTT specific concepts** (18, 19, 20, 21, 22, 23) -> Add specialized concepts
5. **FTT video analysis** (09x3) -> Add video-specific concepts not in synthesis
6. **TPA video analysis** (14x3, 16x2) -> Add TPA-specific concepts not in synthesis
7. **Biomechanics** (24, 25, 26, 27, 28) -> Add muscle mappings, anatomy concepts
8. **User journey** (learning.md) -> Link user problems/breakthroughs to canonical concepts
9. **YouTube notes** (03) -> Miscellaneous concepts

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 9.0.2 |
| Config file | tests/ directory, no config needed |
| Quick run command | `python3 -m pytest tests/test_extraction.py -x -q` |
| Full suite command | `python3 -m pytest tests/ -q` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| EXIST-01 | All 31 research files produce JSON output | integration | `pytest tests/test_extraction.py::test_all_research_files_extracted -x` | Wave 0 |
| EXIST-02 | learning.md concepts linked to canonical registry | integration | `pytest tests/test_extraction.py::test_user_journey_extraction -x` | Wave 0 |
| EXIST-03 | Biomechanics files produce muscle-to-concept mappings | unit | `pytest tests/test_extraction.py::test_biomechanics_muscle_mappings -x` | Wave 0 |
| EXIST-04 | Registry contains 150-300 concepts, no obvious duplicates | integration | `pytest tests/test_extraction.py::test_registry_population -x` | Wave 0 |
| EXIST-05 | Graph has edges, causal chain query works | integration | `pytest tests/test_extraction.py::test_graph_edges_and_queries -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `python3 -m pytest tests/test_extraction.py -x -q`
- **Per wave merge:** `python3 -m pytest tests/ -q`
- **Phase gate:** Full suite green before verification

### Wave 0 Gaps
- [ ] `tests/test_extraction.py` -- covers EXIST-01 through EXIST-05
- [ ] Test fixtures with sample Markdown fragments for each file type

## Canonical Concept Seed List

The following ~50 core concepts should be seeded first (derived from existing JSON + synthesis). This is not exhaustive but establishes the most important canonical IDs:

**Technique concepts (~20):**
`unit_turn`, `unit_swing`, `hip_shoulder_separation`, `shoulder_tilt`, `out_vector`, `windshield_wiper`, `press_slot`, `wrist_lag`, `racket_head_drop`, `contact_point`, `follow_through`, `split_step`, `open_stance`, `semi_western_grip`, `loading_phase`, `forward_swing`, `pronation`, `supination`, `scapular_glide`, `trunk_sequencing`

**Biomechanics concepts (~10):**
`rotational_kinetic_chain`, `ground_reaction_force`, `angular_momentum`, `stretch_shortening_cycle`, `eccentric_loading`, `concentric_contraction`, `deceleration_phase`, `internal_shoulder_rotation`, `external_shoulder_rotation`, `core_activation`

**Connection/Mental Model concepts (~10):**
`back_muscle_connection`, `elbow_space`, `fault_tolerance`, `fundamental_theorem_of_tennis`, `place_pull_forward`, `gravity_driven_elbow_drop`, `rotation_axis`, `chest_engagement`, `arm_body_sync`, `relaxation`

**Symptom concepts (~10):**
`scooping`, `arm_driven_hitting`, `forearm_compensation`, `elbow_flying_out`, `late_contact`, `wrist_flipping`, `over_rotation`, `bracing_at_contact`, `collapsed_wrist`, `shoulder_shrugging`

**Drill concepts (~5):**
`half_swing_drill`, `wall_push_drill`, `shadow_swing`, `toss_and_catch`, `contact_point_drill`

## Open Questions

1. **Extraction approach: Claude-in-loop vs pure Python regex**
   - What we know: Research files are well-structured Markdown with consistent patterns (headers, tables, arrow chains). Regex can handle most structure.
   - What's unclear: Some concept extraction requires semantic understanding (e.g., identifying that a paragraph describes a new concept vs elaborating on an existing one).
   - Recommendation: Use Python regex for structural extraction (tables, causal chains, cross-references). For semantic extraction from prose paragraphs, pre-define the concept list (seed) and use keyword matching to associate text with existing concepts rather than discovering new ones from prose.

2. **Edge confidence scoring**
   - What we know: Existing causal_links have no confidence scores. Phase 1 schema requires confidence 0.0-1.0.
   - What's unclear: What confidence to assign edges extracted from different source types.
   - Recommendation: Edges from synthesis files = 0.9, edges from primary sources = 0.8, edges from single per-video analysis = 0.6, edges from user journey = 0.7 (personal experience, validated through practice).

3. **How to handle the 3 existing JSON files long-term**
   - What we know: They have a legacy schema that doesn't match Pydantic models.
   - Recommendation: Migrate them into the new schema during seeding. After migration, the `knowledge/extracted/` output becomes the single source of truth. The legacy JSONs in `docs/knowledge_graph/` remain as historical artifacts but are no longer authoritative.

## Sources

### Primary (HIGH confidence)
- Direct file system inspection of all 31 research files, 3 JSON files, 1 learning.md
- Phase 1 source code: knowledge/schemas.py, registry.py, graph.py
- Existing test suite: 26 tests passing

### Secondary (MEDIUM confidence)
- Architecture patterns from .planning/research/ARCHITECTURE.md
- Pitfall analysis from .planning/research/PITFALLS.md

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - all libraries already installed and tested
- Architecture: HIGH - file inventory is complete, patterns derive from actual file structure
- Pitfalls: HIGH - based on direct inspection of legacy JSON schema mismatches and concept naming variations
- Extraction approach: MEDIUM - regex vs semantic extraction tradeoff needs validation during implementation

**Research date:** 2026-04-03
**Valid until:** 2026-05-03 (stable domain, no external dependencies)
