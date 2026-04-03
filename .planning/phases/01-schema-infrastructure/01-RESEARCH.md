# Phase 1: Schema & Infrastructure - Research

**Researched:** 2026-04-03
**Domain:** Pydantic data models, NetworkX graph backend, concept deduplication
**Confidence:** HIGH

## Summary

Phase 1 establishes the data foundation for the entire knowledge engineering system. The core deliverables are: (1) Pydantic v2 models for Concept, Edge, and DiagnosticChain, (2) a canonical concept registry with fuzzy dedup, (3) a NetworkX MultiDiGraph as the in-memory knowledge graph, and (4) pytest coverage for all models and registry operations.

The existing codebase already contains three hand-crafted JSON files in `docs/knowledge_graph/` (`ftt_core_concepts.json`, `tpa_kinetic_chain.json`, `user_journey.json`) with ~40 concepts and ~20 problems. These provide concrete field patterns and real data for testing. The existing concept ID schemes (C01-C40 for FTT, T01-T30 for TPA, P01-P21 for problems) need to be migrated to a unified snake_case ID system (e.g., `hip_rotation`, `unit_turn`) for graph interoperability.

**Primary recommendation:** Build schemas that precisely model the existing JSON structures, add `aliases` and `confidence` fields for dedup, use `rapidfuzz` for string matching, and `NetworkX.MultiDiGraph` for the graph backend with `node_link_data` JSON serialization.

## Project Constraints (from CLAUDE.md)

- Chinese-first responses unless user asks in English
- FTT (Fault Tolerant Forehand) system is authoritative; conflicts resolved in its favor
- Core principle: forehand is a rotational whip system; distinguish active actions vs passive results
- Existing codebase: `core/`, `evaluation/`, `analysis/` directories; `docs/research/` has 28+ research files

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| INFRA-01 | Define Pydantic schema for concepts (id, name, aliases, source, type, description, muscles, VLM features) | Concept model validated with Pydantic v2.12.5; field structure derived from existing ftt_core_concepts.json (C01-C40 patterns) |
| INFRA-02 | Define Pydantic schema for edges (source->target, type: causes/fixes/requires/contradicts, confidence, evidence) | Edge/Relationship model with RelationType enum; 7 edge types identified from existing diagnostic_engine.md causal chains |
| INFRA-03 | Define Pydantic schema for diagnostic chains (symptom->root causes->drills, with branching logic) | DiagnosticChain model with ordered check_sequence and branching conditions; patterns from user_journey.json P01-P21 |
| INFRA-04 | Build canonical concept registry with deduplication (fuzzy match + LLM-assisted merge) | rapidfuzz 3.14.3 for string matching; bilingual dedup strategy using English-only matching with Chinese as display |
| INFRA-05 | Set up NetworkX directed multigraph as knowledge graph backend | NetworkX 3.6.1 MultiDiGraph verified: supports multiple typed edges, node_link_data JSON roundtrip, shortest_path for causal chains |
</phase_requirements>

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pydantic | 2.12.5 (installed) | Schema definitions for Concept, Edge, DiagnosticChain | Already installed. v2 API with `model_dump_json()`, Field validators, pattern constraints |
| networkx | 3.6.1 (installed) | In-memory knowledge graph with MultiDiGraph | Already installed. Supports directed multigraph, `node_link_data` JSON serialization, path algorithms |
| rapidfuzz | 3.14.3 (to install) | Fuzzy string matching for concept deduplication | C-optimized, 10-100x faster than thefuzz/fuzzywuzzy. Provides `fuzz.ratio`, `fuzz.partial_ratio`, `process.extractOne` |
| pytest | 8.3.5 (installed) | Test framework | Already in use for existing test_v2_modules.py |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| uuid (stdlib) | - | Generate unique IDs when needed | Fallback for auto-generated concept IDs |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| rapidfuzz | thefuzz/fuzzywuzzy | 10-100x slower, pure Python. rapidfuzz is drop-in compatible |
| MultiDiGraph | DiGraph | DiGraph cannot have multiple edges between same pair (e.g., A causes B AND A requires B) |
| node_link_data JSON | pickle | Pickle is not human-readable, not git-diffable, breaks across Python versions |
| snake_case IDs | Sequential IDs (C01, T01) | Sequential IDs carry no semantic meaning, make graph queries opaque |

**Installation:**
```bash
pip install rapidfuzz>=3.14.0
```

## Architecture Patterns

### Recommended Project Structure
```
knowledge/                    # New top-level package
  __init__.py
  schemas.py                  # Pydantic models (Concept, Edge, DiagnosticChain, enums)
  registry.py                 # Canonical concept registry with dedup
  graph.py                    # NetworkX MultiDiGraph wrapper (build, query, serialize)
tests/
  test_knowledge_schemas.py   # Model instantiation, validation, serialization
  test_knowledge_registry.py  # Dedup, fuzzy match, canonical ID resolution
  test_knowledge_graph.py     # Graph construction, causal chain queries
```

### Pattern 1: Pydantic Models with Snake-Case IDs

**What:** All concept IDs use lowercase snake_case (e.g., `hip_rotation`, `unit_turn`). Validated by regex pattern at the Pydantic level.

**When to use:** Every concept created or imported.

**Example:**
```python
from pydantic import BaseModel, Field
from enum import Enum

class ConceptType(str, Enum):
    TECHNIQUE = "technique"
    BIOMECHANICS = "biomechanics"
    DRILL = "drill"
    SYMPTOM = "symptom"
    MENTAL_MODEL = "mental_model"
    CONNECTION = "connection"

class SourceId(str, Enum):
    FTT = "ftt"
    TPA = "tpa"
    FEEL_TENNIS = "feel_tennis"
    BIOMECHANICS_BOOK = "biomechanics_book"
    USER_EXPERIENCE = "user_experience"

class Concept(BaseModel):
    id: str = Field(pattern=r'^[a-z][a-z0-9_]*$', description="Snake-case canonical ID")
    name: str = Field(description="English display name")
    name_zh: str = Field(description="Chinese display name")
    aliases: list[str] = Field(default_factory=list, description="Alternative names for dedup matching")
    category: ConceptType
    sources: list[str] = Field(default_factory=list, description="Source identifiers")
    description: str
    vlm_features: list[str] = Field(default_factory=list, description="VLM-observable visual features")
    muscles_involved: list[str] = Field(default_factory=list)
    active_or_passive: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
```

### Pattern 2: Typed Directed Edges with Evidence

**What:** Every edge has a direction, a type from a fixed enum, and evidence provenance.

**When to use:** Every relationship in the graph.

**Example:**
```python
class RelationType(str, Enum):
    CAUSES = "causes"           # A biomechanically causes B
    PREVENTS = "prevents"       # A prevents B from occurring
    REQUIRES = "requires"       # A requires B as prerequisite
    CONTRADICTS = "contradicts" # A contradicts B (cross-source conflict)
    SUPPORTS = "supports"       # A reinforces/agrees with B
    DRILLS_FOR = "drills_for"   # Drill A trains Concept B
    VISIBLE_AS = "visible_as"   # Concept A is visible as VLM Feature B

class Edge(BaseModel):
    source_id: str = Field(pattern=r'^[a-z][a-z0-9_]*$')
    target_id: str = Field(pattern=r'^[a-z][a-z0-9_]*$')
    relation: RelationType
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: str = Field(description="Quote, reference, or reasoning")
    source_file: str = Field(description="Which research file or video")
    resolution: str | None = Field(default=None, description="For contradictions: ftt_wins, etc.")
```

### Pattern 3: DiagnosticChain with Branching Logic

**What:** A diagnostic chain models the VLM diagnostic workflow: observe symptom -> check conditions -> identify root cause -> prescribe drill.

**When to use:** Every VLM-diagnosable symptom.

**Key insight from existing data:** The `user_journey.json` shows that problems have multiple root causes with different check sequences. P02 (V-shape Scooping) has 4 distinct root causes, each requiring different checks. The model must support branching.

**Example:**
```python
class DiagnosticStep(BaseModel):
    """A single step in the diagnostic check sequence."""
    check: str = Field(description="What to look for (VLM instruction)")
    check_zh: str = Field(description="Chinese description")
    if_true: str = Field(description="Concept ID of root cause if check is positive")
    if_false: str | None = Field(default=None, description="Next step or None if end")

class DiagnosticChain(BaseModel):
    id: str = Field(pattern=r'^dc_[a-z][a-z0-9_]*$')
    symptom: str = Field(description="VLM-observable symptom description")
    symptom_zh: str
    symptom_concept_id: str = Field(description="Concept ID of the symptom")
    check_sequence: list[DiagnosticStep] = Field(description="Ordered investigation steps with branching")
    root_causes: list[str] = Field(description="All possible root cause concept IDs")
    drills: list[str] = Field(description="Drill concept IDs for remediation")
    priority: int = Field(ge=1, le=5, description="1=most common/important")
    vlm_frame: str | None = Field(default=None, description="Which video frame to analyze: prep_complete, forward_swing_start, contact, etc.")
```

### Pattern 4: Canonical Concept Registry with Fuzzy Dedup

**What:** A registry that stores all canonical concepts and can detect near-duplicates when adding new ones.

**Key design decisions:**
1. Match on English name + aliases only (Chinese names are display-only, not for matching)
2. Use rapidfuzz `token_sort_ratio` (handles word order: "hip rotation" matches "rotation of hips")
3. Threshold: 85 for exact match warning, 70-84 for review suggestion
4. Registry returns the canonical ID for any input string

**Example:**
```python
from rapidfuzz import fuzz, process

class ConceptRegistry:
    def __init__(self):
        self._concepts: dict[str, Concept] = {}
        self._name_index: dict[str, str] = {}  # name/alias -> concept_id

    def add(self, concept: Concept) -> str | None:
        """Add concept. Returns conflicting ID if near-duplicate found."""
        # Check for exact ID collision
        if concept.id in self._concepts:
            return concept.id

        # Check fuzzy match against all known names
        all_names = list(self._name_index.keys())
        if all_names:
            match, score, _ = process.extractOne(
                concept.name.lower(),
                all_names,
                scorer=fuzz.token_sort_ratio
            )
            if score >= 85:
                return self._name_index[match]  # Near-duplicate found

            # Also check all aliases
            for alias in concept.aliases:
                match, score, _ = process.extractOne(
                    alias.lower(),
                    all_names,
                    scorer=fuzz.token_sort_ratio
                )
                if score >= 85:
                    return self._name_index[match]

        # No duplicate — register
        self._concepts[concept.id] = concept
        self._name_index[concept.name.lower()] = concept.id
        for alias in concept.aliases:
            self._name_index[alias.lower()] = concept.id

        return None  # Successfully added

    def get(self, concept_id: str) -> Concept | None:
        return self._concepts.get(concept_id)

    def resolve(self, name: str, threshold: int = 70) -> str | None:
        """Resolve a name/alias to canonical ID via fuzzy match."""
        all_names = list(self._name_index.keys())
        if not all_names:
            return None
        match, score, _ = process.extractOne(
            name.lower(), all_names, scorer=fuzz.token_sort_ratio
        )
        if score >= threshold:
            return self._name_index[match]
        return None
```

### Pattern 5: NetworkX MultiDiGraph with JSON Roundtrip

**What:** Wrap NetworkX MultiDiGraph with typed methods for adding concepts/edges and serializing to/from JSON.

**Example:**
```python
import networkx as nx
import json
from pathlib import Path

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def add_concept(self, concept: Concept):
        self.graph.add_node(concept.id, **concept.model_dump())

    def add_edge(self, edge: Edge):
        self.graph.add_edge(
            edge.source_id, edge.target_id,
            key=edge.relation.value,
            **edge.model_dump()
        )

    def get_causal_chain(self, symptom_id: str, cause_type: str = "causes") -> list[list[str]]:
        """Find all causal paths from symptom to root causes."""
        # Walk backwards through 'causes' edges to find root causes
        paths = []
        def _walk(node, path):
            predecessors = [
                (u, data)
                for u, _, data in self.graph.in_edges(node, data=True)
                if data.get("relation") == cause_type
            ]
            if not predecessors:
                paths.append(list(path))
                return
            for pred, _ in predecessors:
                if pred not in path:  # Avoid cycles
                    _walk(pred, path + [pred])
        _walk(symptom_id, [symptom_id])
        return paths

    def to_json(self, path: Path):
        data = nx.node_link_data(self.graph)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    @classmethod
    def from_json(cls, path: Path) -> "KnowledgeGraph":
        data = json.loads(path.read_text())
        kg = cls()
        kg.graph = nx.node_link_graph(data, multigraph=True, directed=True)
        return kg
```

### Anti-Patterns to Avoid

- **Sequential IDs (C01, T01, P01):** The existing JSON files use these. Migrate to semantic snake_case IDs. Sequential IDs make graph queries unreadable (`graph.successors("C04")` vs `graph.successors("back_muscle_connection")`).
- **Chinese names for matching:** Use English-only for dedup matching. Chinese has too many valid translations for the same concept ("转髋" vs "髋部旋转" vs "旋髋").
- **Undirected edges:** Always use directed edges. "A causes B" is not the same as "B causes A".
- **Single-edge DiGraph:** Use MultiDiGraph. Concept A can both `cause` and `require` Concept B simultaneously.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Fuzzy string matching | Custom Levenshtein | rapidfuzz `fuzz.token_sort_ratio` | Handles word order, Unicode, C-optimized |
| Graph serialization | Custom JSON format | `nx.node_link_data` / `nx.node_link_graph` | Standard format, handles multigraph, tested |
| Data validation | Manual dict checking | Pydantic v2 validators | Type coercion, error messages, JSON schema generation |
| Graph traversal | BFS/DFS from scratch | NetworkX `shortest_path`, `predecessors`, `successors` | Battle-tested, handles edge cases |

## Common Pitfalls

### Pitfall 1: Concept ID Collision from Different Sources
**What goes wrong:** FTT's "Chest Engagement" (C19) and TPA's concept about chest connection get different IDs but represent overlapping concepts.
**Why it happens:** Sources use different terminology for similar biomechanical concepts.
**How to avoid:** The registry's fuzzy match catches this at insert time. Set aliases comprehensively. Include source-specific names as aliases.
**Warning signs:** Graph has > 300 concept nodes after processing FTT alone (expected: ~100-150).

### Pitfall 2: DiagnosticChain Without Frame Context
**What goes wrong:** A diagnostic chain says "check elbow angle" but doesn't specify which video frame (prep, contact, follow-through). Elbow angle is different at each frame.
**Why it happens:** The existing diagnostic_engine.md already specifies frames (prep_complete, forward_swing_start, pre_contact, contact, early_followthrough, followthrough_end).
**How to avoid:** Make `vlm_frame` a required consideration in DiagnosticChain. Each check_sequence step should reference the relevant frame.
**Warning signs:** DiagnosticChain entries with no `vlm_frame` field.

### Pitfall 3: Circular Causal Chains
**What goes wrong:** A -> causes -> B -> causes -> A creates infinite loops in diagnostic traversal.
**Why it happens:** Bidirectional relationships exist (poor technique causes poor timing, poor timing causes poor technique).
**How to avoid:** Cycle detection in the `get_causal_chain` method (already shown in Pattern 5 with `if pred not in path`). Also add a graph validation pass: `nx.find_cycle(graph)` should not find cycles in cause-only edges.
**Warning signs:** `get_causal_chain` returns empty or times out.

### Pitfall 4: Aliases Not Lowercased Consistently
**What goes wrong:** "Hip Rotation" and "hip rotation" treated as different strings in the registry.
**Why it happens:** Pydantic stores the original casing but matching needs normalization.
**How to avoid:** Always `.lower()` before inserting into `_name_index` and before querying. This is already shown in the registry pattern.

## Code Examples

### Complete Model Instantiation and Serialization Test
```python
def test_concept_roundtrip():
    c = Concept(
        id="hip_rotation",
        name="Hip Rotation",
        name_zh="转髋",
        aliases=["hip turn", "hip drive"],
        category=ConceptType.TECHNIQUE,
        sources=["ftt", "tpa"],
        description="Rotational movement of hips initiating kinetic chain",
        vlm_features=["hip angle change in frontal view"],
        muscles_involved=["gluteus maximus", "hip flexors"],
        active_or_passive="active",
        confidence=1.0,
    )
    json_str = c.model_dump_json()
    c2 = Concept.model_validate_json(json_str)
    assert c == c2
    assert c2.id == "hip_rotation"

def test_edge_validation():
    e = Edge(
        source_id="hip_rotation",
        target_id="arm_lag",
        relation=RelationType.CAUSES,
        confidence=0.9,
        evidence="FTT: hip rotation initiates kinetic chain",
        source_file="13_synthesis.md",
    )
    assert e.relation == RelationType.CAUSES

def test_diagnostic_chain():
    dc = DiagnosticChain(
        id="dc_arm_driven_hitting",
        symptom="Arm moves independently of trunk rotation",
        symptom_zh="手臂独立于躯干旋转运动",
        symptom_concept_id="arm_driven_hitting",
        check_sequence=[
            DiagnosticStep(
                check="Does the arm start moving before hip rotation is visible?",
                check_zh="手臂是否在髋部旋转可见之前开始移动?",
                if_true="kinetic_chain_break",
                if_false=None,
            )
        ],
        root_causes=["kinetic_chain_break", "back_tension_loss"],
        drills=["hips_hit_drill", "weighted_shadow_swing"],
        priority=1,
        vlm_frame="forward_swing_start",
    )
    assert dc.priority == 1
```

### Registry Dedup Test
```python
def test_registry_dedup():
    registry = ConceptRegistry()
    c1 = Concept(id="hip_rotation", name="Hip Rotation", name_zh="转髋",
                 category=ConceptType.TECHNIQUE, sources=["ftt"],
                 description="Hip rotational movement")
    result = registry.add(c1)
    assert result is None  # Successfully added

    # Near-duplicate should be detected
    c2 = Concept(id="hip_turn", name="Hip Turn", name_zh="转髋动作",
                 category=ConceptType.TECHNIQUE, sources=["tpa"],
                 description="Turning the hips")
    result = registry.add(c2)
    assert result == "hip_rotation"  # Detected as near-duplicate

    # Resolve by alias
    canonical = registry.resolve("hip drive")
    # May or may not match depending on threshold

def test_registry_no_false_positive():
    registry = ConceptRegistry()
    registry.add(Concept(id="hip_rotation", name="Hip Rotation", name_zh="转髋",
                         category=ConceptType.TECHNIQUE, sources=["ftt"],
                         description="Hip rotation"))
    registry.add(Concept(id="shoulder_tilt", name="Shoulder Tilt", name_zh="肩膀倾斜",
                          category=ConceptType.TECHNIQUE, sources=["ftt"],
                          description="Shoulder tilt"))
    # These are genuinely different concepts — should not match
    assert len(registry._concepts) == 2
```

### Graph Causal Chain Test
```python
def test_causal_chain():
    kg = KnowledgeGraph()
    # Add symptom -> intermediate -> root cause
    kg.add_concept(Concept(id="scooping", name="Scooping", name_zh="向上捞球",
                           category=ConceptType.SYMPTOM, description="V-shape swing path"))
    kg.add_concept(Concept(id="racket_head_drop", name="Excessive Racket Head Drop",
                           name_zh="拍头过度下坠", category=ConceptType.SYMPTOM,
                           description="Racket drops too low"))
    kg.add_concept(Concept(id="active_pat_dog", name="Active Pat the Dog",
                           name_zh="主动下压拍头", category=ConceptType.TECHNIQUE,
                           description="Actively pushing racket head down"))
    # active_pat_dog -> causes -> racket_head_drop -> causes -> scooping
    kg.add_edge(Edge(source_id="active_pat_dog", target_id="racket_head_drop",
                     relation=RelationType.CAUSES, confidence=0.95,
                     evidence="FTT Book Ch3", source_file="01_ftt_book.md"))
    kg.add_edge(Edge(source_id="racket_head_drop", target_id="scooping",
                     relation=RelationType.CAUSES, confidence=0.9,
                     evidence="User P01->P02 chain", source_file="user_journey.json"))
    # Query causal chain
    chains = kg.get_causal_chain("scooping")
    assert len(chains) >= 1
    assert "active_pat_dog" in chains[0]
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Pydantic v1 `.dict()` / `.json()` | Pydantic v2 `.model_dump()` / `.model_dump_json()` | 2023 | Must use v2 API exclusively |
| `nx.readwrite.json_graph.node_link_data` | `nx.node_link_data` (top-level) | NetworkX 3.0 | Import path changed |
| fuzzywuzzy (python-Levenshtein) | rapidfuzz | 2020+ | 10-100x faster, no GPL dependency |

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 8.3.5 |
| Config file | pyproject.toml (no pytest section yet -- add `[tool.pytest.ini_options]` in Wave 0) |
| Quick run command | `pytest tests/test_knowledge_schemas.py tests/test_knowledge_registry.py tests/test_knowledge_graph.py -x` |
| Full suite command | `pytest tests/ -x` |

### Phase Requirements -> Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| INFRA-01 | Concept model instantiation + JSON roundtrip | unit | `pytest tests/test_knowledge_schemas.py::test_concept_roundtrip -x` | Wave 0 |
| INFRA-01 | Concept ID regex validation (reject invalid IDs) | unit | `pytest tests/test_knowledge_schemas.py::test_concept_id_validation -x` | Wave 0 |
| INFRA-02 | Edge model with all RelationType values | unit | `pytest tests/test_knowledge_schemas.py::test_edge_validation -x` | Wave 0 |
| INFRA-03 | DiagnosticChain with branching steps | unit | `pytest tests/test_knowledge_schemas.py::test_diagnostic_chain -x` | Wave 0 |
| INFRA-04 | Registry add + dedup detection | unit | `pytest tests/test_knowledge_registry.py::test_registry_dedup -x` | Wave 0 |
| INFRA-04 | Registry resolve by fuzzy name | unit | `pytest tests/test_knowledge_registry.py::test_registry_resolve -x` | Wave 0 |
| INFRA-04 | Registry no false positives | unit | `pytest tests/test_knowledge_registry.py::test_registry_no_false_positive -x` | Wave 0 |
| INFRA-05 | MultiDiGraph add nodes + edges | unit | `pytest tests/test_knowledge_graph.py::test_graph_add -x` | Wave 0 |
| INFRA-05 | Causal chain query (A->B->C) | unit | `pytest tests/test_knowledge_graph.py::test_causal_chain -x` | Wave 0 |
| INFRA-05 | JSON roundtrip (serialize + deserialize) | unit | `pytest tests/test_knowledge_graph.py::test_graph_json_roundtrip -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/test_knowledge_schemas.py tests/test_knowledge_registry.py tests/test_knowledge_graph.py -x`
- **Per wave merge:** `pytest tests/ -x`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_knowledge_schemas.py` -- covers INFRA-01, INFRA-02, INFRA-03
- [ ] `tests/test_knowledge_registry.py` -- covers INFRA-04
- [ ] `tests/test_knowledge_graph.py` -- covers INFRA-05
- [ ] `knowledge/__init__.py` -- package init
- [ ] `knowledge/schemas.py` -- model definitions
- [ ] `knowledge/registry.py` -- concept registry
- [ ] `knowledge/graph.py` -- graph wrapper
- [ ] `pip install rapidfuzz` -- dedup dependency
- [ ] Add `[tool.pytest.ini_options]` to pyproject.toml with `testpaths = ["tests"]`

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.11 | All | Yes | 3.11.14 | -- |
| pydantic | INFRA-01/02/03 | Yes | 2.12.5 | -- |
| networkx | INFRA-05 | Yes | 3.6.1 | -- |
| pytest | Testing | Yes | 8.3.5 | -- |
| rapidfuzz | INFRA-04 | No (not installed) | 3.14.3 available | Must install |

**Missing dependencies with no fallback:**
- rapidfuzz: Must `pip install rapidfuzz>=3.14.0` before implementing INFRA-04

**Missing dependencies with fallback:**
- None

## Open Questions

1. **Fuzzy match threshold tuning**
   - What we know: 85 is a reasonable default for token_sort_ratio
   - What's unclear: Whether tennis-specific terms like "hip rotation" vs "hip coil" score above 85 (they share only 1 word)
   - Recommendation: Start with 85 for auto-match, 70 for "review suggestion". Test with the 40+ existing concepts in ftt_core_concepts.json to calibrate. Aliases solve the gap -- "hip rotation" gets aliases ["hip turn", "hip drive", "hip coil"].

2. **Existing JSON migration**
   - What we know: ftt_core_concepts.json uses C01-C40 sequential IDs, tpa_kinetic_chain.json uses T01-T30
   - What's unclear: Whether migration should happen in Phase 1 or Phase 2
   - Recommendation: Phase 1 defines the schemas only. Phase 2 (EXIST-01 through EXIST-05) handles actual extraction and migration from existing files. Phase 1 tests should use hand-crafted test data, not load existing JSONs.

3. **DiagnosticStep branching complexity**
   - What we know: user_journey.json P02 has 4 root causes, each with independent check paths
   - What's unclear: Whether a simple linear `check_sequence` with `if_true`/`if_false` is sufficient or a full decision tree is needed
   - Recommendation: Start with the simple `DiagnosticStep` model (linear with branching at each step). If Phase 2 extraction reveals more complex patterns, extend then. YAGNI.

## Sources

### Primary (HIGH confidence)
- Existing codebase: `docs/knowledge_graph/ftt_core_concepts.json` (40 concepts with field patterns)
- Existing codebase: `docs/knowledge_graph/tpa_kinetic_chain.json` (30 concepts with cross-source fields)
- Existing codebase: `docs/knowledge_graph/user_journey.json` (21 problems with diagnostic patterns)
- Existing codebase: `docs/knowledge_graph/diagnostic_engine.md` (6-frame VLM diagnostic guide)
- Verified: Pydantic 2.12.5 installed, v2 API (`model_dump_json`) works
- Verified: NetworkX 3.6.1 installed, MultiDiGraph + `node_link_data` JSON roundtrip works
- Verified: rapidfuzz 3.14.3 available on PyPI

### Secondary (MEDIUM confidence)
- `.planning/research/STACK.md` - Stack decisions (verified against installed packages)
- `.planning/research/ARCHITECTURE.md` - Architecture patterns (validated against codebase)
- `.planning/research/PITFALLS.md` - Domain pitfalls (derived from codebase analysis)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - all packages verified installed or available on PyPI
- Architecture: HIGH - patterns derived from existing JSON data structures and validated with working code
- Pitfalls: HIGH - based on concrete existing data patterns (40+ concepts with known duplication issues)

**Research date:** 2026-04-03
**Valid until:** 2026-05-03 (stable domain, no external API dependencies in this phase)
