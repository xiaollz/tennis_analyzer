"""Pydantic v2 data models for the tennis forehand knowledge engineering system.

Defines three core models:
- Concept: A knowledge node (technique, biomechanics, drill, symptom, etc.)
- Edge: A typed directed relationship between two concepts
- DiagnosticChain: A VLM diagnostic workflow (symptom -> checks -> root cause -> drill)
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


# --- Enums ---


class ConceptType(str, Enum):
    """Category of a knowledge concept."""

    TECHNIQUE = "technique"
    BIOMECHANICS = "biomechanics"
    DRILL = "drill"
    SYMPTOM = "symptom"
    MENTAL_MODEL = "mental_model"
    CONNECTION = "connection"


class SourceId(str, Enum):
    """Canonical source identifiers for knowledge provenance."""

    FTT = "ftt"
    TPA = "tpa"
    FEEL_TENNIS = "feel_tennis"
    BIOMECHANICS_BOOK = "biomechanics_book"
    USER_EXPERIENCE = "user_experience"


class RelationType(str, Enum):
    """Types of directed relationships between concepts."""

    CAUSES = "causes"
    PREVENTS = "prevents"
    REQUIRES = "requires"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"
    DRILLS_FOR = "drills_for"
    VISIBLE_AS = "visible_as"


# --- Models ---


class Concept(BaseModel):
    """A knowledge node in the tennis forehand knowledge graph.

    IDs must be lowercase snake_case (e.g., 'hip_rotation', 'unit_turn').
    """

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$", description="Snake-case canonical ID")
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


class Edge(BaseModel):
    """A typed directed relationship between two concepts."""

    source_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    target_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    relation: RelationType
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: str = Field(description="Quote, reference, or reasoning")
    source_file: str = Field(description="Which research file or video")
    resolution: str | None = Field(default=None, description="For contradictions: ftt_wins, etc.")


class DiagnosticStep(BaseModel):
    """A single step in a diagnostic check sequence."""

    check: str = Field(description="What to look for (VLM instruction)")
    check_zh: str = Field(description="Chinese description")
    if_true: str = Field(description="Concept ID of root cause if check is positive")
    if_false: str | None = Field(default=None, description="Next step or None if end")


class DiagnosticChain(BaseModel):
    """A VLM diagnostic workflow: symptom -> ordered checks -> root cause -> drill.

    IDs must start with 'dc_' prefix (e.g., 'dc_arm_driven_hitting').
    """

    id: str = Field(pattern=r"^dc_[a-z][a-z0-9_]*$")
    symptom: str = Field(description="VLM-observable symptom description")
    symptom_zh: str
    symptom_concept_id: str = Field(description="Concept ID of the symptom")
    check_sequence: list[DiagnosticStep] = Field(description="Ordered investigation steps with branching")
    root_causes: list[str] = Field(description="All possible root cause concept IDs")
    drills: list[str] = Field(description="Drill concept IDs for remediation")
    priority: int = Field(ge=1, le=5, description="1=most common/important")
    vlm_frame: str | None = Field(default=None, description="Which video frame to analyze")


# --- Multi-round VLM diagnostic models ---


class HypothesisStatus(str, Enum):
    """Status of a diagnostic hypothesis through the multi-round loop."""

    ACTIVE = "active"
    CONFIRMED = "confirmed"
    ELIMINATED = "eliminated"


class ObservationJudgment(str, Enum):
    """VLM judgment for a single observation check."""

    YES = "yes"
    NO = "no"
    UNCLEAR = "unclear"


class Observation(BaseModel):
    """A single VLM observation anchored to a specific frame and round."""

    id: str = Field(description="e.g. obs_r1_01")
    round_number: int = Field(ge=0)
    frame: str = Field(description="Which frame in the keyframe grid, e.g. 图3")
    description: str = Field(description="What was seen")
    judgment: ObservationJudgment
    confidence: float = Field(ge=0.0, le=1.0)
    directive_source: str = Field(description="Which check_step or directive generated this")
    is_anchored: bool = Field(default=True, description="Whether observation has valid frame+feature anchoring")
    override_reason: str | None = Field(default=None, description="If overridden by quantitative data, why")


class HypothesisUpdate(BaseModel):
    """A per-round update action on a hypothesis."""

    hypothesis_id: str
    action: str = Field(description="confirm, eliminate, or adjust")
    reason: str


class Hypothesis(BaseModel):
    """A diagnostic hypothesis linking a DiagnosticChain to observations."""

    id: str = Field(description="e.g. hyp_scooping_active_lag")
    chain_id: str = Field(description="DiagnosticChain ID, dc_ prefix")
    root_cause_concept_id: str = Field(description="From chain.root_causes")
    name: str
    name_zh: str
    status: HypothesisStatus = HypothesisStatus.ACTIVE
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    supporting_observations: list[str] = Field(default_factory=list, description="Observation IDs")
    contradicting_observations: list[str] = Field(default_factory=list)
    check_steps_completed: list[int] = Field(default_factory=list, description="Indices into chain.check_sequence")
    round_introduced: int = Field(ge=0)
    round_resolved: int | None = None


class RoundResult(BaseModel):
    """Captures the full input/output of a single VLM diagnostic round."""

    round_number: int = Field(ge=0)
    prompt_sent: str = Field(description="Full prompt text sent to VLM")
    raw_response: str = Field(description="Raw VLM output")
    observations: list[Observation] = Field(default_factory=list)
    hypothesis_updates: list[HypothesisUpdate] = Field(default_factory=list)
    timestamp: str | None = None


class DiagnosticSession(BaseModel):
    """Full state of a multi-round VLM diagnostic session."""

    session_id: str = Field(description="e.g. sess_20260403_143000")
    video_path: str | None = None
    image_b64_hash: str | None = None
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    observations: list[Observation] = Field(default_factory=list)
    rounds: list[RoundResult] = Field(default_factory=list)
    active_chain_ids: list[str] = Field(default_factory=list)
    checked_steps: dict[str, list[int]] = Field(default_factory=dict)
    convergence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    max_rounds: int = Field(default=4, ge=1, le=10)
    final_result: dict | None = None
    supplementary_metrics: dict | None = Field(default=None, description="YOLO kinematic data for cross-validation")
    reobserve_candidates: list[str] = Field(default_factory=list, description="Observation IDs needing re-observation")
    contradictions: list[dict] = Field(default_factory=list, description="Detected contradiction pairs")
    created_at: str | None = None
    completed_at: str | None = None
