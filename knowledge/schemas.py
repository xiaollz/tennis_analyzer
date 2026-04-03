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
