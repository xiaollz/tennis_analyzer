"""Tests for training plan generator — drill recommendations from UserProfile + KnowledgeGraph."""

from __future__ import annotations

import pytest

from knowledge.graph import KnowledgeGraph
from knowledge.schemas import (
    Concept,
    ConceptType,
    DiagnosticChain,
    DiagnosticStep,
    Edge,
    RelationType,
)
from knowledge.user_profile import (
    ConceptLink,
    ConceptStatus,
    SessionEntry,
    UserProfile,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def graph():
    """Small graph with 3 concepts + 2 drills_for edges."""
    kg = KnowledgeGraph()

    # Symptom / technique concepts
    kg.add_concept(Concept(
        id="forearm_compensation",
        name="Forearm Compensation",
        name_zh="小臂代偿",
        category=ConceptType.SYMPTOM,
        description="Arm drives swing instead of body rotation",
        confidence=0.95,
    ))
    kg.add_concept(Concept(
        id="scooping_motion",
        name="Scooping Motion",
        name_zh="捞球动作",
        category=ConceptType.SYMPTOM,
        description="V-shaped swing path",
        confidence=0.9,
    ))
    kg.add_concept(Concept(
        id="unit_turn",
        name="Unit Turn",
        name_zh="整体转身",
        category=ConceptType.TECHNIQUE,
        description="Body rotation as single unit",
        confidence=0.95,
    ))

    # Drill concepts
    kg.add_concept(Concept(
        id="drop_feed_rotation_drill",
        name="Drop Feed Rotation Drill",
        name_zh="喂球旋转练习",
        category=ConceptType.DRILL,
        description="Practice body rotation with drop feeds",
        confidence=1.0,
    ))
    kg.add_concept(Concept(
        id="shadow_swing_drill",
        name="Shadow Swing Drill",
        name_zh="空挥练习",
        category=ConceptType.DRILL,
        description="Shadow swing focusing on rotation",
        confidence=1.0,
    ))

    # Edges: drills_for
    kg.add_edge(Edge(
        source_id="drop_feed_rotation_drill",
        target_id="forearm_compensation",
        relation=RelationType.DRILLS_FOR,
        confidence=0.9,
        evidence="Drop feed drill addresses arm compensation",
        source_file="ftt_video",
    ))
    kg.add_edge(Edge(
        source_id="shadow_swing_drill",
        target_id="forearm_compensation",
        relation=RelationType.DRILLS_FOR,
        confidence=0.85,
        evidence="Shadow swing helps fix arm compensation",
        source_file="ftt_video",
    ))
    kg.add_edge(Edge(
        source_id="shadow_swing_drill",
        target_id="scooping_motion",
        relation=RelationType.DRILLS_FOR,
        confidence=0.8,
        evidence="Shadow swing also helps with scooping",
        source_file="ftt_video",
    ))

    # Causal edge
    kg.add_edge(Edge(
        source_id="unit_turn",
        target_id="forearm_compensation",
        relation=RelationType.CAUSES,
        confidence=0.85,
        evidence="Missing unit turn causes arm-driven hitting",
        source_file="ftt_book",
    ))

    return kg


@pytest.fixture
def chains():
    return [
        DiagnosticChain(
            id="dc_arm_driven_hitting",
            symptom="Arm initiating swing",
            symptom_zh="手臂主导挥拍",
            symptom_concept_id="forearm_compensation",
            check_sequence=[DiagnosticStep(
                check="Is the hip rotating?",
                check_zh="髋部是否旋转？",
                if_true="forearm_compensation",
                if_false=None,
            )],
            root_causes=["forearm_compensation"],
            drills=["drop_feed_rotation_drill"],
            priority=1,
        ),
    ]


@pytest.fixture
def profile_struggling():
    """Profile with 1 struggling concept (forearm_compensation)."""
    return UserProfile(
        sessions=[],
        concept_map={
            "forearm_compensation": ConceptLink(
                concept_id="forearm_compensation",
                status=ConceptStatus.STRUGGLING,
                first_seen="2026-03-15",
                last_seen="2026-03-24",
                cues=["let body rotate first"],
            ),
        },
    )


@pytest.fixture
def profile_multi_issues():
    """Profile with 2 active issues: struggling + regressed."""
    return UserProfile(
        sessions=[],
        concept_map={
            "forearm_compensation": ConceptLink(
                concept_id="forearm_compensation",
                status=ConceptStatus.STRUGGLING,
                first_seen="2026-03-15",
                last_seen="2026-03-24",
                cues=[],
            ),
            "scooping_motion": ConceptLink(
                concept_id="scooping_motion",
                status=ConceptStatus.REGRESSED,
                first_seen="2026-03-15",
                last_seen="2026-03-24",
                cues=[],
            ),
        },
    )


@pytest.fixture
def profile_empty():
    """Profile with no active issues (all mastered)."""
    return UserProfile(
        sessions=[],
        concept_map={
            "forearm_compensation": ConceptLink(
                concept_id="forearm_compensation",
                status=ConceptStatus.MASTERED,
                first_seen="2026-03-15",
                last_seen="2026-03-24",
                cues=[],
            ),
        },
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTrainingPlanGenerator:

    def test_generates_plan_with_drills(self, graph, chains, profile_struggling):
        """Test 1: generate_training_plan returns a TrainingPlan with prioritized drill list."""
        from knowledge.output.training_plan import generate_training_plan

        plan = generate_training_plan(profile_struggling, graph, chains)
        assert len(plan.drills) > 0
        assert plan.generated_date  # not empty
        assert len(plan.focus_areas) > 0

    def test_drills_from_graph_edges(self, graph, chains, profile_struggling):
        """Test 2: Drills sourced from graph edges with relation=drills_for."""
        from knowledge.output.training_plan import generate_training_plan

        plan = generate_training_plan(profile_struggling, graph, chains)
        drill_ids = [d.drill_id for d in plan.drills]
        # Both drills target forearm_compensation
        assert "drop_feed_rotation_drill" in drill_ids
        assert "shadow_swing_drill" in drill_ids

    def test_multi_issue_drill_ranks_higher(self, graph, chains, profile_multi_issues):
        """Test 3: Drills addressing multiple issues rank higher."""
        from knowledge.output.training_plan import generate_training_plan

        plan = generate_training_plan(profile_multi_issues, graph, chains)
        drill_ids = [d.drill_id for d in plan.drills]
        # shadow_swing_drill addresses both forearm_compensation + scooping_motion
        # so it should rank higher than drop_feed_rotation_drill
        assert drill_ids.index("shadow_swing_drill") < drill_ids.index("drop_feed_rotation_drill")

    def test_drill_has_rationale(self, graph, chains, profile_struggling):
        """Test 4: Output includes rationale per drill."""
        from knowledge.output.training_plan import generate_training_plan

        plan = generate_training_plan(profile_struggling, graph, chains)
        for drill in plan.drills:
            assert drill.rationale, f"Drill {drill.drill_id} missing rationale"
            assert len(drill.addresses) > 0

    def test_to_markdown_rendering(self, graph, chains, profile_struggling):
        """Test 5: Output includes Markdown rendering via to_markdown()."""
        from knowledge.output.training_plan import generate_training_plan

        plan = generate_training_plan(profile_struggling, graph, chains)
        md = plan.to_markdown()
        assert isinstance(md, str)
        assert len(md) > 0
        # Should have Chinese headers
        assert "训练计划" in md or "练习" in md

    def test_empty_profile_returns_all_clear(self, graph, chains, profile_empty):
        """Test 6: Empty profile (no struggles) returns empty plan with all clear message."""
        from knowledge.output.training_plan import generate_training_plan

        plan = generate_training_plan(profile_empty, graph, chains)
        assert len(plan.drills) == 0
        assert "mastered" in plan.summary.lower() or "improving" in plan.summary.lower() or "all clear" in plan.summary.lower()
