"""Tests for UserProfile model, session parsing, and concept linking."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from knowledge.user_profile import (
    ConceptLink,
    ConceptStatus,
    SessionEntry,
    UserProfile,
    build_profile_from_learning,
)


# ---------------------------------------------------------------------------
# Model unit tests
# ---------------------------------------------------------------------------

class TestConceptStatus:
    def test_enum_values(self):
        assert ConceptStatus.STRUGGLING == "struggling"
        assert ConceptStatus.IMPROVING == "improving"
        assert ConceptStatus.MASTERED == "mastered"
        assert ConceptStatus.REGRESSED == "regressed"

    def test_all_values_present(self):
        values = {s.value for s in ConceptStatus}
        assert values == {"struggling", "improving", "mastered", "regressed"}


class TestSessionEntry:
    def test_construction(self):
        entry = SessionEntry(
            date="2026-03-15",
            summary="Discovered scooping cause",
            concept_links=[
                ConceptLink(
                    concept_id="scooping_motion",
                    status=ConceptStatus.STRUGGLING,
                    first_seen="2026-03-15",
                    last_seen="2026-03-15",
                    cues=["Unit Turn 后手臂什么都不做"],
                ),
            ],
            breakthroughs=["Found scooping root cause"],
        )
        assert entry.date == "2026-03-15"
        assert len(entry.concept_links) == 1
        assert entry.concept_links[0].concept_id == "scooping_motion"

    def test_serialization_roundtrip(self):
        entry = SessionEntry(
            date="2026-03-15",
            summary="Test",
            concept_links=[],
            breakthroughs=[],
        )
        data = entry.model_dump()
        restored = SessionEntry.model_validate(data)
        assert restored == entry


class TestUserProfile:
    @pytest.fixture
    def profile(self):
        return UserProfile(
            sessions=[
                SessionEntry(
                    date="2026-03-15",
                    summary="Session 1",
                    concept_links=[
                        ConceptLink(
                            concept_id="scooping_motion",
                            status=ConceptStatus.STRUGGLING,
                            first_seen="2026-03-15",
                            last_seen="2026-03-15",
                            cues=["avoid scooping"],
                        ),
                        ConceptLink(
                            concept_id="unit_turn",
                            status=ConceptStatus.IMPROVING,
                            first_seen="2026-03-15",
                            last_seen="2026-03-15",
                            cues=[],
                        ),
                    ],
                    breakthroughs=[],
                ),
                SessionEntry(
                    date="2026-03-21",
                    summary="Session 2",
                    concept_links=[
                        ConceptLink(
                            concept_id="scooping_motion",
                            status=ConceptStatus.MASTERED,
                            first_seen="2026-03-15",
                            last_seen="2026-03-21",
                            cues=["top edge leading"],
                        ),
                        ConceptLink(
                            concept_id="out_vector",
                            status=ConceptStatus.STRUGGLING,
                            first_seen="2026-03-21",
                            last_seen="2026-03-21",
                            cues=[],
                        ),
                    ],
                    breakthroughs=["Scooping eliminated"],
                ),
                SessionEntry(
                    date="2026-03-24",
                    summary="Session 3",
                    concept_links=[
                        ConceptLink(
                            concept_id="scooping_motion",
                            status=ConceptStatus.REGRESSED,
                            first_seen="2026-03-15",
                            last_seen="2026-03-24",
                            cues=[],
                        ),
                        ConceptLink(
                            concept_id="out_vector",
                            status=ConceptStatus.IMPROVING,
                            first_seen="2026-03-21",
                            last_seen="2026-03-24",
                            cues=[],
                        ),
                    ],
                    breakthroughs=["Out vector appeared"],
                ),
            ],
            concept_map={
                "scooping_motion": ConceptLink(
                    concept_id="scooping_motion",
                    status=ConceptStatus.REGRESSED,
                    first_seen="2026-03-15",
                    last_seen="2026-03-24",
                    cues=["avoid scooping", "top edge leading"],
                ),
                "unit_turn": ConceptLink(
                    concept_id="unit_turn",
                    status=ConceptStatus.IMPROVING,
                    first_seen="2026-03-15",
                    last_seen="2026-03-15",
                    cues=[],
                ),
                "out_vector": ConceptLink(
                    concept_id="out_vector",
                    status=ConceptStatus.IMPROVING,
                    first_seen="2026-03-21",
                    last_seen="2026-03-24",
                    cues=[],
                ),
            },
        )

    def test_get_status(self, profile):
        assert profile.get_status("scooping_motion") == ConceptStatus.REGRESSED
        assert profile.get_status("unit_turn") == ConceptStatus.IMPROVING
        assert profile.get_status("nonexistent") is None

    def test_active_issues(self, profile):
        issues = profile.active_issues()
        ids = [c.concept_id for c in issues]
        assert "scooping_motion" in ids  # regressed
        assert "unit_turn" not in ids  # improving, not an issue
        assert "out_vector" not in ids  # improving, not an issue

    def test_recent_breakthroughs(self, profile):
        breakthroughs = profile.recent_breakthroughs(n=2)
        # Should return last 2 sessions' breakthroughs
        assert len(breakthroughs) <= 2
        # Most recent first
        assert breakthroughs[0].date == "2026-03-24"

    def test_json_roundtrip(self, profile, tmp_path):
        path = tmp_path / "profile.json"
        profile.to_json(path)
        loaded = UserProfile.from_json(path)
        assert len(loaded.sessions) == len(profile.sessions)
        assert set(loaded.concept_map.keys()) == set(profile.concept_map.keys())


# ---------------------------------------------------------------------------
# Parsing and fuzzy resolve tests
# ---------------------------------------------------------------------------

MINI_LEARNING_MD = textwrap.dedent("""\
    # 练球问题记录

    ## 2026-03-15（正手）

    ### 今日核心发现

    - 击球前小臂有一个明显的**向上动作**
    - 这就是 **scooping** 的来源
    - **Unit Turn** 之后做 Pat the Dog 时主动用小臂压拍头

    ### 因果链

    ```
    Unit Turn 后主动用小臂压拍头
      → scooping
      → 动力链断裂
    ```

    ### 新口令
    1. `Unit Turn 后手臂什么都不做，等身体转`
    2. `前挥只有一步：蹬地转髋`

    ## 2026-03-21（正手）

    ### 今日核心发现

    - **Scooping 彻底消失**——5 次击球全部未检测到
    - **Out 向量**首次稳定出现
    - 但 **动力链脱节** 仍然存在

    ## 2026-03-24（正手）

    ### 今日重大突破

    - **scooping** 又回来了
    - 但发现了 **Press Slot** 的正确感觉
    - **Out 向量**终于突破
""")

MINI_REGISTRY = [
    {
        "id": "scooping_motion",
        "name": "Scooping Motion",
        "name_zh": "捞球动作",
        "aliases": ["scooping", "v-shape swing", "scoop"],
        "category": "symptom",
        "sources": ["ftt"],
        "description": "V-shaped swing path",
        "vlm_features": [],
        "muscles_involved": [],
        "confidence": 0.9,
    },
    {
        "id": "unit_turn",
        "name": "Unit Turn",
        "name_zh": "整体转身",
        "aliases": ["loading phase", "coiling"],
        "category": "technique",
        "sources": ["ftt"],
        "description": "Body rotation as single unit",
        "vlm_features": [],
        "muscles_involved": [],
        "confidence": 0.95,
    },
    {
        "id": "out_vector",
        "name": "Out Vector",
        "name_zh": "向外向量",
        "aliases": ["outward push", "out direction"],
        "category": "technique",
        "sources": ["ftt"],
        "description": "Outward racket path",
        "vlm_features": [],
        "muscles_involved": [],
        "confidence": 0.9,
    },
    {
        "id": "rotational_kinetic_chain",
        "name": "Rotational Kinetic Chain",
        "name_zh": "旋转动力链",
        "aliases": ["kinetic chain", "whip chain", "动力链"],
        "category": "biomechanics",
        "sources": ["ftt"],
        "description": "Whip system driven by rotation",
        "vlm_features": [],
        "muscles_involved": [],
        "confidence": 0.95,
    },
    {
        "id": "press_slot",
        "name": "Press Slot",
        "name_zh": "按压槽位",
        "aliases": ["pressure slot", "press position"],
        "category": "technique",
        "sources": ["ftt"],
        "description": "Specific press position before contact",
        "vlm_features": [],
        "muscles_involved": [],
        "confidence": 0.9,
    },
]


class TestBuildProfileFromLearning:
    @pytest.fixture
    def mini_data(self, tmp_path):
        learning_path = tmp_path / "learning.md"
        learning_path.write_text(MINI_LEARNING_MD, encoding="utf-8")

        registry_path = tmp_path / "registry.json"
        registry_path.write_text(
            json.dumps(MINI_REGISTRY, ensure_ascii=False), encoding="utf-8"
        )
        return learning_path, registry_path

    def test_parses_sessions_by_date(self, mini_data):
        learning_path, registry_path = mini_data
        profile = build_profile_from_learning(learning_path, registry_path)
        dates = [s.date for s in profile.sessions]
        assert "2026-03-15" in dates
        assert "2026-03-21" in dates
        assert "2026-03-24" in dates
        assert len(profile.sessions) == 3

    def test_resolves_bold_terms_to_concepts(self, mini_data):
        learning_path, registry_path = mini_data
        profile = build_profile_from_learning(learning_path, registry_path)
        # scooping should be resolved across sessions
        assert "scooping_motion" in profile.concept_map

    def test_resolves_causal_chain_terms(self, mini_data):
        learning_path, registry_path = mini_data
        profile = build_profile_from_learning(learning_path, registry_path)
        # "动力链断裂" in causal chain should resolve to rotational_kinetic_chain
        # or "Unit Turn" in causal chain should resolve
        assert "unit_turn" in profile.concept_map

    def test_status_progression(self, mini_data):
        """Concept mentioned as problem early, solved later, problem again = regressed."""
        learning_path, registry_path = mini_data
        profile = build_profile_from_learning(learning_path, registry_path)
        # scooping: struggling (03-15) -> mastered (03-21 "彻底消失") -> regressed (03-24 "又回来了")
        scooping = profile.concept_map.get("scooping_motion")
        assert scooping is not None
        assert scooping.status == ConceptStatus.REGRESSED

    def test_extracts_cues(self, mini_data):
        learning_path, registry_path = mini_data
        profile = build_profile_from_learning(learning_path, registry_path)
        # Should extract backtick-quoted cues from sessions
        all_cues = []
        for session in profile.sessions:
            for link in session.concept_links:
                all_cues.extend(link.cues)
        assert len(all_cues) > 0

    def test_concept_map_has_valid_statuses(self, mini_data):
        learning_path, registry_path = mini_data
        profile = build_profile_from_learning(learning_path, registry_path)
        valid = {s.value for s in ConceptStatus}
        for cid, link in profile.concept_map.items():
            assert link.status.value in valid, f"{cid} has invalid status {link.status}"


# ---------------------------------------------------------------------------
# VLM user context injection tests (07-02)
# ---------------------------------------------------------------------------


class TestVLMUserContextInjection:
    """Tests for VLMPromptCompiler accepting UserProfile and injecting user context."""

    @pytest.fixture
    def graph(self):
        from knowledge.graph import KnowledgeGraph
        from knowledge.schemas import Concept, ConceptType, Edge, RelationType

        kg = KnowledgeGraph()
        kg.add_concept(Concept(
            id="forearm_compensation",
            name="Forearm Compensation",
            name_zh="小臂代偿",
            category=ConceptType.SYMPTOM,
            description="Arm drives swing",
            confidence=0.95,
        ))
        kg.add_concept(Concept(
            id="scooping_motion",
            name="Scooping Motion",
            name_zh="捞球动作",
            category=ConceptType.SYMPTOM,
            description="V-shaped swing",
            confidence=0.9,
        ))
        kg.add_concept(Concept(
            id="unit_turn",
            name="Unit Turn",
            name_zh="整体转身",
            category=ConceptType.TECHNIQUE,
            description="Body rotation",
            confidence=0.95,
        ))
        kg.add_concept(Concept(
            id="drop_feed_rotation_drill",
            name="Drop Feed Rotation Drill",
            name_zh="喂球旋转练习",
            category=ConceptType.DRILL,
            description="Practice body rotation",
            confidence=1.0,
        ))
        kg.add_edge(Edge(
            source_id="unit_turn",
            target_id="forearm_compensation",
            relation=RelationType.CAUSES,
            confidence=0.85,
            evidence="Missing unit turn causes arm-driven hitting",
            source_file="ftt_book",
        ))
        kg.add_edge(Edge(
            source_id="drop_feed_rotation_drill",
            target_id="forearm_compensation",
            relation=RelationType.DRILLS_FOR,
            confidence=0.9,
            evidence="Drill addresses arm compensation",
            source_file="ftt_video",
        ))
        return kg

    @pytest.fixture
    def chains(self):
        from knowledge.schemas import DiagnosticChain, DiagnosticStep

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
    def user_profile(self):
        return UserProfile(
            sessions=[
                SessionEntry(
                    date="2026-03-24",
                    summary="Session 1",
                    concept_links=[],
                    breakthroughs=["Fixed scooping"],
                ),
            ],
            concept_map={
                "forearm_compensation": ConceptLink(
                    concept_id="forearm_compensation",
                    status=ConceptStatus.STRUGGLING,
                    first_seen="2026-03-15",
                    last_seen="2026-03-24",
                    cues=["let body rotate first"],
                ),
                "scooping_motion": ConceptLink(
                    concept_id="scooping_motion",
                    status=ConceptStatus.REGRESSED,
                    first_seen="2026-03-15",
                    last_seen="2026-03-24",
                    cues=["top edge leading"],
                ),
                "unit_turn": ConceptLink(
                    concept_id="unit_turn",
                    status=ConceptStatus.MASTERED,
                    first_seen="2026-03-15",
                    last_seen="2026-03-24",
                    cues=[],
                ),
            },
        )

    def test_compiler_accepts_user_profile(self, graph, chains, user_profile):
        """Test 1: VLMPromptCompiler accepts optional user_profile parameter."""
        from knowledge.output.vlm_prompt import VLMPromptCompiler

        compiler = VLMPromptCompiler(graph, chains, user_profile=user_profile)
        assert compiler.user_profile is user_profile

    def test_compile_pass2_with_user_profile_injects_context(
        self, graph, chains, user_profile
    ):
        """Test 2: compile_pass2_prompt with user_profile injects user context section."""
        from knowledge.output.vlm_prompt import VLMPromptCompiler

        compiler = VLMPromptCompiler(graph, chains, user_profile=user_profile)
        prompt = compiler.compile_pass2_prompt(["dc_arm_driven_hitting"])
        # Should contain user context markers
        assert "用户训练画像" in prompt
        # Should contain user's struggle concepts
        assert "小臂代偿" in prompt or "forearm_compensation" in prompt

    def test_compile_pass2_without_user_profile_unchanged(self, graph, chains):
        """Test 3: compile_pass2_prompt without user_profile works exactly as before."""
        from knowledge.output.vlm_prompt import VLMPromptCompiler

        compiler_no_profile = VLMPromptCompiler(graph, chains)
        compiler_none_profile = VLMPromptCompiler(graph, chains, user_profile=None)
        prompt_no = compiler_no_profile.compile_pass2_prompt(["dc_arm_driven_hitting"])
        prompt_none = compiler_none_profile.compile_pass2_prompt(["dc_arm_driven_hitting"])
        assert prompt_no == prompt_none
        assert "用户训练画像" not in prompt_no

    def test_user_context_budget_enforcement(self, graph, chains):
        """Test 4: User context section stays under 1500 chars even with 50+ concept links."""
        from knowledge.output.vlm_prompt import VLMPromptCompiler

        # Build a profile with 60 struggling concepts
        concept_map = {}
        for i in range(60):
            cid = f"concept_{i:03d}"
            concept_map[cid] = ConceptLink(
                concept_id=cid,
                status=ConceptStatus.STRUGGLING,
                first_seen="2026-01-01",
                last_seen="2026-03-24",
                cues=[f"cue for concept {i}", f"another cue {i}"],
            )
        big_profile = UserProfile(sessions=[], concept_map=concept_map)
        compiler = VLMPromptCompiler(graph, chains, user_profile=big_profile)
        user_ctx = compiler.compile_user_context(["dc_arm_driven_hitting"])
        assert len(user_ctx) <= 1500, f"User context too long: {len(user_ctx)} chars"

    def test_user_context_highlights_detected_chain_symptoms(
        self, graph, chains, user_profile
    ):
        """Test 5: User context highlights concepts matching detected chain symptoms."""
        from knowledge.output.vlm_prompt import VLMPromptCompiler

        compiler = VLMPromptCompiler(graph, chains, user_profile=user_profile)
        user_ctx = compiler.compile_user_context(["dc_arm_driven_hitting"])
        # forearm_compensation is both an active issue and a detected chain symptom
        assert "已知反复出现" in user_ctx or "known recurring" in user_ctx.lower()

    def test_vlm_analyzer_auto_loads_user_profile(self, graph, chains, tmp_path):
        """Test 6: VLMForehandAnalyzer auto-loads user_profile.json if it exists."""
        from evaluation.vlm_analyzer import VLMForehandAnalyzer

        # Create a minimal user profile JSON
        profile = UserProfile(
            sessions=[],
            concept_map={
                "forearm_compensation": ConceptLink(
                    concept_id="forearm_compensation",
                    status=ConceptStatus.STRUGGLING,
                    first_seen="2026-03-15",
                    last_seen="2026-03-24",
                    cues=[],
                ),
            },
        )
        profile_path = tmp_path / "user_profile.json"
        profile.to_json(profile_path)

        cfg = {
            "provider": "openai_compatible",
            "api_key": "test-key",
            "base_url": "http://test",
            "model": "test-model",
        }
        analyzer = VLMForehandAnalyzer(
            config=cfg, graph=graph, chains=chains,
            user_profile_path=profile_path,
        )
        assert analyzer.compiler is not None
        assert analyzer.compiler.user_profile is not None
