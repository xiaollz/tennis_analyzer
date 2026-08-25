import json

from evaluation import diagnosis_engine
from knowledge.external_foundation import (
    ExternalFoundationLibrary,
    FoundationKnowledgeItem,
)
from knowledge.graph import KnowledgeGraph
from knowledge.output.vlm_prompt import VLMPromptCompiler
from knowledge.schemas import Concept, ConceptType, DiagnosticChain
from report.report_generator import ReportGenerator


def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )


def _record(
    item_id,
    *,
    record_type="source_keypoint",
    evidence_status="video_content_analysis",
    title="Build a Connected Forehand",
    content="躯干转动与手臂保持连接，避免小臂独立接管。",
):
    return {
        "record_type": record_type,
        "knowledge_item_id": item_id,
        "source_channel_id": "fault_tolerant_tennis",
        "source_channel_name": "Fault Tolerant Tennis",
        "global_video_id": "fault_tolerant_tennis:abc",
        "source_video_id": "abc",
        "source_video_title": title,
        "source_video_url": "https://www.youtube.com/watch?v=abc",
        "domains_canonical": ["forehand"],
        "atomic_type": record_type,
        "content": content,
        "evidence_status": evidence_status,
        "provenance": {},
        "knowledge_boundary": "test boundary",
    }


def test_external_library_filters_unqualified_and_nonvisual_items(tmp_path):
    records = [
        _record("valid-keypoint"),
        _record(
            "valid-observable",
            record_type="vlm_observable",
            content="观察右肘与胸壁的距离是否在前挥阶段突然增大。",
        ),
        _record(
            "internal-observable",
            record_type="vlm_observable",
            content="观察背部肌肉是否激活并持续发力。",
        ),
        _record("active-elbow", content="训练时主动推肘向前完成击球。"),
        _record("metadata-only", evidence_status="metadata_title_only"),
        _record("serve-only", title="Perfect Tennis Serve", content="发球时保持手臂连接。"),
    ]
    _write_jsonl(tmp_path / "unified" / "all_knowledge_items.jsonl", records)
    _write_jsonl(tmp_path / "unified" / "all_videos.jsonl", [])

    library = ExternalFoundationLibrary(tmp_path)

    assert {item.item_id for item in library.items} == {"valid-keypoint", "valid-observable"}
    result = library.retrieve(
        "手臂主导 小臂代偿 身体连接",
        identifiers=["dc_arm_driven_hitting"],
        limit=5,
    )
    assert result
    assert result[0].video_url == "https://www.youtube.com/watch?v=abc"


class _FakeExternalLibrary:
    def retrieve_for_chains(self, chains, *, categories=None, limit=8):
        category = "observation" if categories == {"observation"} else "mechanism"
        return [
            FoundationKnowledgeItem(
                item_id="external:test",
                category=category,
                text="外部证据：身体与手臂连接需要按可见时序复核。",
                channel_id="fault_tolerant_tennis",
                channel_name="Fault Tolerant Tennis",
                video_id="abc",
                video_title="Build a Connected Forehand",
                video_url="https://www.youtube.com/watch?v=abc",
                evidence_status="video_content_analysis",
                confidence=0.8,
                limitations=("普通视频不能证明肌肉激活。",),
            )
        ]


def _compiler():
    graph = KnowledgeGraph()
    graph.add_concept(Concept(
        id="forearm_compensation",
        name="Forearm compensation",
        name_zh="小臂代偿",
        category=ConceptType.SYMPTOM,
        description="Arm drives independently",
    ))
    chain = DiagnosticChain(
        id="dc_arm_driven_hitting",
        symptom="Arm initiating swing",
        symptom_zh="手臂主导挥拍",
        symptom_concept_id="forearm_compensation",
        check_sequence=[],
        root_causes=["forearm_compensation"],
        drills=[],
        priority=1,
    )
    return VLMPromptCompiler(
        graph, [chain], external_library=_FakeExternalLibrary(),
    )


def test_prompt_compiler_injects_bounded_source_attributed_context():
    compiler = _compiler()
    prompt = compiler.compile_pass2_prompt(["dc_arm_driven_hitting"])
    dynamic = prompt[len(compiler.compile_system_prompt()):]

    assert "外部频道 Foundation 补充证据" in prompt
    assert "Fault Tolerant Tennis" in prompt
    assert "https://www.youtube.com/watch?v=abc" in prompt
    assert "普通视频不能证明肌肉激活" in prompt
    assert len(dynamic) < 10_000


def test_diagnosis_result_exposes_external_foundation(monkeypatch):
    item = FoundationKnowledgeItem(
        item_id="external:diagnosis",
        category="mechanism",
        text="连接性过渡是受控滞后的上游条件。",
        channel_id="tom_allsopp_tennis",
        channel_name="TPA tennis / Tom Allsopp",
        video_id="abc",
        video_url="https://www.youtube.com/watch?v=abc",
        evidence_status="video_content_analysis",
        confidence=0.8,
    )

    class FakeLibrary:
        def retrieve(self, *args, **kwargs):
            return [item]

    monkeypatch.setattr(
        "knowledge.external_foundation.get_external_foundation_library",
        lambda: FakeLibrary(),
    )

    result = diagnosis_engine.diagnose(
        {"raw_answers": {"Q1": "F3 FAIL：手臂先于躯干独立启动。"}},
        {},
    )

    assert result["external_foundation"]["role"] == "supplementary"
    assert result["external_foundation"]["canonical_project_rules_override"] is True
    assert result["external_foundation"]["items"][0]["item_id"] == "external:diagnosis"


def test_report_renders_external_sources_inside_reasoning_details():
    item = FoundationKnowledgeItem(
        item_id="external:report",
        category="mechanism",
        text="连接性过渡是受控滞后的上游条件。",
        channel_id="tom_allsopp_tennis",
        channel_name="TPA tennis / Tom Allsopp",
        video_id="abc",
        video_title="Hit Great Forehands Without Being Loose",
        video_url="https://www.youtube.com/watch?v=abc",
        evidence_status="video_content_analysis",
        confidence=0.8,
    )
    lines = ReportGenerator._vlm_section(
        {
            "root_cause_tree": {"root_cause": "手臂主导", "causal_explanation": "因果说明"},
            "external_foundation": {"items": [item.to_dict()]},
            "evidence_chain": [{"observation": "手先动", "mapped_concept": "problem_p03"}],
        },
        {},
        0,
        1,
    )
    rendered = "\n".join(lines)
    assert "外部频道 Foundation 参考" in rendered
    assert "Hit Great Forehands Without Being Loose" in rendered
    assert "https://www.youtube.com/watch?v=abc" in rendered
