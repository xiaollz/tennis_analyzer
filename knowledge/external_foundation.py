"""Retrieval adapter for the external tennis channel foundation series.

The external package is a discovery and evidence source, not a replacement for
the project's canonical graph.  This module keeps provenance, rejects
metadata-only teaching candidates, and separates visible observations from
mechanism/feel/drill claims before anything reaches a VLM prompt or diagnosis.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_ROOT = _PROJECT_ROOT / "tennis_foundation_series"

_CURATED_SEEDS = (
    "channels/fault_tolerant_tennis/foundation/tennis_knowledge_graph_seed.json",
    "channels/tom_allsopp_tennis/foundation/tennis_knowledge_graph_seed.json",
)

_CHANNEL_NAMES = {
    "fault_tolerant_tennis": "Fault Tolerant Tennis",
    "tom_allsopp_tennis": "TPA tennis / Tom Allsopp",
    "feel_tennis": "Feel Tennis Instruction",
    "one_minute_tennis": "One Minute Tennis",
    "road_to_pro_tennis": "Road to Pro Tennis",
}

_CONFIDENCE = {"high": 0.9, "medium": 0.7, "low": 0.5}

# Project-level conflict: elbow movement may be described as a passive result,
# but it must never be prescribed as an active push/send cue.
_ACTIVE_ELBOW_CUE = re.compile(
    r"(主动.{0,8}(推肘|送肘|肘.{0,4}(前推|前送))|"
    r"(推肘|送肘|肘.{0,4}(前推|前送)).{0,8}(动作|口令|练习|cue)|"
    r"actively\s+(push|send).{0,12}elbow)",
    re.IGNORECASE,
)

# These states cannot be established from ordinary pixels alone.  Such text
# can still be retained as mechanism/feel, but not as a VLM observable.
_INTERNAL_STATE = re.compile(
    r"肌肉|发力|激活|募集|收缩|张力|酸痛|用力|力矩|关节力|"
    r"收紧|支撑机制|是否参与|不参与运动|muscle|activation|contraction|torque",
    re.IGNORECASE,
)

_OTHER_SHOT_TITLE = re.compile(
    r"\bserve\b|serving|backhand|volley|overhead|slice|return of serve|"
    r"发球|反手|截击|高压球|切削|接发",
    re.IGNORECASE,
)

_FOREHAND_TITLE = re.compile(r"forehand|正手", re.IGNORECASE)

_BOILERPLATE = re.compile(
    r"^(视频来源|视频博主|核心人物|动作线索|感觉线索|关键点|"
    r"声明|声明内容|身体关键点|球拍状态|球轨迹|声音|镜头要求)[:：]?\s*$"
)

_CATEGORY_MARKERS = {
    "drill": ("练习", "drill", "训练法", "训练方式"),
    "feel": ("感觉", "感受", "意象", "sensory", "swing thought"),
    "correction": ("禁忌", "严禁", "错误", "纠正", "避免", "不要", "解决问题"),
    "observation": ("画面显示", "观察", "镜头要求", "可见", "轨迹", "位置"),
    "mechanism": ("机制", "动力学", "导致", "产生", "作用", "因果", "核心主题"),
}

_QUERY_ALIASES = {
    "dc_arm_driven_hitting": ("手臂主导", "手臂独立", "小臂代偿", "身体连接", "连接", "整体挥拍"),
    "forearm_compensation": ("手臂主导", "手臂独立", "小臂代偿", "手臂晃荡", "连接"),
    "dc_scooping": ("scooping", "V形", "捞球", "拍头下坠", "掉拍", "自然下降"),
    "racket_drop": ("拍头下坠", "掉拍", "自然下降", "pat the dog", "scooping"),
    "dc_missing_out_vector": ("向前延伸", "向外路径", "穿透", "击球区", "随挥"),
    "swing_out": ("向前延伸", "向外路径", "穿透", "击球区"),
    "dc_over_rotation": ("过度旋转", "过度转体", "制动", "反向旋转"),
    "over_rotation": ("过度旋转", "过度转体", "制动", "反向旋转"),
    "dc_early_release": ("提前释放", "手腕", "滞后", "受控滞后", "拍头释放"),
    "wrist_lag": ("滞后", "受控滞后", "手腕", "拍头释放", "手拍耦合"),
    "dc_trunk_momentum_leak": ("动力链", "躯干时序", "旋转能量", "近端脉冲", "制动"),
    "trunk_sequencing": ("动力链", "躯干时序", "髋部", "近端脉冲", "制动"),
    "problem_p01": ("拍头过度下坠", "掉拍", "自然下降", "节奏"),
    "problem_p02": ("V形", "scooping", "捞球", "从下向上"),
    "problem_p03": ("小臂代偿", "手臂独立", "手臂主导", "身体连接"),
    "problem_p04": ("动力链断裂", "脱节", "身体连接", "躯干时序"),
    "problem_p05": ("击球点偏后", "击球空间", "向前延伸", "甜点"),
    "problem_p07": ("unit turn", "整体转身", "肩部", "准备"),
    "problem_p08": ("击球太急", "时机", "节奏", "准备时间"),
    "problem_p11": ("过度转体", "over rotation", "制动", "反向旋转"),
    "problem_p13": ("手主动引拍", "大臂后拉", "unit turn", "整体转身"),
    "problem_p17": ("握拍", "握力", "手拍耦合", "grip pressure"),
    "problem_p19": ("从下往上", "挥拍路径", "捞球", "向前路径"),
    "problem_p21": ("胸臂连接", "身体连接", "肩内收", "上背稳定"),
    "unit_turn": ("unit turn", "单位转体", "整体转身", "整体挥拍"),
    "f1_hold_up": ("架拍", "hold up", "拍头高位", "拍头下坠", "掉拍头", "过度掉拍", "scooping"),
    "f2_place_pull": ("放置拉向前", "place pull forward", "引拍停顿", "连续前挥"),
    "f3_back_glue": ("背部胶水", "上背稳定", "身体连接", "手臂独立", "整体挥拍"),
    "f4_unit_action": ("整体转", "unit turn", "unit swing", "手臂独立"),
    "f5_right_foot_axis": ("右脚为轴", "支撑脚", "转髋", "重心轴线"),
    "f6_scapular_slot": ("肩胛骨槽", "肩胛", "上背稳定", "近端基础"),
    "f7_hsa": ("肩水平内收", "horizontal shoulder adduction", "胸臂连接", "大臂内收"),
}


@dataclass(frozen=True)
class FoundationKnowledgeItem:
    item_id: str
    category: str
    text: str
    channel_id: str
    channel_name: str
    video_id: str = ""
    video_title: str = ""
    video_url: str = ""
    evidence_status: str = ""
    confidence: float = 0.0
    limitations: tuple[str, ...] = ()
    knowledge_boundary: str = ""
    source_kind: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["limitations"] = list(self.limitations)
        return data


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        parts: list[str] = []
        for key in (
            "problem", "cue_text_or_imagery", "action_or_feel", "mechanism",
            "success_criterion", "misuse_or_boundary",
        ):
            item = value.get(key)
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
        steps = value.get("drill_steps")
        if isinstance(steps, list):
            parts.extend(str(step).strip() for step in steps if str(step).strip())
        return "；".join(parts)
    return ""


def _classify_keypoint(text: str) -> str:
    lowered = text.lower()
    for category in ("drill", "feel", "correction", "observation", "mechanism"):
        if any(marker.lower() in lowered for marker in _CATEGORY_MARKERS[category]):
            return category
    return "principle"


def _is_usable_text(text: str) -> bool:
    stripped = text.strip()
    return len(stripped) >= 12 and not _BOILERPLATE.match(stripped)


def _is_safe_observable(text: str) -> bool:
    return not _INTERNAL_STATE.search(text) and not _ACTIVE_ELBOW_CUE.search(text)


def _extract_terms(query: str, identifiers: Iterable[str] = ()) -> list[str]:
    normalized = query.lower()
    terms: set[str] = set()
    generic_terms = {
        "problem", "drill", "forehand", "正手", "问题", "arm", "body",
        "swing", "rotation", "racket", "hitting", "instead", "initiating",
        "through", "contact", "excessive", "before", "from", "with",
    }

    for identifier in identifiers:
        key = str(identifier).lower()
        terms.update(term.lower() for term in _QUERY_ALIASES.get(key, ()))

    for key, aliases in _QUERY_ALIASES.items():
        if key in normalized:
            terms.update(alias.lower() for alias in aliases)

    terms.update(
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}|[\u4e00-\u9fff]{2,8}", query)
        if token.lower() not in generic_terms
    )
    return sorted(terms, key=len, reverse=True)


class ExternalFoundationLibrary:
    """Lazy, read-only retrieval over ``tennis_foundation_series``."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = Path(root) if root else _DEFAULT_ROOT
        self._items: Optional[tuple[FoundationKnowledgeItem, ...]] = None

    @property
    def available(self) -> bool:
        return (self.root / "unified" / "all_knowledge_items.jsonl").exists()

    @property
    def items(self) -> tuple[FoundationKnowledgeItem, ...]:
        if self._items is None:
            self._items = tuple(self._load_items())
        return self._items

    def _load_items(self) -> list[FoundationKnowledgeItem]:
        if not self.available:
            return []

        video_catalog = self._load_video_catalog()
        items = self._load_curated_nodes(video_catalog)
        items.extend(self._load_video_items())
        return items

    def _load_video_catalog(self) -> dict[str, dict[str, Any]]:
        path = self.root / "unified" / "all_videos.jsonl"
        catalog: dict[str, dict[str, Any]] = {}
        if not path.exists():
            return catalog
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                global_id = str(raw.get("global_video_id") or "")
                if global_id:
                    catalog[global_id] = raw
        return catalog

    def _load_curated_nodes(
        self, video_catalog: dict[str, dict[str, Any]],
    ) -> list[FoundationKnowledgeItem]:
        items: list[FoundationKnowledgeItem] = []
        for relative_path in _CURATED_SEEDS:
            path = self.root / relative_path
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            evidence_map = {item.get("id"): item for item in data.get("evidence", [])}
            channel_id = path.parts[-3]
            for node in data.get("nodes", []):
                domains = set(node.get("domains", []))
                if "forehand" not in domains:
                    continue
                text = "：".join(
                    part for part in (node.get("label", ""), node.get("definition", "")) if part
                )
                if not _is_usable_text(text) or _ACTIVE_ELBOW_CUE.search(text):
                    continue

                evidences = [
                    evidence_map[eid]
                    for eid in node.get("evidence_ids", [])
                    if eid in evidence_map
                ]
                evidence = next(
                    (item for item in evidences if item.get("video_id") not in (None, "channel")),
                    evidences[0] if evidences else {},
                )
                video_id = str(evidence.get("video_id") or "")
                video = video_catalog.get(f"{channel_id}:{video_id}", {})
                items.append(FoundationKnowledgeItem(
                    item_id=f"{channel_id}:foundation:{node.get('id', '')}",
                    category=node.get("node_type", "principle"),
                    text=text,
                    channel_id=channel_id,
                    channel_name=_CHANNEL_NAMES.get(channel_id, channel_id),
                    video_id=video_id,
                    video_title=str(video.get("title") or ""),
                    video_url=str(video.get("url") or evidence.get("source_url") or ""),
                    evidence_status=str(node.get("status") or "curated_channel_synthesis"),
                    confidence=_CONFIDENCE.get(str(node.get("confidence", "medium")), 0.7),
                    limitations=tuple(node.get("limitations", [])),
                    knowledge_boundary="频道级整理；项目 canonical 规则优先。",
                    source_kind="curated_foundation_node",
                ))
        return items

    def _load_video_items(self) -> list[FoundationKnowledgeItem]:
        path = self.root / "unified" / "all_knowledge_items.jsonl"
        items: list[FoundationKnowledgeItem] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Metadata-only candidates are useful for discovery, but not
                # qualified to influence diagnosis or prescribe training.
                if raw.get("evidence_status") != "video_content_analysis":
                    continue
                if "forehand" not in raw.get("domains_canonical", []):
                    continue
                if raw.get("record_type") not in {"source_keypoint", "vlm_observable"}:
                    continue

                title = str(raw.get("source_video_title") or "")
                if _OTHER_SHOT_TITLE.search(title) and not _FOREHAND_TITLE.search(title):
                    continue

                text = _content_text(raw.get("content"))
                if not _is_usable_text(text) or _ACTIVE_ELBOW_CUE.search(text):
                    continue

                category = (
                    "observation" if raw.get("record_type") == "vlm_observable"
                    else _classify_keypoint(text)
                )
                if category == "observation" and not _is_safe_observable(text):
                    continue

                channel_id = str(raw.get("source_channel_id") or "")
                items.append(FoundationKnowledgeItem(
                    item_id=str(raw.get("knowledge_item_id") or ""),
                    category=category,
                    text=text,
                    channel_id=channel_id,
                    channel_name=str(raw.get("source_channel_name") or _CHANNEL_NAMES.get(channel_id, channel_id)),
                    video_id=str(raw.get("source_video_id") or ""),
                    video_title=title,
                    video_url=str(raw.get("source_video_url") or ""),
                    evidence_status="video_content_analysis",
                    confidence=0.68 if category == "observation" else 0.72,
                    limitations=(),
                    knowledge_boundary=str(raw.get("knowledge_boundary") or ""),
                    source_kind="video_observable" if category == "observation" else "video_keypoint",
                ))
        return items

    def retrieve(
        self,
        query: str,
        *,
        identifiers: Iterable[str] = (),
        categories: Optional[set[str]] = None,
        limit: int = 8,
        max_per_video: int = 2,
    ) -> list[FoundationKnowledgeItem]:
        if not self.available or limit <= 0:
            return []
        terms = _extract_terms(query, identifiers)
        if not terms:
            return []

        scored: list[tuple[float, FoundationKnowledgeItem]] = []
        for item in self.items:
            if categories and item.category not in categories:
                continue
            haystack = f"{item.text} {item.video_title}".lower()
            matched = [term for term in terms if term in haystack]
            if not matched:
                continue
            score = sum(min(len(term), 8) / 4 for term in matched)
            score += item.confidence
            if item.source_kind == "curated_foundation_node":
                score += 1.5
            if item.channel_id == "fault_tolerant_tennis":
                score += 0.15
            scored.append((score, item))

        scored.sort(key=lambda pair: (-pair[0], -pair[1].confidence, pair[1].item_id))
        # Reserve up to two slots for curated channel synthesis so thousands
        # of sentence-level keypoints cannot drown out the package's strongest
        # structured conclusions.
        curated = [pair for pair in scored if pair[1].source_kind == "curated_foundation_node"]
        ordered = curated[: min(2, limit)] + scored

        selected: list[FoundationKnowledgeItem] = []
        per_video: dict[str, int] = {}
        seen_text: set[str] = set()
        for _, item in ordered:
            text_key = re.sub(r"\s+", "", item.text.lower())
            if text_key in seen_text:
                continue
            video_key = f"{item.channel_id}:{item.video_id or item.item_id}"
            if per_video.get(video_key, 0) >= max_per_video:
                continue
            selected.append(item)
            seen_text.add(text_key)
            per_video[video_key] = per_video.get(video_key, 0) + 1
            if len(selected) >= limit:
                break
        return selected

    def retrieve_for_chains(
        self,
        chains: Sequence[Any],
        *,
        categories: Optional[set[str]] = None,
        limit: int = 8,
    ) -> list[FoundationKnowledgeItem]:
        parts: list[str] = []
        identifiers: list[str] = []
        for chain in chains:
            for attr in ("id", "symptom", "symptom_zh", "symptom_concept_id"):
                value = getattr(chain, attr, "")
                if value:
                    parts.append(str(value))
                    identifiers.append(str(value))
            for attr in ("root_causes", "drills"):
                values = getattr(chain, attr, ()) or ()
                parts.extend(str(value) for value in values)
                identifiers.extend(str(value) for value in values)
        return self.retrieve(
            " ".join(parts), identifiers=identifiers,
            categories=categories, limit=limit,
        )

    def stats(self) -> dict[str, Any]:
        by_category: dict[str, int] = {}
        by_channel: dict[str, int] = {}
        for item in self.items:
            by_category[item.category] = by_category.get(item.category, 0) + 1
            by_channel[item.channel_id] = by_channel.get(item.channel_id, 0) + 1
        return {
            "available": self.available,
            "item_count": len(self.items),
            "by_category": by_category,
            "by_channel": by_channel,
            "excluded_evidence": ["metadata_title_only", "metadata_title_and_playlist_only"],
        }


@lru_cache(maxsize=1)
def get_external_foundation_library() -> ExternalFoundationLibrary:
    return ExternalFoundationLibrary()


def items_to_dicts(items: Sequence[FoundationKnowledgeItem]) -> list[dict[str, Any]]:
    return [item.to_dict() for item in items]
