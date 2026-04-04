"""Hallucination mitigation for multi-round VLM diagnostic loop.

Provides four mechanisms:
  HM-01: Observation anchoring validation (frame + visual feature required)
  HM-02: Cross-round contradiction detection
  HM-03: Quantitative cross-validation against YOLO kinematic data
  HM-04: Low-confidence re-observation triggers

Used by MultiRoundAnalyzer after each round's observations are parsed.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from knowledge.schemas import DiagnosticSession, Observation


# Frame reference patterns: 图N, Frame N, frame N, 图 N
_FRAME_PATTERN = re.compile(r"(图\s*\d+|[Ff]rame\s*\d+)")


# ---------------------------------------------------------------------------
# HM-01: Observation Anchoring
# ---------------------------------------------------------------------------


def validate_anchoring(observations: list[Observation]) -> list[Observation]:
    """Check each observation for valid frame anchoring.

    Unanchored observations (no frame reference or description too short)
    get confidence set to 0.0 and is_anchored set to False.

    Returns list of unanchored observations (re-observe candidates).
    """
    unanchored: list[Observation] = []
    for obs in observations:
        anchored = True

        # Check frame field matches expected pattern
        if not obs.frame or not _FRAME_PATTERN.search(obs.frame):
            anchored = False

        # Check description has substance (not just "yes"/"no")
        if len(obs.description.strip()) < 10:
            anchored = False

        if not anchored:
            obs.confidence = 0.0
            obs.is_anchored = False
            unanchored.append(obs)

    return unanchored


# ---------------------------------------------------------------------------
# HM-02: Contradiction Detection
# ---------------------------------------------------------------------------


def detect_contradictions(
    session: DiagnosticSession,
    new_observations: list[Observation],
) -> list[dict]:
    """Detect contradictions between new observations and prior rounds.

    A contradiction occurs when two observations reference the same frame
    AND similar visual feature but have opposite judgments (yes <-> no).

    Returns list of contradiction dicts:
      {"obs_new": obs_id, "obs_prior": obs_id, "frame": str, "feature": str}
    """
    contradictions: list[dict] = []

    # Build index of prior observations by frame
    prior_by_frame: dict[str, list[Observation]] = {}
    for obs in session.observations:
        frame_key = _normalize_frame(obs.frame)
        if frame_key:
            prior_by_frame.setdefault(frame_key, []).append(obs)

    for new_obs in new_observations:
        frame_key = _normalize_frame(new_obs.frame)
        if not frame_key:
            continue

        priors = prior_by_frame.get(frame_key, [])
        for prior_obs in priors:
            # Check if they reference similar features (via directive_source)
            if not _similar_directive(new_obs.directive_source, prior_obs.directive_source):
                continue

            # Check for judgment flip
            if _is_judgment_flip(new_obs.judgment.value, prior_obs.judgment.value):
                contradictions.append({
                    "obs_new": new_obs.id,
                    "obs_prior": prior_obs.id,
                    "frame": frame_key,
                    "feature": new_obs.directive_source,
                })
                # Mark both as uncertain
                new_obs.confidence = min(new_obs.confidence, 0.3)
                prior_obs.confidence = min(prior_obs.confidence, 0.3)

    return contradictions


def _normalize_frame(frame: str) -> str:
    """Normalize frame reference to comparable key. e.g., '图3' -> '3'."""
    m = re.search(r"\d+", frame or "")
    return m.group() if m else ""


def _similar_directive(d1: str, d2: str) -> bool:
    """Check if two directive sources reference the same diagnostic feature."""
    if not d1 or not d2:
        return False
    # Same directive source string
    if d1 == d2:
        return True
    # Both reference same hypothesis
    if d1.startswith("hyp_") and d2.startswith("hyp_"):
        return d1 == d2
    # Partial match (one contains the other)
    return d1 in d2 or d2 in d1


def _is_judgment_flip(j1: str, j2: str) -> bool:
    """True if judgments are contradictory (yes<->no)."""
    return (j1 == "yes" and j2 == "no") or (j1 == "no" and j2 == "yes")


# ---------------------------------------------------------------------------
# HM-03: Quantitative Cross-Validation
# ---------------------------------------------------------------------------

# Thresholds for overriding VLM observations
_ELBOW_ANGLE_THRESHOLD = 30  # degrees: if VLM vs YOLO differ by more than this, override
_ELBOW_WIDE_THRESHOLD = 120  # degrees: above this = "wide"
_ELBOW_TIGHT_THRESHOLD = 90  # degrees: below this = "tight"


def cross_validate_with_kinematics(
    observations: list[Observation],
    supplementary_metrics: dict | None,
) -> list[Observation]:
    """Compare VLM observations against kinematic data and override on conflict.

    Currently supports:
    - Elbow angle: VLM says "wide"/"tight" vs measured angle
    - Wrist trajectory: VLM says "sharp drop" vs measured smoothness

    Returns list of overridden observations.
    """
    if not supplementary_metrics:
        return []

    overridden: list[Observation] = []

    elbow_angle = _extract_metric(supplementary_metrics, "elbow_angle")
    wrist_drop = _extract_metric(supplementary_metrics, "wrist_below_elbow")

    for obs in observations:
        desc_lower = obs.description.lower()

        # Elbow angle cross-validation
        if elbow_angle is not None and _mentions_elbow(desc_lower):
            override = _check_elbow_override(obs, elbow_angle)
            if override:
                overridden.append(obs)
                continue

        # Wrist trajectory cross-validation
        if wrist_drop is not None and _mentions_wrist(desc_lower):
            override = _check_wrist_override(obs, wrist_drop)
            if override:
                overridden.append(obs)

    return overridden


def _extract_metric(metrics: dict, key: str) -> float | None:
    """Extract a numeric metric, searching nested dicts."""
    if key in metrics:
        val = metrics[key]
        if isinstance(val, (int, float)):
            return float(val)
        return None

    # Search one level deep (e.g., metrics["contact"]["elbow_angle"])
    for v in metrics.values():
        if isinstance(v, dict) and key in v:
            val = v[key]
            if isinstance(val, (int, float)):
                return float(val)

    return None


def _mentions_elbow(desc: str) -> bool:
    """Check if observation description mentions elbow."""
    return any(kw in desc for kw in ["elbow", "肘", "angle", "角度"])


def _mentions_wrist(desc: str) -> bool:
    """Check if observation description mentions wrist."""
    return any(kw in desc for kw in ["wrist", "手腕", "drop", "下落", "trajectory", "轨迹"])


def _check_elbow_override(obs: "Observation", measured_angle: float) -> bool:
    """Override VLM elbow judgment if contradicted by measured angle."""
    desc_lower = obs.description.lower()

    # VLM says elbow is wide but measurement says tight
    if any(kw in desc_lower for kw in ["wide", "张开", "大角度", "伸展"]):
        if measured_angle < _ELBOW_TIGHT_THRESHOLD:
            obs.judgment = __import__("knowledge.schemas", fromlist=["ObservationJudgment"]).ObservationJudgment.NO
            obs.confidence = 0.9  # High confidence in kinematic data
            obs.override_reason = f"YOLO elbow angle {measured_angle:.0f}deg contradicts 'wide' (threshold {_ELBOW_TIGHT_THRESHOLD}deg)"
            return True

    # VLM says elbow is tight but measurement says wide
    if any(kw in desc_lower for kw in ["tight", "紧", "小角度", "弯曲"]):
        if measured_angle > _ELBOW_WIDE_THRESHOLD:
            from knowledge.schemas import ObservationJudgment
            obs.judgment = ObservationJudgment.NO
            obs.confidence = 0.9
            obs.override_reason = f"YOLO elbow angle {measured_angle:.0f}deg contradicts 'tight' (threshold {_ELBOW_WIDE_THRESHOLD}deg)"
            return True

    return False


def _check_wrist_override(obs: "Observation", wrist_below_elbow: float) -> bool:
    """Override VLM wrist judgment if contradicted by measured trajectory."""
    desc_lower = obs.description.lower()

    # VLM says wrist drops sharply but measurement shows minimal drop
    if any(kw in desc_lower for kw in ["sharp", "急剧", "v形", "v-shape", "猛"]):
        if wrist_below_elbow < 0.05:  # Less than 5% torso height = no significant drop
            from knowledge.schemas import ObservationJudgment
            obs.judgment = ObservationJudgment.NO
            obs.confidence = 0.85
            obs.override_reason = f"YOLO wrist_below_elbow={wrist_below_elbow:.3f} contradicts 'sharp drop' (threshold 0.05)"
            return True

    # VLM says smooth but measurement shows significant drop
    if any(kw in desc_lower for kw in ["smooth", "平滑", "gradual", "缓慢"]):
        if wrist_below_elbow > 0.15:  # More than 15% torso height = significant drop
            from knowledge.schemas import ObservationJudgment
            obs.judgment = ObservationJudgment.NO
            obs.confidence = 0.85
            obs.override_reason = f"YOLO wrist_below_elbow={wrist_below_elbow:.3f} contradicts 'smooth' (threshold 0.15)"
            return True

    return False


# ---------------------------------------------------------------------------
# HM-04: Low-Confidence Re-observation Candidates
# ---------------------------------------------------------------------------


def collect_reobserve_candidates(
    session: DiagnosticSession,
    new_observations: list[Observation],
    confidence_threshold: float = 0.5,
) -> list[str]:
    """Collect observation IDs that need re-observation in the next round.

    Candidates are:
    1. Low-confidence observations (< threshold)
    2. Unanchored observations (is_anchored=False)
    3. Contradicted observations (in session.contradictions)
    """
    candidates: set[str] = set()

    for obs in new_observations:
        # Low confidence
        if obs.confidence < confidence_threshold:
            candidates.add(obs.id)

        # Unanchored
        if not obs.is_anchored:
            candidates.add(obs.id)

    # Add contradicted observation IDs
    for contradiction in session.contradictions:
        candidates.add(contradiction["obs_new"])
        candidates.add(contradiction["obs_prior"])

    return sorted(candidates)


def build_reobserve_prompt_fragment(
    session: DiagnosticSession,
    candidate_ids: list[str],
    max_chars: int = 1000,
) -> str:
    """Build a prompt fragment requesting re-observation of uncertain items.

    Keeps output under max_chars to respect prompt budget.
    """
    if not candidate_ids:
        return ""

    obs_map = {o.id: o for o in session.observations}
    lines = ["【需要重新观察的项目】", "以下项目在前轮观察中不确定或有矛盾，请重新仔细观察："]

    for oid in candidate_ids:
        obs = obs_map.get(oid)
        if not obs:
            continue

        reason = ""
        if not obs.is_anchored:
            reason = "（前次观察缺少具体帧参考）"
        elif obs.override_reason:
            reason = f"（量化数据矛盾: {obs.override_reason[:50]}）"
        elif obs.confidence < 0.5:
            reason = f"（置信度仅{obs.confidence:.1f}）"

        line = f"- 在{obs.frame}中重新观察: {obs.description[:60]}{reason}"
        if sum(len(l) for l in lines) + len(line) > max_chars:
            lines.append("- ...（更多项目因提示预算限制省略）")
            break
        lines.append(line)

    return "\n".join(lines)
