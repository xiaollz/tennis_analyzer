"""UserProfile model linking training sessions to canonical knowledge graph concepts.

Parses learning.md records, resolves mentions to the 732-node concept registry
via fuzzy matching, and tracks per-concept progress status across sessions.
"""

from __future__ import annotations

import json
import re
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from knowledge.registry import ConceptRegistry
from knowledge.schemas import Concept


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ConceptStatus(str, Enum):
    """Progress status for a concept in the user's training journey."""

    STRUGGLING = "struggling"
    IMPROVING = "improving"
    MASTERED = "mastered"
    REGRESSED = "regressed"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ConceptLink(BaseModel):
    """Link between a user session and a canonical concept."""

    concept_id: str
    status: ConceptStatus
    first_seen: str = Field(description="Date first encountered (YYYY-MM-DD)")
    last_seen: str = Field(description="Date last encountered (YYYY-MM-DD)")
    cues: list[str] = Field(default_factory=list, description="Training cues associated with this concept")


class SessionEntry(BaseModel):
    """A single training session parsed from learning.md."""

    date: str = Field(description="YYYY-MM-DD")
    summary: str = Field(default="", description="Brief session summary")
    concept_links: list[ConceptLink] = Field(default_factory=list)
    breakthroughs: list[str] = Field(default_factory=list)


class UserProfile(BaseModel):
    """User training profile linking sessions to knowledge graph concepts."""

    sessions: list[SessionEntry] = Field(default_factory=list)
    concept_map: dict[str, ConceptLink] = Field(
        default_factory=dict,
        description="Aggregated per-concept status (latest state)",
    )

    def get_status(self, concept_id: str) -> ConceptStatus | None:
        """Get the latest status for a concept, or None if not tracked."""
        link = self.concept_map.get(concept_id)
        return link.status if link else None

    def active_issues(self) -> list[ConceptLink]:
        """Return concepts with status struggling or regressed."""
        return [
            link
            for link in self.concept_map.values()
            if link.status in (ConceptStatus.STRUGGLING, ConceptStatus.REGRESSED)
        ]

    def recent_breakthroughs(self, n: int = 3) -> list[SessionEntry]:
        """Return last n sessions that have breakthroughs, most recent first."""
        # Sessions are stored in parse order; sort by date descending
        sorted_sessions = sorted(self.sessions, key=lambda s: s.date, reverse=True)
        result = []
        for session in sorted_sessions:
            if session.breakthroughs:
                result.append(session)
                if len(result) >= n:
                    break
        return result

    def to_json(self, path: Path) -> None:
        """Serialize profile to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def from_json(cls, path: Path) -> UserProfile:
        """Load profile from JSON file."""
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.model_validate(data)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

# Regex patterns
_DATE_HEADER = re.compile(r"^##\s+(\d{4}-\d{2}-\d{2})")
_BOLD_TERM = re.compile(r"\*\*(.+?)\*\*")
_BACKTICK_CUE = re.compile(r"`([^`]+)`")
_ARROW_TERM = re.compile(r"[→←]\s*(.+?)(?:\s*[→←]|$)")
_CODE_BLOCK = re.compile(r"```(.*?)```", re.DOTALL)

# Status keywords
_POSITIVE_KEYWORDS = re.compile(
    r"突破|成功|mastered|解决|消失|改善|进步|稳定|出现|找到|对了|correct|breakthrough|eliminated|solved",
    re.IGNORECASE,
)
_NEGATIVE_KEYWORDS = re.compile(
    r"问题|仍然|还是|still|错误|回来|又|困难|不够|不足|缺|没有|缺少|struggle|issue|wrong|error|失败",
    re.IGNORECASE,
)


def _split_sessions(text: str) -> list[tuple[str, str]]:
    """Split learning.md into (date, section_text) pairs."""
    sessions: list[tuple[str, str]] = []
    lines = text.split("\n")
    current_date: str | None = None
    current_lines: list[str] = []

    for line in lines:
        m = _DATE_HEADER.match(line)
        if m:
            if current_date is not None:
                sessions.append((current_date, "\n".join(current_lines)))
            current_date = m.group(1)
            current_lines = [line]
        elif current_date is not None:
            current_lines.append(line)

    if current_date is not None:
        sessions.append((current_date, "\n".join(current_lines)))

    return sessions


def _extract_bold_terms(text: str) -> list[str]:
    """Extract bold-text terms from markdown."""
    return _BOLD_TERM.findall(text)


def _extract_causal_chain_terms(text: str) -> list[str]:
    """Extract terms from code blocks with arrow chains."""
    terms: list[str] = []
    for block in _CODE_BLOCK.findall(text):
        # Split by arrows and newlines
        for segment in re.split(r"[→←\n]", block):
            segment = segment.strip()
            # Remove leading markers like "→ "
            segment = re.sub(r"^[\s→←]+", "", segment)
            if segment and len(segment) >= 2:
                # Clean up parenthetical explanations
                clean = re.sub(r"[（(].*?[）)]", "", segment).strip()
                if clean and len(clean) >= 2:
                    terms.append(clean)
    return terms


def _extract_cues(text: str) -> list[str]:
    """Extract backtick-quoted cues from text."""
    return _BACKTICK_CUE.findall(text)


def _determine_mention_sentiment(text: str, term: str) -> str:
    """Determine if a concept mention is positive or negative in context.

    Returns: 'positive', 'negative', or 'neutral'
    """
    # Find lines containing the term
    term_lower = term.lower()
    relevant_lines = []
    for line in text.split("\n"):
        if term_lower in line.lower() or term in line:
            relevant_lines.append(line)

    if not relevant_lines:
        return "neutral"

    combined = " ".join(relevant_lines)
    pos_count = len(_POSITIVE_KEYWORDS.findall(combined))
    neg_count = len(_NEGATIVE_KEYWORDS.findall(combined))

    if pos_count > neg_count:
        return "positive"
    elif neg_count > pos_count:
        return "negative"
    return "neutral"


def _build_registry(snapshot_path: Path) -> ConceptRegistry:
    """Load a ConceptRegistry from the snapshot JSON."""
    registry = ConceptRegistry()
    data = json.loads(snapshot_path.read_text(encoding="utf-8"))
    for item in data:
        concept = Concept.model_validate(item)
        registry.add(concept)
    return registry


def _derive_status(
    history: list[tuple[str, str]],
) -> ConceptStatus:
    """Derive concept status from chronological list of (date, sentiment).

    Args:
        history: list of (date, sentiment) sorted by date ascending
    """
    if not history:
        return ConceptStatus.STRUGGLING

    # Track state transitions
    ever_positive = False
    current = ConceptStatus.STRUGGLING

    for _date, sentiment in history:
        if sentiment == "positive":
            ever_positive = True
            current = ConceptStatus.MASTERED
        elif sentiment == "negative":
            if ever_positive:
                current = ConceptStatus.REGRESSED
            else:
                current = ConceptStatus.STRUGGLING
        else:
            # neutral mention
            if ever_positive:
                current = ConceptStatus.IMPROVING
            # if never positive, keep current

    # If ended at mastered but had multiple mentions, might be improving
    # Only mastered if last mention is explicitly positive
    if current == ConceptStatus.MASTERED and len(history) > 1:
        last_sentiment = history[-1][1]
        if last_sentiment != "positive":
            current = ConceptStatus.IMPROVING

    return current


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_profile_from_learning(
    learning_path: Path,
    registry_snapshot_path: Path,
) -> UserProfile:
    """Build a UserProfile by parsing learning.md and linking to registry concepts.

    Args:
        learning_path: Path to docs/record/learning.md
        registry_snapshot_path: Path to knowledge/extracted/_registry_snapshot.json
    """
    registry = _build_registry(registry_snapshot_path)
    text = learning_path.read_text(encoding="utf-8")
    raw_sessions = _split_sessions(text)

    # Concept history: concept_id -> list of (date, sentiment)
    concept_history: dict[str, list[tuple[str, str]]] = {}
    # Concept cues: concept_id -> set of cues
    concept_cues: dict[str, set[str]] = {}
    # First/last seen per concept
    first_seen: dict[str, str] = {}
    last_seen: dict[str, str] = {}

    sessions: list[SessionEntry] = []

    for date, section_text in raw_sessions:
        # Extract candidate terms
        bold_terms = _extract_bold_terms(section_text)
        chain_terms = _extract_causal_chain_terms(section_text)
        cues = _extract_cues(section_text)
        all_terms = bold_terms + chain_terms

        # Resolve terms to concept IDs
        resolved: dict[str, str] = {}  # concept_id -> original term
        seen_ids: set[str] = set()

        for term in all_terms:
            # Try the term directly
            cid = registry.resolve(term, threshold=65)
            if cid and cid not in seen_ids:
                resolved[cid] = term
                seen_ids.add(cid)
                continue

            # For Chinese terms, also try extracting English keywords
            # e.g., "scooping 的根因" -> try "scooping"
            english_parts = re.findall(r"[A-Za-z][A-Za-z\s]+[A-Za-z]", term)
            for eng in english_parts:
                eng = eng.strip()
                if len(eng) >= 3:
                    cid = registry.resolve(eng, threshold=65)
                    if cid and cid not in seen_ids:
                        resolved[cid] = eng
                        seen_ids.add(cid)

        # Build session concept links
        session_links: list[ConceptLink] = []
        for cid, orig_term in resolved.items():
            sentiment = _determine_mention_sentiment(section_text, orig_term)

            # Track history
            if cid not in concept_history:
                concept_history[cid] = []
            concept_history[cid].append((date, sentiment))

            if cid not in first_seen:
                first_seen[cid] = date
            last_seen[cid] = date

            # Find cues near this concept
            related_cues: list[str] = []
            for cue in cues:
                if orig_term.lower() in cue.lower() or (
                    len(orig_term) >= 3
                    and any(
                        orig_term.lower() in line.lower()
                        for line in section_text.split("\n")
                        if cue in line
                    )
                ):
                    related_cues.append(cue)

            if cid not in concept_cues:
                concept_cues[cid] = set()
            concept_cues[cid].update(related_cues)

            # Temporary status (will be replaced by derived status)
            session_links.append(
                ConceptLink(
                    concept_id=cid,
                    status=ConceptStatus.STRUGGLING,  # placeholder
                    first_seen=first_seen[cid],
                    last_seen=date,
                    cues=related_cues,
                )
            )

        # Detect breakthroughs from section text
        breakthroughs: list[str] = []
        for line in section_text.split("\n"):
            if _POSITIVE_KEYWORDS.search(line) and (
                "###" in line or "**" in line
            ):
                clean = line.strip().lstrip("#").strip().strip("*").strip()
                if clean:
                    breakthroughs.append(clean)

        # Summary from first ### section or first discovery
        summary_lines = []
        for line in section_text.split("\n"):
            if line.startswith("### "):
                summary_lines.append(line.lstrip("#").strip())
                if len(summary_lines) >= 2:
                    break
        summary = "; ".join(summary_lines) if summary_lines else f"Session {date}"

        sessions.append(
            SessionEntry(
                date=date,
                summary=summary,
                concept_links=session_links,
                breakthroughs=breakthroughs,
            )
        )

    # Sort concept histories by date for status derivation
    for cid in concept_history:
        concept_history[cid].sort(key=lambda x: x[0])

    # Build aggregated concept map with derived statuses
    concept_map: dict[str, ConceptLink] = {}
    for cid, history in concept_history.items():
        status = _derive_status(history)
        concept_map[cid] = ConceptLink(
            concept_id=cid,
            status=status,
            first_seen=first_seen[cid],
            last_seen=last_seen[cid],
            cues=sorted(concept_cues.get(cid, set())),
        )

    # Update session-level concept links with derived status
    for session in sessions:
        for link in session.concept_links:
            if link.concept_id in concept_map:
                # Use the status as of this session's date
                # For simplicity, use the final derived status
                link.status = concept_map[link.concept_id].status

    profile = UserProfile(sessions=sessions, concept_map=concept_map)
    return profile


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    profile = build_profile_from_learning(
        Path("docs/record/learning.md"),
        Path("knowledge/extracted/_registry_snapshot.json"),
    )
    profile.to_json(Path("knowledge/extracted/user_journey/user_profile.json"))
    print(f"Profile: {len(profile.sessions)} sessions, {len(profile.concept_map)} linked concepts")
    print("\nActive issues:")
    for issue in profile.active_issues():
        print(f"  - {issue.concept_id}: {issue.status.value}")
    print(f"\nRecent breakthroughs ({min(3, len(profile.recent_breakthroughs()))} sessions):")
    for session in profile.recent_breakthroughs(n=3):
        print(f"  - {session.date}: {', '.join(session.breakthroughs[:3])}")
