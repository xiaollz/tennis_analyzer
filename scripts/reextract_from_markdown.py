"""Re-extract concepts from Gemini markdown analyses into structured JSONs.

The original regex-based extractor failed on Chinese-language Gemini output.
This script parses the consistent markdown format to extract:
1. Existing concept references (agreements) from Section 2 and Section 5
2. New concept tables from Section 2
3. Relationship types from Section 5
"""

from __future__ import annotations

import json
import re
from pathlib import Path


def _parse_existing_concepts(text: str) -> list[dict]:
    """Extract references to existing FTT concepts from markdown."""
    concepts = []
    seen_ids = set()

    # Pattern: **C01 旋转动力链** (Supports): description
    # Also matches T01, DC01, D01 patterns
    pattern = r'\*\*\s*(C\d{1,2}|T\d{1,2}|DC\d{1,2}|D\d{1,2})\s+([^*]+?)\*\*\s*\((\w+)\)'
    for match in re.finditer(pattern, text):
        ref_id = match.group(1)
        name_zh = match.group(2).strip()
        relationship = match.group(3).strip()
        if ref_id not in seen_ids:
            seen_ids.add(ref_id)
            concepts.append({
                "ref_id": ref_id,
                "name_zh": name_zh,
                "relationship": relationship,
            })

    return concepts


def _parse_new_concepts(text: str, source_label: str) -> list[dict]:
    """Extract new concept tables from Section 2 of markdown."""
    concepts = []

    # Pattern for new concept table rows:
    # | V01-01 | Name (EN) | Name (ZH) | Definition | Category | Active/Passive |
    # Also matches V1-01 pattern
    pattern = r'\|\s*(V\d+-\d+)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|'
    for match in re.finditer(pattern, text):
        vid = match.group(1).strip()
        name_en = match.group(2).strip()
        name_zh = match.group(3).strip()
        definition = match.group(4).strip()
        category_raw = match.group(5).strip()
        active_passive = match.group(6).strip()

        # Skip header rows
        if name_en.startswith("Name") or name_en.startswith("---"):
            continue

        # Map Chinese category to ConceptType
        category = _map_category(category_raw)

        # Generate a snake_case ID from the English name
        concept_id = re.sub(r'[^a-z0-9]+', '_', name_en.lower()).strip('_')
        # Ensure valid ID
        if not concept_id or not concept_id[0].isalpha():
            concept_id = "c_" + concept_id

        concepts.append({
            "id": concept_id,
            "name": name_en,
            "name_zh": name_zh,
            "aliases": [],
            "category": category,
            "sources": [source_label],
            "description": definition,
            "vlm_features": [],
            "muscles_involved": [],
            "active_or_passive": active_passive if active_passive != "---" else None,
            "confidence": 0.8,
        })

    return concepts


def _parse_relationships(text: str) -> list[dict]:
    """Extract relationship entries from Section 5 tables."""
    edges = []
    # Pattern for Section 5 relationship table:
    # | what video said | C01 concept name | `relationship_type` | meaning |
    pattern = r'\|\s*([^|]+?)\s*\|\s*(C\d{1,2}|T\d{1,2}|DC\d{1,2}|D\d{1,2})\s+([^|]+?)\s*\|\s*`(\w+)`\s*\|\s*([^|]+?)\s*\|'
    for match in re.finditer(pattern, text):
        ref_id = match.group(2).strip()
        concept_name = match.group(3).strip()
        rel_type = match.group(4).strip()
        meaning = match.group(5).strip()
        edges.append({
            "target_ref_id": ref_id,
            "target_name": concept_name,
            "relation": rel_type,
            "evidence": meaning,
        })
    return edges


def _map_category(raw: str) -> str:
    """Map Chinese/mixed category strings to ConceptType values."""
    raw_lower = raw.lower().strip()
    mapping = {
        "technique": "technique",
        "biomechanics": "biomechanics",
        "drill": "drill",
        "symptom": "symptom",
        "mental_model": "mental_model",
        "connection": "connection",
        "发力模型": "biomechanics",
        "动力链": "biomechanics",
        "连接机制": "connection",
        "心理模型": "mental_model",
        "训练方法": "drill",
        "移动/地基": "technique",
        "移动": "technique",
        "地基": "technique",
        "时机": "technique",
        "容错性": "technique",
        "诊断标准": "symptom",
    }
    for key, val in mapping.items():
        if key in raw_lower:
            return val
    return "technique"


def extract_from_markdown(md_path: Path, source_label: str) -> dict:
    """Parse a Gemini analysis markdown and return structured extraction."""
    text = md_path.read_text()
    video_id = md_path.stem

    existing_refs = _parse_existing_concepts(text)
    new_concepts = _parse_new_concepts(text, source_label)
    relationships = _parse_relationships(text)

    # Build concept list: new concepts as full Concept-like dicts
    concepts = new_concepts

    # Build edges from relationships
    edges = []
    for rel in relationships:
        edges.append({
            "source_id": "video_" + video_id,
            "target_id": rel["target_ref_id"],
            "relation": rel["relation"],
            "confidence": 0.8,
            "evidence": rel["evidence"],
            "source_file": f"{source_label}_video_{video_id}",
        })

    return {
        "video_id": video_id,
        "concepts": concepts,
        "edges": edges,
        "diagnostic_chains": [],
        "existing_concept_refs": existing_refs,
    }


def reextract_channel(
    md_dir: Path, json_dir: Path, source_label: str
) -> dict:
    """Re-extract all videos in a channel from markdown to JSON."""
    json_dir.mkdir(parents=True, exist_ok=True)
    stats = {"total": 0, "with_concepts": 0, "total_new_concepts": 0, "total_refs": 0}

    for md_file in sorted(md_dir.glob("*.md")):
        extraction = extract_from_markdown(md_file, source_label)
        out_path = json_dir / f"{md_file.stem}.json"
        out_path.write_text(json.dumps(extraction, ensure_ascii=False, indent=2))
        stats["total"] += 1
        if extraction["concepts"]:
            stats["with_concepts"] += 1
        stats["total_new_concepts"] += len(extraction["concepts"])
        stats["total_refs"] += len(extraction.get("existing_concept_refs", []))

    return stats


if __name__ == "__main__":
    base = Path(__file__).parent.parent

    print("Re-extracting TomAllsopp...")
    ta_stats = reextract_channel(
        md_dir=base / "docs/research/tomallsopp_video_analyses",
        json_dir=base / "knowledge/extracted/tomallsopp_videos",
        source_label="tomallsopp",
    )
    print(f"  {ta_stats}")

    print("Re-extracting Feel Tennis...")
    ft_stats = reextract_channel(
        md_dir=base / "docs/research/feeltennis_video_analyses",
        json_dir=base / "knowledge/extracted/feeltennis_videos",
        source_label="feeltennis",
    )
    print(f"  {ft_stats}")

    print("Done!")
