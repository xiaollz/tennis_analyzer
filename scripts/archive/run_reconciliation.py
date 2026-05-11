"""Run cross-source reconciliation against actual extracted data.

Loads the FTT-primary registry, runs reconcile_all() against TomAllsopp and
Feel Tennis extractions, saves the reconciliation report, and persists the
updated registry snapshot.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

from knowledge.pipeline.reconciliation import reconcile_all
from knowledge.registry import ConceptRegistry
from knowledge.schemas import Concept


def load_registry_from_snapshot(snapshot_path: Path) -> ConceptRegistry:
    """Load a ConceptRegistry from the JSON snapshot file."""
    data = json.loads(snapshot_path.read_text())
    registry = ConceptRegistry()

    # Snapshot is a flat list of concept dicts
    if isinstance(data, list):
        concepts = data
    elif isinstance(data, dict) and "concepts" in data:
        concepts = data["concepts"]
    else:
        raise ValueError(f"Unknown snapshot format: {type(data)}")

    for c_data in concepts:
        try:
            concept = Concept(**c_data)
            registry.add(concept)
        except Exception as e:
            print(f"  Warning: skipping concept {c_data.get('id', '?')}: {e}")
    return registry


def save_registry_snapshot(registry: ConceptRegistry, snapshot_path: Path) -> None:
    """Persist the registry back to JSON snapshot."""
    concepts = [c.model_dump() for c in registry.all_concepts()]
    snapshot_path.write_text(json.dumps(concepts, ensure_ascii=False, indent=2))


def _build_ref_id_map(base: Path, registry: ConceptRegistry) -> dict[str, str]:
    """Build mapping from short IDs (C01, T09, etc.) to registry concept IDs.

    Parses markdown analyses to extract short-ID + Chinese name pairs,
    then matches Chinese names against registry's name_zh field.
    """
    # Extract short-ID -> Chinese name from all markdown analyses
    short_id_to_zh: dict[str, str] = {}
    for md_dir in [
        base / "docs/research/tomallsopp_video_analyses",
        base / "docs/research/feeltennis_video_analyses",
    ]:
        if not md_dir.exists():
            continue
        for md_file in md_dir.glob("*.md"):
            text = md_file.read_text()
            for m in re.finditer(
                r"\*\*\s*(C\d{1,2}|T\d{1,2}|DC\d{1,2}|D\d{1,2})\s+([^*]+?)\*\*",
                text,
            ):
                short_id = m.group(1)
                name_zh = m.group(2).strip()
                if short_id not in short_id_to_zh:
                    short_id_to_zh[short_id] = name_zh

    # Build name_zh -> registry_id index
    zh_to_rid: dict[str, str] = {}
    for concept in registry.all_concepts():
        zh_to_rid[concept.name_zh] = concept.id

    # Map short IDs to registry IDs via Chinese name matching
    ref_map: dict[str, str] = {}
    for short_id, name_zh in short_id_to_zh.items():
        rid = zh_to_rid.get(name_zh)
        if rid:
            ref_map[short_id] = rid
        else:
            # Try partial match
            for zh, rid2 in zh_to_rid.items():
                if name_zh in zh or zh in name_zh:
                    ref_map[short_id] = rid2
                    break
    return ref_map


def main():
    base = Path(__file__).parent.parent

    snapshot_path = base / "knowledge/extracted/_registry_snapshot.json"
    ta_dir = base / "knowledge/extracted/tomallsopp_videos"
    ft_dir = base / "knowledge/extracted/feeltennis_videos"
    report_path = base / "knowledge/state/secondary_reconciliation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading FTT-primary registry...")
    registry = load_registry_from_snapshot(snapshot_path)
    size_before = len(registry)
    print(f"  Registry size: {size_before} concepts")

    # Build short-ID (C01, T09, etc.) to registry-ID mapping from markdown analyses
    print("\nBuilding short-ID mapping from markdown analyses...")
    ref_id_map = _build_ref_id_map(base, registry)
    print(f"  Mapped {len(ref_id_map)} short IDs to registry concepts")

    print("\nRunning reconciliation...")
    result = reconcile_all(registry, ta_dir, ft_dir, ref_id_map=ref_id_map)
    size_after = len(registry)

    print(f"\n{'='*50}")
    print(f"RECONCILIATION RESULTS")
    print(f"{'='*50}")
    print(f"  Total concepts processed: {result['total']}")
    print(f"  Agreements:  {result['agreements']} ({result['agreements']/max(result['total'],1):.0%})")
    print(f"  Complements: {result['complements']} ({result['complements']/max(result['total'],1):.0%})")
    print(f"  Conflicts:   {result['conflicts']} ({result['conflicts']/max(result['total'],1):.0%})")
    print(f"\n  Registry: {size_before} -> {size_after} (+{size_after - size_before} concepts)")

    conflict_rate = result['conflicts'] / max(result['total'], 1)
    if conflict_rate > 0.3:
        print(f"\n  WARNING: Conflict rate {conflict_rate:.0%} exceeds 30% -- manual review needed!")

    if size_after > 700:
        print(f"\n  WARNING: Registry size {size_after} exceeds 700 -- concept explosion risk!")

    # Build full report
    report = {
        "run_at": str(date.today()),
        "totals": {
            "agreements": result["agreements"],
            "complements": result["complements"],
            "conflicts": result["conflicts"],
            "total": result["total"],
        },
        "registry_size_before": size_before,
        "registry_size_after": size_after,
        "conflict_log": result["conflict_log"],
        "complement_concepts": result["complement_concepts"],
        "confidence_boosted": result["confidence_boosted"],
    }

    print(f"\nSaving report to {report_path}...")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    print(f"Saving updated registry to {snapshot_path}...")
    save_registry_snapshot(registry, snapshot_path)

    # Print conflict details if any
    if result["conflict_log"]:
        print(f"\nConflict details ({len(result['conflict_log'])} conflicts):")
        for c in result["conflict_log"]:
            print(f"  - {c['concept_name']} ({c['source']}) vs FTT:{c['matched_ftt_id']}")
            print(f"    Reason: {c['reason'][:100]}...")

    print("\nDone!")


if __name__ == "__main__":
    main()
