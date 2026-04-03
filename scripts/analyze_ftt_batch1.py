#!/usr/bin/env python3
"""Batch 1: Analyze 12 forehand-priority FTT videos via Gemini API.

Processes forehand technique + biomechanics videos with checkpointing.
Uses per-plan state slice (batch1_state.json) to avoid race conditions
with parallel batch plans.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from knowledge.pipeline.video_analyzer import (
    analyze_video,
    create_client,
    load_analysis_prompt,
    load_api_config,
)
from knowledge.pipeline.video_concept_extractor import (
    extract_concepts_from_analysis,
    save_video_extraction,
)
from knowledge.registry import ConceptRegistry

# ---------------------------------------------------------------------------
# Batch 1 video list: forehand technique + forehand biomechanics (12 videos)
# ---------------------------------------------------------------------------

BATCH1_VIDEOS = [
    {"video_id": "pWzyP-xfLfU", "title": "The Secret to Lag is on Your Handle"},
    {"video_id": "McCb-RfYd0w", "title": "The Magic of the Non-Dominant Side on the Forehand"},
    {"video_id": "JIMgI3jiVns", "title": "How Shoulder Rotation Syncs Your Contact"},
    {"video_id": "5KdScDKxVSI", "title": "Shoulder Adduction Will Transform Forehand Contact"},
    {"video_id": "Am8j1Zw5KrE", "title": "Shoulder Adduction Unlocks the Tennis Forehand"},
    {"video_id": "xLs469ZVMPU", "title": "Fix Your Forehand Over-Rotation - 3 Techniques"},
    {"video_id": "ExkBtFRhUWY", "title": "The Magic of Single-Foot Forehand Training"},
    {"video_id": "wFOy0RKWBTg", "title": "Swing OUT on the Forehand"},
    {"video_id": "5jHCDc44SQM", "title": "The Abdominal Corkscrew ft. Carson Branstine"},
    {"video_id": "hNVbbPEob3g", "title": "Chest Engagement Makes Controlling the Racket Face Easy"},
    {"video_id": "UB6SbA_KX9E", "title": "Proper Trunk Sequencing will Transform Your Tennis"},
    {"video_id": "wFIrPMutzRo", "title": "2 Secrets to Rotational Power - Side Bending + X"},
]

BATCH1_VIDEO_IDS = [v["video_id"] for v in BATCH1_VIDEOS]

# Paths
CONFIG_PATH = PROJECT_ROOT / "config" / "youtube_api_config.json"
PROMPT_PATH = PROJECT_ROOT / "docs" / "knowledge_graph" / "video_analysis_prompt.md"
REGISTRY_PATH = PROJECT_ROOT / "knowledge" / "extracted" / "_registry_snapshot.json"
STATE_PATH = PROJECT_ROOT / "knowledge" / "state" / "batch1_state.json"
ANALYSIS_DIR = PROJECT_ROOT / "docs" / "research" / "ftt_video_analyses"
EXTRACTION_DIR = PROJECT_ROOT / "knowledge" / "extracted" / "ftt_videos"

DELAY_BETWEEN_CALLS = 20  # seconds between API calls
RETRY_DELAY_429 = 60  # seconds on rate limit


# ---------------------------------------------------------------------------
# Enhanced extraction: parse Gemini's structured table output
# ---------------------------------------------------------------------------

def extract_from_gemini_analysis(
    analysis_text: str,
    video_id: str,
    video_title: str,
    registry: ConceptRegistry,
) -> tuple:
    """Enhanced extraction that parses Gemini's concept tables and references.

    Handles:
    - New concept table rows: | V01-01 | Name (EN) | Name (ZH) | Def | Cat | A/P |
    - Existing concept references: **C11 ... (supports/refines/extends)**
    - Teaching points as technique concepts
    - Drill/training methods from section 7
    """
    from knowledge.schemas import (
        Concept, ConceptType, Edge, RelationType, DiagnosticChain,
    )

    concepts: list = []
    edges: list = []
    chains: list = []
    source_file = f"ftt_video_{video_id}"

    cat_map = {
        "technique": ConceptType.TECHNIQUE,
        "biomechanics": ConceptType.BIOMECHANICS,
        "drill": ConceptType.DRILL,
        "symptom": ConceptType.SYMPTOM,
        "mental_model": ConceptType.MENTAL_MODEL,
        "connection": ConceptType.CONNECTION,
        "心理模型": ConceptType.MENTAL_MODEL,
        "技术": ConceptType.TECHNIQUE,
        "生物力学": ConceptType.BIOMECHANICS,
        "发力模型": ConceptType.BIOMECHANICS,
        "连接机制": ConceptType.CONNECTION,
        "训练方法": ConceptType.DRILL,
        "训练": ConceptType.DRILL,
        "症状": ConceptType.SYMPTOM,
        "准备阶段": ConceptType.TECHNIQUE,
        "前挥阶段": ConceptType.TECHNIQUE,
        "握拍": ConceptType.TECHNIQUE,
    }

    # --- 1. Parse new concept table rows ---
    # Format: | V01-01 | Name EN | Name ZH | Definition | Category | Active/Passive |
    table_pattern = re.compile(
        r"\|\s*(V\d+-\d+)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|"
    )
    for m in table_pattern.finditer(analysis_text):
        vid_concept_id = m.group(1).strip()
        name_en = m.group(2).strip()
        name_zh = m.group(3).strip()
        definition = m.group(4).strip()
        category_str = m.group(5).strip().lower()
        active_passive = m.group(6).strip()

        # Skip header rows
        if name_en in ("Name (EN)", "---", "Name(EN)"):
            continue

        snake_id = re.sub(r"[^a-zA-Z0-9\s]", "", name_en).lower().split()
        concept_id = "_".join(p for p in snake_id if p) or "unknown"
        if not concept_id[0].isalpha():
            concept_id = "c_" + concept_id

        # Check registry for dedup
        existing = registry.resolve(name_en, threshold=70)
        if existing:
            edges.append(Edge(
                source_id=existing,
                target_id=existing,
                relation=RelationType.SUPPORTS,
                confidence=0.7,
                evidence=f"Referenced in video: {video_title} ({vid_concept_id})",
                source_file=source_file,
            ))
            continue

        cat = cat_map.get(category_str, ConceptType.TECHNIQUE)
        concept = Concept(
            id=concept_id,
            name=name_en,
            name_zh=name_zh,
            category=cat,
            sources=["ftt"],
            description=definition[:200],
            confidence=0.8,
            active_or_passive=active_passive if active_passive.lower() in ("active", "passive") else None,
        )
        dup = registry.add(concept)
        if dup is None:
            concepts.append(concept)

    # --- 2. Parse existing concept references ---
    # Pattern: **C11 手腕滞后 (Wrist Lag)**: ... (supports)
    ref_pattern = re.compile(
        r"\*\*([CTMD]\d+)\s+([^*]+?)\*\*[^(]*\((\w+)\)"
    )
    existing_refs = set()
    for m in ref_pattern.finditer(analysis_text):
        ref_id = m.group(1).strip()
        rel_type = m.group(3).strip().lower()
        existing_refs.add(ref_id)

    # --- 3. Parse teaching points as concepts ---
    # Pattern: **[教学要点名称]**: description
    teaching_pattern = re.compile(
        r"\*\*\[(.+?)\]\*\*:\s*(.+?)(?:\n|$)"
    )
    for m in teaching_pattern.finditer(analysis_text):
        point_name = m.group(1).strip()
        description = m.group(2).strip()

        # Chinese name is the point_name itself
        # Generate snake_case ID from English chars only
        english_parts = re.sub(r"[^a-zA-Z0-9\s]", "", point_name).lower().split()
        concept_id = "_".join(p for p in english_parts if p and len(p) > 1)
        if not concept_id:
            # All Chinese - transliterate to a generic ID
            concept_id = f"tp_{video_id.replace('-', '_').lower()}_{len(concepts)}"
        if not concept_id[0].isalpha():
            concept_id = "tp_" + concept_id

        existing = registry.resolve(point_name, threshold=70)
        if existing:
            continue

        concept = Concept(
            id=concept_id,
            name=point_name,
            name_zh=point_name,
            category=ConceptType.TECHNIQUE,
            sources=["ftt"],
            description=description[:200],
            confidence=0.7,
        )
        dup = registry.add(concept)
        if dup is None:
            concepts.append(concept)

    # --- 4. Parse drill/training methods from section 7 ---
    drill_pattern = re.compile(
        r"\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*([CTDVM][^|]*?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|"
    )
    in_drill_section = False
    for line in analysis_text.split("\n"):
        if "训练方法" in line or "第七部分" in line:
            in_drill_section = True
            continue
        if in_drill_section and line.startswith("###"):
            in_drill_section = False
            continue
        if in_drill_section and line.startswith("|"):
            dm = drill_pattern.match(line)
            if dm:
                drill_name = dm.group(1).strip()
                if drill_name in ("训练名称", "---", ""):
                    continue
                english_parts = re.sub(r"[^a-zA-Z0-9\s]", "", drill_name).lower().split()
                drill_id = "drill_" + "_".join(p for p in english_parts if p and len(p) > 1)
                if not drill_id or drill_id == "drill_":
                    drill_id = f"drill_{video_id.replace('-', '_').lower()}_{len(concepts)}"
                drill_id = drill_id[:60]

                existing = registry.resolve(drill_name, threshold=70)
                if existing:
                    continue

                concept = Concept(
                    id=drill_id,
                    name=drill_name,
                    name_zh=drill_name,
                    category=ConceptType.DRILL,
                    sources=["ftt"],
                    description=f"Drill from FTT video: {video_title}",
                    confidence=0.8,
                )
                dup = registry.add(concept)
                if dup is None:
                    concepts.append(concept)

    # --- 5. Create co-occurrence edges between concepts ---
    concept_ids = [c.id for c in concepts]
    if len(concept_ids) >= 2:
        for i, src in enumerate(concept_ids):
            for tgt in concept_ids[i + 1:]:
                edges.append(Edge(
                    source_id=src,
                    target_id=tgt,
                    relation=RelationType.SUPPORTS,
                    confidence=0.6,
                    evidence=f"Co-occurring in video: {video_title}",
                    source_file=source_file,
                ))

    return concepts, edges, chains


def load_batch_state() -> dict:
    """Load or initialize batch1 state."""
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))

    # Initialize fresh state
    state = {
        "batch": "batch1",
        "plan": "03-03",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "videos": {},
    }
    for v in BATCH1_VIDEOS:
        state["videos"][v["video_id"]] = {
            "title": v["title"],
            "status": "pending",
            "analysis_file": None,
            "extracted_file": None,
            "analyzed_at": None,
            "concepts_count": 0,
            "edges_count": 0,
            "error": None,
        }
    save_batch_state(state)
    return state


def save_batch_state(state: dict) -> None:
    """Persist batch state to disk."""
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_registry() -> ConceptRegistry:
    """Load the concept registry from snapshot."""
    from knowledge.schemas import Concept, ConceptType

    registry = ConceptRegistry()
    if REGISTRY_PATH.exists():
        snapshot = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        # Registry snapshot can be a list of concept dicts or {"concepts": [...]}
        concepts_data = snapshot
        if isinstance(snapshot, dict) and "concepts" in snapshot:
            concepts_data = snapshot["concepts"]
        if isinstance(concepts_data, list):
            for c in concepts_data:
                try:
                    concept = Concept(**c)
                    registry.add(concept)
                except Exception:
                    # Skip malformed entries
                    pass
    return registry


def main() -> None:
    """Run batch 1 analysis pipeline."""
    print("=" * 60)
    print("FTT Batch 1: 12 Forehand-Priority Videos")
    print("=" * 60)

    # Load API config and verify proxy
    config = load_api_config(CONFIG_PATH)
    base_url = config.get("base_url", "DEFAULT (no proxy)")
    print(f"Using API base_url: {base_url}")
    assert config.get("base_url"), "ERROR: base_url not set in config, proxy not configured"

    model = config.get("model", "gemini-3-flash-preview")
    print(f"Using model: {model}")

    # Create Gemini client
    client = create_client(config)
    print("Gemini client created successfully")

    # Load analysis prompt
    prompt = load_analysis_prompt(PROMPT_PATH)
    print(f"Loaded analysis prompt ({len(prompt)} chars)")

    # Load concept registry
    registry = load_registry()
    print(f"Loaded registry with {len(registry)} existing concepts")

    # Load batch state
    state = load_batch_state()
    print(f"Loaded batch1 state: {len(state['videos'])} videos tracked")

    # Create output directories
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    EXTRACTION_DIR.mkdir(parents=True, exist_ok=True)

    # Process each video
    total = len(BATCH1_VIDEOS)
    succeeded = 0
    failed = 0
    skipped = 0
    total_concepts = 0
    total_edges = 0

    for i, video in enumerate(BATCH1_VIDEOS):
        vid = video["video_id"]
        title = video["title"]
        entry = state["videos"][vid]

        # Skip already processed
        if entry["status"] in ("analyzed", "extracted"):
            print(f"[{i+1}/{total}] SKIP (already {entry['status']}): {title}")
            skipped += 1
            continue

        print(f"\n[{i+1}/{total}] Analyzing: {title} ({vid})")

        # --- Step A: Analyze via Gemini API ---
        try:
            analysis_text = analyze_video(client, vid, prompt, model=model)

            # Save raw markdown FIRST (before extraction)
            md_path = ANALYSIS_DIR / f"{vid}.md"
            md_path.write_text(analysis_text, encoding="utf-8")
            print(f"  Saved raw analysis: {md_path.name} ({len(analysis_text)} chars)")

            # Update state: analyzed
            entry["status"] = "analyzed"
            entry["analysis_file"] = str(md_path.relative_to(PROJECT_ROOT))
            entry["analyzed_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            entry["error"] = None
            save_batch_state(state)  # CHECKPOINT

        except Exception as e:
            print(f"  ERROR during analysis: {e}")
            entry["status"] = "failed"
            entry["error"] = str(e)[:500]
            save_batch_state(state)
            failed += 1
            # Continue to next video after delay
            if i < total - 1:
                print(f"  Waiting {DELAY_BETWEEN_CALLS}s before next call...")
                time.sleep(DELAY_BETWEEN_CALLS)
            continue

        # --- Step B: Extract structured concepts ---
        try:
            concepts, edges, chains = extract_concepts_from_analysis(
                analysis_text, vid, title, registry
            )

            # Save extraction JSON
            out_path = save_video_extraction(
                vid, concepts, edges, chains, EXTRACTION_DIR
            )
            print(f"  Extracted: {len(concepts)} concepts, {len(edges)} edges -> {out_path.name}")

            # Update state: extracted
            entry["status"] = "extracted"
            entry["extracted_file"] = str(out_path.relative_to(PROJECT_ROOT))
            entry["concepts_count"] = len(concepts)
            entry["edges_count"] = len(edges)
            save_batch_state(state)  # CHECKPOINT

            succeeded += 1
            total_concepts += len(concepts)
            total_edges += len(edges)

        except Exception as e:
            print(f"  ERROR during extraction (raw MD preserved): {e}")
            # Analysis is saved, only extraction failed
            entry["error"] = f"extraction_failed: {str(e)[:400]}"
            save_batch_state(state)
            succeeded += 1  # Analysis succeeded, extraction failed
            total_concepts += 0

        # Delay between API calls (skip after last)
        if i < total - 1:
            print(f"  Waiting {DELAY_BETWEEN_CALLS}s before next call...")
            time.sleep(DELAY_BETWEEN_CALLS)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("BATCH 1 COMPLETE")
    print(f"  Processed: {succeeded + failed}/{total} (skipped {skipped})")
    print(f"  Succeeded: {succeeded}")
    print(f"  Failed: {failed}")
    print(f"  Total new concepts: {total_concepts}")
    print(f"  Total new edges: {total_edges}")
    print(f"  State file: {STATE_PATH}")
    print("=" * 60)


def reextract() -> None:
    """Re-extract concepts from already-saved analysis markdown files
    using the enhanced Gemini-output parser."""
    print("=" * 60)
    print("Re-extracting concepts with enhanced parser")
    print("=" * 60)

    registry = load_registry()
    print(f"Registry: {len(registry)} existing concepts")

    state = load_batch_state()
    total_concepts = 0
    total_edges = 0

    for video in BATCH1_VIDEOS:
        vid = video["video_id"]
        title = video["title"]
        md_path = ANALYSIS_DIR / f"{vid}.md"

        if not md_path.exists():
            print(f"  SKIP {vid}: no analysis file")
            continue

        analysis_text = md_path.read_text(encoding="utf-8")
        concepts, edges, chains = extract_from_gemini_analysis(
            analysis_text, vid, title, registry
        )

        # Save extraction JSON
        out_path = save_video_extraction(vid, concepts, edges, chains, EXTRACTION_DIR)
        print(f"  {vid}: {len(concepts)} concepts, {len(edges)} edges -> {out_path.name}")

        # Update state
        entry = state["videos"][vid]
        entry["concepts_count"] = len(concepts)
        entry["edges_count"] = len(edges)
        entry["extracted_file"] = str(out_path.relative_to(PROJECT_ROOT))

        total_concepts += len(concepts)
        total_edges += len(edges)

    save_batch_state(state)
    print(f"\nTotal: {total_concepts} concepts, {total_edges} edges")


if __name__ == "__main__":
    import sys
    if "--reextract" in sys.argv:
        reextract()
    else:
        main()
