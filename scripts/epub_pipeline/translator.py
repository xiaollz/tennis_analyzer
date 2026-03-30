"""Translate English FTT book text into Chinese using Gemini API."""

import json
import os
import re
import time

from google import genai
from google.genai import types

from .glossary import build_glossary_prompt

# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

_CFG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "config", "youtube_api_config.json"
)

with open(_CFG_PATH) as _f:
    _cfg = {k: v for k, v in json.load(_f).items() if not k.startswith("_")}

client = genai.Client(
    api_key=_cfg["api_key"],
    http_options={"api_version": "v1beta", "base_url": _cfg["base_url"]},
)
model = _cfg.get("model", "gemini-3-flash-preview")

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a professional translator specializing in tennis biomechanics literature.
Translate the following English text into Simplified Chinese.

RULES:
1. Use the tennis terminology glossary below EXACTLY.
2. Preserve the author's conversational, coaching tone - not stiff academic Chinese.
3. Keep proper nouns in English: player names (Federer, Djokovic, Brady), brand names.
4. Technical terms: translate but add English in parentheses on FIRST occurrence per chapter.
5. Numbered/bullet lists: preserve structure exactly.
6. English coaching cues in quotes: keep the English, translate surrounding text.
7. Return ONLY translations, no explanations or notes.
8. Match each [P1], [P2]... marker exactly.

{glossary_text}"""

TRANSLATABLE_TYPES = {"p", "blockquote", "h2", "h3", "h4", "figcaption"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_system_prompt() -> str:
    return SYSTEM_PROMPT.format(glossary_text=build_glossary_prompt())


def _word_count(text: str) -> int:
    return len(text.split())


def _batch_elements(elements: list[dict], batch_size: int = 9, max_words: int = 1800):
    """Yield batches of (index, element) tuples, ~batch_size items or ~max_words."""
    batch: list[tuple[int, dict]] = []
    words = 0
    for idx, el in enumerate(elements):
        if el.get("type") not in TRANSLATABLE_TYPES:
            continue
        text = el.get("text", "").strip()
        if not text:
            continue
        w = _word_count(text)
        if batch and (len(batch) >= batch_size or words + w > max_words):
            yield batch
            batch = []
            words = 0
        batch.append((idx, el))
        words += w
    if batch:
        yield batch


def _parse_markers(response_text: str, count: int) -> list[str]:
    """Extract translations keyed by [P1]..[Pn] markers."""
    results = []
    for i in range(1, count + 1):
        start_tag = f"[P{i}]"
        if i < count:
            end_tag = f"[P{i + 1}]"
        else:
            end_tag = None

        start_pos = response_text.find(start_tag)
        if start_pos == -1:
            return []  # marker missing – signal parse failure
        start_pos += len(start_tag)

        if end_tag:
            end_pos = response_text.find(end_tag, start_pos)
            if end_pos == -1:
                return []
            results.append(response_text[start_pos:end_pos].strip())
        else:
            results.append(response_text[start_pos:].strip())
    return results


# ---------------------------------------------------------------------------
# Core translation
# ---------------------------------------------------------------------------


def _translate_batch(
    elements: list[tuple[int, dict]], batch_idx: int, chapter_id: str
) -> list[str]:
    """Send one batch to Gemini and return list of Chinese translations."""
    # Build user prompt with markers
    lines = []
    for i, (_, el) in enumerate(elements, start=1):
        lines.append(f"[P{i}] {el['text'].strip()}")
    user_prompt = "\n\n".join(lines)
    count = len(elements)

    system = _build_system_prompt()

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=0.3,
                ),
            )
            text = response.text or ""

            # Try marker-based parsing first
            translations = _parse_markers(text, count)
            if len(translations) == count:
                return translations

            # Fallback: split by blank lines / newlines
            parts = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
            # Strip any leftover [Pn] prefixes
            parts = [re.sub(r"^\[P\d+\]\s*", "", p) for p in parts]
            if len(parts) == count:
                return parts

            # Retry with simpler prompt on parse failure
            if attempt < 2:
                print(
                    f"  [!] Parse mismatch ch={chapter_id} batch={batch_idx}: "
                    f"expected {count}, got {len(parts)}. Retrying..."
                )
                time.sleep(2 ** attempt)
                continue
            else:
                print(
                    f"  [!] Parse failed after retries ch={chapter_id} batch={batch_idx}. "
                    f"Using raw split (got {len(parts)}/{count})."
                )
                # Pad or truncate
                if len(parts) < count:
                    parts.extend(["[TRANSLATION MISSING]"] * (count - len(parts)))
                return parts[:count]

        except Exception as e:
            if attempt < 2:
                wait = 3 * (2 ** attempt)
                print(
                    f"  [!] API error ch={chapter_id} batch={batch_idx}: {e}. "
                    f"Retrying in {wait}s..."
                )
                time.sleep(wait)
            else:
                print(f"  [!] API error ch={chapter_id} batch={batch_idx}: {e}. Giving up.")
                return ["[TRANSLATION ERROR]"] * count

    return ["[TRANSLATION ERROR]"] * count  # should not reach here


# ---------------------------------------------------------------------------
# Chapter-level
# ---------------------------------------------------------------------------


def translate_chapter(chapter_data: dict, output_dir: str, force: bool = False) -> dict:
    """Translate one chapter. Returns cache dict with translations."""
    chapter_id = chapter_data.get("id", chapter_data.get("chapter_id", "unknown"))
    cache_dir = os.path.join(output_dir, "translations")
    cache_path = os.path.join(cache_dir, f"{chapter_id}.json")

    # Check cache
    if not force and os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        if cached.get("status") == "complete":
            print(f"  [cache] {chapter_id} — loaded from cache")
            return cached

    elements = chapter_data.get("elements", [])
    translations: dict[str, str] = {}

    batches = list(_batch_elements(elements))
    if not batches:
        result = {
            "chapter_id": chapter_id,
            "translations": {},
            "status": "complete",
        }
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return result

    for batch_idx, batch in enumerate(batches):
        print(f"  Batch {batch_idx + 1}/{len(batches)} ({len(batch)} elements)...")
        translated = _translate_batch(batch, batch_idx, chapter_id)

        for (elem_idx, _), zh_text in zip(batch, translated):
            translations[str(elem_idx)] = zh_text

        # Rate limiting between batches
        if batch_idx < len(batches) - 1:
            time.sleep(3)

    result = {
        "chapter_id": chapter_id,
        "translations": translations,
        "status": "complete",
    }

    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


# ---------------------------------------------------------------------------
# All chapters
# ---------------------------------------------------------------------------


def translate_all(
    structured_data: dict, output_dir: str, force: bool = False, from_chapter: str = None
) -> None:
    """Translate all chapters. Skip chapters with skip_translate=True."""
    chapters = structured_data.get("chapters", structured_data) if isinstance(structured_data, dict) else structured_data
    total = len(chapters)
    failed: list[str] = []
    started = from_chapter is None

    for i, chapter in enumerate(chapters):
        chapter_id = chapter.get("id", chapter.get("chapter_id", f"unknown_{i}"))

        if not started:
            if chapter_id == from_chapter:
                started = True
            else:
                print(f"[{i + 1}/{total}] {chapter_id} — skipped (before --from-chapter)")
                continue

        if chapter.get("skip_translate"):
            print(f"[{i + 1}/{total}] {chapter_id} — skipped (skip_translate)")
            continue

        print(f"[{i + 1}/{total}] Translating {chapter_id}...")
        try:
            translate_chapter(chapter, output_dir, force=force)
            print(f"  Done.")
        except Exception as e:
            print(f"  [!] FAILED: {e}")
            failed.append(chapter_id)

    if failed:
        print(f"\n{'=' * 50}")
        print(f"Translation completed with errors.")
        print(f"Failed chapters ({len(failed)}): {', '.join(failed)}")
    else:
        print(f"\nAll {total} chapters translated successfully.")
