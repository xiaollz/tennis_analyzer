"""Gemini API video analyzer for FTT YouTube videos.

Wraps the google-genai SDK to send YouTube URLs directly to Gemini
for video understanding analysis. Supports proxy URL via config.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

from knowledge.pipeline.video_state import mark_video, save_state


def load_api_config(config_path: Path) -> dict:
    """Load Gemini API configuration from JSON file."""
    return json.loads(config_path.read_text(encoding="utf-8"))


def create_client(config: dict) -> genai.Client:
    """Create a google-genai Client with proxy URL and API key from config.

    The base_url MUST be threaded through from config for proxy support.
    """
    return genai.Client(
        api_key=config["api_key"],
        http_options=types.HttpOptions(base_url=config["base_url"]),
    )


@retry(
    wait=wait_exponential(min=5, max=60),
    stop=stop_after_attempt(3),
)
def analyze_video(
    client: genai.Client,
    video_id: str,
    prompt: str,
    model: str = "gemini-3-flash-preview",
) -> str:
    """Send YouTube URL + prompt to Gemini and return raw analysis markdown.

    Args:
        client: Configured genai.Client instance.
        video_id: YouTube video ID.
        prompt: Analysis prompt text.
        model: Gemini model name.

    Returns:
        Raw markdown analysis text from Gemini response.
    """
    url = f"https://www.youtube.com/watch?v={video_id}"

    response = client.models.generate_content(
        model=model,
        contents=types.Content(
            parts=[
                types.Part(file_data=types.FileData(file_uri=url)),
                types.Part(text=prompt),
            ]
        ),
        config=types.GenerateContentConfig(temperature=0),
    )
    return response.text


def analyze_batch(
    client: genai.Client,
    video_ids: list[str],
    prompt: str,
    state: dict,
    state_path: Path,
    delay: float = 20.0,
    output_dir: Path | None = None,
) -> list[tuple[str, str]]:
    """Process videos sequentially with delay between API calls.

    Checkpoints state after each video. Skips already-analyzed videos.

    Args:
        client: Configured genai.Client.
        video_ids: List of video IDs to analyze.
        prompt: Analysis prompt text.
        state: Video state dict (modified in-place).
        state_path: Path to save state checkpoints.
        delay: Seconds between API calls (default 20s, conservative).
        output_dir: Directory for raw analysis markdown files.

    Returns:
        List of (video_id, analysis_text) tuples for successful analyses.
    """
    if output_dir is None:
        output_dir = Path("docs/research/ftt_video_analyses")
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[tuple[str, str]] = []

    for i, video_id in enumerate(video_ids):
        # Skip already processed
        entry = state["videos"].get(video_id, {})
        if entry.get("status") in ("analyzed", "extracted"):
            continue

        try:
            analysis_text = analyze_video(client, video_id, prompt)

            # Save raw markdown
            md_path = output_dir / f"{video_id}.md"
            md_path.write_text(analysis_text, encoding="utf-8")

            # Update state
            from datetime import datetime, timezone
            mark_video(
                state, video_id, "analyzed",
                analysis_file=str(md_path),
                analyzed_at=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            )
            save_state(state, state_path)

            results.append((video_id, analysis_text))

        except Exception as e:
            mark_video(state, video_id, "failed", error=str(e))
            save_state(state, state_path)

        # Delay between calls (skip after last video)
        if i < len(video_ids) - 1:
            time.sleep(delay)

    return results


def load_analysis_prompt(prompt_path: Path | None = None) -> str:
    """Read the video analysis prompt template.

    Extracts the prompt text between ``` markers from the markdown file.
    """
    if prompt_path is None:
        prompt_path = Path("docs/knowledge_graph/video_analysis_prompt.md")

    text = prompt_path.read_text(encoding="utf-8")

    # Extract text between ``` markers (the prompt block)
    match = re.search(r"```\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback: return everything after "## Prompt" header
    return text
