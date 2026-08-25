#!/usr/bin/env python3
"""Analyze a YouTube video with Gemini native file_data.

This follows the local youtube-gemini skill: fetch real YouTube metadata first,
then pass the canonical YouTube URL as file_data to Gemini.
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

from google import genai
from google.genai import types


_LOCAL_SECRET_ENV_FILES = (
    Path("/Users/qsy/.codex/skills/analyze-tennis/config.env"),
    Path("/Users/qsy/.agents/skills/analyze-tennis/config.env"),
    Path("/Users/qsy/.codex/skills/youtube-gemini/config.env"),
    Path("/Users/qsy/.agents/skills/youtube-gemini/config.env"),
)


def video_id_from_url(video_url: str) -> str:
    if "youtu.be/" in video_url:
        return video_url.split("youtu.be/", 1)[1].split("?", 1)[0].split("&", 1)[0]
    if "v=" in video_url:
        return video_url.split("v=", 1)[1].split("&", 1)[0]
    return video_url.rstrip("/").rsplit("/", 1)[-1].split("?", 1)[0]


def get_youtube_metadata(video_url: str) -> dict[str, str]:
    video_id = video_id_from_url(video_url)
    canonical_url = f"https://www.youtube.com/watch?v={video_id}"
    req = urllib.request.Request(
        canonical_url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36"
            )
        },
    )
    page_html = urllib.request.urlopen(req, timeout=20).read().decode(
        "utf-8", errors="replace"
    )
    title_match = re.search(
        r"<title>(.*?)(?:\s*-\s*YouTube)?</title>", page_html
    )
    channel_match = re.search(r'"ownerChannelName":"([^"]+)"', page_html)
    return {
        "video_id": video_id,
        "url": canonical_url,
        "title": html.unescape(title_match.group(1).strip()) if title_match else "Unknown",
        "channel": html.unescape(channel_match.group(1)) if channel_match else "Unknown",
    }


def _load_local_secret_env(var_name: str) -> str:
    for env_path in _LOCAL_SECRET_ENV_FILES:
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == var_name:
                return value.strip().strip("\"'")
    return ""


def _resolve_secret_ref(value):
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        var_name = value[2:-1]
        return os.environ.get(var_name, "") or _load_local_secret_env(var_name)
    if isinstance(value, list):
        return [_resolve_secret_ref(item) for item in value]
    if isinstance(value, dict):
        return {key: _resolve_secret_ref(item) for key, item in value.items()}
    return value


def load_config(path: Path) -> dict:
    with path.open() as f:
        cfg = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    return _resolve_secret_ref(cfg)


def _configured_endpoints(cfg: dict) -> list[dict[str, str]]:
    raw_endpoints = cfg.get("endpoints") or [
        {
            "name": "primary",
            "api_key": cfg.get("api_key", ""),
            "base_url": cfg.get("base_url", ""),
        }
    ]
    endpoints = []
    for index, endpoint in enumerate(raw_endpoints):
        api_key = str(endpoint.get("api_key", "") or "")
        if not api_key:
            continue
        endpoints.append(
            {
                "name": str(endpoint.get("name") or f"endpoint-{index + 1}"),
                "api_key": api_key,
                "base_url": str(endpoint.get("base_url", "") or ""),
            }
        )
    return endpoints


def _redact_error(exc: Exception, endpoints: list[dict[str, str]]) -> str:
    text = str(exc)
    for endpoint in endpoints:
        secret = endpoint.get("api_key", "")
        if secret:
            text = text.replace(secret, "***")
    return text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument(
        "--prompt",
        default=(
            "请详细分析这个 YouTube 视频。先给核心论点，再给关键时间段、"
            "技术要点、可观察证据、与已有知识库的关系、可能误读和训练建议。"
        ),
    )
    parser.add_argument(
        "--config",
        default="config/youtube_api_config.json",
        help="Path to youtube_api_config.json",
    )
    parser.add_argument(
        "--out-dir",
        default="/Users/qsy/.gemini/transcripts",
        help="Directory for saved analysis text",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    endpoints = _configured_endpoints(cfg)
    if not endpoints:
        raise RuntimeError("Gemini 配置中没有可用的 API key")

    meta = get_youtube_metadata(args.url)
    contents = [
        types.Content(
            parts=[
                types.Part(
                    file_data=types.FileData(
                        file_uri=meta["url"],
                        mime_type="video/mp4",
                    )
                ),
                types.Part(
                    text=(
                        f"视频标题：{meta['title']}\n"
                        f"频道：{meta['channel']}\n"
                        f"URL：{meta['url']}\n\n"
                        f"{args.prompt}"
                    )
                ),
            ]
        )
    ]
    models = [cfg.get("model", "gemini-3.5-flash")]
    models.extend(
        model for model in cfg.get("fallback_models", [])
        if model and model not in models
    )
    response = None
    used_model = None
    used_endpoint = None
    last_error = None
    for endpoint_index, endpoint in enumerate(endpoints):
        client_kwargs = {"api_key": endpoint["api_key"]}
        if endpoint["base_url"]:
            client_kwargs["http_options"] = {
                "api_version": "v1beta",
                "base_url": endpoint["base_url"],
            }
        client = genai.Client(**client_kwargs)
        for model in models:
            try:
                candidate = client.models.generate_content(
                    model=model,
                    contents=contents,
                )
                if not (candidate.text or "").strip():
                    raise RuntimeError("Gemini 返回空文本")
                response = candidate
                used_model = model
                used_endpoint = endpoint["name"]
                break
            except Exception as exc:
                last_error = exc
                safe_error = _redact_error(exc, endpoints)
                print(
                    f"[Gemini] 通道 {endpoint['name']} / 模型 {model} 失败: "
                    f"{safe_error}",
                    file=sys.stderr,
                )
        if response is not None:
            break
        if endpoint_index + 1 < len(endpoints):
            print(
                f"[Gemini] 切换备用通道: {endpoints[endpoint_index + 1]['name']}",
                file=sys.stderr,
            )
    if response is None:
        safe_error = _redact_error(last_error, endpoints) if last_error else "未知错误"
        raise RuntimeError(f"所有 Gemini 通道均失败: {safe_error}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    date = dt.date.today().isoformat()
    out_path = out_dir / f"{date}_{meta['video_id']}_youtube_gemini.txt"
    text = (
        f"Title: {meta['title']}\n"
        f"Channel: {meta['channel']}\n"
        f"URL: {meta['url']}\n"
        f"Model: {used_model}\n"
        f"Endpoint: {used_endpoint}\n\n"
        f"{response.text}"
    )
    out_path.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n[Saved] {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
