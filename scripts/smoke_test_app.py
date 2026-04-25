#!/usr/bin/env python3
"""End-to-end smoke test for the Baseline app stack.

Boots the FastAPI server (or uses a running one), uploads the canonical
test video, polls segmentation, runs diagnose with stroke=forehand override,
verifies the expected behaviour against the three bugs we fixed:

  Bug 1: Single-hit short video must produce exactly 1 clip
  Bug 2: User-supplied stroke must be honoured (not auto-classified)
  Bug 3: Forehand path must produce a real diagnosis (root_cause + narrative)

Usage:
    # against a server that's already running
    python3 scripts/smoke_test_app.py

    # boot a fresh server, run the test, shut it down
    python3 scripts/smoke_test_app.py --boot
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_VIDEO = PROJECT_ROOT / "videos" / "test_user_clip.mp4"
HOST = "127.0.0.1"
PORT = 8765
BASE = f"http://{HOST}:{PORT}/api"


def http_get(path: str) -> dict:
    with urllib.request.urlopen(BASE + path, timeout=10) as r:
        return json.loads(r.read())


def http_post_form(path: str, fields: dict, files: dict | None = None) -> dict:
    """multipart POST. files dict is {name: (filename, bytes)}."""
    boundary = f"----baseline-smoke-{int(time.time() * 1000)}"
    parts = []
    for name, value in fields.items():
        parts.append(f"--{boundary}\r\n")
        parts.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n')
        parts.append(f"{value}\r\n")
    payload_parts: list[bytes] = []
    for s in parts:
        payload_parts.append(s.encode())
    for name, (fn, content) in (files or {}).items():
        payload_parts.append(f'--{boundary}\r\n'.encode())
        payload_parts.append(
            f'Content-Disposition: form-data; name="{name}"; filename="{fn}"\r\n'.encode()
        )
        payload_parts.append(b"Content-Type: video/mp4\r\n\r\n")
        payload_parts.append(content)
        payload_parts.append(b"\r\n")
    payload_parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(payload_parts)
    req = urllib.request.Request(
        BASE + path,
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def http_post(path: str) -> dict:
    req = urllib.request.Request(BASE + path, method="POST")
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def wait_health(max_s: int = 30) -> bool:
    for _ in range(max_s):
        try:
            http_get("/health")
            return True
        except (urllib.error.URLError, ConnectionRefusedError):
            time.sleep(1)
    return False


def poll_job(job_id: str, max_s: int = 240, label: str = "job") -> dict:
    start = time.time()
    last_msg = ""
    while time.time() - start < max_s:
        j = http_get(f"/jobs/{job_id}")
        msg = j.get("message", "")
        if msg != last_msg:
            print(f"  [{label} {j.get('progress', 0) * 100:.0f}%] {msg}")
            last_msg = msg
        if j["status"] in ("done", "error"):
            return j
        time.sleep(2)
    raise TimeoutError(f"{label} timed out after {max_s}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", action="store_true", help="boot a fresh server for the test")
    args = ap.parse_args()

    if not TEST_VIDEO.exists():
        print(f"FAIL: test video not found at {TEST_VIDEO}")
        return 1

    server_proc = None
    if args.boot:
        print(f"→ booting server on {HOST}:{PORT}")
        server_proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.main:app",
             "--host", HOST, "--port", str(PORT), "--log-level", "warning"],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    try:
        if not wait_health(30):
            print("FAIL: server did not become healthy")
            return 1
        print("✓ server healthy")

        # ── Upload + segment ──────────────────────────────────────
        print(f"→ uploading {TEST_VIDEO.name} with stroke=forehand")
        with open(TEST_VIDEO, "rb") as f:
            content = f.read()
        upload = http_post_form("/videos", {"stroke": "forehand"},
                                {"file": (TEST_VIDEO.name, content)})
        video_id = upload["video_id"]
        seg_job = upload["job_id"]
        assert upload.get("stroke") == "forehand", f"stroke not echoed: {upload}"
        print(f"  video_id={video_id}  segmentation job={seg_job}")

        seg = poll_job(seg_job, label="segment")
        if seg["status"] != "done":
            print(f"FAIL: segmentation: {seg.get('error', seg)}")
            return 2

        meta = http_get(f"/videos/{video_id}")
        clips = meta["clips"]
        print(f"✓ segmentation: {len(clips)} clip(s), {meta['total_onsets']} onset(s)")

        # ── Bug 1 assertion ──────────────────────────────────────
        if len(clips) != 1:
            print(f"FAIL Bug#1: expected 1 clip for the single-hit test video, got {len(clips)}")
            print(f"  clips: {[c['clip_id'] for c in clips]}")
            return 3
        clip = clips[0]
        if len(clip["impact_times_s"]) != 1:
            print(f"FAIL Bug#1: clip should have 1 impact, got {clip['impact_times_s']}")
            return 3
        print(f"✓ Bug#1 fixed: 1 clip / 1 impact at t={clip['impact_times_s'][0]:.2f}s")

        # ── Diagnose with stroke=forehand ────────────────────────
        clip_id = clip["clip_id"]
        print(f"→ starting diagnosis with stroke=forehand on {clip_id}")
        diag_resp = http_post(f"/clips/{clip_id}/diagnose")
        if diag_resp["status"] == "already_done":
            print("  diagnosis already cached")
        else:
            assert diag_resp.get("stroke") == "forehand", f"stroke not echoed: {diag_resp}"
            diag_job = diag_resp["job_id"]
            done = poll_job(diag_job, max_s=240, label="diagnose")
            if done["status"] != "done":
                print(f"FAIL: diagnosis: {done.get('error', done)}")
                return 4

        # ── Bug 2 + 3 assertions ──────────────────────────────────
        result = http_get(f"/diagnoses/{clip_id}")["result"]
        if result.get("stroke_type") != "forehand":
            print(f"FAIL Bug#2: stroke_type was {result.get('stroke_type')}, expected forehand")
            return 5
        print(f"✓ Bug#2 fixed: stroke_type=forehand respected (score={result.get('overall_score'):.1f})")

        diag = result.get("diagnosis") or {}
        has_root = bool(diag.get("root_cause"))
        has_narr = bool(diag.get("narrative"))
        if not (has_root and has_narr):
            print(f"FAIL Bug#3: diagnosis incomplete (root_cause={has_root}, narrative={has_narr})")
            return 6
        print(f"✓ Bug#3 fixed: root_cause + narrative produced")

        # ── Artifacts sanity check ───────────────────────────────
        art = result.get("artifacts") or {}
        for key in ("annotated_video", "thumbnail", "clip_video"):
            if not art.get(key):
                print(f"WARN: missing artifact {key}")

        print()
        print("ALL THREE BUGS FIXED ✓")
        print(f"  video_id   = {video_id}")
        print(f"  clip_id    = {clip_id}")
        print(f"  stroke     = {result.get('stroke_type')}")
        print(f"  score      = {result.get('overall_score'):.1f}")
        print(f"  swings     = {len(result.get('swings', []))}")
        print(f"  vlm obs    = {len((result.get('vlm') or {}).get('observations', []))}")
        print(f"  root cause = {bool((result.get('diagnosis') or {}).get('root_cause'))}")
        return 0

    finally:
        if server_proc:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_proc.kill()


if __name__ == "__main__":
    sys.exit(main())
