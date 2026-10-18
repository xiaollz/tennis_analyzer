#!/usr/bin/env python3
"""usage: gem.py <mode yt|inline> <id> <video_or_url> <out_prefix>; primary key only (MICU_GEMINI_KEY); key never printed."""
import base64, json, os, sys, time, urllib.request, urllib.error
mode, vid, src, outp = sys.argv[1:5]
KEY = os.environ["MICU_GEMINI_KEY"]
MODEL = "gemini-3.8-flash"
BASE = "https://www.micuapi.ai"
prompt = open("/workspace/ftt_pilot/prompt_v1.txt", encoding="utf-8").read()
if mode == "yt":
    vpart = {"fileData": {"fileUri": src, "mimeType": "video/mp4"}}
else:
    vpart = {"inline_data": {"mime_type": "video/mp4", "data": base64.b64encode(open(src, "rb").read()).decode()}}
body = {"contents": [{"parts": [{"text": prompt}, vpart]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 32000}}
req = urllib.request.Request(f"{BASE}/v1beta/models/{MODEL}:generateContent", data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json", "x-goog-api-key": KEY, "Authorization": f"Bearer {KEY}", "User-Agent": "Mozilla/5.0"})
t0 = time.time(); status = "ok"; text = ""; usage = None
try:
    with urllib.request.urlopen(req, timeout=900) as r:
        payload = json.loads(r.read().decode())
    parts = payload["candidates"][0]["content"]["parts"]
    text = "".join(p.get("text", "") for p in parts)
    usage = payload.get("usageMetadata")
except urllib.error.HTTPError as e:
    status = f"HTTP {e.code}: " + e.read().decode("utf-8", "replace")[:500]
except Exception as e:
    status = f"{type(e).__name__}: {e}"
dt = time.time() - t0
log = {"id": vid, "mode": mode, "model": MODEL, "prompt_version": "v1", "status": status, "elapsed_s": round(dt, 1),
       "usage": usage, "ts": time.strftime("%Y-%m-%d %H:%M:%S %z")}
open(outp + ".log.json", "w").write(json.dumps(log, ensure_ascii=False, indent=1))
if text: open(outp + ".md", "w", encoding="utf-8").write(text)
print(json.dumps(log, ensure_ascii=False))
