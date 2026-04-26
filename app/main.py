"""FastAPI app factory for Baseline.

Run with:

    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import mimetypes

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


class NoCacheStaticFiles(StaticFiles):
    """StaticFiles that adds proper Cache-Control headers.

    The default Starlette StaticFiles sends only Etag + Last-Modified, with
    no Cache-Control. iOS Safari (and PWAs) interpret missing Cache-Control
    as a license to use heuristic caching — typically (now - Last-Modified)
    × 10% — which means index.html and sw.js can stick in the local HTTP
    cache for hours or days. The result: even a hard refresh can serve
    stale code, and the service worker itself never gets a byte-different
    fetch so it never updates.

    This subclass forces:
      • index.html / sw.js → no-store (every page load goes to network)
      • everything else      → public, immutable for 1 day (icons, manifest)
    """

    # Starlette normalises '/' → '.' (via os.path.normpath), so we have to
    # match by content-type instead of relying on a path whitelist.
    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        norm = path.lstrip("/")
        ctype = resp.headers.get("content-type", "")
        is_html = "text/html" in ctype
        is_sw = norm.endswith("sw.js")
        if is_html or is_sw:
            # Belt + suspenders: every directive that tells iOS Safari /
            # service workers / browser cache to skip the local copy.
            resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            resp.headers["Pragma"] = "no-cache"
            resp.headers["Expires"] = "0"
        else:
            # Static assets: cache for a day, but allow re-validation.
            resp.headers.setdefault("Cache-Control", "public, max-age=86400")
        return resp

# Ensure project root is on sys.path so `main` module (TennisAnalysisPipeline) imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Make sure manifest + sw.js get correct MIME types (some setups default to
# application/octet-stream which breaks PWA installability)
mimetypes.add_type("application/manifest+json", ".webmanifest")
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/javascript", ".mjs")

from app import routes, storage


def create_app() -> FastAPI:
    app = FastAPI(
        title="Baseline — Tennis Analyzer API",
        description=(
            "Audio-based video segmentation + per-clip pose / VLM / diagnosis. "
            "Pairs with the Baseline iOS app designed in Claude Desktop."
        ),
        version="1.0.0",
    )

    # CORS: local frontend dev server runs on a different port.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=False,
    )

    storage.ensure_storage()
    app.include_router(routes.router)

    # /reset — kill switch for stale PWA installs. iOS Safari + PWA is
    # notorious for pinning a stale shell when the original response had
    # no Cache-Control (the previous bug). Even after fixing the headers,
    # already-cached entries can persist for hours. Visiting this route
    # serves a tiny HTML page that:
    #   1. unregisters every service worker on this origin
    #   2. deletes every Cache API store
    #   3. redirects to / with a fresh timestamp query string
    # The redirect URL is unique each time, so the browser's HTTP cache
    # treats it as a new resource and goes to the network.
    @app.get("/reset", response_class=HTMLResponse)
    def reset_pwa() -> HTMLResponse:
        html = """<!doctype html>
<html><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Resetting Baseline…</title>
<style>
  body{font-family:-apple-system,system-ui,sans-serif;background:#F7F3EC;color:#2A2925;
       margin:0;padding:48px 24px;text-align:center;line-height:1.5}
  h1{font-size:18px;font-weight:600;margin:0 0 12px}
  p{font-size:14px;color:#6F6B66;margin:0 0 8px}
  code{background:#EAE4D9;padding:2px 6px;border-radius:4px;font-size:12px}
  .ok{color:#3F6E3F}
</style>
</head><body>
<h1>Resetting Baseline…</h1>
<p id="msg">Clearing service worker and caches…</p>
<p><code id="log"></code></p>
<script>
(async () => {
  const log = (m) => { document.getElementById('log').textContent = m; };
  try {
    if ('serviceWorker' in navigator) {
      const regs = await navigator.serviceWorker.getRegistrations();
      log('found ' + regs.length + ' SW; unregistering…');
      await Promise.all(regs.map(r => r.unregister()));
    }
    if ('caches' in window) {
      const keys = await caches.keys();
      log('found ' + keys.length + ' caches; deleting…');
      await Promise.all(keys.map(k => caches.delete(k)));
    }
    document.getElementById('msg').innerHTML =
      '<span class="ok">Reset complete.</span> Loading fresh app in 1s…';
    setTimeout(() => { location.replace('/?fresh=' + Date.now()); }, 1000);
  } catch (e) {
    document.getElementById('msg').textContent = 'Reset error: ' + e.message;
  }
})();
</script>
</body></html>"""
        return HTMLResponse(
            content=html,
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    # /version — quick verifier for what the SERVER is currently shipping.
    # Reads BUILD_TAG out of frontend/dist/index.html so we can prove the
    # server is on the latest code without depending on the browser cache.
    @app.get("/version")
    def version() -> JSONResponse:
        index_path = PROJECT_ROOT / "frontend" / "dist" / "index.html"
        tag = "unknown"
        try:
            text = index_path.read_text(encoding="utf-8", errors="replace")
            import re
            m = re.search(r"BUILD_TAG\s*=\s*'([^']+)'", text)
            if m:
                tag = m.group(1)
        except Exception:
            pass
        return JSONResponse(
            {"build_tag": tag},
            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
        )

    # Optional static frontend — if a `frontend/dist/` exists, serve it at /
    static_root = PROJECT_ROOT / "frontend" / "dist"
    if static_root.exists():
        app.mount("/", NoCacheStaticFiles(directory=str(static_root), html=True), name="frontend")
    else:
        @app.get("/")
        def index():
            return {
                "service": "baseline",
                "message": "Frontend not built yet. API is live at /api/*.",
                "docs": "/docs",
            }

    return app


app = create_app()
