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
        # No automatic redirect — auto-redirects can leave the user on a
        # blank screen if the next page hits ngrok's interstitial or a
        # network blip. Show explicit status + a big manual link instead.
        html = """<!doctype html>
<html><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Reset Baseline</title>
<style>
  *{box-sizing:border-box}
  body{font-family:-apple-system,system-ui,sans-serif;background:#F7F3EC;color:#2A2925;
       margin:0;padding:32px 20px;line-height:1.45;-webkit-font-smoothing:antialiased}
  .wrap{max-width:420px;margin:0 auto}
  h1{font-size:22px;font-weight:600;margin:0 0 18px;letter-spacing:-0.2px}
  .step{padding:10px 0;border-bottom:0.5px solid rgba(0,0,0,0.08);font-size:14px;
        display:flex;justify-content:space-between;align-items:center}
  .label{color:#2A2925}
  .state{font-family:ui-monospace,Menlo,monospace;font-size:11px;color:#6F6B66}
  .state.ok{color:#3F6E3F;font-weight:600}
  .state.err{color:#A8432F;font-weight:600}
  .cta{display:block;margin-top:24px;padding:16px;background:#A8432F;color:#fff;
       text-align:center;text-decoration:none;border-radius:12px;font-weight:600;
       font-size:15px;letter-spacing:0.2px}
  .cta:active{opacity:0.85}
  .hint{font-size:12px;color:#6F6B66;text-align:center;margin-top:14px}
</style>
</head><body>
<div class="wrap">
  <h1>Reset Baseline</h1>
  <div class="step"><span class="label">Service workers</span><span class="state" id="sw-state">checking…</span></div>
  <div class="step"><span class="label">Caches</span><span class="state" id="cache-state">checking…</span></div>
  <div class="step"><span class="label">Server build</span><span class="state" id="ver-state">checking…</span></div>
  <a class="cta" id="continue" href="/">Open Baseline</a>
  <p class="hint">After tap → Home 屏底部应显示当前版本号</p>
</div>
<script>
(async () => {
  const set = (id, txt, cls) => {
    const el = document.getElementById(id);
    el.textContent = txt;
    el.className = 'state' + (cls ? ' ' + cls : '');
  };
  // Build a unique URL for the continue button so it always bypasses
  // the browser HTTP cache for '/'.
  const cta = document.getElementById('continue');
  cta.href = '/?fresh=' + Date.now();

  // Step 1: unregister service workers
  try {
    if ('serviceWorker' in navigator) {
      const regs = await navigator.serviceWorker.getRegistrations();
      await Promise.all(regs.map(r => r.unregister()));
      set('sw-state', regs.length + ' removed', 'ok');
    } else {
      set('sw-state', 'unsupported', 'ok');
    }
  } catch (e) { set('sw-state', 'error: ' + e.message, 'err'); }

  // Step 2: delete caches
  try {
    if ('caches' in window) {
      const keys = await caches.keys();
      await Promise.all(keys.map(k => caches.delete(k)));
      set('cache-state', keys.length + ' deleted', 'ok');
    } else {
      set('cache-state', 'unsupported', 'ok');
    }
  } catch (e) { set('cache-state', 'error: ' + e.message, 'err'); }

  // Step 3: ask server what version it's serving
  try {
    const r = await fetch('/version', {
      headers: { 'ngrok-skip-browser-warning': 'true' },
      cache: 'no-store',
    });
    const j = await r.json();
    set('ver-state', j.build_tag || 'unknown', 'ok');
  } catch (e) { set('ver-state', 'error: ' + e.message, 'err'); }
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
