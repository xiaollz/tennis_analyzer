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
