"""FastAPI app factory for Baseline.

Run with:

    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Ensure project root is on sys.path so `main` module (TennisAnalysisPipeline) imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
        app.mount("/", StaticFiles(directory=str(static_root), html=True), name="frontend")
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
