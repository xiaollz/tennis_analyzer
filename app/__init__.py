"""Baseline backend — FastAPI app wrapping the tennis analysis pipeline.

Layers:
- storage.py : filesystem layout + paths + persistence helpers
- jobs.py    : in-process background job queue
- services.py: thin wrappers around segmenter + TennisAnalysisPipeline
- routes.py  : HTTP endpoints
- main.py    : FastAPI app factory
"""
