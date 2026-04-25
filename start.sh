#!/usr/bin/env bash
# Baseline server one-click launcher.
# Usage: ./start.sh
set -e

cd "$(dirname "$0")"

PORT=8765

# Kill any zombie server
pkill -f "uvicorn app.main:app" 2>/dev/null || true
sleep 0.5

# Make sure deps are installed
python3 -c "import fastapi, uvicorn, multipart" 2>/dev/null || {
  echo "→ Installing fastapi / uvicorn / python-multipart"
  pip3 install -q fastapi 'uvicorn[standard]' python-multipart
}

# Boot server
echo ""
echo "  Baseline starting on http://localhost:$PORT"
echo ""
echo "  ┌────────────────────────────────────────────────────────┐"
echo "  │  http://localhost:$PORT/             functional app      │"
echo "  │  http://localhost:$PORT/design.html  design canvas       │"
echo "  │  http://localhost:$PORT/docs         API documentation   │"
echo "  └────────────────────────────────────────────────────────┘"
echo ""
echo "  Press Ctrl-C to stop."
echo ""

exec python3 -m uvicorn app.main:app --host 0.0.0.0 --port $PORT --log-level info
