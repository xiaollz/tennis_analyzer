#!/usr/bin/env bash
# Baseline — public-access launcher.
#
# Boots the FastAPI server AND a Cloudflare quick tunnel so you can open
# the App from your phone over any network (球场 / 移动数据 / 别人家 WiFi).
#
# Usage:  ./start-public.sh
#
# Prints a https://*.trycloudflare.com URL — open that in mobile Safari,
# then "Share → Add to Home Screen". Done. App icon on your home screen.
#
# Stop both processes with Ctrl-C.
set -e

cd "$(dirname "$0")"
PORT=${PORT:-8765}

# ── Sanity checks ───────────────────────────────────────────────────

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "✗ cloudflared not installed."
  echo
  echo "  Install with:  brew install cloudflared"
  exit 1
fi

python3 -c "import fastapi, uvicorn, multipart" 2>/dev/null || {
  echo "→ Installing fastapi / uvicorn / python-multipart"
  pip3 install -q fastapi 'uvicorn[standard]' python-multipart
}

# ── Kill any zombies ────────────────────────────────────────────────

pkill -f "uvicorn app.main:app" 2>/dev/null || true
pkill -f "cloudflared tunnel" 2>/dev/null || true
sleep 0.5

# ── Boot uvicorn + tunnel ───────────────────────────────────────────

echo
echo "  Booting Baseline server on http://127.0.0.1:$PORT ..."
python3 -m uvicorn app.main:app --host 127.0.0.1 --port $PORT --log-level warning &
SERVER_PID=$!

# Wait for server
for i in $(seq 1 20); do
  if curl -sSf "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1; then
    echo "  ✓ server up"
    break
  fi
  sleep 0.5
done

if ! curl -sSf "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1; then
  echo "  ✗ server failed to start"
  kill $SERVER_PID 2>/dev/null || true
  exit 1
fi

echo
echo "  Opening Cloudflare tunnel — wait for the public URL ..."
echo

# cloudflared prints the URL to stderr; pipe both, grep the URL out.
cloudflared tunnel --url "http://127.0.0.1:$PORT" 2>&1 | tee /tmp/baseline-tunnel.log &
TUNNEL_PID=$!

# Print the URL once we see it (in a sub-shell)
(
  for i in $(seq 1 30); do
    URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' /tmp/baseline-tunnel.log 2>/dev/null | head -1)
    if [ -n "$URL" ]; then
      echo
      echo "  ┌──────────────────────────────────────────────────────────────┐"
      printf "  │  %-60s│\n" "Public URL:"
      printf "  │  %-60s│\n" "$URL"
      echo "  ├──────────────────────────────────────────────────────────────┤"
      printf "  │  %-60s│\n" "On your phone:"
      printf "  │  %-60s│\n" "  1. Open the URL in Safari"
      printf "  │  %-60s│\n" "  2. Tap Share → Add to Home Screen"
      printf "  │  %-60s│\n" "  3. Open Baseline from your home screen"
      echo "  └──────────────────────────────────────────────────────────────┘"
      echo
      break
    fi
    sleep 1
  done
) &

trap "echo; echo '  Stopping...'; kill $SERVER_PID $TUNNEL_PID 2>/dev/null || true; exit 0" INT TERM

# Wait for either process to exit
wait $TUNNEL_PID
kill $SERVER_PID 2>/dev/null || true
