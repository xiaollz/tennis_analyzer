#!/usr/bin/env bash
# Baseline — public-access launcher.
#
# Boots the FastAPI server AND a public tunnel so you can open the App
# from your phone over any network (球场 / 移动数据 / 别人家 WiFi).
#
# Two tunnel modes (auto-detected):
#
#   1. ngrok with reserved domain (RECOMMENDED for daily use)
#      Requires `.tunnel.config` with NGROK_AUTHTOKEN + NGROK_DOMAIN.
#      Stable URL — install once on your phone, works forever.
#      Setup: copy .tunnel.config.example → .tunnel.config, fill values.
#
#   2. Cloudflare quick tunnel (FALLBACK, no setup)
#      Random *.trycloudflare.com URL, changes every restart.
#
# Usage:  ./start-public.sh
#
# Stop both processes with Ctrl-C.
set -e

cd "$(dirname "$0")"
PORT=${PORT:-8765}

# ── Load tunnel config if present ──────────────────────────────────

NGROK_AUTHTOKEN=""
NGROK_DOMAIN=""
if [ -f ".tunnel.config" ]; then
  # shellcheck disable=SC1091
  . ./.tunnel.config
fi

# ── Pick tunnel mode ───────────────────────────────────────────────

TUNNEL_MODE=""
if [ -n "$NGROK_AUTHTOKEN" ] && [ -n "$NGROK_DOMAIN" ]; then
  if command -v ngrok >/dev/null 2>&1; then
    TUNNEL_MODE="ngrok"
  else
    echo "✗ ngrok not installed but .tunnel.config has ngrok values."
    echo "  Install:  brew install --cask ngrok"
    echo "  or:       curl -sSL https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-darwin-arm64.zip -o /tmp/ngrok.zip && unzip /tmp/ngrok.zip -d /opt/homebrew/bin/"
    exit 1
  fi
elif command -v cloudflared >/dev/null 2>&1; then
  TUNNEL_MODE="cloudflared"
else
  echo "✗ No tunnel tool installed."
  echo
  echo "  Recommended:"
  echo "    1. brew install --cask ngrok"
  echo "    2. cp .tunnel.config.example .tunnel.config"
  echo "    3. Fill NGROK_AUTHTOKEN + NGROK_DOMAIN (see file for links)"
  echo
  echo "  Or:  brew install cloudflared  (random URL each run)"
  exit 1
fi

# ── Sanity: Python deps ────────────────────────────────────────────

python3 -c "import fastapi, uvicorn, multipart" 2>/dev/null || {
  echo "→ Installing fastapi / uvicorn / python-multipart"
  pip3 install -q fastapi 'uvicorn[standard]' python-multipart
}

# ── Kill any zombies ───────────────────────────────────────────────

pkill -f "uvicorn app.main:app" 2>/dev/null || true
pkill -f "cloudflared tunnel" 2>/dev/null || true
pkill -f "ngrok " 2>/dev/null || true
sleep 0.5

# ── Boot uvicorn ───────────────────────────────────────────────────

echo
echo "  Booting Baseline server on http://127.0.0.1:$PORT ..."
python3 -m uvicorn app.main:app --host 127.0.0.1 --port $PORT --log-level warning &
SERVER_PID=$!

for i in $(seq 1 20); do
  if curl -sSf "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1; then
    echo "  ✓ server up"
    break
  fi
  sleep 0.5
done

if ! curl -sSf "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1; then
  echo "  ✗ server failed to start"
  kill "$SERVER_PID" 2>/dev/null || true
  exit 1
fi

# ── Boot tunnel ────────────────────────────────────────────────────

trap "echo; echo '  Stopping...'; kill \$SERVER_PID \$TUNNEL_PID 2>/dev/null || true; exit 0" INT TERM

if [ "$TUNNEL_MODE" = "ngrok" ]; then
  echo
  echo "  Starting ngrok tunnel → https://$NGROK_DOMAIN ..."
  echo

  # Configure authtoken (idempotent — overwrites if existing)
  ngrok config add-authtoken "$NGROK_AUTHTOKEN" >/dev/null 2>&1 || true

  # Start tunnel pinned to your reserved domain
  ngrok http --url="$NGROK_DOMAIN" --log=stdout "$PORT" 2>&1 | tee /tmp/baseline-ngrok.log &
  TUNNEL_PID=$!

  # Wait for "started tunnel" event then print URL
  (
    for i in $(seq 1 30); do
      if grep -q "started tunnel" /tmp/baseline-ngrok.log 2>/dev/null \
        || grep -q "Forwarding" /tmp/baseline-ngrok.log 2>/dev/null \
        || grep -q "url=https://$NGROK_DOMAIN" /tmp/baseline-ngrok.log 2>/dev/null; then
        URL="https://$NGROK_DOMAIN"
        echo
        echo "  ┌──────────────────────────────────────────────────────────────┐"
        printf "  │  %-60s│\n" "Public URL (永久):"
        printf "  │  %-60s│\n" "$URL"
        echo "  ├──────────────────────────────────────────────────────────────┤"
        printf "  │  %-60s│\n" "On your phone (first time):"
        printf "  │  %-60s│\n" "  1. Open the URL in Safari"
        printf "  │  %-60s│\n" "  2. Tap Share → Add to Home Screen"
        printf "  │  %-60s│\n" ""
        printf "  │  %-60s│\n" "After that the URL never changes —"
        printf "  │  %-60s│\n" "just tap the home-screen icon."
        echo "  └──────────────────────────────────────────────────────────────┘"
        echo
        break
      fi
      sleep 1
    done
  ) &

else
  echo
  echo "  Starting Cloudflare quick tunnel (random URL — install ngrok"
  echo "  to get a stable one). Wait for the public URL ..."
  echo

  cloudflared tunnel --url "http://127.0.0.1:$PORT" --no-autoupdate 2>&1 | tee /tmp/baseline-tunnel.log &
  TUNNEL_PID=$!

  (
    for i in $(seq 1 30); do
      URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' /tmp/baseline-tunnel.log 2>/dev/null | head -1)
      if [ -n "$URL" ]; then
        echo
        echo "  ┌──────────────────────────────────────────────────────────────┐"
        printf "  │  %-60s│\n" "Public URL (random, this session only):"
        printf "  │  %-60s│\n" "$URL"
        echo "  ├──────────────────────────────────────────────────────────────┤"
        printf "  │  %-60s│\n" "Heads up: this URL changes each time you restart."
        printf "  │  %-60s│\n" "For a permanent URL, set up ngrok in .tunnel.config."
        echo "  └──────────────────────────────────────────────────────────────┘"
        echo
        break
      fi
      sleep 1
    done
  ) &
fi

wait "$TUNNEL_PID"
kill "$SERVER_PID" 2>/dev/null || true
