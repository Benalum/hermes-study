#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -x .venv/bin/hermes-study ]]; then
  echo "Hermes Study is not installed yet. Run: bash scripts/install_mac.sh" >&2
  exit 1
fi

set -a
[[ -f .env ]] && source .env
set +a

PORT="${HERMES_STUDY_PORT:-8787}"
HOST="${HERMES_STUDY_HOST:-127.0.0.1}"
HERMES_BASE="${HERMES_AGENT_BASE_URL:-http://127.0.0.1:8642/v1}"
HERMES_HEALTH="${HERMES_BASE%/v1}/health"

if command -v curl >/dev/null 2>&1; then
  if curl -fsS --max-time 3 "$HERMES_HEALTH" >/dev/null 2>&1; then
    echo "Hermes Agent API: online (${HERMES_BASE%/v1})"
  else
    echo "Warning: Hermes Agent API is not answering at ${HERMES_BASE%/v1}."
    echo "Hermes Study will still start, but tutoring requests need Hermes online."
  fi
fi

TAILSCALE_BIN=""
if command -v tailscale >/dev/null 2>&1; then
  TAILSCALE_BIN="$(command -v tailscale)"
elif [[ -x /Applications/Tailscale.app/Contents/MacOS/Tailscale ]]; then
  TAILSCALE_BIN="/Applications/Tailscale.app/Contents/MacOS/Tailscale"
fi

if [[ "${HERMES_STUDY_TAILSCALE_SERVE:-1}" == "1" && -n "$TAILSCALE_BIN" ]]; then
  "$TAILSCALE_BIN" serve --bg "localhost:${PORT}" >/dev/null 2>&1 || true
  echo "Tailscale Serve status:"
  "$TAILSCALE_BIN" serve status || true
fi

echo "Hermes Study local UI: http://${HOST}:${PORT}"
echo "Press Ctrl-C to stop Hermes Study."
exec .venv/bin/hermes-study --host "$HOST" --port "$PORT"
