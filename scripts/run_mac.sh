#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -d .venv ]] || { echo "Run ./scripts/install_mac.sh first" >&2; exit 1; }
set -a
[[ -f .env ]] && source .env
set +a
. .venv/bin/activate

PORT="${HERMES_STUDY_PORT:-8787}"
HOST="${HERMES_STUDY_HOST:-127.0.0.1}"

if [[ "${HERMES_STUDY_TAILSCALE_SERVE:-1}" == "1" ]] && command -v tailscale >/dev/null 2>&1; then
  tailscale serve --bg "localhost:${PORT}" >/dev/null || true
  echo "Tailscale Serve status:"
  tailscale serve status || true
fi

echo "Hermes Study local UI: http://${HOST}:${PORT}"
exec hermes-study --host "$HOST" --port "$PORT"
