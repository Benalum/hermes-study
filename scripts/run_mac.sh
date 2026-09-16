#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
[[ -x .venv/bin/python ]] || { echo "Run: bash scripts/install_mac.sh" >&2; exit 1; }
set -a
[[ -f .env ]] && source .env
set +a
. .venv/bin/activate

find_hermes() {
  local c
  for c in \
    "${HERMES_BIN:-}" \
    "$(command -v hermes 2>/dev/null || true)" \
    "$HOME/.local/bin/hermes" \
    "$HOME/.hermes/hermes-agent/.venv/bin/hermes"; do
    [[ -n "$c" && -x "$c" ]] || continue
    printf '%s\n' "$c"
    return 0
  done
  return 1
}

HERMES_CLI="$(find_hermes || true)"
if ! curl -fsS http://127.0.0.1:8642/health >/dev/null 2>&1; then
  if [[ -n "$HERMES_CLI" ]]; then
    echo "Starting Hermes API gateway..."
    "$HERMES_CLI" gateway start >/dev/null 2>&1 || "$HERMES_CLI" gateway restart >/dev/null 2>&1 || true
  fi
  for _ in {1..20}; do
    curl -fsS http://127.0.0.1:8642/health >/dev/null 2>&1 && break
    sleep 0.5
  done
fi

if ! curl -fsS http://127.0.0.1:8642/health >/dev/null 2>&1; then
  echo "Hermes API is not reachable on 127.0.0.1:8642." >&2
  echo "Run 'hermes gateway status' and 'hermes gateway restart', then try again." >&2
  exit 1
fi

PORT="${HERMES_STUDY_PORT:-8787}"
HOST="${HERMES_STUDY_HOST:-127.0.0.1}"

if [[ "${HERMES_STUDY_TAILSCALE_SERVE:-1}" == "1" ]] && command -v tailscale >/dev/null 2>&1; then
  tailscale serve --bg "localhost:${PORT}" >/dev/null 2>&1 || true
  echo "Tailscale Serve status:"
  tailscale serve status || true
fi

echo "Hermes Study local UI: http://${HOST}:${PORT}"
echo "Hermes Agent API: http://127.0.0.1:8642"
exec hermes-study --host "$HOST" --port "$PORT"
