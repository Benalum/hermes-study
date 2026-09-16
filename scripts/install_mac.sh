#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

version_ok() {
  "$1" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 11) else 1)
PY
}

find_python() {
  local c
  for c in \
    "$(command -v python3.12 2>/dev/null || true)" \
    "$(command -v python3.13 2>/dev/null || true)" \
    "$(command -v python3.11 2>/dev/null || true)" \
    "$HOME/.hermes/hermes-agent/.venv/bin/python" \
    "$HOME/.hermes/hermes-agent/.venv/bin/python3" \
    "$(command -v python3 2>/dev/null || true)"; do
    [[ -n "$c" && -x "$c" ]] || continue
    if version_ok "$c"; then
      printf '%s\n' "$c"
      return 0
    fi
  done
  return 1
}

find_hermes() {
  local c
  for c in \
    "$(command -v hermes 2>/dev/null || true)" \
    "$HOME/.local/bin/hermes" \
    "$HOME/.hermes/hermes-agent/.venv/bin/hermes"; do
    [[ -n "$c" && -x "$c" ]] || continue
    printf '%s\n' "$c"
    return 0
  done
  return 1
}

set_env_value() {
  local file="$1" key="$2" value="$3" tmp
  mkdir -p "$(dirname "$file")"
  touch "$file"
  tmp="$(mktemp)"
  awk -v k="$key" -v v="$value" '
    BEGIN { done=0 }
    index($0, k "=")==1 { if (!done) { print k "=" v; done=1 } next }
    { print }
    END { if (!done) print k "=" v }
  ' "$file" > "$tmp"
  mv "$tmp" "$file"
}

read_env_value() {
  local file="$1" key="$2"
  [[ -f "$file" ]] || return 1
  awk -v k="$key" '
    index($0, k "=")==1 {
      sub("^[^=]*=", "");
      gsub(/^[\x27\"]|[\x27\"]$/, "");
      print;
      exit
    }
  ' "$file"
}

PYTHON_BIN="$(find_python || true)"
if [[ -z "$PYTHON_BIN" ]]; then
  if command -v brew >/dev/null 2>&1; then
    echo "Python 3.11+ was not found. Installing Homebrew Python 3.12..."
    brew install python@3.12
    BREW_PY="$(brew --prefix python@3.12)/bin/python3.12"
    if [[ -x "$BREW_PY" ]] && version_ok "$BREW_PY"; then
      PYTHON_BIN="$BREW_PY"
    fi
  fi
fi
if [[ -z "$PYTHON_BIN" ]]; then
  echo "Could not find or install Python 3.11+. Install Homebrew, then run: brew install python@3.12" >&2
  exit 1
fi
PYVER="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')"
echo "Using Python $PYVER at $PYTHON_BIN"

HERMES_BIN="$(find_hermes || true)"
if [[ -z "$HERMES_BIN" ]]; then
  echo "Hermes Agent was not found in PATH or ~/.hermes. Hermes Study expects your existing Hermes installation." >&2
  echo "Try: command -v hermes && hermes --version" >&2
  exit 1
fi
echo "Using Hermes at $HERMES_BIN"

if command -v brew >/dev/null 2>&1 && ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Installing ffmpeg for browser microphone audio..."
  brew install ffmpeg
fi

rm -rf .venv
"$PYTHON_BIN" -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[voice]'

if [[ "${INSTALL_SPEAKER_GATE:-0}" == "1" ]]; then
  echo "Installing optional SpeechBrain speaker verification..."
  python -m pip install -e '.[speaker]'
fi

mkdir -p data data/logs
[[ -f .env ]] || cp .env.example .env

# Hermes Agent exposes its fully tooled agent through an authenticated localhost API.
HERMES_HOME_DIR="${HERMES_HOME:-$HOME/.hermes}"
HERMES_ENV="$HERMES_HOME_DIR/.env"
API_KEY="$(read_env_value "$HERMES_ENV" API_SERVER_KEY || true)"
if [[ -z "$API_KEY" ]]; then
  API_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')"
fi
set_env_value "$HERMES_ENV" API_SERVER_ENABLED true
set_env_value "$HERMES_ENV" API_SERVER_HOST 127.0.0.1
set_env_value "$HERMES_ENV" API_SERVER_PORT 8642
set_env_value "$HERMES_ENV" API_SERVER_KEY "$API_KEY"

set_env_value "$ROOT/.env" HERMES_AGENT_BASE_URL http://127.0.0.1:8642/v1
set_env_value "$ROOT/.env" HERMES_AGENT_API_KEY "$API_KEY"
set_env_value "$ROOT/.env" HERMES_AGENT_MODEL hermes-agent
set_env_value "$ROOT/.env" HERMES_AGENT_SESSION_KEY hermes-study

chmod +x scripts/*.sh

echo "Enabling the local Hermes API server..."
"$HERMES_BIN" gateway restart >/dev/null 2>&1 || "$HERMES_BIN" gateway start >/dev/null 2>&1 || true

HERMES_OK=0
for _ in {1..20}; do
  if curl -fsS http://127.0.0.1:8642/health >/dev/null 2>&1; then
    HERMES_OK=1
    break
  fi
  sleep 0.5
done

if [[ "$HERMES_OK" == "1" ]]; then
  echo "Hermes API is online at http://127.0.0.1:8642"
else
  echo "Hermes Study installed, but the Hermes API is not online yet."
  echo "Run: $HERMES_BIN gateway status"
  echo "Then: $HERMES_BIN gateway restart"
fi

echo
echo "Installed. Next: ./scripts/run_mac.sh"
echo "No hermes-voice or Ollama installation is required."
echo "For speaker filtering later: INSTALL_SPEAKER_GATE=1 bash scripts/install_mac.sh"
