#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
chmod +x scripts/*.sh 2>/dev/null || true

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3.11+ is required. Install it first (Homebrew: brew install python@3.13)." >&2
  exit 1
fi

PYVER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Using Python $PYVER"
python3 - <<'PY'
import sys
if sys.version_info < (3,11):
    raise SystemExit("Hermes Study requires Python 3.11+")
PY

if command -v brew >/dev/null 2>&1 && ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Installing ffmpeg for browser microphone audio..."
  brew install ffmpeg
fi

python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[voice]'

if [[ "${INSTALL_SPEAKER_GATE:-0}" == "1" ]]; then
  echo "Installing optional SpeechBrain speaker verification..."
  python -m pip install -e '.[speaker]'
fi

if command -v ollama >/dev/null 2>&1; then
  EMBED_MODEL="${OLLAMA_EMBED_MODEL:-nomic-embed-text}"
  echo "Ensuring embedding model is available: $EMBED_MODEL"
  ollama pull "$EMBED_MODEL"
else
  echo "Ollama was not found. Install/start Ollama before using the local tutor model."
fi

mkdir -p data
[[ -f .env ]] || cp .env.example .env

echo
echo "Installed. Next: ./scripts/run_mac.sh"
echo "For speaker filtering: INSTALL_SPEAKER_GATE=1 bash scripts/install_mac.sh, then set HERMES_STUDY_SPEAKER_GATE=true in .env."
