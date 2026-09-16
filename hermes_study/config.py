from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class Settings:
    data_dir: Path = Path(os.getenv("HERMES_STUDY_DATA_DIR", "./data")).expanduser()
    host: str = os.getenv("HERMES_STUDY_HOST", "127.0.0.1")
    port: int = int(os.getenv("HERMES_STUDY_PORT", "8787"))
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    chat_model: str = os.getenv("OLLAMA_CHAT_MODEL", "gemma4:31b-mlx")
    embed_model: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    ollama_api_key: str = os.getenv("OLLAMA_API_KEY", "")
    top_k: int = int(os.getenv("HERMES_STUDY_TOP_K", "6"))
    speaker_gate: bool = _bool("HERMES_STUDY_SPEAKER_GATE", False)
    speaker_threshold: float = float(os.getenv("HERMES_STUDY_SPEAKER_THRESHOLD", "0.72"))
    whisper_model: str = os.getenv("HERMES_STUDY_WHISPER_MODEL", "base.en")

    @property
    def db_path(self) -> Path:
        return self.data_dir / "study.sqlite3"

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def voice_dir(self) -> Path:
        return self.data_dir / "voice"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.voice_dir.mkdir(parents=True, exist_ok=True)
