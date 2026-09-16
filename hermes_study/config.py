from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _dotenv_value(path: Path, key: str) -> str:
    """Read one simple KEY=value entry without importing or executing the file."""
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() == key:
                return value.strip().strip('"').strip("'")
    except OSError:
        pass
    return ""


def _default_hermes_key() -> str:
    env = os.getenv("HERMES_AGENT_API_KEY", "").strip()
    if env:
        return env
    hermes_home = Path(os.getenv("HERMES_HOME", "~/.hermes")).expanduser()
    return _dotenv_value(hermes_home / ".env", "API_SERVER_KEY")


@dataclass(slots=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("HERMES_STUDY_DATA_DIR", "./data")).expanduser())
    host: str = field(default_factory=lambda: os.getenv("HERMES_STUDY_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.getenv("HERMES_STUDY_PORT", "8787")))
    hermes_base_url: str = field(
        default_factory=lambda: os.getenv("HERMES_AGENT_BASE_URL", "http://127.0.0.1:8642/v1").rstrip("/")
    )
    hermes_api_key: str = field(default_factory=_default_hermes_key)
    hermes_model: str = field(default_factory=lambda: os.getenv("HERMES_AGENT_MODEL", "hermes-agent"))
    hermes_session_key: str = field(default_factory=lambda: os.getenv("HERMES_AGENT_SESSION_KEY", "hermes-study"))
    top_k: int = field(default_factory=lambda: int(os.getenv("HERMES_STUDY_TOP_K", "6")))
    speaker_gate: bool = field(default_factory=lambda: _bool("HERMES_STUDY_SPEAKER_GATE", False))
    speaker_threshold: float = field(
        default_factory=lambda: float(os.getenv("HERMES_STUDY_SPEAKER_THRESHOLD", "0.72"))
    )
    whisper_model: str = field(default_factory=lambda: os.getenv("HERMES_STUDY_WHISPER_MODEL", "base.en"))

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
