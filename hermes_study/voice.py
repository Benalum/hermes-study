from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


class VoiceUnavailable(RuntimeError):
    pass


def _to_wav(source: Path, dest: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise VoiceUnavailable("ffmpeg is required for browser-recorded audio. Install it with: brew install ffmpeg")
    subprocess.run(
        [ffmpeg, "-y", "-i", str(source), "-ac", "1", "-ar", "16000", str(dest)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


class SpeakerGate:
    def __init__(self, voice_dir: Path, threshold: float = 0.72):
        self.voice_dir = Path(voice_dir)
        self.voice_dir.mkdir(parents=True, exist_ok=True)
        self.threshold = threshold
        self.reference = self.voice_dir / "speaker-reference.wav"
        self._model: Any = None

    @property
    def enrolled(self) -> bool:
        return self.reference.exists()

    def enroll(self, audio_path: Path) -> None:
        _to_wav(audio_path, self.reference)

    def verify(self, audio_path: Path) -> tuple[bool, float]:
        if not self.enrolled:
            raise VoiceUnavailable("No speaker reference has been enrolled yet.")
        try:
            from speechbrain.inference.speaker import SpeakerRecognition
        except Exception as exc:
            raise VoiceUnavailable("Speaker verification extra is not installed. Run: pip install -e '.[speaker]'") from exc
        if self._model is None:
            self._model = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=str(self.voice_dir / "speechbrain-ecapa"),
            )
        candidate = self.voice_dir / "candidate.wav"
        _to_wav(audio_path, candidate)
        score, prediction = self._model.verify_files(str(self.reference), str(candidate))
        value = float(score.squeeze().item())
        passed = bool(prediction.squeeze().item()) and value >= self.threshold
        return passed, value


class WhisperSTT:
    def __init__(self, model_name: str = "base.en"):
        self.model_name = model_name
        self._model: Any = None

    def transcribe(self, audio_path: Path) -> str:
        try:
            from faster_whisper import WhisperModel
        except Exception as exc:
            raise VoiceUnavailable("Voice STT extra is not installed. Run: pip install -e '.[voice]'") from exc
        if self._model is None:
            self._model = WhisperModel(self.model_name, device="auto", compute_type="auto")
        segments, _info = self._model.transcribe(str(audio_path), vad_filter=True, beam_size=5)
        return " ".join(seg.text.strip() for seg in segments).strip()


def save_upload_bytes(data: bytes, suffix: str = ".webm") -> Path:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data)
    tmp.close()
    return Path(tmp.name)
