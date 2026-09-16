from __future__ import annotations

import asyncio
import io
import wave
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

MODEL_ID = "mlx-community/Kokoro-82M-bf16"
SAMPLE_RATE = 24_000

# Deliberately small, curated set: no browser/system voices and no giant catalog.
# Heart is the default used by Hermes Voice on Apple Silicon.
VOICES = (
    {"id": "af_heart", "name": "Heart", "description": "Warm American female", "recommended": True},
    {"id": "af_bella", "name": "Bella", "description": "Bright American female", "recommended": False},
    {"id": "af_sarah", "name": "Sarah", "description": "Soft American female", "recommended": False},
    {"id": "am_michael", "name": "Michael", "description": "Deep American male", "recommended": False},
    {"id": "bf_emma", "name": "Emma", "description": "Natural British female", "recommended": False},
    {"id": "bm_george", "name": "George", "description": "Natural British male", "recommended": False},
)
VOICE_IDS = {item["id"] for item in VOICES}
DEFAULT_VOICE = "af_heart"


class NeuralTtsUnavailable(RuntimeError):
    pass


class NeuralKokoroTts:
    """Lazy Apple-Silicon Kokoro TTS using the same MLX model as Hermes Voice."""

    def __init__(self, model_id: str = MODEL_ID) -> None:
        self.model_id = model_id
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hermes-study-tts")
        self._model: Any = None

    @property
    def available(self) -> bool:
        try:
            import mlx_audio  # noqa: F401
        except Exception:
            return False
        return True

    async def synthesize_wav(self, text: str, *, voice: str = DEFAULT_VOICE, speed: float = 1.0) -> bytes:
        clean = text.strip()
        if not clean:
            return b""
        if voice not in VOICE_IDS:
            raise ValueError(f"Unsupported neural voice: {voice}")
        speed = float(speed)
        if not 0.65 <= speed <= 1.40:
            raise ValueError("speed must be between 0.65 and 1.40")
        try:
            return await asyncio.get_running_loop().run_in_executor(
                self._executor,
                self._synthesize_wav_sync,
                clean,
                voice,
                speed,
            )
        except NeuralTtsUnavailable:
            raise
        except Exception as exc:
            raise NeuralTtsUnavailable(f"Kokoro neural TTS failed: {exc}") from exc

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from mlx_audio.tts.utils import load_model
        except Exception as exc:
            raise NeuralTtsUnavailable(
                "Neural TTS is not installed. Run scripts/install_mac.sh to install the MLX Kokoro voice stack."
            ) from exc
        _patch_sine_gen_length_bug()
        self._model = load_model(self.model_id)
        return self._model

    def _synthesize_wav_sync(self, text: str, voice: str, speed: float) -> bytes:
        model = self._load()
        chunks: list[np.ndarray[Any, np.dtype[np.float32]]] = []
        kwargs = {"text": text, "voice": voice, "speed": speed, "lang_code": voice[0]}
        try:
            generated = model.generate(**kwargs)
        except TypeError:
            # Compatibility with older mlx-audio builds used by Hermes Voice.
            kwargs.pop("lang_code", None)
            generated = model.generate(**kwargs)
        for segment in generated:
            audio = np.asarray(segment.audio, dtype=np.float32).reshape(-1)
            if audio.size:
                chunks.append(audio)
        if not chunks:
            return b""
        pcm = (np.clip(np.concatenate(chunks), -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        return _pcm_to_wav(pcm)


def _pcm_to_wav(pcm: bytes) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm)
    return output.getvalue()


def _patch_sine_gen_length_bug() -> None:
    """Carry forward the MLX Kokoro compatibility patch from Hermes Voice."""
    try:
        import mlx.core as mx
        from mlx_audio.tts.models.kokoro import istftnet
    except Exception:
        return
    if getattr(istftnet.SineGen, "_hermes_study_patched", False):
        return

    def patched_call(self: Any, f0: Any) -> tuple[Any, Any, Any]:
        fn = f0 * mx.arange(1, self.harmonic_num + 2)[None, None, :]
        sine_waves = self._f02sine(fn) * self.sine_amp
        uv = self._f02uv(f0)
        target = uv.shape[1]
        if sine_waves.shape[1] < target:
            pad = mx.zeros((sine_waves.shape[0], target - sine_waves.shape[1], sine_waves.shape[2]))
            sine_waves = mx.concatenate([sine_waves, pad], axis=1)
        elif sine_waves.shape[1] > target:
            sine_waves = sine_waves[:, :target, :]
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * mx.random.normal(sine_waves.shape)
        return sine_waves * uv + noise, uv, noise

    istftnet.SineGen.__call__ = patched_call
    istftnet.SineGen._hermes_study_patched = True
