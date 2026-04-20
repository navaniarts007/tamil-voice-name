"""SarvamAI Tamil speech-to-text client.

Replaces the local IndicConformer path for the voice WebSocket. When
SARVAM_API_KEY is set in .env, the server pipes the buffered audio
through Sarvam's `saarika:v2.5` model at end-of-utterance and romanizes
the Tamil-script result.

Sarvam accepts a complete WAV file per call, so this is end-of-utterance
transcription (not streaming). Latency is dominated by the upload +
model round-trip, typically ~1–2s for a short name.
"""

from __future__ import annotations

import io
import os
import threading
import time
import wave
from typing import Optional

import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except Exception:
    pass

_KEY = os.environ.get("SARVAM_API_KEY", "").strip()
_MODEL = os.environ.get("SARVAM_MODEL", "saarika:v2.5")
_LANG = os.environ.get("SARVAM_LANG", "ta-IN")
# mode options: transcribe (Tamil script) | translit (English letters direct)
#   | verbatim | translate | codemix
# translit is usually what we want: Sarvam's built-in transliteration is far
# more accurate than our rule-based romanize.py for personal names.
_MODE = os.environ.get("SARVAM_MODE", "transcribe")
_RETURNS_LATIN = _MODE in ("translit", "translate")

_lock = threading.Lock()
_client = None
_disabled_reason: Optional[str] = None


def _get_client():
    global _client, _disabled_reason
    if _client is not None:
        return _client
    if _disabled_reason is not None:
        return None
    with _lock:
        if _client is not None:
            return _client
        if not _KEY:
            _disabled_reason = "SARVAM_API_KEY not set"
            return None
        try:
            from sarvamai import SarvamAI
        except Exception as exc:
            _disabled_reason = f"sarvamai import failed: {exc}"
            return None
        try:
            _client = SarvamAI(api_subscription_key=_KEY)
        except Exception as exc:
            _disabled_reason = f"SarvamAI() failed: {exc}"
            return None
        return _client


def is_available() -> bool:
    return _get_client() is not None


def disabled_reason() -> Optional[str]:
    _get_client()
    return _disabled_reason


def _float32_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Pack a float32 mono array into a 16-bit PCM WAV blob."""
    if audio.size == 0:
        return b""
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16).tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def transcribe_wav(wav_bytes: bytes, content_type: str = "audio/wav") -> str:
    """Send a ready-made WAV blob to Sarvam and return the transcribed string."""
    client = _get_client()
    if client is None or not wav_bytes:
        return ""
    start = time.perf_counter()
    kwargs = dict(
        file=("audio.wav", wav_bytes, content_type),
        model=_MODEL,
        language_code=_LANG,
        input_audio_codec="wav",
    )
    if _MODE and _MODE != "transcribe":
        kwargs["mode"] = _MODE
    try:
        resp = client.speech_to_text.transcribe(**kwargs)
    except Exception as exc:
        print(f"[sarvam] transcribe_wav failed ({_MODE}): {exc}")
        return ""
    dt_ms = int((time.perf_counter() - start) * 1000)
    text = getattr(resp, "transcript", None) or getattr(resp, "text", "") or ""
    text = (text or "").strip()
    print(f"[sarvam] wav {_MODE} {dt_ms}ms -> {text!r}")
    return text


def transcribe_array(audio: np.ndarray, sample_rate: int) -> str:
    """Transcribe a mono float32 audio buffer via Sarvam. Returns Tamil script."""
    client = _get_client()
    if client is None:
        return ""
    if audio is None or audio.size == 0:
        return ""
    wav_bytes = _float32_to_wav_bytes(audio.astype(np.float32, copy=False), sample_rate)
    if not wav_bytes:
        return ""
    start = time.perf_counter()
    kwargs = dict(
        file=("audio.wav", wav_bytes, "audio/wav"),
        model=_MODEL,
        language_code=_LANG,
        input_audio_codec="wav",
    )
    if _MODE and _MODE != "transcribe":
        kwargs["mode"] = _MODE
    try:
        resp = client.speech_to_text.transcribe(**kwargs)
    except Exception as exc:
        print(f"[sarvam] transcribe failed ({_MODE}): {exc}")
        return ""
    dt_ms = int((time.perf_counter() - start) * 1000)
    text = getattr(resp, "transcript", None) or getattr(resp, "text", "") or ""
    text = (text or "").strip()
    print(f"[sarvam] {_MODE} {dt_ms}ms -> {text!r}")
    return text


def returns_latin() -> bool:
    """True if the configured mode produces English letters directly
    (so callers should not run romanize.py on the output)."""
    return _RETURNS_LATIN
