"""Gemini-based spelling generator for Tamil-English names.

Given an ASR transcript (a Tamil name romanized into English by our
romanizer), ask Gemini for 4 plausible English spellings of that name.
This complements the corpus: the corpus tells us which names exist in
the dataset; Gemini tells us the likely spellings a human would use.

The API key is read from the GEMINI_API_KEY env var (loaded from .env
at import time). No key is hardcoded anywhere in source.
"""

from __future__ import annotations

import os
import threading
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except Exception:
    pass

_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
_KEY_ENV = "GEMINI_API_KEY"
# Free tier of Gemini 2.5 Flash has tight quotas (10 RPM / 250 RPD).
# Retrying aggressively burns through the quota faster than it recovers, so
# we take just one swing per query and cache results for repeat queries.
_RETRY_ATTEMPTS = 1
_RETRY_BACKOFF_S = 0.0
_CACHE_MAX = 256
_cache: dict[str, list[str]] = {}

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
        api_key = os.environ.get(_KEY_ENV, "").strip()
        if not api_key:
            _disabled_reason = f"{_KEY_ENV} not set"
            return None
        try:
            from google import genai
        except Exception as exc:
            _disabled_reason = f"google-genai import failed: {exc}"
            return None
        try:
            _client = genai.Client(api_key=api_key)
        except Exception as exc:
            _disabled_reason = f"genai.Client failed: {exc}"
            return None
        return _client


def is_available() -> bool:
    return _get_client() is not None


def disabled_reason() -> Optional[str]:
    _get_client()
    return _disabled_reason


_PROMPT_TEMPLATE = (
    "You are a Tamil-name transliteration assistant for a hospital voice app.\n"
    "An Indian-language ASR system transcribed a Tamil name as:\n\n"
    "    {transcript}\n\n"
    "The transcription may be imperfect (wrong vowels, dropped syllables, "
    "merged/split words). Produce exactly {k} plausible English spellings "
    "of the Tamil name the speaker most likely said.\n"
    "Rules:\n"
    "  - Each spelling on its own line.\n"
    "  - No numbering, bullets, or explanations.\n"
    "  - Use common Tamil-English romanizations (e.g. 'Preethi', 'Thangamuthu', "
    "'Vadivel Murugan').\n"
    "  - Prefer the most common spelling variants a Tamil speaker would use.\n"
    "  - Do not invent names — each line must be a real Tamil personal name.\n"
)


def suggest_spellings(transcript: str, k: int = 4) -> list[str]:
    client = _get_client()
    transcript = transcript.strip()
    if client is None or not transcript:
        return []
    cache_key = f"{transcript.lower()}|{k}"
    if cache_key in _cache:
        return _cache[cache_key]
    import time

    prompt = _PROMPT_TEMPLATE.format(transcript=transcript, k=k)
    resp = None
    last_exc: Exception | None = None
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            resp = client.models.generate_content(model=_MODEL, contents=prompt)
            break
        except Exception as exc:
            last_exc = exc
            msg = str(exc)
            print(f"[gemini] {_MODEL} attempt {attempt+1} failed: {msg.splitlines()[0][:180]}")
            transient = "UNAVAILABLE" in msg or "503" in msg or "overloaded" in msg.lower()
            if transient and attempt + 1 < _RETRY_ATTEMPTS:
                time.sleep(_RETRY_BACKOFF_S * (2 ** attempt))
                continue
            break
    if resp is None:
        return []
    text = (getattr(resp, "text", "") or "").strip()
    if not text:
        return []
    lines = []
    for raw in text.splitlines():
        line = raw.strip().lstrip("-*0123456789.) ").strip()
        if not line:
            continue
        line = line.strip('"\'')
        lines.append(line)
        if len(lines) >= k:
            break
    result = lines[:k]
    if result:
        if len(_cache) >= _CACHE_MAX:
            _cache.pop(next(iter(_cache)))
        _cache[cache_key] = result
    return result
