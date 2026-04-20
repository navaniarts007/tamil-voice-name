"""SarvamAI LLM client — generates plausible English spellings for a Tamil name.

Uses `sarvam-m` / `sarvam-30b` / `sarvam-105b` via the chat.completions
endpoint. Same Sarvam API key as the STT client — set SARVAM_API_KEY in
.env. Replaces Gemini for the suggester's generative spelling layer.
"""

from __future__ import annotations

import os
import re
import threading
import time
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except Exception:
    pass

from .phonetic import encode as _phon

_KEY = os.environ.get("SARVAM_API_KEY", "").strip()
_MODEL = os.environ.get("SARVAM_LLM_MODEL", "sarvam-m")
_RETRY_ATTEMPTS = 2
_RETRY_BACKOFF_S = 1.2
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


_SYSTEM = (
    "You are a Tamil-name transliteration expert for a hospital voice app. "
    "Given an ASR transcript of a spoken Tamil name (already romanized into "
    "English letters), produce exactly the requested number of plausible "
    "English spellings of the Tamil name the speaker most likely said.\n"
    "Rules:\n"
    " - Each spelling on its own line, nothing else.\n"
    " - No numbering, bullets, quotes, or explanations.\n"
    " - Use common Tamil-English romanizations (e.g. 'Preethi', 'Thangamuthu', "
    "'Vadivel Murugan').\n"
    " - Each line must be a real Tamil personal name spelled in English letters."
)


def suggest_spellings(transcript: str, k: int = 4) -> list[str]:
    client = _get_client()
    transcript = (transcript or "").strip()
    if client is None or not transcript:
        return []
    cache_key = f"{transcript.lower()}|{k}"
    if cache_key in _cache:
        return _cache[cache_key]

    user_msg = (
        f"The ASR heard: \"{transcript}\"\n\n"
        f"Give exactly {k} plausible English spellings of this Tamil name."
    )

    resp = None
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            resp = client.chat.completions(
                model=_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.3,
                max_tokens=2000,
                reasoning_effort="low",
            )
            break
        except Exception as exc:
            msg = str(exc)
            print(f"[sarvam-llm] {_MODEL} attempt {attempt + 1} failed: "
                  f"{msg.splitlines()[0][:180]}")
            transient = ("503" in msg or "UNAVAILABLE" in msg
                         or "overloaded" in msg.lower() or "timeout" in msg.lower())
            if transient and attempt + 1 < _RETRY_ATTEMPTS:
                time.sleep(_RETRY_BACKOFF_S * (2 ** attempt))
                continue
            break
    if resp is None:
        return []

    raw = _extract_text(resp)
    if not raw:
        return []
    # Prefer the post-<think> answer, but keep the reasoning as a fallback
    # since the model sometimes runs out of tokens mid-think.
    after_think = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE)
    after_think = re.sub(r"<think>.*$", "", after_think, flags=re.DOTALL | re.IGNORECASE)
    after_think = after_think.strip()
    text = after_think if after_think else raw

    name_line_re = re.compile(r"^[A-Za-z][A-Za-z .'-]{1,39}$")
    inline_name_re = re.compile(r"\b([A-Z][a-z]{2,}(?:[ '\-][A-Z][a-z]{2,}){0,3})\b")
    # English words that appear at the start of sentences in reasoning prose.
    FILLER = {
        "ok", "okay", "sure", "here", "tamil", "english", "name", "names",
        "spelling", "spellings", "pronounced", "common", "possible", "first",
        "second", "third", "fourth", "next", "then", "also", "let", "lets",
        "the", "this", "these", "those", "that", "there", "they", "them",
        "okay", "actually", "maybe", "perhaps", "however", "another", "since",
        "because", "given", "user", "wants", "asr", "transcription", "think",
        "okay", "tamil", "transliteration", "variations", "some", "others",
        "start", "start", "yes", "now", "let", "now", "well", "but", "and",
        "or", "so", "if", "in", "on", "at", "by", "for", "with", "from",
        "about", "what", "when", "where", "which", "who", "how", "why",
        "alternatively", "additionally", "similarly", "finally", "lastly",
        "example", "examples", "note", "notes", "option", "options",
        "candidate", "candidates", "suggestion", "suggestions",
    }

    # Query's letter set + phonetic code — real spellings of the same name
    # share most letters AND most of the consonant skeleton.
    q_letters = {c for c in transcript.lower() if c.isalpha()}
    q_phon = _phon(transcript)
    q_phon_prefix = q_phon[:2] if q_phon else ""

    results: list[str] = []
    seen: set[str] = set()

    def _accept(candidate: str, strict: bool = False) -> None:
        canon = " ".join(w.capitalize() for w in candidate.split())
        key = canon.lower()
        if key in seen or key in FILLER:
            return
        # Strict pass: the candidate must share a meaningful number of
        # letters with the query AND its phonetic code prefix must match
        # the query's. Rejects prose words like "Alternatively" / "Kavitha".
        if strict and q_letters:
            cand_letters = {c for c in key if c.isalpha()}
            overlap = len(q_letters & cand_letters)
            if overlap < max(4, len(q_letters) // 2):
                return
            if q_phon_prefix:
                cand_phon = _phon(canon)
                if not cand_phon or not cand_phon.startswith(q_phon_prefix[:1]):
                    return
        seen.add(key)
        results.append(canon)

    # Pass 1: clean one-per-line outputs (the happy path).
    for raw_line in text.splitlines():
        line = raw_line.strip().lstrip("-*0123456789.)\u2022 ").strip().strip("\"'*`")
        if not line or not name_line_re.match(line):
            continue
        _accept(line)
        if len(results) >= k:
            break

    # Pass 2: if the model embedded names inside prose (common for
    # reasoning-mode outputs), extract capitalised tokens that look like
    # names AND share enough letters with the query to plausibly be the
    # same name.
    if len(results) < k:
        for match in inline_name_re.finditer(text):
            _accept(match.group(1), strict=True)
            if len(results) >= k:
                break

    result = results[:k]
    if result:
        if len(_cache) >= _CACHE_MAX:
            _cache.pop(next(iter(_cache)))
        _cache[cache_key] = result
    return result


def _extract_text(resp) -> str:
    """Read message.content first; fall back to reasoning_content when the
    model exhausts max_tokens during reasoning (sarvam-30b/105b return the
    working notes under reasoning_content, which still contain candidate
    spellings our regex can salvage)."""
    parts: list[str] = []
    try:
        choice = resp.choices[0]
        msg = choice.message
        if getattr(msg, "content", None):
            parts.append(str(msg.content))
        if getattr(msg, "reasoning_content", None):
            parts.append(str(msg.reasoning_content))
    except Exception:
        pass
    if parts:
        return "\n".join(parts).strip()
    for attr in ("text", "content", "output"):
        val = getattr(resp, attr, None)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""
