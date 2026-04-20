"""Name suggester: 1 ASR card + up to 3 LLM-generated English spellings.

Strategy:
  1. Sarvam ASR transcript is the "pakka" primary card.
  2. Sarvam LLM (sarvam-m) generates 3 plausible alternate English
     spellings. These appear as three separate cards below.
  3. Rule-based variants are the last-resort fallback ONLY when the LLM
     is unavailable (missing API key or network failure) — not part of
     the default presentation.

No corpus lookup. No hardcoded name list. Tamil script is generated
client-side from the Latin name via indic-transliteration for display.
"""

from __future__ import annotations

import os
from typing import Any

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

from . import sarvam_llm, variants

_LLM_ENABLED = os.environ.get("SARVAM_LLM_ENABLED", "true").lower() in ("1", "true", "yes")


def _title(name: str) -> str:
    return " ".join(w.capitalize() for w in name.split() if w)


def _to_tamil_script(latin: str) -> str:
    if not latin:
        return ""
    try:
        return transliterate(latin.lower(), sanscript.ITRANS, sanscript.TAMIL)
    except Exception:
        return ""


def primary_card(query: str) -> dict[str, Any]:
    """Fast: return just the ASR-echo card. No LLM call."""
    asr_name = _title(query)
    return {
        "name": asr_name,
        "name_ta": _to_tamil_script(asr_name),
        "score": 98.0,
        "source": "asr",
    }


def suggest(query: str, k: int = 4) -> dict[str, Any]:
    query = (query or "").strip()
    if not query:
        return {
            "query": query,
            "suggestions": [],
            "sources": {"asr": 0, "sarvam": 0, "variant": 0},
            "sarvam_llm_enabled": _LLM_ENABLED,
        }

    asr_name = _title(query)
    suggestions: list[dict[str, Any]] = [{
        "name": asr_name,
        "name_ta": _to_tamil_script(asr_name),
        "score": 98.0,
        "source": "asr",
    }]
    seen_lower = {asr_name.lower()}

    # Primary alternate-spelling source: Sarvam LLM.
    need = k - len(suggestions)
    llm_used = False
    if need > 0 and _LLM_ENABLED and sarvam_llm.is_available():
        llm_names = sarvam_llm.suggest_spellings(query, k=need + 2)
        for i, name in enumerate(llm_names):
            if len(suggestions) >= k:
                break
            canon = _title(name)
            if canon.lower() in seen_lower:
                continue
            seen_lower.add(canon.lower())
            suggestions.append({
                "name": canon,
                "name_ta": _to_tamil_script(canon),
                "score": round(92.0 - i * 3.0, 1),
                "source": "sarvam",
            })
        llm_used = True

    # Fallback only if LLM unavailable or returned nothing.
    need = k - len(suggestions)
    if need > 0 and not llm_used:
        fill = variants.generate(asr_name, [s["name"] for s in suggestions], k=need)
        for i, name in enumerate(fill):
            if len(suggestions) >= k:
                break
            suggestions.append({
                "name": name,
                "name_ta": _to_tamil_script(name),
                "score": round(82.0 - i * 2.0, 1),
                "source": "variant",
            })

    return {
        "query": query,
        "suggestions": suggestions[:k],
        "sources": {
            "asr": sum(1 for s in suggestions if s["source"] == "asr"),
            "sarvam": sum(1 for s in suggestions if s["source"] == "sarvam"),
            "variant": sum(1 for s in suggestions if s["source"] == "variant"),
        },
        "sarvam_llm_enabled": _LLM_ENABLED,
    }
