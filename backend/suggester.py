"""Name suggester for the Tamil-voice → English-name flow.

Pipeline (corpus removed per user's instruction — Sarvam ASR is the
source of truth):

  1. Seat the Sarvam ASR transcript as suggestion #1. ASR is "pakka"
     in the user's words — it's the accurate form of what they said.
  2. Ask Sarvam LLM for 3 alternate English spellings. This covers the
     common case of name spelling variation (Preethi/Preethy/Prithi).
  3. If the LLM returns fewer than we need, backfill with rule-based
     spelling variants from backend/variants.py. Guarantees k cards.
"""

from __future__ import annotations

from typing import Any

from . import sarvam_llm, variants
from .phonetic import encode


def _title(name: str) -> str:
    return " ".join(w.capitalize() for w in name.split() if w)


def suggest(query: str, k: int = 4) -> dict[str, Any]:
    query = (query or "").strip()
    if not query:
        return {
            "query": query,
            "suggestions": [],
            "sources": {"asr": 0, "sarvam": 0, "variant": 0},
            "sarvam_llm_available": sarvam_llm.is_available(),
        }

    asr_name = _title(query)
    suggestions: list[dict[str, Any]] = [{
        "name": asr_name,
        "name_ta": "",
        "score": 98.0,
        "source": "asr",
    }]
    # Dedupe by exact (case-folded) spelling. We WANT spellings that share the
    # same phonetic code as the ASR — they're the alternate romanizations
    # (Preethi/Preethy/Prithi) that the user needs to pick between.
    seen_lower = {asr_name.lower()}

    # Ask Sarvam LLM for (k-1) alternate spellings.
    need = k - len(suggestions)
    if need > 0 and sarvam_llm.is_available():
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
                "name_ta": "",
                "score": round(92.0 - i * 4.0, 1),
                "source": "sarvam",
            })

    # Rule-based backstop — guarantees we hit k cards.
    need = k - len(suggestions)
    if need > 0:
        fill = variants.generate(asr_name, [s["name"] for s in suggestions], k=need)
        for i, name in enumerate(fill):
            if len(suggestions) >= k:
                break
            suggestions.append({
                "name": name,
                "name_ta": "",
                "score": round(75.0 - i * 3.0, 1),
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
        "sarvam_llm_available": sarvam_llm.is_available(),
    }
