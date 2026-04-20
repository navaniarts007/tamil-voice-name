"""Rule-based Tamil-English spelling variants.

Used as a backstop when the LLM returns fewer than k suggestions.
Produces plausible alternate romanizations of a Tamil name by applying
common Tamil-English transliteration substitutions (th↔t, ee↔i, v↔w,
etc.) to the ASR transcript. No network, no model — just deterministic
spelling mutations.
"""

from __future__ import annotations

import re


# Ordered list of substitution pairs. Each tuple is (pattern, replacement).
# We apply them forwards AND backwards so each rule produces 2 variants.
_SUBS: list[tuple[str, str]] = [
    ("th", "t"),
    ("dh", "d"),
    ("ee", "i"),
    ("ee", "y"),
    ("i", "y"),
    ("oo", "u"),
    ("u", "oo"),
    ("v", "w"),
    ("k", "g"),
    ("sh", "s"),
    ("zh", "l"),
    ("ph", "f"),
]


def _apply(name: str, pat: str, repl: str) -> str:
    return re.sub(pat, repl, name, flags=re.IGNORECASE)


def _title(name: str) -> str:
    return " ".join(w.capitalize() for w in name.split())


def generate(query: str, existing: list[str], k: int = 3) -> list[str]:
    """Generate up to k spelling variants of `query` not already in existing."""
    query = (query or "").strip()
    if not query:
        return []

    existing_lower = {n.lower() for n in existing}
    out: list[str] = []

    for pat, repl in _SUBS:
        for p, r in ((pat, repl), (repl, pat)):
            variant = _apply(query, p, r)
            if variant.lower() == query.lower():
                continue
            variant = _title(variant)
            if variant.lower() in existing_lower:
                continue
            existing_lower.add(variant.lower())
            out.append(variant)
            if len(out) >= k:
                return out

    if len(out) < k:
        for i, (pat_a, repl_a) in enumerate(_SUBS):
            for pat_b, repl_b in _SUBS[i + 1:]:
                v = _apply(_apply(query, pat_a, repl_a), pat_b, repl_b)
                if v.lower() == query.lower():
                    continue
                v = _title(v)
                if v.lower() in existing_lower:
                    continue
                existing_lower.add(v.lower())
                out.append(v)
                if len(out) >= k:
                    return out
    return out
