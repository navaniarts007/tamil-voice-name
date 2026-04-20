"""Rule-based Tamil-English spelling variants (<10ms, pure Python).

Given the ASR transcript of a Tamil name, generate plausible alternate
English romanizations by applying common Tamil-English transliteration
substitutions (th↔t, ee↔i, v↔w, single↔double consonants, suffixes, etc.).

Every rule is applied both GLOBALLY (all occurrences at once) AND
POSITIONALLY (one occurrence at a time). The positional expansion is
important for names like "Prithi" where `i→ee` globally gives
"Preethee" but the more natural "Preethi" comes from replacing only
the first `i`.
"""

from __future__ import annotations

import re
from typing import Iterable


# Ordered by how "natural" the resulting spelling looks. First hits win,
# so the top k suggestions feel like what a Tamil speaker would actually write.
_SUBS: list[tuple[str, str]] = [
    ("th", "t"),          # Thangamuthu / Tangamuthu
    ("ee", "i"),          # Preethi / Prithi
    ("ee", "e"),          # Preethi / Prethi
    ("th", "dh"),         # Vadhivel / Vadivel (voicing alternate)
    ("dh", "d"),          # Vadhivel / Vadivel
    ("oo", "u"),          # Muthu / Moothoo (tighten)
    ("u", "oo"),          # Muthu / Moothoo (stretch)
    ("v", "w"),           # Vadivel / Wadivel
    ("sh", "s"),          # Shankar / Sankar
    ("zh", "l"),          # Pazhani / Palani
    ("ai", "ay"),         # Vairam / Vayram
    ("bh", "b"),          # Bharath / Barath
    ("a", "aa"),          # Anandhi / Aanandhi
    ("ee", "y"),          # Preethi / Preethy
    ("i", "ee"),          # Prithi / Preethi  (positional single-swap is key)
    ("k", "c"),           # Kavitha / Cavitha
    ("t", "d"),           # voicing
]

# Simplify-only (never add). Adding doubles produces noise like "Tthangamutthu".
_DOUBLE_SIMPLIFICATIONS: list[tuple[str, str]] = [
    ("tt", "t"), ("pp", "p"), ("kk", "k"),
    ("nn", "n"), ("mm", "m"), ("ll", "l"),
]

# Tamil-name endings we can legitimately append.
_SUFFIX_ADDS = ["u", "an", "n"]

# Patterns that indicate a mechanical-noise output — reject these candidates.
_BAD_PATTERNS = [
    re.compile(r"(.)\1{2,}"),     # 3+ same letter in a row
    re.compile(r"yy"),             # double-y is not a Tamil-English pattern
    re.compile(r"hh"),             # th+h artefacts from naive substitution
    re.compile(r"[bcdfghjklmnpqrstvwxz]y[bcdfghjklmnpqrstvwxz]"),
    re.compile(r"^y"),             # word-initial lone y is rare
]


def _is_ugly(name: str) -> bool:
    lower = name.lower()
    return any(p.search(lower) for p in _BAD_PATTERNS)


def _title(name: str) -> str:
    return " ".join(w.capitalize() for w in name.split() if w)


def _safe_add_h(name: str, base: str, replacement: str) -> str:
    """For rules that ADD an `h` after a consonant, skip positions that are
    already followed by `h` (avoids "Prithi"→"Prithhi")."""
    if not (replacement.endswith("h") and len(replacement) == len(base) + 1
            and base[-1] != "h"):
        return re.sub(base, replacement, name, flags=re.IGNORECASE)
    return re.sub(base + r"(?!h)", replacement, name, flags=re.IGNORECASE)


def _iter_occurrences(name: str, pat: str) -> Iterable[int]:
    lower = name.lower()
    low_pat = pat.lower()
    i = 0
    while True:
        j = lower.find(low_pat, i)
        if j < 0:
            return
        yield j
        i = j + 1


def _replace_at(name: str, pat: str, repl: str, pos: int) -> str:
    """Replace exactly the occurrence of `pat` starting at `pos`."""
    return name[:pos] + repl + name[pos + len(pat):]


def _variants_from_rule(query: str, pat: str, repl: str) -> list[str]:
    """Apply rule globally AND per-occurrence, returning all distinct
    non-identical results."""
    results: list[str] = []
    seen: set[str] = {query.lower()}
    # Global (all occurrences at once, with h-lookahead safety)
    g = _safe_add_h(query, pat, repl)
    if g.lower() not in seen:
        seen.add(g.lower())
        results.append(g)
    # Per-occurrence (one position at a time)
    for pos in _iter_occurrences(query, pat):
        # Keep the same-char lookahead for h-adding rules.
        if (repl.endswith("h") and len(repl) == len(pat) + 1
                and pat[-1] != "h"
                and pos + len(pat) < len(query)
                and query[pos + len(pat)].lower() == "h"):
            continue
        v = _replace_at(query, pat, repl, pos)
        if v.lower() not in seen:
            seen.add(v.lower())
            results.append(v)
    return results


def generate(query: str, existing: list[str], k: int = 7) -> list[str]:
    """Return up to k plausible alternate spellings of `query`, excluding any
    name already in `existing` (case-insensitive)."""
    query = (query or "").strip()
    if not query:
        return []

    seen = {n.strip().lower() for n in existing}
    out: list[str] = []

    def _push(candidate: str) -> bool:
        c = _title(candidate.strip())
        k_ = c.lower()
        if not c or k_ in seen or k_ == query.strip().lower():
            return False
        if _is_ugly(c):
            return False
        seen.add(k_)
        out.append(c)
        return True

    def _done() -> bool:
        return len(out) >= k

    # Pass 1: single-rule substitutions, both directions, global + positional.
    for pat, repl in _SUBS:
        for p, r in ((pat, repl), (repl, pat)):
            if not re.search(p, query, flags=re.IGNORECASE):
                continue
            for v in _variants_from_rule(query, p, r):
                if _push(v) and _done():
                    return out

    # Pass 2: collapse double consonants (only simplify, never add).
    for pat, repl in _DOUBLE_SIMPLIFICATIONS:
        if re.search(pat, query, flags=re.IGNORECASE):
            v = re.sub(pat, repl, query, flags=re.IGNORECASE)
            if _push(v) and _done():
                return out

    # Pass 3: Tamil name suffix/trim operations.
    base = query.rstrip()
    ends_with_vowel = bool(base) and base[-1].lower() in "aeiou"
    for suf in _SUFFIX_ADDS:
        if base.lower().endswith(suf):
            continue
        if ends_with_vowel and suf[0].lower() in "aeiou":
            continue
        if _push(base + suf) and _done():
            return out
    if ends_with_vowel and _push(base[:-1]) and _done():
        return out

    # Pass 4: pair-wise rule combos (more creative, ugly-filtered).
    for i, (pat_a, repl_a) in enumerate(_SUBS):
        for pat_b, repl_b in _SUBS[i + 1:]:
            v = _safe_add_h(_safe_add_h(query, pat_a, repl_a), pat_b, repl_b)
            if _push(v) and _done():
                return out
    return out
