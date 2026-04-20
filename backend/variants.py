"""Rule-based Tamil-English spelling variants (<10ms, pure Python).

Given the ASR transcript of a Tamil name, generate plausible alternate
English romanizations by applying common Tamil-English transliteration
substitutions (th↔t, ee↔i, v↔w, single↔double consonants, etc.).

Applied in priority order — the first rules produce the most "natural"
alternate spellings, so the top k suggestions feel close to what a Tamil
speaker would actually write.
"""

from __future__ import annotations

import re


# Substitution pairs in priority order. The first ones produce the most
# natural Tamil-English spelling alternates — reach k=3 before the rule
# list ever gets to the noisier tail.
_SUBS: list[tuple[str, str]] = [
    ("th", "t"),          # Thangamuthu / Tangamuthu   (very common)
    ("ee", "i"),          # Preethi / Prithi
    ("ee", "e"),          # Preethi / Prethi
    ("dh", "d"),          # Vadhivel / Vadivel
    ("oo", "u"),          # Muthu / Mooth u
    ("u", "oo"),          # Muthu / Moothoo
    ("v", "w"),           # Vadivel / Wadivel
    ("sh", "s"),          # Shankar / Sankar
    ("zh", "l"),          # Pazhani / Palani
    ("ai", "ay"),         # Vairam / Vayram
    ("bh", "b"),          # Bharath / Barath
    # Rules below are noisier — only used if we still haven't hit k.
    ("ee", "y"),          # Preethi / Preethy
    ("i", "ee"),          # Prithi / Preethi  (reverse of first rule in context)
]

# Double-consonant tweaks (gemination) — Tamil often uses doubled consonants
# (pp, tt, kk) where English spellers drop one.
_DOUBLE_SUBS: list[tuple[str, str]] = [
    ("tt", "t"), ("pp", "p"), ("kk", "k"),
    ("nn", "n"), ("mm", "m"), ("ll", "l"),
]

# Patterns that indicate a bad mechanical output (reject the candidate).
_BAD_PATTERNS = [
    re.compile(r"(.)\1{2,}"),   # 3+ same letter in a row ("hhh", "eee")
    re.compile(r"yy"),           # double-y never occurs in Tamil-English names
    re.compile(r"hh"),           # "thh", "phh" are non-words
    re.compile(r"[bcdfghjklmnpqrstvwxz]y[bcdfghjklmnpqrstvwxz]"),  # consonant-y-consonant
    re.compile(r"^y"),           # word-initial lone y (rare in Tamil names)
]


def _apply_safe(name: str, pat: str, repl: str) -> str:
    """Apply substitution without creating double-h artefacts.

    Turning "t" into "th" on "Prithi" would otherwise produce "Prithhi"
    (the existing 'h' gets a new 'th' prefix dumped in front of it).
    Use a negative lookahead so consonant→consonant+h only fires on
    consonants that aren't already followed by h.
    """
    if repl.endswith("h") and len(repl) == len(pat) + 1 and pat[-1] != "h":
        safe = pat + r"(?!h)"
        return re.sub(safe, repl, name, flags=re.IGNORECASE)
    return re.sub(pat, repl, name, flags=re.IGNORECASE)


def _is_ugly(name: str) -> bool:
    lower = name.lower()
    return any(p.search(lower) for p in _BAD_PATTERNS)

# Common Tamil name endings — appending or removing them yields plausible
# variants (Vadivel / Vadivelu / Vadivelan).
_SUFFIX_ADDS = ["u", "an", "n"]


def _title(name: str) -> str:
    return " ".join(w.capitalize() for w in name.split() if w)


def _apply(name: str, pat: str, repl: str) -> str:
    return _apply_safe(name, pat, repl)


def generate(query: str, existing: list[str], k: int = 3) -> list[str]:
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

    # Pass 1: single-rule substitutions (most natural).
    for pat, repl in _SUBS:
        for p, r in ((pat, repl), (repl, pat)):
            if re.search(p, query, flags=re.IGNORECASE):
                v = _apply(query, p, r)
                if _push(v) and _done():
                    return out

    # Pass 2: double-consonant simplifications. Only the "remove a double"
    # direction — adding doubles produces noise like "Tthangamutthu".
    for pat, repl in _DOUBLE_SUBS:
        if re.search(pat, query, flags=re.IGNORECASE):
            v = _apply(query, pat, repl)
            if _push(v) and _done():
                return out

    # Pass 3: add a Tamil suffix (or remove a trailing vowel).
    base = query.rstrip()
    ends_with_vowel = bool(base) and base[-1].lower() in "aeiou"
    for suf in _SUFFIX_ADDS:
        if base.lower().endswith(suf):
            continue
        # Don't append a vowel-suffix to a word already ending in a vowel
        # ("Prithi" + "u" = "Prithiu" — awkward).
        if ends_with_vowel and suf[0].lower() in "aeiou":
            continue
        if _push(base + suf) and _done():
            return out
    if ends_with_vowel:
        if _push(base[:-1]) and _done():
            return out

    # Pass 4: pair-wise rule combinations for extra coverage.
    for i, (pat_a, repl_a) in enumerate(_SUBS):
        for pat_b, repl_b in _SUBS[i + 1:]:
            v = _apply(_apply(query, pat_a, repl_a), pat_b, repl_b)
            if _push(v) and _done():
                return out

    return out
