"""
Tamil-aware phonetic encoder.

Produces a compact consonant-skeleton code that is invariant to common
Tamil-English transliteration spelling variants:

    encode("Preethi") == encode("Preethy") == encode("Prithi")
        == encode("Prethi") == encode("Prithee") == encode("பிரீதி")

Strategy (Metaphone-style for Tamil-English):
  1. Lowercase, normalise Tamil script via indic-transliteration (ITRANS).
  2. Collapse consonant digraphs that share an articulation point
     (th/dh/t/d -> T; zh/l -> L; sh/s -> S; ch -> C; ph/f/p/b -> P;
     v/w -> V; k/g/c/q/kh/gh -> K; n/m kept distinct).
  3. Keep the FIRST character's category (vowel or consonant code) so
     "Anbu" and "Banbu" don't collide, then drop every subsequent vowel
     — Tamil speakers freely insert epenthetic vowels between
     consonant clusters ("Preethi" vs "Piriiti" via Tamil script) and
     freely swap vowels ("Preethi"/"Prethi"/"Prithi"). The consonant
     skeleton is the stable part.
  4. Silent 'h' after a consonant is dropped.
  5. Collapse consecutive duplicate codes.

The code does not try to capture tone, vowel length, or retroflex
contrasts — those aren't reliable in Tamil-English transliteration
anyway. The goal is recall, not linguistic fidelity.
"""

from __future__ import annotations

import re

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate


_TAMIL_RANGE = (0x0B80, 0x0BFF)

# Consonant digraphs — longest match first.
_CONSONANT_DIGRAPHS: list[tuple[str, str]] = [
    ("zh", "L"),  # ழ — retroflex approximant, collapses with 'l'
    ("th", "T"),
    ("dh", "T"),
    ("sh", "S"),
    ("ch", "C"),
    ("ph", "P"),
    ("kh", "K"),
    ("gh", "K"),
    ("bh", "P"),
    ("jh", "J"),
    ("ng", "N"),  # word-internal ng cluster collapses to N
    ("nj", "N"),
    ("ck", "K"),
]

# Single consonant code map.
_CONSONANT_MAP: dict[str, str] = {
    "t": "T",
    "d": "T",
    "p": "P",
    "b": "P",
    "f": "P",
    "v": "V",
    "w": "V",
    "k": "K",
    "g": "K",
    "q": "K",
    "c": "K",
    "r": "R",
    "l": "L",
    "n": "N",
    "m": "M",
    "s": "S",
    "z": "S",
    "x": "S",
    "j": "J",
    "y": "",   # semivowel — always dropped, behaves as a vowel glide
    "h": "",   # silent after a consonant; standalone 'h' is rare
}

_VOWELS = set("aeiou")


def _has_tamil(text: str) -> bool:
    return any(_TAMIL_RANGE[0] <= ord(ch) <= _TAMIL_RANGE[1] for ch in text)


def _tamil_to_latin(text: str) -> str:
    """Transliterate Tamil script -> ASCII approximation."""
    try:
        return transliterate(text, sanscript.TAMIL, sanscript.ITRANS)
    except Exception:
        return text


def _normalise(text: str) -> str:
    s = text.strip().lower()
    if _has_tamil(s):
        mixed = []
        buf_tamil = []
        for ch in s:
            if _TAMIL_RANGE[0] <= ord(ch) <= _TAMIL_RANGE[1]:
                buf_tamil.append(ch)
            else:
                if buf_tamil:
                    mixed.append(_tamil_to_latin("".join(buf_tamil)).lower())
                    buf_tamil = []
                mixed.append(ch)
        if buf_tamil:
            mixed.append(_tamil_to_latin("".join(buf_tamil)).lower())
        s = "".join(mixed)
    s = re.sub(r"[^a-z]+", "", s)
    return s


def encode(name: str) -> str:
    if not name:
        return ""
    s = _normalise(name)
    if not s:
        return ""

    for pat, repl in _CONSONANT_DIGRAPHS:
        s = s.replace(pat, repl)

    tokens: list[str] = []
    first_seen = False
    for ch in s:
        if not first_seen:
            first_seen = True
            if ch in _VOWELS:
                if ch == "a":
                    tokens.append("A")
                elif ch in ("e", "i"):
                    tokens.append("I")
                else:
                    tokens.append("U")
                continue
            if ch.isupper():
                tokens.append(ch)
                continue
            mapped = _CONSONANT_MAP.get(ch, ch.upper())
            if mapped:
                tokens.append(mapped)
            continue

        if ch in _VOWELS:
            continue
        if ch.isupper():
            tokens.append(ch)
            continue
        mapped = _CONSONANT_MAP.get(ch, ch.upper())
        if mapped:
            tokens.append(mapped)

    collapsed: list[str] = []
    for t in tokens:
        if not collapsed or collapsed[-1] != t:
            collapsed.append(t)
    return "".join(collapsed)


if __name__ == "__main__":
    tests = [
        "Preethi", "Preethy", "Prithi", "Prethi", "Prithee", "பிரீதி",
        "Thangamuthu", "Thangamudhu",
        "Vadivel", "Wadivel",
        "Ponnuswamy", "Ponusamy",
        "Kaliyaperumal", "Kaliaperumal",
        "Meenakshiammal", "Minakshiamal",
    ]
    for t in tests:
        print(f"{t:>20s} -> {encode(t)}")
