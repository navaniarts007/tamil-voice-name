"""Tamil script -> readable English letters.

This is a display-time romanizer, not a linguistic one. The goal is that
the nurse sees "Thangamuthu" instead of "தங்கமுத்து" — a readable
English spelling, not IPA/IAST. The matcher runs on the same text and
works fine either way because the phonetic encoder normalises both
forms to the same consonant skeleton.

Rules used (informal Tamil-English spelling, not ISO 15919):
  - consonants default to the unvoiced stop (த→th, ப→p, க→k, ட→d)
  - every consonant without a following vowel sign carries inherent 'a'
  - virama (் U+0BCD) suppresses the inherent 'a'
  - homorganic nasal + stop clusters collapse (ங்க → ng, not "ngk";
    ந்த → nth, ம்ப → mp, ண்ட → nd)
  - gemination (த்த, ப்ப, க்க, ட்ட) collapses to a single stop
"""

from __future__ import annotations

import re

_CONSONANTS = {
    "\u0B95": "k",   # க
    "\u0B99": "ng",  # ங
    "\u0B9A": "s",   # ச
    "\u0B9C": "j",   # ஜ
    "\u0B9E": "nj",  # ஞ
    "\u0B9F": "d",   # ட
    "\u0BA3": "n",   # ண
    "\u0BA4": "th",  # த
    "\u0BA8": "n",   # ந
    "\u0BA9": "n",   # ன
    "\u0BAA": "p",   # ப
    "\u0BAE": "m",   # ம
    "\u0BAF": "y",   # ய
    "\u0BB0": "r",   # ர
    "\u0BB1": "r",   # ற
    "\u0BB2": "l",   # ல
    "\u0BB3": "l",   # ள
    "\u0BB4": "zh",  # ழ
    "\u0BB5": "v",   # வ
    "\u0BB6": "sh",  # ஶ
    "\u0BB7": "sh",  # ஷ
    "\u0BB8": "s",   # ஸ
    "\u0BB9": "h",   # ஹ
}

_VOWELS = {
    "\u0B85": "a",    # அ
    "\u0B86": "aa",   # ஆ
    "\u0B87": "i",    # இ
    "\u0B88": "ee",   # ஈ
    "\u0B89": "u",    # உ
    "\u0B8A": "oo",   # ஊ
    "\u0B8E": "e",    # எ
    "\u0B8F": "e",    # ஏ
    "\u0B90": "ai",   # ஐ
    "\u0B92": "o",    # ஒ
    "\u0B93": "o",    # ஓ
    "\u0B94": "au",   # ஔ
}

_VOWEL_SIGNS = {
    "\u0BBE": "a",    # ா
    "\u0BBF": "i",    # ி
    "\u0BC0": "ee",   # ீ
    "\u0BC1": "u",    # ு
    "\u0BC2": "oo",   # ூ
    "\u0BC6": "e",    # ெ
    "\u0BC7": "e",    # ே
    "\u0BC8": "ai",   # ை
    "\u0BCA": "o",    # ொ
    "\u0BCB": "o",    # ோ
    "\u0BCC": "au",   # ௌ
}

_VIRAMA = "\u0BCD"   # ்
_AYTHAM = "\u0B83"   # ஃ

_TAMIL_RANGE = (0x0B80, 0x0BFF)

# Conjunct clusters that don't round-trip through per-character mapping.
# ட்ச encodes the Sanskrit retroflex /kʂ/ in Tamil script (e.g. மீனாட்சி =
# Meenakshi). ட்ஸ is a less common variant. க்ஷ is the Grantha form.
_CONJUNCTS = {
    "\u0B9F\u0BCD\u0B9A": "ksh",   # ட்ச
    "\u0B9F\u0BCD\u0BB8": "ksh",   # ட்ஸ
    "\u0B95\u0BCD\u0BB7": "ksh",   # க்ஷ
    "\u0BB6\u0BCD\u0BB0\u0BC0": "shri",  # ஶ்ரீ
    "\u0BB8\u0BCD\u0BB0\u0BC0": "shri",  # ஸ்ரீ
}

# Homorganic nasal + stop -> collapsed cluster.
_CLUSTER_FIXES: list[tuple[str, str]] = [
    ("ngk", "ng"), ("ngg", "ng"),
    ("njs", "nj"), ("njc", "nj"),
    ("nt", "nth"),   # ந்த already produced as "nth"; leave as-is
    # gemination collapse
    ("thth", "th"), ("ththth", "th"),
    ("pp", "p"), ("kk", "k"),
    ("tt", "t"), ("dd", "d"),
    ("mm", "m"), ("nn", "n"),
    ("ll", "l"), ("rr", "r"),
    ("ss", "s"),
]


def _is_tamil(ch: str) -> bool:
    return _TAMIL_RANGE[0] <= ord(ch) <= _TAMIL_RANGE[1]


def romanize(text: str) -> str:
    if not text:
        return ""
    if not any(_is_tamil(c) for c in text):
        return text

    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        matched_conjunct = None
        for key, romanized in _CONJUNCTS.items():
            if text.startswith(key, i):
                matched_conjunct = (key, romanized)
                break
        if matched_conjunct is not None:
            key, romanized = matched_conjunct
            nxt_idx = i + len(key)
            if nxt_idx < n and text[nxt_idx] in _VOWEL_SIGNS:
                out.append(romanized + _VOWEL_SIGNS[text[nxt_idx]])
                i = nxt_idx + 1
            else:
                out.append(romanized + "a")
                i = nxt_idx
            continue
        ch = text[i]
        if ch in _CONSONANTS:
            nxt = text[i + 1] if i + 1 < n else ""
            if nxt == _VIRAMA:
                out.append(_CONSONANTS[ch])
                i += 2
                continue
            if nxt in _VOWEL_SIGNS:
                out.append(_CONSONANTS[ch] + _VOWEL_SIGNS[nxt])
                i += 2
                continue
            out.append(_CONSONANTS[ch] + "a")
            i += 1
            continue
        if ch in _VOWELS:
            out.append(_VOWELS[ch])
            i += 1
            continue
        if ch == _AYTHAM:
            out.append("h")
            i += 1
            continue
        if ch == _VIRAMA:
            i += 1
            continue
        out.append(ch)
        i += 1

    result = "".join(out)
    for pat, repl in _CLUSTER_FIXES:
        result = result.replace(pat, repl)
    result = re.sub(r"\s+", " ", result).strip()
    if result:
        result = " ".join(w[:1].upper() + w[1:] for w in result.split(" "))
    return result


if __name__ == "__main__":
    samples = [
        "\u0ba4\u0b99\u0bcd\u0b95\u0bae\u0bc1\u0ba4\u0bcd\u0ba4\u0bc1",
        "\u0baa\u0bbf\u0bb0\u0bc0\u0ba4\u0bbf",
        "\u0bb5\u0b9f\u0bbf\u0bb5\u0bc7\u0bb2\u0bcd \u0bae\u0bc1\u0bb0\u0bc1\u0b95\u0ba9\u0bcd",
        "\u0b95\u0bbe\u0bb3\u0bbf\u0baf\u0baa\u0bc6\u0bb0\u0bc1\u0bae\u0bbe\u0bb3\u0bcd",
        "\u0bae\u0bc0\u0ba9\u0bbe\u0b9f\u0bcd\u0b9a\u0bbf\u0baf\u0bae\u0bcd\u0bae\u0bbe\u0bb3\u0bcd",
    ]
    for s in samples:
        print(repr(s), "->", romanize(s))
