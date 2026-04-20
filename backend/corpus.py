"""Dataset-backed Tamil-name corpus.

Names come from `ai4bharat/naamapadam` (Apache-2.0) — a Tamil NER
corpus with ~500k person-name tagged spans. At first startup we stream
the dataset once, extract every PER span, romanize via romanize.py,
and cache the result as a plain-text file under backend/cache/. No
hand-maintained Python name list: the corpus is dataset-derived.

Runtime just reads the cache (one name per line); no HuggingFace
network calls after the first build.
"""

from __future__ import annotations

import os
import re
import threading
from typing import Optional

from .romanize import romanize


_BAD_CHARS = re.compile(r"[^A-Za-z\u0B80-\u0BFF ]")


def _is_clean(name_en: str) -> bool:
    """Drop names that contain punctuation/digits — naamapadam's PER spans
    sometimes include initials like 'A .', commas, quoted phrases, etc."""
    if not name_en or len(name_en) < 2 or len(name_en) > 40:
        return False
    if _BAD_CHARS.search(name_en):
        return False
    if name_en.count(" ") > 3:
        return False
    return True

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
CACHE_PATH = os.path.join(CACHE_DIR, "tamil_names.txt")

_lock = threading.Lock()
_names_en: list[str] | None = None
_names_ta_by_en: dict[str, str] = {}


def _read_cache() -> Optional[tuple[list[str], dict[str, str]]]:
    if not os.path.exists(CACHE_PATH):
        return None
    names_en: list[str] = []
    mapping: dict[str, str] = {}
    with open(CACHE_PATH, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t", 1)
            name_en = parts[0].strip()
            name_ta = parts[1].strip() if len(parts) > 1 else ""
            if not _is_clean(name_en):
                continue
            names_en.append(name_en)
            if name_ta and name_en not in mapping:
                mapping[name_en] = name_ta
    return names_en, mapping


def _build_cache_from_dataset(max_names: int = 20_000) -> tuple[list[str], dict[str, str]]:
    """One-time build: pull naamapadam[ta], extract PER, romanize."""
    from collections import Counter
    from datasets import load_dataset

    print("Building Tamil name corpus from ai4bharat/naamapadam (one-time)…")
    freq: Counter[str] = Counter()
    for split in ("train", "validation", "test"):
        try:
            ds = load_dataset(
                "ai4bharat/naamapadam", "ta", split=split,
                trust_remote_code=True,
            )
        except Exception as exc:
            print(f"  skipping split={split}: {exc}")
            continue
        label_names = ds.features["ner_tags"].feature.names
        b_per = next(i for i, n in enumerate(label_names) if n.startswith("B") and "PER" in n)
        i_per = next(i for i, n in enumerate(label_names) if n.startswith("I") and "PER" in n)
        for row in ds:
            words = row.get("words") or row.get("tokens") or []
            tags = row["ner_tags"]
            current: list[str] = []
            for w, t in zip(words, tags):
                if t == b_per:
                    if current:
                        freq[" ".join(current).strip()] += 1
                    current = [w]
                elif t == i_per and current:
                    current.append(w)
                else:
                    if current:
                        freq[" ".join(current).strip()] += 1
                        current = []
            if current:
                freq[" ".join(current).strip()] += 1

    ranked = [n for n, _ in freq.most_common() if 2 <= len(n) <= 60 and not any(c.isdigit() for c in n)]
    ranked = ranked[:max_names]

    os.makedirs(CACHE_DIR, exist_ok=True)
    names_en: list[str] = []
    mapping: dict[str, str] = {}
    seen_en: set[str] = set()
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        f.write("# source: ai4bharat/naamapadam [ta]\n")
        f.write("# format: <romanized_english>\\t<tamil_script>\n")
        for name_ta in ranked:
            name_en = romanize(name_ta).strip()
            if not name_en or name_en in seen_en:
                continue
            seen_en.add(name_en)
            names_en.append(name_en)
            mapping[name_en] = name_ta
            f.write(f"{name_en}\t{name_ta}\n")
    print(f"  wrote {len(names_en):,} names -> {CACHE_PATH}")
    return names_en, mapping


def load(max_names: int = 20_000, *, rebuild: bool = False) -> tuple[list[str], dict[str, str]]:
    """Return (english-names, {english -> tamil_script})."""
    global _names_en, _names_ta_by_en
    with _lock:
        if _names_en is not None and not rebuild:
            return _names_en, _names_ta_by_en
        if not rebuild:
            cached = _read_cache()
            if cached is not None:
                _names_en, _names_ta_by_en = cached
                return _names_en, _names_ta_by_en
        names_en, mapping = _build_cache_from_dataset(max_names=max_names)
        _names_en, _names_ta_by_en = names_en, mapping
        return _names_en, _names_ta_by_en


def names() -> list[str]:
    return load()[0]


def tamil_script_for(name_en: str) -> str:
    return load()[1].get(name_en, "")
