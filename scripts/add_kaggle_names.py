"""Extend the Tamil-name corpus with Kaggle datasets.

Usage:
    # First, set Kaggle credentials (once):
    #   export KAGGLE_USERNAME=yourname
    #   export KAGGLE_KEY=your_kaggle_api_key
    # OR put them in .env.
    # (Get a key at https://www.kaggle.com/settings → Create New API Token.)

    python scripts/add_kaggle_names.py

Appends rows to backend/cache/tamil_names.txt (one name per line,
tab-separated english<TAB>tamil — tamil may be empty). No hardcoded names.

Datasets pulled:
  - younusmohamed/tanglish-and-tamil-transliterated-words-dataset
  - extra datasets can be added to the DATASETS list below or via
    the --dataset CLI flag.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Iterable

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
except Exception:
    pass

from backend import corpus  # noqa: E402
from backend.romanize import romanize  # noqa: E402

DATASETS: list[tuple[str, str]] = [
    # (kaggle_slug, preferred file_path — empty string = first file in dataset)
    ("younusmohamed/tanglish-and-tamil-transliterated-words-dataset", ""),
]

_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z .'-]{1,39}$")


def _looks_like_name(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    if not _NAME_RE.match(t):
        return False
    if t.lower() in {"the", "and", "of", "in", "to", "for", "is"}:
        return False
    return True


def _collect_names_from_df(df) -> list[tuple[str, str]]:
    """Scan every column for plausible name strings. Returns (en, ta) pairs.

    The dataset column names aren't known up-front, so we walk columns
    and keep strings that look like Latin-letter names. If a Tamil-script
    column sits alongside, we pair them by row index.
    """
    import pandas as pd

    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()

    latin_cols: list[str] = []
    tamil_cols: list[str] = []
    for col in df.columns:
        try:
            sample = df[col].dropna().astype(str).head(30).tolist()
        except Exception:
            continue
        if not sample:
            continue
        latin_ratio = sum(1 for s in sample if _looks_like_name(s)) / len(sample)
        tamil_ratio = sum(1 for s in sample if any("\u0b80" <= c <= "\u0bff" for c in s)) / len(sample)
        if latin_ratio >= 0.3:
            latin_cols.append(col)
        if tamil_ratio >= 0.3:
            tamil_cols.append(col)

    print(f"  latin-looking cols: {latin_cols}")
    print(f"  tamil-looking cols: {tamil_cols}")

    if latin_cols and tamil_cols:
        lcol, tcol = latin_cols[0], tamil_cols[0]
        for l, t in zip(df[lcol].astype(str), df[tcol].astype(str)):
            l = l.strip(); t = t.strip()
            if not _looks_like_name(l):
                continue
            key = l.lower()
            if key in seen:
                continue
            seen.add(key)
            pairs.append((l.title(), t if any("\u0b80" <= c <= "\u0bff" for c in t) else ""))
    elif latin_cols:
        for l in df[latin_cols[0]].astype(str):
            l = l.strip()
            if not _looks_like_name(l):
                continue
            key = l.lower()
            if key in seen:
                continue
            seen.add(key)
            pairs.append((l.title(), ""))
    elif tamil_cols:
        for t in df[tamil_cols[0]].astype(str):
            t = t.strip()
            if not t:
                continue
            en = romanize(t).strip()
            if not _looks_like_name(en):
                continue
            key = en.lower()
            if key in seen:
                continue
            seen.add(key)
            pairs.append((en.title(), t))
    return pairs


def _load_dataset(slug: str, file_path: str = "") -> list[tuple[str, str]]:
    import kagglehub
    from kagglehub import KaggleDatasetAdapter

    print(f"Loading Kaggle dataset: {slug} (file={file_path!r})")
    try:
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS, slug, file_path,
        )
    except TypeError:
        # older kagglehub positional signature
        df = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, slug, file_path)
    print(f"  rows: {len(df):,}, cols: {list(df.columns)}")
    return _collect_names_from_df(df)


def _existing_keys() -> set[str]:
    """Lowercased English names already in the corpus cache."""
    keys: set[str] = set()
    if os.path.exists(corpus.CACHE_PATH):
        with open(corpus.CACHE_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line or line.startswith("#"):
                    continue
                name_en = line.split("\t", 1)[0].strip().lower()
                if name_en:
                    keys.add(name_en)
    return keys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", default=[],
                        help="Extra Kaggle slug(s) to pull (repeat).")
    parser.add_argument("--limit", type=int, default=50_000,
                        help="Cap on new names added per run (default 50000).")
    args = parser.parse_args()

    targets = list(DATASETS) + [(slug, "") for slug in args.dataset]
    existing = _existing_keys()
    print(f"Existing corpus size: {len(existing):,}")

    all_new: list[tuple[str, str]] = []
    for slug, path in targets:
        try:
            pairs = _load_dataset(slug, path)
        except Exception as exc:
            print(f"  ⚠️  failed: {exc}")
            continue
        print(f"  extracted {len(pairs):,} name candidates")
        for en, ta in pairs:
            key = en.lower()
            if key in existing:
                continue
            existing.add(key)
            all_new.append((en, ta))
            if len(all_new) >= args.limit:
                break

    if not all_new:
        print("Nothing new to add.")
        return

    os.makedirs(corpus.CACHE_DIR, exist_ok=True)
    with open(corpus.CACHE_PATH, "a", encoding="utf-8") as f:
        f.write(f"# appended from Kaggle: {', '.join(s for s, _ in targets)}\n")
        for en, ta in all_new:
            f.write(f"{en}\t{ta}\n")
    print(f"✅ appended {len(all_new):,} names to {corpus.CACHE_PATH}")
    print("Restart the server to pick up the new corpus.")


if __name__ == "__main__":
    main()
