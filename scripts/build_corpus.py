"""One-time: build the Tamil name corpus from ai4bharat/naamapadam.

Usage:
    python scripts/build_corpus.py --max 20000

Writes backend/cache/tamil_names.txt (tab-separated english<TAB>tamil),
which is the dataset-derived corpus the suggester consumes at runtime.
No hardcoded names anywhere.
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from backend import corpus  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=20_000,
                        help="Maximum unique names to keep (default 20000).")
    args = parser.parse_args()
    names, mapping = corpus.load(max_names=args.max, rebuild=True)
    print(f"Done. {len(names):,} names written to {corpus.CACHE_PATH}")


if __name__ == "__main__":
    main()
