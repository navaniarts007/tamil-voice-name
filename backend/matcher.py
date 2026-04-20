"""Fuzzy + phonetic ranking over patients.json.

Designed to stay interactive (<100ms) as the patient pool grows from
the 30-row seed to tens of thousands of rows (e.g. after running
scripts/build_patients.py against ai4bharat/naamapadam). The trick is
a phonetic-prefix inverted index: rapidfuzz only scores the shortlist
of patients whose phonetic code shares a prefix with the query.
"""

from __future__ import annotations

import json
import os
from typing import Any

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from rapidfuzz import fuzz

from .phonetic import encode
from .romanize import romanize

_PATIENTS_PATH = os.path.join(os.path.dirname(__file__), "patients.json")

_PHON_W = 0.5
_EN_W = 0.3
_TA_W = 0.2

# Bucket prefilter: if the phonetic-prefix bucket yields at least this
# many candidates, skip the full scan.
_PREFILTER_MIN = 20
_PREFILTER_PREFIX_LEN = 2


def _transliterate_tamil(name_ta: str) -> str:
    if not name_ta:
        return ""
    try:
        latin = transliterate(name_ta, sanscript.TAMIL, sanscript.ITRANS)
    except Exception:
        return name_ta
    return latin.lower()


def _token_codes(name: str) -> list[str]:
    codes = [encode(tok) for tok in name.split() if tok.strip()]
    return [c for c in codes if c]


def _display_name_en(name_en: str, name_ta: str) -> str:
    if name_en and name_en.strip():
        return name_en.strip()
    if name_ta:
        return romanize(name_ta)
    return ""


class PatientIndex:
    def __init__(self, patients: list[dict[str, Any]]):
        self.patients: list[dict[str, Any]] = []
        self._enriched: list[dict[str, Any]] = []
        # phonetic-prefix inverted index -> list of entry indices
        self._bucket_2: dict[str, list[int]] = {}
        self._bucket_1: dict[str, list[int]] = {}

        for raw in patients:
            name_ta = raw.get("name_ta", "") or ""
            name_en = _display_name_en(raw.get("name_en", "") or "", name_ta)
            # Persist the derived display name back onto the patient record.
            patient = dict(raw)
            patient["name_en"] = name_en

            name_ta_latin = _transliterate_tamil(name_ta)
            entry = {
                "patient": patient,
                "name_en": name_en,
                "name_ta_latin": name_ta_latin,
                "code_full_en": encode(name_en),
                "code_tokens_en": _token_codes(name_en),
                "code_full_ta": encode(name_ta) if name_ta else "",
                "code_tokens_ta": _token_codes(name_ta) if name_ta else [],
            }
            idx = len(self._enriched)
            self.patients.append(patient)
            self._enriched.append(entry)

            all_codes = {entry["code_full_en"], entry["code_full_ta"]}
            all_codes.update(entry["code_tokens_en"])
            all_codes.update(entry["code_tokens_ta"])
            for c in all_codes:
                if not c:
                    continue
                p2 = c[:_PREFILTER_PREFIX_LEN]
                p1 = c[:1]
                self._bucket_2.setdefault(p2, []).append(idx)
                self._bucket_1.setdefault(p1, []).append(idx)

        # Deduplicate bucket lists.
        for buckets in (self._bucket_2, self._bucket_1):
            for k, v in buckets.items():
                if len(v) > 1:
                    buckets[k] = sorted(set(v))

    def _candidate_ids(self, q_codes: list[str]) -> list[int]:
        if not q_codes:
            return list(range(len(self._enriched)))
        hits: set[int] = set()
        for c in q_codes:
            if not c:
                continue
            p2 = c[:_PREFILTER_PREFIX_LEN]
            hits.update(self._bucket_2.get(p2, ()))
        if len(hits) < _PREFILTER_MIN:
            for c in q_codes:
                if not c:
                    continue
                hits.update(self._bucket_1.get(c[:1], ()))
        if len(hits) < _PREFILTER_MIN:
            return list(range(len(self._enriched)))
        return sorted(hits)

    def _phonetic_score(self, q_code: str, q_token_codes: list[str], entry: dict) -> float:
        if not q_code and not q_token_codes:
            return 0.0
        candidates = [entry["code_full_en"], entry["code_full_ta"]] + \
                     entry["code_tokens_en"] + entry["code_tokens_ta"]
        candidates = [c for c in candidates if c]
        if not candidates:
            return 0.0

        probe = [c for c in ([q_code] + q_token_codes) if c]
        best = 0.0
        for p in probe:
            for c in candidates:
                if p == c:
                    return 100.0
                best = max(best, fuzz.ratio(p, c))
                best = max(best, fuzz.partial_ratio(p, c))
        return best

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        query = (query or "").strip()
        if not query:
            return []
        q_code = encode(query)
        q_token_codes = _token_codes(query)
        probe_codes = [c for c in ([q_code] + q_token_codes) if c]

        candidate_ids = self._candidate_ids(probe_codes)
        scored: list[tuple[float, dict[str, Any], dict[str, float]]] = []
        for idx in candidate_ids:
            entry = self._enriched[idx]
            phon = self._phonetic_score(q_code, q_token_codes, entry)
            fuzzy_en = float(fuzz.WRatio(query, entry["name_en"])) if entry["name_en"] else 0.0
            fuzzy_ta = float(fuzz.WRatio(query, entry["name_ta_latin"])) if entry["name_ta_latin"] else 0.0
            total = _PHON_W * phon + _EN_W * fuzzy_en + _TA_W * fuzzy_ta
            scored.append((total, entry["patient"], {
                "phonetic": round(phon, 1),
                "fuzzy_en": round(fuzzy_en, 1),
                "fuzzy_ta": round(fuzzy_ta, 1),
            }))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, patient, breakdown in scored[:k]:
            results.append({
                **patient,
                "score": round(score, 1),
                "score_breakdown": breakdown,
            })
        return results


def _load_patients() -> list[dict[str, Any]]:
    with open(_PATIENTS_PATH, encoding="utf-8") as f:
        return json.load(f)


_index: PatientIndex | None = None


def get_index() -> PatientIndex:
    global _index
    if _index is None:
        _index = PatientIndex(_load_patients())
    return _index


def search(query: str, k: int = 5) -> list[dict[str, Any]]:
    return get_index().search(query, k=k)
