from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


# Cosine similarities below this value are normally background matches for the
# multilingual embedding model used by this project.  The value is deliberately
# configurable: callers can tune it on a labelled evaluation set without
# changing the aggregation algorithm.
DEFAULT_MIN_VECTOR_SCORE = 0.45
DEFAULT_MIN_TEXT_SCORE = 0.01
DEFAULT_TEXT_SCORE_SATURATION = 0.15
DEFAULT_MIN_DOCUMENT_SCORE = 0.05
DEFAULT_RELATIVE_CUTOFF = 0.55
DEFAULT_MIN_COMPETITIVE_SIGNAL = 0.25

_MISSING_RANK = 999_999
_DOCUMENT_FIELDS = (
    "doc_id",
    "source_name",
    "index_code",
    "document_title",
    "version",
)


def rank_documents(
    rows: Sequence[Mapping[str, Any]],
    *,
    limit: int = 20,
    min_vector_score: float = DEFAULT_MIN_VECTOR_SCORE,
    min_text_score: float = DEFAULT_MIN_TEXT_SCORE,
    text_score_saturation: float = DEFAULT_TEXT_SCORE_SATURATION,
    min_document_score: float = DEFAULT_MIN_DOCUMENT_SCORE,
    relative_cutoff: float = DEFAULT_RELATIVE_CUTOFF,
    min_competitive_signal: float = DEFAULT_MIN_COMPETITIVE_SIGNAL,
) -> list[dict[str, Any]]:
    """Aggregate hybrid-search chunks and return only relevant documents.

    ``hybrid_score`` is reciprocal-rank based, so every vector candidate gets a
    positive value even when its cosine similarity is weak.  Treating that
    value as relevance is what produces a long, unrelated document tail.  This
    function instead:

    * aggregates all returned chunks by document;
    * applies absolute floors to the real vector and full-text scores;
    * normalizes each signal above its own floor;
    * keeps documents that remain competitive with the best document on either
      signal, so a strong vector-only match is not lost merely because another
      document also has a lexical match.

    The returned dictionaries contain document metadata plus deterministic
    aggregate diagnostics.  They are directly consumable by
    ``documents_by_topic_answer`` and useful when tuning cutoffs.
    """

    _validate_cutoffs(
        limit=limit,
        min_vector_score=min_vector_score,
        min_text_score=min_text_score,
        text_score_saturation=text_score_saturation,
        min_document_score=min_document_score,
        relative_cutoff=relative_cutoff,
        min_competitive_signal=min_competitive_signal,
    )
    if limit == 0 or not rows:
        return []

    aggregates: dict[str, _DocumentAggregate] = {}
    for position, row in enumerate(rows):
        document_key = _document_key(row)
        if not document_key:
            continue
        aggregate = aggregates.get(document_key)
        if aggregate is None:
            aggregate = _DocumentAggregate(document_key, position)
            aggregates[document_key] = aggregate
        aggregate.add(row, position)

    if not aggregates:
        return []

    scored: list[dict[str, Any]] = []
    for aggregate in aggregates.values():
        vector_signal = _signal_above_floor(
            aggregate.best_vector_score,
            floor=min_vector_score,
            saturation=1.0,
        )
        text_signal = _signal_above_floor(
            aggregate.best_text_score,
            floor=min_text_score,
            saturation=text_score_saturation,
        )

        # RRF affects ordering, but it must never make an otherwise irrelevant
        # document eligible.  Multiple supporting chunks receive only a small,
        # saturating bonus so long documents cannot dominate by chunk count.
        primary_signal = max(vector_signal, text_signal)
        corroborating_signal = min(vector_signal, text_signal)
        evidence_chunks = sum(
            vector_score > min_vector_score or text_score > min_text_score
            for vector_score, text_score in aggregate.chunk_scores
        )
        support_strength = min(1.0, math.log2(max(1, evidence_chunks)) / 2.0)
        document_score = min(
            1.0,
            primary_signal * (1.0 + 0.06 * support_strength)
            + 0.10 * corroborating_signal,
        )

        has_absolute_evidence = (
            aggregate.best_vector_score > min_vector_score
            or aggregate.best_text_score > min_text_score
        )
        if not has_absolute_evidence or document_score < min_document_score:
            continue

        record = aggregate.as_record()
        record.update(
            {
                "document_score": round(document_score, 8),
                "vector_signal": round(vector_signal, 8),
                "text_signal": round(text_signal, 8),
                "evidence_chunks": evidence_chunks,
            }
        )
        scored.append(record)

    if not scored:
        return []

    best_score = max(float(record["document_score"]) for record in scored)
    best_vector_signal = max(float(record["vector_signal"]) for record in scored)
    best_text_signal = max(float(record["text_signal"]) for record in scored)

    filtered = [
        record
        for record in scored
        if _passes_relative_cutoff(
            record,
            relative_cutoff=relative_cutoff,
            best_score=best_score,
            best_vector_signal=best_vector_signal,
            best_text_signal=best_text_signal,
            min_competitive_signal=min_competitive_signal,
        )
    ]
    filtered.sort(key=_document_sort_key)
    return filtered[:limit]


class _DocumentAggregate:
    def __init__(self, key: str, first_position: int) -> None:
        self.key = key
        self.first_position = first_position
        self.metadata: dict[str, Any] = {}
        self.best_vector_score = 0.0
        self.best_text_score = 0.0
        self.best_hybrid_score = 0.0
        self.best_vector_rank = _MISSING_RANK
        self.best_text_rank = _MISSING_RANK
        self.matched_chunks = 0
        self.vector_matched_chunks = 0
        self.text_matched_chunks = 0
        self.chunk_scores: list[tuple[float, float]] = []

    def add(self, row: Mapping[str, Any], position: int) -> None:
        del position  # Position is kept at document creation for stable tie-breaking.
        self.matched_chunks += 1

        for field in _DOCUMENT_FIELDS:
            if not self.metadata.get(field) and row.get(field) not in (None, ""):
                self.metadata[field] = row[field]

        vector_score = _finite_float(row.get("vector_score"))
        text_score = _finite_float(row.get("text_score"))
        hybrid_score = _finite_float(row.get("hybrid_score"))
        vector_rank = _rank(row.get("vector_rank"))
        text_rank = _rank(row.get("text_rank"))

        self.best_vector_score = max(self.best_vector_score, vector_score)
        self.best_text_score = max(self.best_text_score, text_score)
        self.best_hybrid_score = max(self.best_hybrid_score, hybrid_score)
        self.best_vector_rank = min(self.best_vector_rank, vector_rank)
        self.best_text_rank = min(self.best_text_rank, text_rank)
        self.chunk_scores.append((vector_score, text_score))
        if vector_score > 0.0 and vector_rank < _MISSING_RANK:
            self.vector_matched_chunks += 1
        if text_score > 0.0 and text_rank < _MISSING_RANK:
            self.text_matched_chunks += 1

    def as_record(self) -> dict[str, Any]:
        record = {field: self.metadata.get(field) for field in _DOCUMENT_FIELDS}
        record.update(
            {
                "best_vector_score": self.best_vector_score,
                "best_text_score": self.best_text_score,
                "best_hybrid_score": self.best_hybrid_score,
                "best_vector_rank": self.best_vector_rank,
                "best_text_rank": self.best_text_rank,
                "matched_chunks": self.matched_chunks,
                "vector_matched_chunks": self.vector_matched_chunks,
                "text_matched_chunks": self.text_matched_chunks,
                "first_search_position": self.first_position,
            }
        )
        return record


def _passes_relative_cutoff(
    record: Mapping[str, Any],
    *,
    relative_cutoff: float,
    best_score: float,
    best_vector_signal: float,
    best_text_signal: float,
    min_competitive_signal: float,
) -> bool:
    if float(record["document_score"]) >= best_score * relative_cutoff:
        return True
    if best_vector_signal >= min_competitive_signal:
        if float(record["vector_signal"]) >= best_vector_signal * relative_cutoff:
            return True
    if best_text_signal >= min_competitive_signal:
        if float(record["text_signal"]) >= best_text_signal * relative_cutoff:
            return True
    return False


def _document_sort_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        -float(record["document_score"]),
        -float(record["best_hybrid_score"]),
        -float(record["best_vector_score"]),
        -float(record["best_text_score"]),
        int(record["first_search_position"]),
        str(record.get("doc_id") or record.get("source_name") or ""),
    )


def _document_key(row: Mapping[str, Any]) -> str:
    for field in ("doc_id", "source_name", "document_title"):
        value = str(row.get(field) or "").strip()
        if value:
            return f"{field}:{value}"
    return ""


def _signal_above_floor(value: float, *, floor: float, saturation: float) -> float:
    if value <= floor:
        return 0.0
    return min(1.0, (value - floor) / (saturation - floor))


def _finite_float(value: Any) -> float:
    try:
        result = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(result):
        return 0.0
    return result


def _rank(value: Any) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        return _MISSING_RANK
    if result <= 0 or result >= _MISSING_RANK:
        return _MISSING_RANK
    return result


def _validate_cutoffs(
    *,
    limit: int,
    min_vector_score: float,
    min_text_score: float,
    text_score_saturation: float,
    min_document_score: float,
    relative_cutoff: float,
    min_competitive_signal: float,
) -> None:
    if limit < 0:
        raise ValueError("limit must be non-negative")
    if not 0.0 <= min_vector_score < 1.0:
        raise ValueError("min_vector_score must be in [0, 1)")
    if min_text_score < 0.0:
        raise ValueError("min_text_score must be non-negative")
    if text_score_saturation <= min_text_score:
        raise ValueError("text_score_saturation must be greater than min_text_score")
    if not 0.0 <= min_document_score <= 1.0:
        raise ValueError("min_document_score must be in [0, 1]")
    if not 0.0 <= relative_cutoff <= 1.0:
        raise ValueError("relative_cutoff must be in [0, 1]")
    if not 0.0 <= min_competitive_signal <= 1.0:
        raise ValueError("min_competitive_signal must be in [0, 1]")
