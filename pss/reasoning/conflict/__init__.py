"""Observation-vs-prediction conflict detection (Phase 7)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ConflictResult:
    conflict: bool
    missing: list[tuple]
    unexpected: list[tuple]
    score: int


def detect_conflict(
    observed,
    predicted,
    critical: set[tuple] | None = None,
) -> ConflictResult:
    observed = set(observed)
    predicted = set(predicted)
    missing = sorted(predicted - observed)
    unexpected = sorted(observed - predicted)
    return ConflictResult(
        conflict=bool(missing or unexpected),
        missing=missing,
        unexpected=unexpected,
        score=len(missing) + len(unexpected),
    )


def should_replan(
    result: ConflictResult,
    critical: set[tuple] | None = None,
    threshold: int = 1,
) -> bool:
    if critical is not None:
        touched = set(result.missing) | set(result.unexpected)
        if touched & set(critical):
            return True
    return result.score >= threshold
