"""Benchmark adapter protocol for the PSS evaluation suite.

An adapter is the only place that knows a benchmark's native format. It
produces ``EvalEpisode`` objects and scores ``Prediction`` objects. The runner
never imports benchmark-specific code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

from .schema import EvalEpisode, EvalResult, Prediction


class BenchmarkUnavailable(RuntimeError):
    """The benchmark data is not present; the adapter can explain how to get it."""


class BenchmarkAdapter(ABC):
    name: str = "unknown"
    layer: str = "unknown"
    license: str = "unknown"
    homepage: str = ""

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root is not None else None

    def available(self) -> bool:
        try:
            self.prepare()
            return True
        except BenchmarkUnavailable:
            return False

    @abstractmethod
    def prepare(self) -> None:
        """Check the local data; raise ``BenchmarkUnavailable`` with instructions."""

    @abstractmethod
    def episodes(self, split: str = "test") -> Iterator[EvalEpisode]:
        """Yield the official split episodes."""

    @abstractmethod
    def evaluate(self, predictions: list[Prediction]) -> list[EvalResult]:
        """Score predictions; return one result per metric."""

    def sample_episodes(self) -> list[EvalEpisode]:
        """Tiny built-in episodes used by ``--smoke`` and tests."""
        return []
