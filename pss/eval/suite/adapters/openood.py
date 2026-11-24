"""OpenOOD-style open-set adapter (skeleton).

``prepare`` expects a ``features.npz`` with:
- ``train_x``, ``train_y``: in-distribution features/labels;
- ``test_x``, ``test_y``: test features/labels;
- ``test_ood``: boolean OOD flags for ``test_x``.

The adapter never downloads data. ``sample_episodes`` builds a tiny synthetic
feature set so the runner can smoke-test the full pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from pss.eval.suite.adapter import BenchmarkAdapter, BenchmarkUnavailable
from pss.eval.suite.metrics import accuracy, auroc, fpr_at_95tpr
from pss.eval.suite.schema import EvalEpisode, EvalResult, Prediction


class OpenOODAdapter(BenchmarkAdapter):
    name = "openood"
    layer = "open_set"
    license = "mixed"
    homepage = "https://github.com/JNetHao/OpenOOD"

    def _load(self):
        if self.root is None:
            raise BenchmarkUnavailable(
                "OpenOOD features not found. Export features.npz under "
                "data/public/openood/ (train_x/train_y/test_x/test_y/test_ood)."
            )
        path = Path(self.root) / "features.npz"
        if not path.exists():
            raise BenchmarkUnavailable(f"missing {path}")
        data = np.load(path)
        return data

    def prepare(self) -> None:
        self._load()

    def _prototypes(self, x, y) -> np.ndarray:
        classes = sorted(set(int(v) for v in y))
        return np.stack([x[y == c].mean(axis=0) for c in classes])

    def episodes(self, split: str = "test") -> Iterator[EvalEpisode]:
        data = self._load()
        train_x, train_y = data["train_x"], data["train_y"]
        test_x, test_y, test_ood = data["test_x"], data["test_y"], data["test_ood"]
        prototypes = self._prototypes(train_x, train_y)
        for i in range(len(test_x)):
            yield EvalEpisode(
                id=f"openood_{i:06d}",
                benchmark=self.name,
                layer=self.layer,
                split=split,
                capabilities={"features"},
                expected={"is_ood": bool(test_ood[i]), "label": int(test_y[i])},
                meta={
                    "features": np.asarray(test_x[i], dtype=np.float32),
                    "prototypes": prototypes,
                },
            )

    def evaluate(self, predictions: list[Prediction]) -> list[EvalResult]:
        scores = [p.output.get("score", float("nan")) for p in predictions]
        is_ood = [p.meta.get("expected", {}).get("is_ood", False) for p in predictions]
        in_dist = [not v for v in is_ood]
        threshold = 0.5
        pred = [s >= threshold for s in scores]
        return [
            EvalResult(self.name, self.layer, "test", "auroc", auroc(scores, in_dist), len(scores)),
            EvalResult(self.name, self.layer, "test", "fpr_at_95tpr", fpr_at_95tpr(scores, in_dist), len(scores)),
            EvalResult(self.name, self.layer, "test", "accuracy", accuracy(pred, in_dist), len(pred)),
        ]

    def sample_episodes(self) -> list[EvalEpisode]:
        rng = np.random.default_rng(0)
        prototypes = rng.normal(size=(3, 8)).astype(np.float32)
        episodes = []
        for i in range(9):
            ood = i >= 6
            if ood:
                x = rng.normal(scale=2.5, size=8).astype(np.float32)
            else:
                x = prototypes[i % 3] + 0.1 * rng.normal(size=8).astype(np.float32)
            episodes.append(
                EvalEpisode(
                    id=f"openood_sample_{i}",
                    benchmark=self.name,
                    layer=self.layer,
                    capabilities={"features"},
                    expected={"is_ood": ood, "label": i % 3},
                    meta={"features": x, "prototypes": prototypes},
                )
            )
        return episodes
