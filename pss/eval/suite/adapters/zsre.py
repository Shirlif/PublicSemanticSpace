"""ZsRE-style fact-edit adapter (skeleton).

``prepare`` expects a JSONL file under ``data/public/zsre/`` with at least
``subject``, ``property``/``relation``, ``old_value`` and ``new_value``.
``locality`` facts (other subject/property pairs) are optional.

The predictor uses the PSS ``FactStore`` directly; the adapter measures edit
success, locality and rollback. No model is involved.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from pss.eval.suite.adapter import BenchmarkAdapter, BenchmarkUnavailable
from pss.eval.suite.metrics import accuracy
from pss.eval.suite.schema import EvalEpisode, EvalResult, FactSpec, Prediction


class ZsREAdapter(BenchmarkAdapter):
    name = "zsre"
    layer = "memory"
    license = "check-source"
    homepage = "https://github.com/edenb/zsre"

    def _path(self) -> Path:
        if self.root is None:
            raise BenchmarkUnavailable(
                "ZsRE data not found. Put the official JSONL under data/public/zsre/."
            )
        for name in ("zsre.jsonl", "edits.jsonl"):
            path = Path(self.root) / name
            if path.exists():
                return path
        raise BenchmarkUnavailable(f"no zsre.jsonl/edits.jsonl under {self.root}")

    def prepare(self) -> None:
        self._path()

    def episodes(self, split: str = "test") -> Iterator[EvalEpisode]:
        path = self._path()
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            row = json.loads(line)
            subject = str(row.get("subject") or row.get("entity"))
            prop = str(row.get("property") or row.get("relation") or row.get("attribute"))
            old = row.get("old_value", row.get("old"))
            new = row.get("new_value", row.get("new"))
            locality = row.get("locality") or []
            facts = [FactSpec(subject, prop, old)]
            for item in locality:
                if isinstance(item, dict):
                    facts.append(
                        FactSpec(
                            str(item.get("subject")),
                            str(item.get("property")),
                            item.get("value"),
                        )
                    )
            yield EvalEpisode(
                id=f"zsre_{i:06d}",
                benchmark=self.name,
                layer=self.layer,
                split=split,
                capabilities={"facts"},
                facts=facts,
                expected={"subject": subject, "property": prop, "old": old, "new": new},
                meta={"source": str(path)},
            )

    def evaluate(self, predictions: list[Prediction]) -> list[EvalResult]:
        edited = [bool(p.output.get("edited")) for p in predictions]
        locality = [bool(p.output.get("locality")) for p in predictions]
        rollback = [bool(p.output.get("rollback")) for p in predictions]
        return [
            EvalResult(self.name, self.layer, "test", "edit_success", accuracy(edited, [True] * len(edited)), len(edited)),
            EvalResult(self.name, self.layer, "test", "locality", accuracy(locality, [True] * len(locality)), len(locality)),
            EvalResult(self.name, self.layer, "test", "rollback_ok", accuracy(rollback, [True] * len(rollback)), len(rollback)),
        ]

    def sample_episodes(self) -> list[EvalEpisode]:
        return [
            EvalEpisode(
                id="zsre_sample_0",
                benchmark=self.name,
                layer=self.layer,
                capabilities={"facts"},
                facts=[
                    FactSpec("Q1", "capital", "A"),
                    FactSpec("Q2", "capital", "C"),
                ],
                expected={"subject": "Q1", "property": "capital", "old": "A", "new": "B"},
            ),
            EvalEpisode(
                id="zsre_sample_1",
                benchmark=self.name,
                layer=self.layer,
                capabilities={"facts"},
                facts=[
                    FactSpec("Q3", "founder", "X"),
                    FactSpec("Q4", "founder", "Y"),
                ],
                expected={"subject": "Q3", "property": "founder", "old": "X", "new": "Z"},
            ),
        ]
