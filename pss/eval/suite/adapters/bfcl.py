"""BFCL-style tool-call adapter (skeleton).

The official Berkeley Function Calling Leaderboard format varies by release.
This adapter accepts a JSON list where each item has a question/prompt, a
tool/function schema list, and an expected call. ``prepare`` only checks that a
file exists; it never downloads data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from pss.eval.suite.adapter import BenchmarkAdapter, BenchmarkUnavailable
from pss.eval.suite.metrics import accuracy, exact_match
from pss.eval.suite.schema import EvalEpisode, EvalResult, Prediction


class BFCLAdapter(BenchmarkAdapter):
    name = "bfcl"
    layer = "language"
    license = "check-source"
    homepage = "https://gorilla.cs.berkeley.edu/leaderboard.html"

    def prepare(self) -> None:
        if self.root is None or not Path(self.root).exists():
            raise BenchmarkUnavailable(
                "BFCL data not found. Put the official JSON under "
                "data/public/bfcl/ and rerun without --smoke."
            )
        if not any(Path(self.root).glob("*.json")):
            raise BenchmarkUnavailable(f"no *.json under {self.root}")

    def episodes(self, split: str = "test") -> Iterator[EvalEpisode]:
        self.prepare()
        path = sorted(Path(self.root).glob("*.json"))[0]
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else payload.get("questions", [])
        for i, row in enumerate(rows):
            query = row.get("question") or row.get("prompt") or row.get("query") or ""
            tools = row.get("function") or row.get("functions") or row.get("tools") or []
            truth = row.get("ground_truth") or row.get("expected") or row.get("answer")
            if isinstance(truth, list) and truth:
                truth = truth[0]
            expected = {}
            if isinstance(truth, dict):
                expected = {
                    "tool": truth.get("name") or truth.get("tool"),
                    "args": truth.get("arguments") or truth.get("args") or {},
                }
            yield EvalEpisode(
                id=f"bfcl_{i:05d}",
                benchmark=self.name,
                layer=self.layer,
                split=split,
                capabilities={"query", "tools"},
                query=query,
                tools=list(tools),
                expected=expected,
                meta={"source": str(path)},
            )

    def evaluate(self, predictions: list[Prediction]) -> list[EvalResult]:
        tool_ok, args_ok = [], []
        for pred in predictions:
            truth = pred.meta.get("expected", {})
            out = pred.output or {}
            tool_ok.append(out.get("tool") == truth.get("tool"))
            args_ok.append(out.get("args") == truth.get("args"))
        return [
            EvalResult(self.name, self.layer, "test", "tool_call_accuracy", accuracy(tool_ok, [True] * len(tool_ok)), len(tool_ok)),
            EvalResult(self.name, self.layer, "test", "argument_accuracy", accuracy(args_ok, [True] * len(args_ok)), len(args_ok)),
        ]

    def sample_episodes(self) -> list[EvalEpisode]:
        return [
            EvalEpisode(
                id="bfcl_sample_0",
                benchmark=self.name,
                layer=self.layer,
                capabilities={"query", "tools"},
                query="What is the weather in Paris?",
                tools=[
                    {"name": "get_weather", "description": "Get weather", "parameters": {"city": "str"}},
                    {"name": "get_time", "description": "Get time", "parameters": {"zone": "str"}},
                ],
                expected={"tool": "get_weather", "args": {"city": "Paris"}},
            ),
            EvalEpisode(
                id="bfcl_sample_1",
                benchmark=self.name,
                layer=self.layer,
                capabilities={"query", "tools"},
                query="Convert 10 USD to EUR",
                tools=[
                    {"name": "convert_currency", "description": "Convert money", "parameters": {"amount": "float", "from": "str", "to": "str"}},
                    {"name": "get_weather", "description": "Get weather", "parameters": {"city": "str"}},
                ],
                expected={"tool": "convert_currency", "args": {"amount": 10, "from": "USD", "to": "EUR"}},
            ),
            EvalEpisode(
                id="bfcl_sample_2",
                benchmark=self.name,
                layer=self.layer,
                capabilities={"query", "tools"},
                query="List files in /tmp",
                tools=[
                    {"name": "list_files", "description": "List files in a directory", "parameters": {"path": "str"}},
                    {"name": "read_file", "description": "Read a file", "parameters": {"path": "str"}},
                ],
                expected={"tool": "list_files", "args": {"path": "/tmp"}},
            ),
        ]
