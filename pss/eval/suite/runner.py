"""Suite runner for the PSS evaluation suite.

Example:
    python -m pss.eval.suite.runner --smoke
    python -m pss.eval.suite.runner --benchmarks bfcl,openood,zsre
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from pss._console import ensure_utf8
from pss.eval.suite.adapter import BenchmarkUnavailable
from pss.eval.suite.predictors import predict_episode
from pss.eval.suite.registry import get_adapter, list_adapters


def _fmt(value, digits=3):
    if value is None:
        return "n/a"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if value != value:  # NaN
        return "n/a"
    return f"{value:.{digits}f}"


def run(
    benchmarks: list[str] | None = None,
    root: str = "data/public",
    smoke: bool = False,
    out: str = "eval/results/suite",
    allow_missing: bool = True,
) -> dict:
    benchmarks = benchmarks or list_adapters()
    payload = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "smoke": bool(smoke),
        "root": root,
        "benchmarks": {},
        "skipped": {},
    }
    for name in benchmarks:
        adapter_root = Path(root) / name
        adapter = get_adapter(name, root=adapter_root)
        try:
            if smoke:
                episodes = adapter.sample_episodes()
            else:
                adapter.prepare()
                episodes = list(adapter.episodes("test"))
            predictions = [predict_episode(episode) for episode in episodes]
            results = adapter.evaluate(predictions)
            payload["benchmarks"][name] = {
                "layer": adapter.layer,
                "license": adapter.license,
                "homepage": adapter.homepage,
                "n_episodes": len(episodes),
                "results": [
                    {
                        "metric": r.metric,
                        "value": r.value,
                        "n": r.n,
                        "split": r.split,
                    }
                    for r in results
                ],
            }
        except BenchmarkUnavailable as exc:
            if not allow_missing:
                raise
            payload["skipped"][name] = str(exc)
    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_path / "report.md").write_text(render(payload), encoding="utf-8")
    return payload


def render(payload: dict) -> str:
    lines = [
        "# PSS evaluation suite",
        "",
        f"> generated: {payload['generated']}",
        f"> smoke: {payload['smoke']}; root: `{payload['root']}`",
        "",
        "| benchmark | layer | split | metric | value | n | license |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for name, row in payload["benchmarks"].items():
        for result in row["results"]:
            lines.append(
                f"| {name} | {row['layer']} | {result['split']} | {result['metric']} | "
                f"{_fmt(result['value'])} | {result['n']} | {row['license']} |"
            )
    if payload["skipped"]:
        lines += ["", "## Skipped", ""]
        for name, reason in payload["skipped"].items():
            lines.append(f"- `{name}`: {reason}")
    lines += [
        "",
        "## Notes",
        "",
        "- This is the Phase A skeleton: schema + adapters + metrics + runner.",
        "- `--smoke` uses tiny built-in episodes; it validates plumbing, not",
        "  benchmark performance.",
        "- Real runs require the official data under `data/public/<bench>/`; raw",
        "  data is never committed.",
        "- The BFCL predictor is a keyword baseline; the OpenOOD predictor is a",
        "  prototype baseline; the ZsRE predictor uses the PSS FactStore.",
        "",
    ]
    return "\n".join(lines)


def main():
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", default=",".join(list_adapters()))
    ap.add_argument("--root", default="data/public")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="eval/results/suite")
    ap.add_argument("--strict", action="store_true", help="fail on missing data")
    args = ap.parse_args()
    payload = run(
        [b.strip() for b in args.benchmarks.split(",") if b.strip()],
        root=args.root,
        smoke=args.smoke,
        out=args.out,
        allow_missing=not args.strict,
    )
    for name, row in payload["benchmarks"].items():
        summary = ", ".join(
            f"{r['metric']}={_fmt(r['value'])}" for r in row["results"]
        )
        print(f"{name}: {summary}")
    for name, reason in payload["skipped"].items():
        print(f"{name}: skipped ({reason})")
    print(f"output: {args.out}")


if __name__ == "__main__":
    main()
