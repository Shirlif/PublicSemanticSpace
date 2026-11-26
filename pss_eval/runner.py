"""Run a property suite against any set of ``SystemUnderTest`` factories."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .fixtures import Stream
from .preregistration import check_preregistration
from .properties import FAIL, PASS, SKIP, Property, SystemFactory

__all__ = [
    "SystemSpec",
    "make_default_systems",
    "make_protocol_systems",
    "run",
    "run_bundle_suite",
    "run_suite",
    "write_report",
]


@dataclass(frozen=True)
class SystemSpec:
    """A named factory.  The factory must return a fresh system each call."""

    name: str
    factory: SystemFactory

    def __post_init__(self) -> None:
        if not str(self.name):
            raise ValueError("SystemSpec.name must be non-empty")


def make_default_systems(dim: int = 32) -> list[SystemSpec]:
    """The two external-style systems in the F5 pre-registration."""

    from .adapters.flat import FlatNonPSSAdapter
    from .adapters.transformer import TransformerStateAdapter

    return [
        SystemSpec("e5-flat", lambda: FlatNonPSSAdapter(dim=dim)),
        SystemSpec("transformer-state", lambda: TransformerStateAdapter(dim=dim)),
    ]


def make_protocol_systems(
    *,
    dim: int = 32,
    input_dim: int | None = None,
    public_dim: int = 32,
    budget_ms: float | None = 50.0,
    tier: str = "frozen_real",
) -> list[SystemSpec]:
    """The three-way comparison: E5 flat, transformer-state, and PSS runtime.

    ``dim`` is the feature width of the stream (384 for the real bundles);
    ``input_dim`` is the detector-embedding width the PSS adapter projects into
    its public space (defaults to ``dim``).
    """

    from .adapters.flat import FlatNonPSSAdapter
    from .adapters.pss_runtime import PSSRuntimeAdapter
    from .adapters.transformer import TransformerStateAdapter

    resolved_input_dim = int(dim if input_dim is None else input_dim)
    return [
        SystemSpec("e5-flat", lambda: FlatNonPSSAdapter(dim=dim)),
        SystemSpec("transformer-state", lambda: TransformerStateAdapter(dim=dim)),
        SystemSpec(
            "pss-runtime",
            lambda: PSSRuntimeAdapter(
                input_dim=resolved_input_dim,
                public_dim=public_dim,
                budget_ms=budget_ms,
                tier=tier,
            ),
        ),
    ]


def _stream_summary(stream: Stream) -> dict:
    return {
        "n": len(stream),
        "dim": int(stream.dim),
        "seed": int(stream.seed),
        "categories": list(stream.categories),
    }


def _summarise(rows: list[dict], systems: Sequence[SystemSpec]) -> dict:
    systems_summary: dict[str, dict] = {}
    for spec in systems:
        own = [row for row in rows if row["system"] == spec.name]
        statuses = {row["property"]: row["status"] for row in own}
        counts = {
            status: sum(1 for row in own if row["status"] == status)
            for status in (PASS, FAIL, SKIP)
        }
        systems_summary[spec.name] = {
            "statuses": statuses,
            "counts": counts,
            "smoke_pass": all(
                row["status"] == PASS for row in own if row["smoke"]
            ),
            "properties": [row["property"] for row in own],
        }
    return {"systems": systems_summary}


def run(
    systems: Sequence[SystemSpec],
    properties: Sequence[Property],
    stream: Stream,
    *,
    seed: int = 0,
) -> dict:
    """Evaluate every property against every system for one seed."""

    rows: list[dict] = []
    for spec in systems:
        for prop in properties:
            result = prop.evaluate(spec.factory, stream, seed=seed)
            row = {"system": spec.name, **result.to_dict()}
            rows.append(row)
    return {
        "seed": int(seed),
        "stream": _stream_summary(stream),
        "results": rows,
        "summary": _summarise(rows, systems),
    }


def _aggregate_runs(
    runs: Sequence[dict],
    systems: Sequence[SystemSpec],
    properties: Sequence[Property],
) -> dict:
    systems_summary: dict[str, dict] = {}
    for spec in systems:
        statuses: dict[str, str] = {}
        metrics: dict[str, dict[str, float]] = {}
        smoke_flags: dict[str, bool] = {}
        for prop in properties:
            own = [
                row
                for run_report in runs
                for row in run_report["results"]
                if row["system"] == spec.name and row["property"] == prop.name
            ]
            observed = [row["status"] for row in own]
            if not observed:
                overall = SKIP
            elif all(status == SKIP for status in observed):
                overall = SKIP
            elif all(status == PASS for status in observed):
                overall = PASS
            elif any(status == FAIL for status in observed):
                overall = FAIL
            else:
                overall = "mixed"
            statuses[prop.name] = overall
            smoke_flags[prop.name] = bool(prop.smoke)

            keys = sorted({key for row in own for key in row["metrics"]})
            metrics[prop.name] = {
                key: float(
                    sum(row["metrics"][key] for row in own if key in row["metrics"])
                    / max(1, sum(1 for row in own if key in row["metrics"]))
                )
                for key in keys
            }
        systems_summary[spec.name] = {
            "statuses": statuses,
            "metrics": metrics,
            "smoke": smoke_flags,
            "smoke_pass": all(
                statuses[name] == PASS
                for name, is_smoke in smoke_flags.items()
                if is_smoke
            ),
        }
    return {"systems": systems_summary}


def run_suite(
    systems: Sequence[SystemSpec],
    properties: Sequence[Property],
    stream: Stream,
    *,
    seeds: Sequence[int] = (0,),
    preregistration_expectations: dict[str, dict[str, str]] | None = None,
) -> dict:
    """Run every seed, aggregate statuses, and check the pre-registration."""

    seed_list = [int(seed) for seed in seeds]
    if not seed_list:
        raise ValueError("seeds must be non-empty")
    runs = [run(systems, properties, stream, seed=seed) for seed in seed_list]
    report = {
        "seeds": seed_list,
        "stream": _stream_summary(stream),
        "runs": runs,
        "summary": _aggregate_runs(runs, systems, properties),
    }
    report["preregistration"] = check_preregistration(
        report, expectations=preregistration_expectations
    )
    return report


def _combine_statuses(statuses: Sequence[str]) -> str:
    if not statuses:
        return SKIP
    if all(status == SKIP for status in statuses):
        return SKIP
    if all(status == PASS for status in statuses):
        return PASS
    if any(status == FAIL for status in statuses):
        return FAIL
    return "mixed"


def _aggregate_bundle_summaries(
    per_bundle: dict[str, dict],
    systems: Sequence[SystemSpec],
    properties: Sequence[Property],
) -> dict:
    systems_summary: dict[str, dict] = {}
    for spec in systems:
        statuses: dict[str, str] = {}
        metrics: dict[str, dict[str, float]] = {}
        smoke_flags: dict[str, bool] = {}
        for prop in properties:
            summaries = [
                report["summary"]["systems"][spec.name]
                for report in per_bundle.values()
                if spec.name in report["summary"]["systems"]
            ]
            observed = [
                summary["statuses"][prop.name]
                for summary in summaries
                if prop.name in summary["statuses"]
            ]
            statuses[prop.name] = _combine_statuses(observed)
            smoke_flags[prop.name] = bool(prop.smoke)
            keys = sorted(
                {
                    key
                    for summary in summaries
                    for key in summary["metrics"].get(prop.name, {})
                }
            )
            metrics[prop.name] = {
                key: float(
                    np.mean(
                        [
                            summary["metrics"][prop.name][key]
                            for summary in summaries
                            if key in summary["metrics"].get(prop.name, {})
                        ]
                    )
                )
                for key in keys
            }
        systems_summary[spec.name] = {
            "statuses": statuses,
            "metrics": metrics,
            "smoke": smoke_flags,
            "smoke_pass": all(
                statuses[name] == PASS
                for name, is_smoke in smoke_flags.items()
                if is_smoke
            ),
        }
    return {"systems": systems_summary}


def run_bundle_suite(
    systems: Sequence[SystemSpec],
    properties: Sequence[Property],
    bundle_streams: dict[str, Stream],
    *,
    seeds: Sequence[int] = (0,),
    preregistration_expectations: dict[str, dict[str, str]] | None = None,
) -> dict:
    """Run the same suite on every bundle stream and aggregate the results.

    Each bundle gets its own report (the runtime is reset per bundle), so a
    tracker's frame indices never leak across bundles.  A property passes the
    aggregate only if it passes on every bundle.
    """

    if not bundle_streams:
        raise ValueError("bundle_streams must be non-empty")
    per_bundle = {
        name: run_suite(
            systems,
            properties,
            stream,
            seeds=seeds,
            preregistration_expectations=preregistration_expectations,
        )
        for name, stream in bundle_streams.items()
    }
    checks = {name: report["preregistration"] for name, report in per_bundle.items()}
    mismatches = [
        f"{name}: {mismatch}"
        for name, check in checks.items()
        for mismatch in check.get("mismatches", [])
    ]
    return {
        "seeds": [int(seed) for seed in seeds],
        "stream": {
            "kind": "bundles",
            "bundles": list(per_bundle),
            "n": int(sum(report["stream"]["n"] for report in per_bundle.values())),
            "dim": sorted({report["stream"]["dim"] for report in per_bundle.values()}),
        },
        "bundles": {
            name: {
                "stream": report["stream"],
                "summary": report["summary"],
                "preregistration": report["preregistration"],
            }
            for name, report in per_bundle.items()
        },
        "summary": _aggregate_bundle_summaries(per_bundle, systems, properties),
        "preregistration": {
            "ok": all(check.get("ok") for check in checks.values()),
            "mismatches": mismatches,
        },
    }


def _fmt_metrics(metrics: dict[str, float]) -> str:
    return ", ".join(f"{key}={value:.3f}" for key, value in sorted(metrics.items()))


def _report_markdown(report: dict) -> str:
    stream = report.get("stream", {})
    dim = stream.get("dim", "?")
    if isinstance(dim, list):
        dim = "/".join(str(value) for value in dim)
    lines = [
        "# pss_eval: property report",
        "",
        f"> seeds: {', '.join(str(seed) for seed in report['seeds'])}; "
        f"stream: n={stream.get('n', '?')}, dim={dim}"
        + (
            f"; bundles: {', '.join(stream.get('bundles', []))}"
            if stream.get("kind") == "bundles"
            else ""
        ),
        "",
        "## Aggregate",
        "",
        "| system | property | smoke | status | metrics |",
        "|---|---|---|---|---|",
    ]
    for system, summary in report["summary"]["systems"].items():
        for prop, status in summary["statuses"].items():
            lines.append(
                f"| {system} | {prop} | {summary['smoke'][prop]} | {status} | "
                f"{_fmt_metrics(summary['metrics'].get(prop, {}))} |"
            )

    if stream.get("kind") == "bundles":
        lines += ["", "## Per-bundle statuses", ""]
        bundle_names = list(report.get("bundles", {}))
        header = "| system | property | " + " | ".join(bundle_names) + " |"
        lines.append(header)
        lines.append("|---|---|" + "---|" * len(bundle_names))
        for system, summary in report["summary"]["systems"].items():
            for prop in summary["statuses"]:
                cells = [
                    report["bundles"][name]["summary"]["systems"][system][
                        "statuses"
                    ].get(prop, "?")
                    for name in bundle_names
                ]
                lines.append(
                    f"| {system} | {prop} | " + " | ".join(cells) + " |"
                )

    lines += ["", "## Pre-registration", ""]
    check = report.get("preregistration", {})
    lines.append(f"ok: **{bool(check.get('ok'))}**")
    for mismatch in check.get("mismatches", []):
        lines.append(f"- {mismatch}")
    lines += [
        "",
        "Smoke properties are protocol hygiene checks.  `contract` and",
        "`swap_rollback` are expected to fail or be skipped for contract-free",
        "external systems; `budget` / `identity_retention` skip when the system",
        "or stream lacks the required capability.",
        "",
    ]
    return "\n".join(lines)


def write_report(report: dict, out_dir: str | Path) -> dict[str, str]:
    """Write ``report.json`` and ``REPORT.md``; return their absolute paths."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "report.json"
    md_path = out / "REPORT.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md_path.write_text(_report_markdown(report), encoding="utf-8")
    return {
        "json": str(json_path.resolve()),
        "markdown": str(md_path.resolve()),
    }
