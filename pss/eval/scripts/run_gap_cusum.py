"""CUSUM detection of slow targeted attacks across streaming windows.

A single window cannot separate a ~1.5-point attack from ~2-point noise.  This
script simulates a stream of windows and measures how many windows a CUSUM needs,
plus the clean false-alarm rate on the same number of windows.

Usage:
    python -m pss.eval.scripts.run_gap_cusum --source real --max-per-class 12
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pss.projection.calibration_pipeline import (
    CalibrationBatch,
    RobustCalibrationPipeline,
    _consensus_with_weights,
    _modalities,
    _per_class_mismatch,
    _source_mismatch,
    _vote_weights,
    bootstrap_params,
    fit_params,
    score_samples,
    source_health,
)
from pss.projection.cusum import (
    CusumBank,
    CusumDetector,
    calibrate_cusum,
    calibrate_per_class,
)
from pss.projection.source_registry import SourceRegistry
from pss.projection.online import (
    poison_agree_with_prediction,
    poison_labels,
    poison_single_class,
)

WINDOWS = 20

SCENARIOS = [
    {"name": "clean", "kind": "none"},
    {"name": "random_label_flip@0.05", "kind": "label_flip", "fraction": 0.05},
    {"name": "targeted_agree@0.10", "kind": "agree", "fraction": 0.10},
    {"name": "targeted_single_class@0.20", "kind": "single", "fraction": 0.20},
    {"name": "drift_sigma@0.30", "kind": "drift", "sigma": 0.30},
    {"name": "source_outlier_stream", "kind": "source", "fraction": 0.35},
]


def _load(args) -> dict:
    if args.source == "synthetic":
        from pss.eval.scripts.run_gap_online_calibration import synthetic_context

        return synthetic_context(seed=args.seed)
    from pss.eval.scripts.run_gap_online_calibration import load_real_context

    return load_real_context(
        args.data, args.experts, args.projections,
        device=args.device, max_per_class=args.max_per_class,
    )


def _split(ctx):
    from pss.eval.scripts.run_gap_calibration_robust import _split as split

    return split(ctx)


def _batch(ctx, idx, labels, embeddings=None, sources=None) -> CalibrationBatch:
    idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
    modalities = ctx["modalities"]
    protos = {m: np.asarray(ctx["prototypes"][m], dtype=np.float64) for m in modalities}
    if embeddings is None:
        embs = {m: np.asarray(ctx["Z"][m][idx], dtype=np.float64) for m in modalities}
    else:
        embs = embeddings
    return CalibrationBatch(
        embeddings=embs,
        prototypes=protos,
        labels=np.asarray(labels, dtype=np.int64),
        sources=list(sources) if sources is not None else ["s0"] * len(idx),
        sample_ids=[f"w{int(i)}" for i in idx],
    )


def _agreement(batch: CalibrationBatch, params) -> float:
    consensus, _ = _consensus_with_weights(
        batch, _vote_weights(params, _modalities(batch))
    )
    if consensus is None:
        return float("nan")
    labels = np.asarray(batch.labels, dtype=np.int64)
    valid = labels >= 0
    if not valid.any():
        return float("nan")
    return float(np.mean(labels[valid] == consensus[valid]))


def _stable_seed(name: str) -> int:
    """Deterministic across runs (Python's hash() is randomised per process)."""
    return sum((i + 1) * ord(ch) for i, ch in enumerate(name)) % 10_000


def _scenario_windows(ctx, spec, fit_idx, y_fit, rng):
    modalities = ctx["modalities"]
    protos = {m: np.asarray(ctx["prototypes"][m], dtype=np.float64) for m in modalities}
    embs = {m: np.asarray(ctx["Z"][m][fit_idx], dtype=np.float64) for m in modalities}
    labels = np.asarray(y_fit, dtype=np.int64).copy()
    n = len(labels)
    sources = None
    kind = spec["kind"]

    if kind == "none":
        return _batch(ctx, fit_idx, labels)
    if kind == "label_flip":
        labels = poison_labels(labels, spec["fraction"], len(ctx["classes"]), rng=rng)
    elif kind == "agree":
        labels = poison_agree_with_prediction(embs, protos, labels, spec["fraction"], rng=rng)
    elif kind == "single":
        labels = poison_single_class(labels, spec["fraction"], 0, len(ctx["classes"]), rng=rng)
    elif kind == "drift":
        embs = {m: embs[m] + spec["sigma"] * rng.normal(size=embs[m].shape) for m in modalities}
    elif kind == "source":
        good = int(round(n * 0.4))
        sources = ["good"] * good + ["bad"] * (n - good)
        bad_idx = np.arange(good, n)
        flips = rng.choice(
            bad_idx, size=max(1, int(round(spec["fraction"] * len(bad_idx)))), replace=False
        )
        labels = labels.copy()
        labels[flips] = (labels[flips] + 1) % len(ctx["classes"])
    return _batch(ctx, fit_idx, labels, embeddings=embs, sources=sources)


def run(args) -> dict:
    ctx = _load(args)
    (fit_idx, y_fit), (hold_idx, y_hold), (gate_idx, y_gate) = _split(ctx)
    modalities = ctx["modalities"]

    clean_fit = _batch(ctx, fit_idx, y_fit)
    gate = _batch(ctx, gate_idx, y_gate)
    seed_params = bootstrap_params(clean_fit)
    incumbent = fit_params(
        clean_fit, score_samples(clean_fit, seed_params, source_health(clean_fit))
    )

    # Calibrate the CUSUM on clean windows of the same size.
    rng0 = np.random.default_rng([int(args.seed), 999])
    clean_vals = []
    clean_mixups = []
    clean_sources = []
    for _ in range(50):
        pick = rng0.integers(0, len(fit_idx), size=len(fit_idx))
        idx = np.asarray(fit_idx)[pick]
        lab = np.asarray(y_fit)[pick]
        b = _batch(ctx, idx, lab)
        clean_vals.append(_agreement(b, incumbent))
        clean_mixups.append(_per_class_mismatch(b, incumbent))
        # Clean windows with the same two-source layout used by the source
        # scenario, so both source detectors are calibrated.
        n = len(idx)
        good = int(round(n * 0.4))
        srcs = ["good"] * good + ["bad"] * (n - good)
        clean_sources.append(
            _source_mismatch(_batch(ctx, idx, lab, sources=srcs), incumbent)
        )
    reference, slack, threshold = calibrate_cusum(np.asarray(clean_vals))

    rows = []
    for spec in SCENARIOS:
        bank = CusumBank(
            agreement_down=CusumDetector(reference, slack, threshold, direction=-1.0),
            agreement_up=CusumDetector(reference, slack, threshold, direction=+1.0),
        )
        per_class = calibrate_per_class(clean_mixups)
        registry = SourceRegistry.calibrate(clean_sources)
        pipeline = RobustCalibrationPipeline(
            params=incumbent,
            gate_set=gate,
            cusum=bank,
            per_class_cusum=per_class,
            source_registry=registry,
        )
        first_alarm = None
        first_per_class = None
        first_quarantine = None
        fit_idx_arr = np.asarray(fit_idx)
        y_fit_arr = np.asarray(y_fit)
        for w in range(WINDOWS):
            rng = np.random.default_rng([int(args.seed), _stable_seed(spec["name"]), w])
            # Bootstrap the window: without resampling there is no window-to-window
            # noise and the CUSUM calibration degenerates (sd = 0).
            pick = rng.integers(0, len(fit_idx_arr), size=len(fit_idx_arr))
            batch = _scenario_windows(
                ctx, spec, fit_idx_arr[pick], y_fit_arr[pick], rng
            )
            record = pipeline.step(batch)
            if record["cusum_alarm"] and first_alarm is None:
                first_alarm = w + 1
            if record["per_class_alarm"] and first_per_class is None:
                first_per_class = w + 1
            if record["quarantined_sources"] and first_quarantine is None:
                first_quarantine = w + 1
        rows.append(
            {
                "scenario": spec["name"],
                "windows": WINDOWS,
                "alarms": int(bank.agreement_down.alarms + bank.agreement_up.alarms),
                "first_alarm": first_alarm,
                "detected": first_alarm is not None,
                "per_class_first_alarm": first_per_class,
                "per_class_detected": first_per_class is not None,
                "per_class_alarms": int(sum(per_class.alarms.values())),
                "first_quarantine": first_quarantine,
                "quarantined_at_end": registry.quarantined(),
            }
        )

    return {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "source": ctx.get("kind", "unknown"),
        "cusum": {
            "reference": reference,
            "slack": slack,
            "threshold": threshold,
            "clean_windows": 50,
        },
        "rows": rows,
    }


def _report(metrics: dict) -> str:
    c = metrics["cusum"]
    lines = [
        "# Gap: CUSUM detection across streaming windows",
        "",
        f"> generated: {metrics['generated']}",
        f"> source: {metrics['source']}",
        f"> reference={c['reference']:.4f}  slack={c['slack']:.4f}  "
        f"threshold={c['threshold']:.4f}  (calibrated on {c['clean_windows']} clean windows)",
        "",
        "| scenario | windows | global first | per-class first | first quarantine | quarantined at end |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for r in metrics["rows"]:
        lines.append(
            f"| {r['scenario']} | {r['windows']} | {r['first_alarm']} | "
            f"{r['per_class_first_alarm']} | {r['first_quarantine']} | "
            f"{r['quarantined_at_end']} |"
        )
    lines += [
        "",
        "A single window cannot separate the targeted attacks from noise; the",
        "question is how many windows the CUSUM needs, and whether clean/drift",
        "stay silent.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("synthetic", "real"), default="synthetic")
    parser.add_argument("--data", default="data/category_pretrain")
    parser.add_argument("--experts", default="checkpoints/category_pretrain_experts")
    parser.add_argument("--projections", default="checkpoints/category_pretrain_projection")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="reports/claim_gaps/cusum")
    args = parser.parse_args()

    metrics = run(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / "REPORT.md").write_text(_report(metrics), encoding="utf-8")
    print(_report(metrics))


if __name__ == "__main__":
    main()
