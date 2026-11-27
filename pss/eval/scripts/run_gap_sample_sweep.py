"""Gap: sample-size sweep for the calibration robustness pipeline.

Question: how much labelled data does each mechanism need?

The answer is expressed per class, because that is what drives the statistics:
a window of N samples over K classes gives N/K examples per class, and the
per-class signals (single-class attacks, coverage) are the first to break.

Outputs, per window size:
- clean calibration error of the confidence map on a held-out test split;
- one-window detection rate for label poisoning at 10% and 20%;
- signal-to-noise ratio of the targeted attacks, in within-window standard
  deviations.  CUSUM slack is 0.4 sd, so ``snr < 0.4`` means undetectable by
  the streaming detector at that window size.
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
    _vote_weights,
    bootstrap_params,
    evaluate_params,
    fit_confidence_maps,
    fit_params,
    score_samples,
    source_health,
)
from pss.projection.cusum import calibrate_cusum
from pss.projection.online import (
    poison_agree_with_prediction,
    poison_labels,
    poison_single_class,
)

SIZES = (24, 48, 96, 128)
SEEDS = 5
DRAWS = 40


def _load(args) -> dict:
    if args.source == "synthetic":
        from pss.eval.scripts.run_gap_online_calibration import synthetic_context

        return synthetic_context(seed=args.seed)
    from pss.eval.scripts.run_gap_online_calibration import load_real_context

    return load_real_context(
        args.data, args.experts, args.projections,
        device=args.device, max_per_class=args.max_per_class,
    )


def _batch(ctx, idx, labels, embeddings=None) -> CalibrationBatch:
    idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
    modalities = ctx["modalities"]
    protos = {m: np.asarray(ctx["prototypes"][m], dtype=np.float64) for m in modalities}
    embs = embeddings or {
        m: np.asarray(ctx["Z"][m][idx], dtype=np.float64) for m in modalities
    }
    return CalibrationBatch(
        embeddings=embs,
        prototypes=protos,
        labels=np.asarray(labels, dtype=np.int64),
        sources=["s0"] * len(idx),
        sample_ids=[f"s{int(i)}" for i in idx],
    )


def _agreement(batch, params) -> float:
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


def _fit(window):
    seed = bootstrap_params(window)
    return fit_params(window, score_samples(window, seed, source_health(window)))


def run(args) -> dict:
    ctx = _load(args)
    from pss.eval.scripts.run_gap_calibration_robust import _split

    (fit_idx, y_fit), (hold_idx, y_hold), (gate_idx, y_gate) = _split(ctx)
    modalities = ctx["modalities"]
    n_classes = len(ctx["classes"])
    protos = {m: np.asarray(ctx["prototypes"][m], dtype=np.float64) for m in modalities}

    # Pool = everything except the frozen gate set.
    pool = np.concatenate([np.asarray(fit_idx), np.asarray(hold_idx)])
    y_pool = np.concatenate([np.asarray(y_fit), np.asarray(y_hold)])
    gate = _batch(ctx, gate_idx, y_gate)
    test = _batch(ctx, ctx["test_idx"], ctx["y_test"])

    rows = []
    for n in SIZES:
        n = int(min(n, len(pool)))
        clean_ece: list[float] = []
        det10: list[bool] = []
        det20: list[bool] = []
        agree_delta: list[float] = []
        single_delta: list[float] = []
        clean_agreement: list[float] = []

        for seed in range(SEEDS):
            rng = np.random.default_rng([int(args.seed), n, seed])
            # Bootstrap when the window equals the pool: sampling without
            # replacement would give every draw the same set and zero variance.
            replace = n >= len(pool)
            pick = rng.choice(len(pool), size=n, replace=replace)
            idx, labels = pool[pick], y_pool[pick]
            window = _batch(ctx, idx, labels)
            params = _fit(window)
            maps = fit_confidence_maps(params, window)
            metrics = evaluate_params(params, test, confidence_maps=maps)
            clean_ece.append(metrics["ece_map_fused"])

            for frac, bucket in ((0.10, det10), (0.20, det20)):
                poisoned = _batch(
                    ctx, idx, poison_labels(labels, frac, n_classes, rng=rng)
                )
                pipeline = RobustCalibrationPipeline(params=params, gate_set=gate)
                record = pipeline.step(poisoned)
                bucket.append(record["state"] != "normal")

        # Signal-to-noise of targeted attacks at this window size.
        ref_rng = np.random.default_rng([int(args.seed), n, 99])
        embs_pool = {
            m: np.asarray(ctx["Z"][m][pool], dtype=np.float64) for m in modalities
        }
        for _ in range(DRAWS):
            pick = ref_rng.choice(len(pool), size=n, replace=n >= len(pool))
            idx, labels = pool[pick], y_pool[pick]
            embs = {m: embs_pool[m][pick] for m in modalities}
            base = _batch(ctx, idx, labels, embeddings=embs)
            params = _fit(base)
            clean_agreement.append(_agreement(base, params))
            agree_lab = poison_agree_with_prediction(
                embs, protos, labels, 0.10, rng=ref_rng
            )
            single_lab = poison_single_class(labels, 0.20, 0, n_classes, rng=ref_rng)
            agree_delta.append(
                _agreement(_batch(ctx, idx, agree_lab, embeddings=embs), params)
                - clean_agreement[-1]
            )
            single_delta.append(
                _agreement(_batch(ctx, idx, single_lab, embeddings=embs), params)
                - clean_agreement[-1]
            )

        _, _, _ = calibrate_cusum(np.asarray(clean_agreement))
        sd = float(np.std(clean_agreement, ddof=1)) if len(clean_agreement) > 1 else 0.0
        rows.append(
            {
                "window": n,
                "per_class": n / float(n_classes),
                "clean_ece_map": float(np.mean(clean_ece)),
                "clean_ece_std": float(np.std(clean_ece)),
                "flip10_detect_rate": float(np.mean(det10)),
                "flip20_detect_rate": float(np.mean(det20)),
                "agreement_sd": sd,
                "agree_snr": float(abs(np.mean(agree_delta)) / sd) if sd > 0 else float("nan"),
                "single_snr": float(abs(np.mean(single_delta)) / sd) if sd > 0 else float("nan"),
            }
        )

    return {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "source": ctx.get("kind", "unknown"),
        "classes": n_classes,
        "pool": int(len(pool)),
        "seeds": SEEDS,
        "draws": DRAWS,
        "rows": rows,
    }


def _report(metrics: dict) -> str:
    lines = [
        "# Gap: sample-size sweep",
        "",
        f"> generated: {metrics['generated']}",
        f"> source: {metrics['source']}",
        f"> classes: {metrics['classes']}  pool: {metrics['pool']}  "
        f"seeds: {metrics['seeds']}  draws: {metrics['draws']}",
        "",
        "CUSUM slack is 0.4 within-window sd, so `snr < 0.4` means the streaming",
        "detector cannot separate the attack from window noise at that size.",
        "",
        "| window | per class | ECE(map) | flip10 detect | flip20 detect | agreement sd | agree snr | single snr |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in metrics["rows"]:
        lines.append(
            f"| {r['window']} | {r['per_class']:.1f} | "
            f"{r['clean_ece_map']:.4f}+/-{r['clean_ece_std']:.4f} | "
            f"{r['flip10_detect_rate']:.2f} | {r['flip20_detect_rate']:.2f} | "
            f"{r['agreement_sd']:.4f} | {r['agree_snr']:.2f} | {r['single_snr']:.2f} |"
        )
    lines.append("")
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
    parser.add_argument("--out", default="reports/claim_gaps/sample_sweep")
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
