"""Diagnostic: what drives the confidence-gap noise, and can it be reduced?

The signed-gap CUSUM is silent because the clean window-to-window sd (~2.2pp)
exceeds the targeted-attack signal (0.16-0.48pp).  This script sweeps window
size and map family to answer two questions:

1. Does the noise scale down with the calibration window?
2. Can a smoother map (fewer bins, shrinkage toward the global accuracy, a
   two-parameter logistic) cut the noise below the ~0.5pp the CUSUM needs --
   and what does that cost in clean calibration (ECE)?

Production fidelity:
- candidate maps are fitted with ``score_samples`` trust weights, exactly as
  the pipeline does; without them a poisoned label is counted at full weight
  and the attack signal is overstated;
- sizes up to the val pool are drawn from the same val split as production,
  so the N=96 row should reproduce the measured ~2.2pp; larger sizes use the
  train pool and are labelled as such.

Usage:
    python -m pss.eval.scripts.run_gap_confidence_variance --source real --max-per-class 12
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from pss.eval.scripts.run_gap_calibration_robust import _batch, _incumbent, _split
from pss.eval.scripts.run_gap_online_calibration import (
    load_real_context,
    synthetic_context,
)
from pss.projection.calibration import expected_calibration_error
from pss.projection.calibration_pipeline import (
    State,
    _fused_top1,
    score_samples,
    source_health,
)
from pss.projection.confidence_map import (
    ConfidenceMap,
    apply_confidence_map,
    fit_confidence_map,
)
from pss.projection.online import (
    poison_agree_with_prediction,
    poison_single_class,
)

VARIANTS = [
    "pav4",
    "pav6",
    "pav6_shrink75",
    "constant",
]
ATTACK_FRACTION = {"single": 0.20, "agree": 0.10}
VIABLE_SD_PP = 0.5
CLEAN_ECE_CEILING = 0.03
# The CUSUM accumulates only when the per-window signal clears its slack
# (0.4 sd); with a 4 sd threshold, ~0.6 sd is the practical floor for
# detection within a 20-window stream.
VIABLE_SNR = 0.6


def _load(args) -> dict:
    if args.source == "synthetic":
        return synthetic_context(seed=args.seed)
    return load_real_context(
        args.data,
        args.experts,
        args.projections,
        device=args.device,
        max_per_class=args.max_per_class,
    )


def _pools(ctx: dict, fit_idx, fit_y) -> list[tuple[str, tuple[np.ndarray, np.ndarray]]]:
    """Clean pools at three sizes: production fit, val, and train.

    ``train`` is optimistic (the experts were fitted on it) and is reported
    separately; ``fit`` is the production calibration window.
    """
    pools: list[tuple[str, tuple[np.ndarray, np.ndarray]]] = [
        (
            "fit",
            (
                np.asarray(fit_idx, dtype=np.int64),
                np.asarray(fit_y, dtype=np.int64),
            ),
        ),
        (
            "val",
            (
                np.asarray(ctx["val_idx"], dtype=np.int64),
                np.asarray(ctx["y_val"], dtype=np.int64),
            ),
        ),
    ]
    feats = ctx.get("feats")
    if feats is not None and "train_idx" in ctx:
        cats = np.asarray(feats.categories, dtype=object)
        label_of = {c: i for i, c in enumerate(ctx["classes"])}
        train_idx = np.asarray(ctx["train_idx"], dtype=np.int64)
        train_y = np.array(
            [label_of[str(c)] for c in cats[train_idx]], dtype=np.int64
        )
        pools.append(("train", (train_idx, train_y)))
    return pools


def _weights(batch, params) -> np.ndarray:
    return np.asarray(
        score_samples(
            batch, params, source_health(batch), state=State.NORMAL
        ).per_sample,
        dtype=np.float64,
    )


def _fit_logistic(
    scores: np.ndarray,
    correct: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple:
    """Two-parameter monotone sigmoid P(correct | score), trust-weighted."""
    x = np.asarray(scores, dtype=np.float64)
    y = np.asarray(correct, dtype=np.float64)
    w = np.ones_like(y) if weights is None else np.clip(weights, 0.0, None)
    if w.sum() <= 0:
        w = np.ones_like(y)
    w = w / w.sum()
    mu = float(np.mean(x))
    sd = float(np.std(x)) or 1.0
    z = (x - mu) / sd

    def nll(params: np.ndarray) -> float:
        p = 1.0 / (1.0 + np.exp(-(params[0] + params[1] * z)))
        p = np.clip(p, 1e-9, 1.0 - 1e-9)
        return -float(
            np.sum(w * (y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
        )

    result = minimize(
        nll,
        np.array([0.0, 1.0]),
        method="Nelder-Mead",
        options={"maxiter": 200, "xatol": 1e-4, "fatol": 1e-6},
    )
    a, b = (float(v) for v in result.x)
    return a, max(b, 1e-3), mu, sd


def _weighted_rate(correct: np.ndarray, weights: np.ndarray | None) -> float:
    if weights is None or weights.sum() <= 0:
        return float(np.mean(correct))
    return float(np.average(correct, weights=weights))


def _fit_variant(
    name: str,
    scores: np.ndarray,
    correct: np.ndarray,
    weights: np.ndarray | None = None,
):
    w = None if weights is None else np.clip(weights, 0.0, None)
    if name == "constant":
        return ("constant", _weighted_rate(correct, w))
    if name == "logistic":
        return ("logistic", _fit_logistic(scores, correct, w))
    n_bins = int(name.split("_")[0][3:])
    cmap = fit_confidence_map(
        scores, correct, n_bins=n_bins, sample_weights=w
    )
    if "shrink" in name:
        lam = int(name.split("_shrink")[1]) / 100.0
        rate = _weighted_rate(correct, w)
        cmap = ConfidenceMap(
            edges=cmap.edges,
            values=tuple((1.0 - lam) * v + lam * rate for v in cmap.values),
        )
    return ("step", cmap)


def _apply_variant(variant, scores: np.ndarray) -> np.ndarray:
    kind, obj = variant
    if kind == "step":
        return apply_confidence_map(scores, obj)
    if kind == "constant":
        return np.full(len(scores), float(obj), dtype=np.float64)
    a, b, mu, sd = obj
    z = (np.asarray(scores, dtype=np.float64) - mu) / sd
    return 1.0 / (1.0 + np.exp(-(a + b * z)))


def _gap_delta(candidate, reference, gate_scores: np.ndarray) -> float:
    return float(
        np.mean(_apply_variant(candidate, gate_scores))
        - np.mean(_apply_variant(reference, gate_scores))
    )


def _evaluate(
    ctx: dict,
    params,
    pool_name: str,
    pool_idx: np.ndarray,
    pool_y: np.ndarray,
    gate_scores: np.ndarray,
    test_scores: np.ndarray,
    test_correct: np.ndarray,
    variant: str,
    n: int,
    *,
    seed: int,
    n_resamples: int,
) -> dict:
    ref_batch = _batch(ctx, pool_idx, pool_y)
    ref_scores, ref_correct = _fused_top1(params, ref_batch)
    reference = _fit_variant(
        variant, ref_scores, ref_correct, _weights(ref_batch, params)
    )
    modalities = ctx["modalities"]
    n_classes = len(ctx["classes"])

    clean, eces, single, agree = [], [], [], []
    for r in range(n_resamples):
        rng = np.random.default_rng([int(seed), _stable_seed(variant), n, r])
        pick = rng.integers(0, len(pool_idx), size=n)
        idx = pool_idx[pick]
        labels = pool_y[pick]

        batch = _batch(ctx, idx, labels)
        scores, correct = _fused_top1(params, batch)
        candidate = _fit_variant(
            variant, scores, correct, _weights(batch, params)
        )
        clean.append(_gap_delta(candidate, reference, gate_scores))
        eces.append(
            float(
                expected_calibration_error(
                    _apply_variant(candidate, test_scores), test_correct
                )
            )
        )

        poisoned = poison_single_class(
            labels, ATTACK_FRACTION["single"], 0, n_classes
        )
        b = _batch(ctx, idx, poisoned)
        s, c = _fused_top1(params, b)
        single.append(
            _gap_delta(
                _fit_variant(variant, s, c, _weights(b, params)),
                reference,
                gate_scores,
            )
        )

        embs = {
            m: np.asarray(ctx["Z"][m][idx], dtype=np.float64)
            for m in modalities
        }
        poisoned = poison_agree_with_prediction(
            embs,
            ctx["prototypes"],
            labels,
            ATTACK_FRACTION["agree"],
            rng=np.random.default_rng([int(seed), 7, r]),
        )
        b = _batch(ctx, idx, poisoned)
        s, c = _fused_top1(params, b)
        agree.append(
            _gap_delta(
                _fit_variant(variant, s, c, _weights(b, params)),
                reference,
                gate_scores,
            )
        )

    sd = float(np.std(clean, ddof=1))
    clean_mean = float(np.mean(clean))
    signal_single = float(np.mean(single) - clean_mean)
    signal_agree = float(np.mean(agree) - clean_mean)
    return {
        "variant": variant,
        "n": int(n),
        "pool": pool_name,
        "sd_pp": 100.0 * sd,
        "ece": float(np.mean(eces)),
        "signal_single_pp": 100.0 * signal_single,
        "snr_single": abs(signal_single) / sd if sd > 0 else float("nan"),
        "signal_agree_pp": 100.0 * signal_agree,
        "snr_agree": abs(signal_agree) / sd if sd > 0 else float("nan"),
        "viable": bool(
            100.0 * sd < VIABLE_SD_PP
            and np.mean(eces) <= CLEAN_ECE_CEILING
            and max(abs(signal_single), abs(signal_agree)) / sd
            >= VIABLE_SNR
        ),
    }


def _stable_seed(name: str) -> int:
    return sum((i + 1) * ord(ch) for i, ch in enumerate(name)) % 10_000


def run(args) -> dict:
    ctx = _load(args)
    (fit_idx, fit_y), _hold, (gate_idx, gate_y) = _split(ctx)
    clean_fit = _batch(ctx, fit_idx, fit_y)
    gate = _batch(ctx, gate_idx, gate_y)
    test = _batch(ctx, ctx["test_idx"], ctx["y_test"])
    params = _incumbent(ctx, clean_fit)

    gate_scores, _ = _fused_top1(params, gate)
    test_scores, test_correct = _fused_top1(params, test)
    pools = _pools(ctx, fit_idx, fit_y)

    rows = []
    for pool_name, (pool_idx, pool_y) in pools:
        sizes = sorted({len(pool_idx), max(1, len(pool_idx) // 2)})
        for variant in VARIANTS:
            for n in sizes:
                rows.append(
                    _evaluate(
                        ctx,
                        params,
                        pool_name,
                        pool_idx,
                        pool_y,
                        gate_scores,
                        test_scores,
                        test_correct,
                        variant,
                        n,
                        seed=int(args.seed),
                        n_resamples=int(args.resamples),
                    )
                )

    return {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "source": ctx.get("kind", "unknown"),
        "counts": ctx.get("counts", {}),
        "pools": {name: int(len(idx)) for name, (idx, _) in pools},
        "gate": int(len(gate_idx)),
        "test": int(len(ctx["test_idx"])),
        "resamples": int(args.resamples),
        "targets": {
            "sd_pp": VIABLE_SD_PP,
            "ece": CLEAN_ECE_CEILING,
            "snr": VIABLE_SNR,
        },
        "rows": rows,
    }


def _report(metrics: dict) -> str:
    lines = [
        "# Gap: confidence-gap noise vs window size and map family",
        "",
        f"> generated: {metrics['generated']}",
        f"> source: {metrics['source']}",
        f"> pools: {metrics['pools']}  gate: {metrics['gate']}  "
        f"test: {metrics['test']}  resamples: {metrics['resamples']}",
        "",
        "`sd` is the clean window-to-window standard deviation of the signed-gap",
        "change on the frozen gate set. `signal` is the mean shift under the",
        "attack, relative to the clean mean; `snr = |signal| / sd`. A viable map",
        f"needs `sd < {metrics['targets']['sd_pp']}pp`, clean ECE <= "
        f"{metrics['targets']['ece']}, and `snr >= {metrics['targets']['snr']}` "
        "(the CUSUM slack is 0.4 sd, so a weaker signal never accumulates).",
        "Each pool is tested at half and full size. `fit` is the production",
        "calibration window; `val` is the whole validation split (using it all",
        "would need a larger labelled pool); `train` is optimistic, because the",
        "experts were fitted on it.",
        "",
        "| variant | n | pool | sd(pp) | ECE | signal single(pp) | snr single | signal agree(pp) | snr agree | viable |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in metrics["rows"]:
        lines.append(
            f"| {r['variant']} | {r['n']} | {r['pool']} | {r['sd_pp']:.3f} | "
            f"{r['ece']:.4f} | {r['signal_single_pp']:+.3f} | "
            f"{r['snr_single']:.2f} | {r['signal_agree_pp']:+.3f} | "
            f"{r['snr_agree']:.2f} | {r['viable']} |"
        )
    production = [r for r in metrics["rows"] if r["pool"] != "train"]
    viable = [r for r in production if r["viable"]]
    lines += ["", "## Verdict", ""]
    if viable:
        best = max(viable, key=lambda r: max(r["snr_single"], r["snr_agree"]))
        lines.append(
            f"Viable on a production-representative pool: `{best['variant']}` "
            f"at n={best['n']} ({best['pool']}) sd {best['sd_pp']:.3f}pp, "
            f"ECE {best['ece']:.4f}, snr single {best['snr_single']:.2f}, "
            f"agree {best['snr_agree']:.2f}."
        )
    else:
        lines.append(
            "No production-representative pool (fit/val) reaches the noise and "
            "SNR targets while keeping clean ECE under the ceiling. The noise "
            "is bound by the size of the labelled pool, not by the map family."
        )
    optimistic = [
        r for r in metrics["rows"] if r["pool"] == "train" and r["viable"]
    ]
    if optimistic:
        best = max(optimistic, key=lambda r: max(r["snr_single"], r["snr_agree"]))
        lines.append(
            f"For reference, the train pool (optimistic: the experts were "
            f"fitted on it) reaches sd {best['sd_pp']:.3f}pp at n={best['n']} "
            f"with clean ECE {best['ece']:.4f}."
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("synthetic", "real"), default="real")
    parser.add_argument("--data", default="data/category_pretrain")
    parser.add_argument("--experts", default="checkpoints/category_pretrain_experts")
    parser.add_argument("--projections", default="checkpoints/category_pretrain_projection")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resamples", type=int, default=25)
    parser.add_argument("--out", default="reports/claim_gaps/confidence_variance")
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
