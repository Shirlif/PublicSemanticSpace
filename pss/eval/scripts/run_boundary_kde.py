"""Boundary benchmark, KDE/mixture posterior: the exact empirical-Bayes rule.

The grid Bayes boundary reports a stress failure (near-worst ~0.11) while the
information ceiling reports a high overall (~0.92).  This stage evaluates the
exact empirical-Bayes rule on genuinely noisy observations to separate the
approximation from the information.  The answer is that the exact KDE matches
the grid on the same inputs, including the near cells: the approximation was
not the cause, and the ceiling's high overall comes from the diagonal majority,
not from a recoverable near cell under the raw prior.

The rule is the exact empirical-Bayes weighted vote over train-pair particles:

    p(k | observed) ∝ exp(-||observed - prior_k||^2 / (2 sigma^2))

Same splits, same ``eps=0.02``, same pair labels/types and the same metric
definitions as ``run_boundary_bayes.py``.  It evaluates the contract noise
(``sigma=0.01``, the pre-registered level) and the stress noise
(``sigma=0.04``, reported but not judged) over three noise seeds.

Noise convention: ``--noise-space object`` (default) matches
``run_boundary_bench`` -- each object coordinate gets ``N(0, sigma)``, so the
pair difference (and the KDE kernel) uses ``sigma*sqrt(2)``.
``--noise-space difference`` matches ``run_boundary_ceiling``, which adds
``sigma`` directly to the difference.  The two are different noise levels and
must not be compared as if they were the same run.

Three arms are reported, all on the *same* noisy observations:

- ``kde_raw``: the specified empirical-Bayes rule with the raw train prior.
- ``kde_type_balanced``: diagnostic.  Same kernel, but prior particles carry
  equal mass per pair type.  This answers whether a low near cell is a
  representation problem or a prior/cost imbalance problem.
- ``grid_noisy``: diagnostic.  The existing ``GridBayesBoundary`` re-evaluated
  on the same noisy observations.  The published ``boundary_bayes.json`` fed
  it *clean* coordinates and used sigma only as a prior smoothing width, so
  its contract near-worst is not comparable to the bench arms; this arm makes
  the comparison apples-to-apples without modifying that script.

The KDE decision is the empirical-Bayes decision given its prior and
likelihood, so its per-cell accuracy is a per-cell ceiling.  With the raw
prior the ceiling at stress is near zero because near-aligned pairs are only
~3.2% of the prior; the type-balanced diagnostic shows the information is
still in the coordinates, which is the prior/cost finding.

Example
-------
python -u -m pss.eval.scripts.run_boundary_kde \
  --out eval/results/boundary_bench/kde
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pss._console import ensure_utf8
from pss.eval.readout_reasoning_preregistration import (
    check_readout_reasoning_preregistration,
)
from pss.eval.scripts.run_attribution_grid import _write_json
from pss.eval.scripts.run_upper_bound_u3 import _load_cache, _normalise, _pairs
from pss.eval.upper_bound import cell_accuracy, exact4, pair_labels, pair_types
from pss.reasoning.relations.bayes_boundary import GridBayesBoundary
from pss.reasoning.relations.kde_boundary import KDEBayesBoundary

EPS = 0.02
CONTRACT_SIGMA = 0.01
STRESS_SIGMA = 0.04
LEVELS = (CONTRACT_SIGMA, STRESS_SIGMA)
CELL_NAMES = ("horizontal", "vertical", "diagonal")
ARMS = ("kde_raw", "kde_type_balanced", "grid_noisy")
NOISE_SPACES = ("object", "difference")


def _metrics(bits: np.ndarray, y: np.ndarray, types: np.ndarray) -> dict:
    cells = cell_accuracy(types, y, bits)
    h = float(cells.get("horizontal", 0.0))
    v = float(cells.get("vertical", 0.0))
    return {
        "overall": exact4(y, bits),
        "cells": cells,
        "near_worst": min(h, v),
        "absence_error": max(1.0 - h, 1.0 - v),
    }


def _agg(rows: list[dict]) -> dict:
    """Mean/std across noise seeds for one arm and sigma level."""

    def _mean(key: str) -> float:
        return float(np.mean([r[key] for r in rows]))

    def _std(key: str) -> float:
        return float(np.std([r[key] for r in rows]))

    cells = {
        name: float(np.mean([r["cells"].get(name, 0.0) for r in rows]))
        for name in CELL_NAMES
    }
    cells_std = {
        name: float(np.std([r["cells"].get(name, 0.0) for r in rows]))
        for name in CELL_NAMES
    }
    return {
        "overall": _mean("overall"),
        "overall_std": _std("overall"),
        "cells": cells,
        "cells_std": cells_std,
        "near_worst": _mean("near_worst"),
        "near_worst_std": _std("near_worst"),
        "absence_error": _mean("absence_error"),
        "absence_error_std": _std("absence_error"),
        "mean_confidence": _mean("mean_confidence"),
        "latency_ms_per_1000": _mean("latency_ms_per_1000"),
        "per_seed": rows,
    }


def _load_comparison(bench_path: Path, bayes_path: Path) -> dict:
    comparison: dict[str, dict] = {}
    if bench_path.exists():
        bench = json.loads(bench_path.read_text(encoding="utf-8"))
        for arm in ("hard_clean", "hard_noise_aug", "soft_estimated"):
            if arm not in bench.get("arms", {}):
                continue
            comparison[arm] = {}
            for level, row in bench["arms"][arm].items():
                if level not in ("0.0", "0.01", "0.04"):
                    continue
                cells = row.get("cells", {})
                comparison[arm][level] = {
                    "near_worst": float(row["near_worst"]["mean"]),
                    "overall": float(row["overall"]["mean"]),
                    "cells": {
                        name: float(cells.get(name, 0.0))
                        for name in CELL_NAMES
                    },
                }
    if bayes_path.exists():
        bayes = json.loads(bayes_path.read_text(encoding="utf-8"))
        comparison["bayes_grid_published"] = {}
        for level, row in bayes["levels"].items():
            comparison["bayes_grid_published"][level] = {
                "near_worst": float(row["near_worst"]),
                "overall": float(row["overall"]),
                "cells": {
                    name: float(row["cells"].get(name, 0.0))
                    for name in CELL_NAMES
                },
            }
    return comparison


def run(args) -> dict:
    root = Path(args.clevr_root)
    train = _load_cache(root, args.train_split)
    test = _load_cache(root, args.test_split)
    pa_tr, pb_tr, _, _ = _pairs(train, args.train_pairs, args.seed)
    pa_te, pb_te, _, _ = _pairs(test, args.test_pairs, args.seed + 1)
    lo = np.minimum(pa_tr.min(0), pb_tr.min(0))
    hi = np.maximum(pa_tr.max(0), pb_tr.max(0))
    pa_tr, pb_tr = _normalise(pa_tr, lo, hi), _normalise(pb_tr, lo, hi)
    pa_te, pb_te = _normalise(pa_te, lo, hi), _normalise(pb_te, lo, hi)

    dx_tr = pb_tr[:, 0] - pa_tr[:, 0]
    dy_tr = pb_tr[:, 1] - pa_tr[:, 1]
    dx_te = pb_te[:, 0] - pa_te[:, 0]
    dy_te = pb_te[:, 1] - pa_te[:, 1]
    Yte = pair_labels(pa_te, pb_te, eps=EPS)
    Tte = pair_types(pa_te, pb_te, eps=EPS)
    print(
        f"[data] train {len(pa_tr)} test {len(pa_te)} "
        f"cells {dict(zip(*np.unique(Tte, return_counts=True)))}",
        flush=True,
    )

    models = {}
    fit_ms = {}
    t0 = time.perf_counter()
    models["kde_raw"] = KDEBayesBoundary(
        eps=EPS,
        prior_samples=args.prior_samples,
        chunk=args.chunk,
        seed=args.seed,
        balance_types=False,
    ).fit(dx_tr, dy_tr)
    fit_ms["kde_raw"] = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    models["kde_type_balanced"] = KDEBayesBoundary(
        eps=EPS,
        prior_samples=args.prior_samples,
        chunk=args.chunk,
        seed=args.seed,
        balance_types=True,
    ).fit(dx_tr, dy_tr)
    fit_ms["kde_type_balanced"] = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    models["grid_noisy"] = GridBayesBoundary(
        eps=EPS, bins=args.bins
    ).fit(dx_tr, dy_tr)
    fit_ms["grid_noisy"] = (time.perf_counter() - t0) * 1e3

    raw = models["kde_raw"]
    abs_dx = np.abs(raw.prior[:, 0])
    abs_dy = np.abs(raw.prior[:, 1])
    is_horizontal = abs_dy <= EPS
    is_vertical = (abs_dx <= EPS) & (abs_dy > EPS)
    is_diagonal = (abs_dx > EPS) & (abs_dy > EPS)
    prior_freq = {
        "horizontal": float(is_horizontal.mean()),
        "vertical": float(is_vertical.mean()),
        "diagonal": float(is_diagonal.mean()),
    }
    print(
        f"[prior] {len(raw.prior)} particles "
        f"type freq { {k: round(v, 4) for k, v in prior_freq.items()} }",
        flush=True,
    )

    results: dict[str, dict[str, dict]] = {
        arm: {} for arm in ARMS
    }
    # The bench adds noise to each object coordinate, so a pair difference has
    # sqrt(2) times that sigma.  The ceiling script adds noise directly to the
    # difference.  ``--noise-space`` selects which convention the requested
    # ``sigma`` means; the difference sigma (and the KDE kernel) is scaled
    # accordingly.
    diff_factor = float(np.sqrt(2.0)) if args.noise_space == "object" else 1.0
    for sigma in LEVELS:
        diff_sigma = float(sigma) * diff_factor
        rows: dict[str, list[dict]] = {arm: [] for arm in ARMS}
        for repeat in range(args.repeats):
            rng = np.random.default_rng(
                [args.seed, int(round(sigma * 10000)), repeat]
            )
            observed = np.stack(
                [
                    dx_te + rng.normal(0.0, diff_sigma, size=dx_te.shape),
                    dy_te + rng.normal(0.0, diff_sigma, size=dy_te.shape),
                ],
                axis=1,
            )
            for arm in ARMS:
                t0 = time.perf_counter()
                bits, conf = models[arm].predict(
                    observed[:, 0], observed[:, 1], diff_sigma
                )
                elapsed_ms = (time.perf_counter() - t0) * 1e3
                row = _metrics(bits, Yte, Tte)
                row["mean_confidence"] = float(conf.mean())
                row["latency_ms_per_1000"] = float(
                    elapsed_ms / len(observed) * 1000.0
                )
                row["seed"] = int(repeat)
                row["difference_sigma"] = diff_sigma
                rows[arm].append(row)
            print(
                f"[sigma {sigma:.2f} seed {repeat}] "
                + " | ".join(
                    f"{arm} near {rows[arm][-1]['near_worst']:.3f} "
                    f"overall {rows[arm][-1]['overall']:.3f}"
                    for arm in ARMS
                ),
                flush=True,
            )
        for arm in ARMS:
            results[arm][str(sigma)] = _agg(rows[arm])
            results[arm][str(sigma)]["difference_sigma"] = diff_sigma

    raw_contract = results["kde_raw"][str(CONTRACT_SIGMA)]
    raw_stress = results["kde_raw"][str(STRESS_SIGMA)]
    bal_contract = results["kde_type_balanced"][str(CONTRACT_SIGMA)]
    bal_stress = results["kde_type_balanced"][str(STRESS_SIGMA)]
    primary = float(raw_contract["near_worst"])
    check = check_readout_reasoning_preregistration(
        {"u_boundary_soft": {"worst_cell_acc": primary}}
    )
    check_balanced = check_readout_reasoning_preregistration(
        {"u_boundary_soft": {"worst_cell_acc": float(bal_contract["near_worst"])}}
    )

    comparison = _load_comparison(
        Path(args.bench_json), Path(args.bayes_json)
    )

    per_cell_ceiling = {
        "rule": "KDE weighted vote (empirical Bayes) per observed pair",
        "raw_prior": {
            "contract_sigma": {
                "cells": raw_contract["cells"],
                "near_worst": raw_contract["near_worst"],
            },
            "stress_sigma": {
                "cells": raw_stress["cells"],
                "near_worst": raw_stress["near_worst"],
            },
        },
        "type_balanced_prior": {
            "contract_sigma": {
                "cells": bal_contract["cells"],
                "near_worst": bal_contract["near_worst"],
            },
            "stress_sigma": {
                "cells": bal_stress["cells"],
                "near_worst": bal_stress["near_worst"],
            },
        },
        "note": (
            "The KDE vote is the empirical-Bayes rule for its prior, so its "
            "per-cell accuracy is a ceiling.  With the raw train prior "
            "(~95% diagonal) the overall-optimal vote essentially never says "
            "'neither' at sigma=0.04, so the near cells collapse.  With equal "
            "prior mass per pair type the same kernel recovers the near cells, "
            "which locates the failure in the prior/cost balance rather than "
            "in the axis information."
        ),
    }

    payload = {
        "stage": "boundary_kde",
        "clevr_root": str(root),
        "train_split": args.train_split,
        "test_split": args.test_split,
        "train_pairs": int(len(pa_tr)),
        "test_pairs": int(len(pa_te)),
        "prior_samples": int(len(raw.prior)),
        "prior_type_freq": prior_freq,
        "eps": EPS,
        "noise_space": args.noise_space,
        "difference_sigma_factor": float(diff_factor),
        "bins": int(args.bins),
        "chunk": int(args.chunk),
        "repeats": int(args.repeats),
        "seed": int(args.seed),
        "fit_ms": fit_ms,
        "arms": results,
        "primary_contract_near_worst": primary,
        "stress_near_worst": float(raw_stress["near_worst"]),
        "stress_near_at_least_0_60": bool(raw_stress["near_worst"] >= 0.60),
        "type_balanced_contract_near_worst": float(bal_contract["near_worst"]),
        "type_balanced_stress_near_worst": float(bal_stress["near_worst"]),
        "type_balanced_stress_near_at_least_0_60": bool(
            bal_stress["near_worst"] >= 0.60
        ),
        "per_cell_ceiling": per_cell_ceiling,
        "comparison": comparison,
        "preregistration": check,
        "preregistration_type_balanced": check_balanced,
        "note": (
            "KDE/mixture posterior boundary: prior particles from train "
            "pairs, Gaussian kernel with the difference noise sigma, per-axis "
            "3-way weighted vote.  Same splits/labels/metrics as "
            "run_boundary_bayes; no learned parameters.  Diagnostic ceiling, "
            "not a deployment path (O(N*K) per pair).  kde_raw is the "
            "pre-registered arm; kde_type_balanced and grid_noisy are "
            "diagnostics on the same noisy observations.  noise_space='object' "
            "scales both the observation noise and the kernel by sqrt(2) to "
            "match the bench; 'difference' matches the ceiling script."
        ),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "boundary_kde.json", payload)
    return payload


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--clevr-root",
        default="data/public/clevr_cogent/CLEVR_CoGenT_v1.0",
    )
    ap.add_argument("--train-split", default="traina")
    ap.add_argument("--test-split", default="vala")
    ap.add_argument("--train-pairs", type=int, default=30000)
    ap.add_argument("--test-pairs", type=int, default=10000)
    ap.add_argument("--prior-samples", type=int, default=30000)
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument(
        "--noise-space",
        choices=NOISE_SPACES,
        default="object",
        help=(
            "'object' matches run_boundary_bench (per-object noise sigma, so "
            "the pair difference and KDE kernel use sigma*sqrt(2)); "
            "'difference' matches run_boundary_ceiling (noise sigma is "
            "already the difference sigma)."
        ),
    )
    ap.add_argument("--bins", type=int, default=257)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--bench-json",
        default="eval/results/boundary_bench/boundary_bench.json",
    )
    ap.add_argument(
        "--bayes-json",
        default="eval/results/boundary_bench/bayes/boundary_bayes.json",
    )
    ap.add_argument("--out", default="eval/results/boundary_bench/kde")
    args = ap.parse_args()
    payload = run(args)
    print(
        json.dumps(
            {
                "primary_contract_near_worst": payload[
                    "primary_contract_near_worst"
                ],
                "stress_near_worst": payload["stress_near_worst"],
                "stress_near_at_least_0_60": payload[
                    "stress_near_at_least_0_60"
                ],
                "type_balanced_contract_near_worst": payload[
                    "type_balanced_contract_near_worst"
                ],
                "type_balanced_stress_near_worst": payload[
                    "type_balanced_stress_near_worst"
                ],
                "preregistration": payload["preregistration"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
