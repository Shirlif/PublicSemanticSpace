"""Boundary benchmark, corrected intervention: empirical-Bayes boundary.

The global-temperature soft dead zone failed (NO-GO), but the ceiling analysis
showed the contract is reachable (Bayes 0.971 at sigma=0.01 vs hard/soft
0.68).  This script evaluates the corrected intervention on the same data and
metrics: a grid Bayes boundary with the empirical train-pair prior.

No training and no hard-baseline rerun: it reuses the splits and metrics from
``run_boundary_bench`` and compares against its JSON.

Example
-------
python -u -m pss.eval.scripts.run_boundary_bayes \
  --out eval/results/boundary_bench/bayes --device cpu
"""

from __future__ import annotations

import argparse
import json
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

EPS = 0.02
CONTRACT_SIGMA = 0.01
STRESS_SIGMA = 0.04
LEVELS = (0.0, CONTRACT_SIGMA, STRESS_SIGMA)


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


def _balance_weights(
    pa: np.ndarray, pb: np.ndarray, strength: float
) -> tuple[np.ndarray, dict[str, int]]:
    types = pair_types(pa, pb, eps=EPS)
    counts = {
        kind: max(int((types == kind).sum()), 1)
        for kind in ("horizontal", "vertical", "diagonal")
    }
    weights = np.asarray(
        [
            (1.0 / 3.0 / (counts[kind] / len(types))) ** float(strength)
            for kind in types
        ],
        dtype=np.float64,
    )
    weights /= weights.sum()
    return weights, counts


def _eval_boundary(
    boundary: GridBayesBoundary,
    pa: np.ndarray,
    pb: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    level: float,
    rng: np.random.Generator,
    repeats: int,
) -> dict:
    rows = []
    for _ in range(repeats):
        if level > 0:
            pa_obs = pa + rng.normal(0.0, level, pa.shape)
            pb_obs = pb + rng.normal(0.0, level, pb.shape)
        else:
            pa_obs, pb_obs = pa, pb
        dx_obs = pb_obs[:, 0] - pa_obs[:, 0]
        dy_obs = pb_obs[:, 1] - pa_obs[:, 1]
        # Per-object noise sigma -> difference noise sigma * sqrt(2).
        bits, conf = boundary.predict(dx_obs, dy_obs, sigma=level * np.sqrt(2.0))
        row = _metrics(bits, Y, T)
        row["mean_confidence"] = float(conf.mean())
        rows.append(row)
    return {
        "overall": float(np.mean([r["overall"] for r in rows])),
        "near_worst": float(np.mean([r["near_worst"] for r in rows])),
        "absence_error": float(np.mean([r["absence_error"] for r in rows])),
        "cells": {
            cell: float(np.mean([r["cells"].get(cell, 0.0) for r in rows]))
            for cell in ("horizontal", "vertical", "diagonal")
        },
        "mean_confidence": float(np.mean([r["mean_confidence"] for r in rows])),
        "noise_repeats": int(repeats),
    }


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
    Yte = pair_labels(pa_te, pb_te, eps=EPS)
    Tte = pair_types(pa_te, pb_te, eps=EPS)

    chosen_strength = float(args.balance_strength)
    calibration = None
    if args.calibrate_strength:
        rng_cal = np.random.default_rng(args.seed + 77)
        order = rng_cal.permutation(len(pa_tr))
        n_cal = max(1, int(round(0.10 * len(order))))
        cal_idx, fit_idx = order[:n_cal], order[n_cal:]
        pa_fit, pb_fit = pa_tr[fit_idx], pb_tr[fit_idx]
        pa_cal, pb_cal = pa_tr[cal_idx], pb_tr[cal_idx]
        dx_fit = pb_fit[:, 0] - pa_fit[:, 0]
        dy_fit = pb_fit[:, 1] - pa_fit[:, 1]
        Ycal = pair_labels(pa_cal, pb_cal, eps=EPS)
        Tcal = pair_types(pa_cal, pb_cal, eps=EPS)
        base_boundary = GridBayesBoundary(eps=EPS, bins=args.bins).fit(
            dx_fit, dy_fit
        )
        base = _eval_boundary(
            base_boundary, pa_cal, pb_cal, Ycal, Tcal,
            CONTRACT_SIGMA, rng_cal, args.noise_repeats,
        )
        rows = []
        best = None
        for strength in args.balance_strengths:
            weights, counts = _balance_weights(pa_fit, pb_fit, strength)
            candidate = GridBayesBoundary(eps=EPS, bins=args.bins).fit(
                dx_fit, dy_fit, weights=weights
            )
            metrics = _eval_boundary(
                candidate, pa_cal, pb_cal, Ycal, Tcal,
                CONTRACT_SIGMA, rng_cal, args.noise_repeats,
            )
            drop = base["overall"] - metrics["overall"]
            ok = metrics["near_worst"] >= 0.80 and drop <= 0.02
            rows.append(
                {
                    "strength": float(strength),
                    "near_worst": metrics["near_worst"],
                    "overall": metrics["overall"],
                    "overall_drop": drop,
                    "ok": bool(ok),
                    "counts": counts,
                }
            )
            if ok and (best is None or metrics["near_worst"] > best["near_worst"]):
                best = rows[-1]
        if best is None:
            best = max(rows, key=lambda r: r["near_worst"])
        chosen_strength = float(best["strength"])
        calibration = {"base": base, "rows": rows, "chosen_strength": chosen_strength}
        print(
            f"[cal] base overall {base['overall']:.3f} near {base['near_worst']:.3f}; "
            f"chosen strength {chosen_strength}",
            flush=True,
        )

    weights = None
    if args.balance or args.calibrate_strength:
        weights, counts = _balance_weights(pa_tr, pb_tr, chosen_strength)
        print(
            f"[balance] pair counts {counts}; strength {chosen_strength}",
            flush=True,
        )

    boundary = GridBayesBoundary(eps=EPS, bins=args.bins).fit(
        dx_tr, dy_tr, weights=weights
    )
    rng = np.random.default_rng(args.seed + 123)
    results = {}
    for level in LEVELS:
        results[str(level)] = _eval_boundary(
            boundary, pa_te, pb_te, Yte, Tte, level, rng, args.noise_repeats
        )
        row = results[str(level)]
        print(
            f"[sigma {level:.2f}] overall {row['overall']:.3f} "
            f"h {row['cells'].get('horizontal', 0):.3f} "
            f"v {row['cells'].get('vertical', 0):.3f} "
            f"d {row['cells'].get('diagonal', 0):.3f} "
            f"near {row['near_worst']:.3f}",
            flush=True,
        )

    primary = float(results[str(CONTRACT_SIGMA)]["near_worst"])
    check = check_readout_reasoning_preregistration(
        {"u_boundary_soft": {"worst_cell_acc": primary}}
    )

    comparison = None
    bench_path = Path(args.bench_json)
    if bench_path.exists():
        bench = json.loads(bench_path.read_text(encoding="utf-8"))
        comparison = {
            arm: {
                level: {
                    "near_worst": bench["arms"][arm][level]["near_worst"]["mean"],
                    "overall": bench["arms"][arm][level]["overall"]["mean"],
                }
                for level in ("0.0", str(CONTRACT_SIGMA), str(STRESS_SIGMA))
            }
            for arm in ("hard_clean", "hard_noise_aug", "soft_estimated")
            if arm in bench.get("arms", {})
        }

    payload = {
        "stage": "boundary_bayes",
        "clevr_root": str(root),
        "train_pairs": int(len(pa_tr)),
        "test_pairs": int(len(pa_te)),
        "eps": EPS,
        "bins": int(args.bins),
        "balance_strength": float(chosen_strength),
        "calibration": calibration,
        "levels": results,
        "primary_contract_near_worst": primary,
        "comparison": comparison,
        "preregistration": check,
        "note": (
            "empirical-Bayes boundary: prior from train pairs, Gaussian "
            "posterior mean/variance per observed grid cell, 3-way probit; "
            "no learned parameters and no hard-baseline training.  Test "
            "observations are noised per object at the requested sigma; the "
            "boundary table uses sigma*sqrt(2) for the difference."
        ),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "boundary_bayes.json", payload)
    (out / "BOUNDARY_BAYES_REPORT.md").write_text(
        _render(payload), encoding="utf-8"
    )
    return payload


def _render(p: dict) -> str:
    lines = [
        "# Boundary benchmark — empirical-Bayes boundary",
        "",
        f"- CLEVR: `{p['clevr_root']}` · train {p['train_pairs']} / test {p['test_pairs']} pairs",
        f"- eps={p['eps']} · bins={p['bins']}",
        "",
        "## 1. Main table",
        "",
        "| level | overall | horizontal | vertical | diagonal | near worst | absence |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for level in ("0.0", str(CONTRACT_SIGMA), str(STRESS_SIGMA)):
        r = p["levels"][level]
        lines.append(
            f"| σ={level} | {r['overall']:.3f} | {r['cells'].get('horizontal', 0):.3f} | "
            f"{r['cells'].get('vertical', 0):.3f} | {r['cells'].get('diagonal', 0):.3f} | "
            f"**{r['near_worst']:.3f}** | {r['absence_error']:.3f} |"
        )
    if p.get("comparison"):
        lines += [
            "",
            "## 2. Comparison with the failed arms",
            "",
            "| arm | clean near | contract near | stress near |",
            "|---|---:|---:|---:|",
        ]
        for arm, rows in p["comparison"].items():
            lines.append(
                f"| `{arm}` | {rows['0.0']['near_worst']:.3f} | "
                f"{rows[str(CONTRACT_SIGMA)]['near_worst']:.3f} | "
                f"{rows[str(STRESS_SIGMA)]['near_worst']:.3f} |"
            )
        lines.append(
            f"| **`bayes`** | {p['levels']['0.0']['near_worst']:.3f} | "
            f"**{p['levels'][str(CONTRACT_SIGMA)]['near_worst']:.3f}** | "
            f"{p['levels'][str(STRESS_SIGMA)]['near_worst']:.3f} |"
        )
    check = p["preregistration"]
    lines += [
        "",
        "## 3. Preregistration",
        "",
        f"- checked: {check['checked']} · ok: **{check['ok']}**",
    ]
    if check["mismatches"]:
        lines.append("")
        for mismatch in check["mismatches"]:
            lines.append(f"- `{mismatch}`")
    lines += [
        "",
        "## 4. Interpretation",
        "",
        "- The global-temperature arm failed because one scalar cannot represent",
        "  the local prior; the Bayes layer uses the prior mean/variance per cell.",
        "- `σ=0` is the clean check; `σ=0.01` is the contract; `σ=0.04` is stress.",
        "- Noise is added to each object; the difference has σ√2, and the table",
        "  is indexed with that difference sigma.",
        "- This is a closed-form layer, not a learned model; it is the corrected",
        "  intervention indicated by the information ceiling.",
        "",
        "## 5. Boundaries",
        "",
        "- CLEVR rendered geometry; σ is pixel-coordinate noise, not RGB-D calibration.",
        "- The prior is the train-pair empirical distribution; a deployment shift",
        "  would require re-estimating it.",
        "- The Gaussian posterior approximation is not the exact Bayes rule; the",
        "  exact ceiling is reported separately in `ceiling.json`.",
        "",
    ]
    return "\n".join(lines)


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
    ap.add_argument("--bins", type=int, default=257)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise-repeats", type=int, default=3)
    ap.add_argument("--balance", action="store_true")
    ap.add_argument("--balance-strength", type=float, default=1.0)
    ap.add_argument("--calibrate-strength", action="store_true")
    ap.add_argument(
        "--balance-strengths", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0]
    )
    ap.add_argument(
        "--bench-json",
        default="eval/results/boundary_bench/boundary_bench.json",
    )
    ap.add_argument("--out", default="eval/results/boundary_bench/bayes")
    args = ap.parse_args()
    payload = run(args)
    print(json.dumps(payload["preregistration"], indent=2))


if __name__ == "__main__":
    main()
