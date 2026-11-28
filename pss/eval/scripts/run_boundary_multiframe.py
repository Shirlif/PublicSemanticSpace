"""Multi-frame filtering / depth fusion for the boundary layer.

The empirical-Bayes boundary (``run_boundary_bayes``) reaches near-worst 0.948
at the contract noise ``sigma=0.01`` but collapses to 0.109 at stress
``sigma=0.04``.  This script asks whether *more frames* recover the contract
operating point: average ``M`` independent noisy observations of both
endpoints, feed the averaged relative coordinates to the same boundary layer,
and use the reduced noise ``sigma_eff = sigma / sqrt(M)``.

The prior is fitted once on the clean train pairs ``(dx, dy)``; every test
pair is then observed ``M`` times with independent Gaussian noise on ``pa``
and ``pb``.  We report overall, horizontal / vertical / diagonal, near-worst
and absence error for

    sigma in {0.01, 0.04}  x  M in {1, 2, 4, 8, 16}  x  3 noise seeds.

Targets
-------
- stress ``sigma=0.04, M=16`` -> ``sigma_eff=0.01``: near-worst >= 0.80.
- contract ``sigma=0.01, M=4`` -> ``sigma_eff=0.005``: near-worst close to 0.97.

The simulation perturbs each endpoint independently.  The difference of two
independent noisy endpoints has standard deviation ``sigma * sqrt(2)`` per
frame, so the *measured* noise on ``dx_bar`` is close to
``sigma * sqrt(2) / sqrt(M)``.  The script reports that measured value next to
the nominal ``sigma / sqrt(M)`` used as the boundary likelihood width; the
main grid follows the task specification exactly.

Example
-------
python -u -m pss.eval.scripts.run_boundary_multiframe \
  --out eval/results/boundary_bench/multiframe
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pss._console import ensure_utf8
from pss.eval.scripts.run_attribution_grid import _write_json
from pss.eval.scripts.run_upper_bound_u3 import _load_cache, _normalise, _pairs
from pss.eval.upper_bound import cell_accuracy, exact4, pair_labels, pair_types
from pss.reasoning.relations.bayes_boundary import GridBayesBoundary

EPS = 0.02
SIGMAS_DEFAULT = (0.01, 0.04)
FRAMES_DEFAULT = (1, 2, 4, 8, 16)
NOISE_SEEDS_DEFAULT = (0, 1, 2)

# The two reported operating points.
STRESS_SIGMA = 0.04
STRESS_FRAMES = 16
CONTRACT_SIGMA = 0.01
CONTRACT_FRAMES = 4
STRESS_NEAR_BAR = 0.80


def _simulate(
    pa: np.ndarray,
    pb: np.ndarray,
    sigma: float,
    frames: int,
    noise_seed: int,
    sample_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Average ``frames`` independent noisy endpoint observations.

    Returns the averaged ``(pa_bar, pb_bar)`` for every pair.
    """

    rng = np.random.default_rng(
        [int(sample_seed), int(noise_seed), int(frames), int(round(sigma * 1e6)), 17]
    )
    shape = (len(pa), int(frames), pa.shape[1])
    pa_frames = pa[:, None, :] + rng.normal(0.0, sigma, size=shape)
    pb_frames = pb[:, None, :] + rng.normal(0.0, sigma, size=shape)
    return pa_frames.mean(axis=1), pb_frames.mean(axis=1)


def _metrics(bits: np.ndarray, y: np.ndarray, types: np.ndarray) -> dict:
    cells = cell_accuracy(types, y, bits)
    h = float(cells.get("horizontal", 0.0))
    v = float(cells.get("vertical", 0.0))
    return {
        "overall": exact4(y, bits),
        "cells": cells,
        "near_worst": min(h, v),
        "absence_horizontal_error": 1.0 - h,
        "absence_vertical_error": 1.0 - v,
        "absence_error": max(1.0 - h, 1.0 - v),
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
    dx_te = pb_te[:, 0] - pa_te[:, 0]
    dy_te = pb_te[:, 1] - pa_te[:, 1]

    Yte = pair_labels(pa_te, pb_te, eps=EPS)
    Tte = pair_types(pa_te, pb_te, eps=EPS)

    boundary = GridBayesBoundary(eps=EPS, bins=args.bins).fit(dx_tr, dy_tr)

    rows: list[dict] = []
    for sigma in args.sigmas:
        for frames in args.frames:
            sigma_eff = float(sigma) / np.sqrt(float(frames))
            for noise_seed in args.noise_seeds:
                pa_bar, pb_bar = _simulate(
                    pa_te, pb_te, sigma, frames, noise_seed, args.seed + 1
                )
                dx_bar = pb_bar[:, 0] - pa_bar[:, 0]
                dy_bar = pb_bar[:, 1] - pa_bar[:, 1]
                measured_dx = float(np.std(dx_bar - dx_te))
                measured_dy = float(np.std(dy_bar - dy_te))
                bits, conf = boundary.predict(dx_bar, dy_bar, sigma=sigma_eff)
                row = _metrics(bits, Yte, Tte)
                # Diagnostic only: re-run the same observations with the
                # likelihood width set to the *measured* dx noise.  Independent
                # endpoint noise makes this sigma*sqrt(2)/sqrt(M), not the
                # nominal sigma/sqrt(M); this shows the calibration gap.
                matched_bits, _ = boundary.predict(
                    dx_bar, dy_bar, sigma=measured_dx
                )
                matched = _metrics(matched_bits, Yte, Tte)
                row.update(
                    {
                        "sigma": float(sigma),
                        "frames": int(frames),
                        "noise_seed": int(noise_seed),
                        "sigma_eff": sigma_eff,
                        "mean_confidence": float(conf.mean()),
                        "measured_dx_noise": measured_dx,
                        "measured_dy_noise": measured_dy,
                        "matched_sigma": measured_dx,
                        "matched_overall": matched["overall"],
                        "matched_near_worst": matched["near_worst"],
                        "matched_horizontal": matched["cells"].get("horizontal", 0.0),
                        "matched_vertical": matched["cells"].get("vertical", 0.0),
                    }
                )
                rows.append(row)
                print(
                    f"[sigma {sigma:.2f} M {frames:2d} seed {noise_seed}] "
                    f"overall {row['overall']:.3f} "
                    f"h {row['cells'].get('horizontal', 0):.3f} "
                    f"v {row['cells'].get('vertical', 0):.3f} "
                    f"d {row['cells'].get('diagonal', 0):.3f} "
                    f"near {row['near_worst']:.3f} "
                    f"obs_dx {row['measured_dx_noise']:.4f}",
                    flush=True,
                )

    aggregate = _aggregate(rows)
    targets = _targets(aggregate)

    payload = {
        "stage": "boundary_multiframe",
        "clevr_root": str(root),
        "train_split": args.train_split,
        "test_split": args.test_split,
        "train_pairs": int(len(pa_tr)),
        "test_pairs": int(len(pa_te)),
        "eps": EPS,
        "bins": int(args.bins),
        "sigmas": [float(s) for s in args.sigmas],
        "frames": [int(m) for m in args.frames],
        "noise_seeds": [int(s) for s in args.noise_seeds],
        "rows": rows,
        "aggregate": aggregate,
        "targets": targets,
        "note": (
            "multi-frame filtering: prior fitted on clean train (dx,dy); each "
            "test pair observed M times with independent N(0,sigma) noise on "
            "both endpoints; averaged relative coordinates predicted with "
            "sigma_eff=sigma/sqrt(M). The measured dx noise is also reported "
            "because independent endpoint noise gives sigma*sqrt(2)/sqrt(M)."
        ),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "boundary_multiframe.json", payload)
    (out / "MULTIFRAME_REPORT.md").write_text(_render(payload), encoding="utf-8")
    return payload


def _aggregate(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    keys = sorted({(r["sigma"], r["frames"]) for r in rows})
    for sigma, frames in keys:
        group = [r for r in rows if r["sigma"] == sigma and r["frames"] == frames]

        def mean(key: str) -> float:
            return float(np.mean([r[key] for r in group]))

        def cell(name: str) -> float:
            return float(np.mean([r["cells"].get(name, 0.0) for r in group]))

        near = np.asarray([r["near_worst"] for r in group], dtype=np.float64)
        out.append(
            {
                "sigma": float(sigma),
                "frames": int(frames),
                "sigma_eff": float(sigma) / np.sqrt(float(frames)),
                "overall": mean("overall"),
                "horizontal": cell("horizontal"),
                "vertical": cell("vertical"),
                "diagonal": cell("diagonal"),
                "near_worst": float(near.mean()),
                "near_worst_std": float(near.std(ddof=0)),
                "matched_near_worst": mean("matched_near_worst"),
                "matched_overall": mean("matched_overall"),
                "absence_error": mean("absence_error"),
                "mean_confidence": mean("mean_confidence"),
                "measured_dx_noise": mean("measured_dx_noise"),
                "measured_dy_noise": mean("measured_dy_noise"),
                "n_seeds": len(group),
            }
        )
    return out


def _find(aggregate: list[dict], sigma: float, frames: int) -> dict:
    for row in aggregate:
        if abs(row["sigma"] - sigma) < 1e-12 and row["frames"] == frames:
            return row
    raise KeyError((sigma, frames))


def _targets(aggregate: list[dict]) -> dict:
    stress = _find(aggregate, STRESS_SIGMA, STRESS_FRAMES)
    contract = _find(aggregate, CONTRACT_SIGMA, CONTRACT_FRAMES)
    return {
        "stress_sigma_0.04_M16": {
            "near_worst": stress["near_worst"],
            "near_worst_std": stress["near_worst_std"],
            "bar": STRESS_NEAR_BAR,
            "ok": bool(stress["near_worst"] >= STRESS_NEAR_BAR),
            "sigma_eff": stress["sigma_eff"],
        },
        "contract_sigma_0.01_M4": {
            "near_worst": contract["near_worst"],
            "near_worst_std": contract["near_worst_std"],
            "sigma_eff": contract["sigma_eff"],
            "reference_ceiling_0.005": 0.975,
        },
    }


def _render(p: dict) -> str:
    lines = [
        "# Boundary — 多帧滤波/深度融合",
        "",
        f"- CLEVR: `{p['clevr_root']}` · train {p['train_pairs']} / test "
        f"{p['test_pairs']} pairs · eps={p['eps']} · bins={p['bins']}",
        f"- σ ∈ {p['sigmas']} · M ∈ {p['frames']} · noise seeds {p['noise_seeds']}",
        "- 每个 pair 的 pa、pb 独立加噪 M 帧后取平均；预测用 σ_eff=σ/√M；"
        "但独立端点噪声使实测 dx 噪声 ≈ σ√2/√M（表中同时给出）。",
        "",
        "## 1. 主表（3 噪声 seed 的 mean）",
        "",
        "| σ | M | σ_eff | overall | horizontal | vertical | diagonal | near-worst | near-worst (匹配 σ) | absence | 实测 dx 噪声 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in p["aggregate"]:
        lines.append(
            f"| {row['sigma']:.2f} | {row['frames']} | {row['sigma_eff']:.4f} | "
            f"{row['overall']:.3f} | {row['horizontal']:.3f} | "
            f"{row['vertical']:.3f} | {row['diagonal']:.3f} | "
            f"**{row['near_worst']:.3f}** ± {row['near_worst_std']:.3f} | "
            f"{row['matched_near_worst']:.3f} | "
            f"{row['absence_error']:.3f} | {row['measured_dx_noise']:.4f} |"
        )
    t = p["targets"]
    stress = t["stress_sigma_0.04_M16"]
    contract = t["contract_sigma_0.01_M4"]
    lines += [
        "",
        "## 2. 目标检查",
        "",
        f"- σ=0.04, M=16 (σ_eff={stress['sigma_eff']:.4f}): near "
        f"{stress['near_worst']:.3f} ± {stress['near_worst_std']:.3f} · "
        f"bar {stress['bar']:.2f} · **{'PASS' if stress['ok'] else 'FAIL'}**",
        f"- σ=0.01, M=4 (σ_eff={contract['sigma_eff']:.4f}): near "
        f"{contract['near_worst']:.3f} ± {contract['near_worst_std']:.3f} · "
        f"契约 σ=0.01 ceiling {contract['reference_ceiling_0.005']:.3f}",
        "",
        "## 3. 判读",
        "",
        "- 若 near-worst 随 M 单调上升并回到契约档，则 stress 崩溃是**观测噪声**问题，"
        "多帧平均是有效干预。",
        "- σ_eff 曲线用来检查 near-worst 是否主要由 σ_eff 决定；若同一 σ_eff 下"
        "σ=0.01 与 σ=0.04 两条曲线重合，说明噪声已被平均掉。",
        "- 实测 dx 噪声是 σ√2/√M，比名义 σ/√M 大 √2；这会让 M 帧的收益比"
        "理想公式略慢，但只要仍过 bar 就不影响结论。",
        "",
        "## 4. 边界",
        "",
        "- 帧间噪声**独立同分布**；真实序列有运动、遮挡和配准误差，未测。",
        "- pair 是**静态**的：每帧观测同一个真值 pair，没有跨帧运动补偿。",
        "- σ 是像素坐标代理，不是真实 RGB-D 标定；结论不能直接外推到真实传感器。",
        "- 先验仍是 train-pair 经验分布；场景漂移需要重估先验。",
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
    ap.add_argument("--sigmas", nargs="+", type=float, default=list(SIGMAS_DEFAULT))
    ap.add_argument("--frames", nargs="+", type=int, default=list(FRAMES_DEFAULT))
    ap.add_argument(
        "--noise-seeds", nargs="+", type=int, default=list(NOISE_SEEDS_DEFAULT)
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="eval/results/boundary_bench/multiframe")
    args = ap.parse_args()
    payload = run(args)
    print(json.dumps(payload["targets"], indent=2))


if __name__ == "__main__":
    main()
