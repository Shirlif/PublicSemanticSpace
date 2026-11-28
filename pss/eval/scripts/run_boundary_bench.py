"""PSS-Boundary: the single end-to-end benchmark for the noise-boundary question.

Executable form of ``docs/plans/PSS-问题重构.md`` D1–D5.

The only open question after U1–U6 is whether the explicit-geometry skeleton
can survive the sensor contract.  This script fixes one task, one primary
metric and four guards, then runs six arms:

  hard_clean          U6 baseline: clean-trained hard dead zone
  hard_noise_aug      capacity/data control: hard head + noise augmentation
  soft_fixed          soft dead zone, temperature fixed at contract noise
  soft_oracle         soft dead zone with the true sigma
  soft_estimated      soft dead zone with sigma estimated from calibration
  soft_oracle_abstain oracle soft + confidence abstention

Primary: near-aligned worst-cell at the contract noise (sigma=0.01).
Guards: clean boundary, coverage, open-set, overall, latency.  The coverage
and open-set guards are constants from U2 / U1b; they are not rerun here.

Example
-------
python -u -m pss.eval.scripts.run_boundary_bench \
  --out eval/results/boundary_bench --device cuda
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from pss._console import ensure_utf8
from pss.eval.readout_reasoning_preregistration import (
    check_readout_reasoning_preregistration,
)
from pss.eval.scripts.run_attribution_grid import _write_json
from pss.eval.scripts.run_geometry_validation import _predict, _train
from pss.eval.scripts.run_typed_relations import _pair_feature
from pss.eval.scripts.run_upper_bound_u3 import (
    _load_cache,
    _normalise,
    _pairs,
)
from pss.eval.upper_bound import cell_accuracy, exact4, pair_labels, pair_types
from pss.reasoning.relations.soft_deadzone import (
    calibrate_abstain,
    calibrate_k,
    calibrate_tau,
    predict as soft_predict,
)

CONTRACT_SIGMA = 0.01
STRESS_SIGMA = 0.04
LEVELS = (0.0, CONTRACT_SIGMA, STRESS_SIGMA)
EPS = 0.02
ARMS = (
    "hard_clean",
    "hard_noise_aug",
    "soft_fixed",
    "soft_oracle",
    "soft_estimated",
    "soft_oracle_abstain",
)


def _features(pa: np.ndarray, pb: np.ndarray) -> np.ndarray:
    return np.asarray(
        [_pair_feature(pa[i], pb[i], "relative") for i in range(len(pa))],
        dtype=np.float32,
    )


def _noisy(
    pa: np.ndarray, pb: np.ndarray, sigma: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    if sigma <= 0:
        return pa.copy(), pb.copy()
    return (
        pa + rng.normal(0.0, sigma, pa.shape),
        pb + rng.normal(0.0, sigma, pb.shape),
    )


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


def _eval_hard(head, pa, pb, y, types, device) -> dict:
    bits = _predict(head, _features(pa, pb), device)
    return _metrics(bits, y, types)


def _eval_soft(pa, pb, y, types, tau, threshold=None) -> dict:
    bits, conf = soft_predict(pa, pb, eps=EPS, tau=tau)
    out = _metrics(bits, y, types)
    if threshold is not None:
        keep = conf >= float(threshold)
        correct = np.all(bits == np.asarray(y).astype(bool), axis=1)
        out["selective_coverage"] = float(keep.mean())
        out["selective_accuracy"] = float(correct[keep].mean()) if keep.any() else 0.0
    return out


def _agg(rows: list[dict]) -> dict:
    keys = ("overall", "near_worst", "absence_error")
    out = {
        key: {
            "mean": float(np.mean([r[key] for r in rows])),
            "std": float(np.std([r[key] for r in rows])),
        }
        for key in keys
    }
    out["cells"] = {
        cell: float(np.mean([r["cells"].get(cell, 0.0) for r in rows]))
        for cell in ("horizontal", "vertical", "diagonal")
    }
    return out


def run(args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train = _load_cache(Path(args.clevr_root), args.train_split)
    test = _load_cache(Path(args.clevr_root), args.test_split)
    pa_tr, pb_tr, _, _ = _pairs(train, args.train_pairs, args.seed)
    pa_te, pb_te, _, _ = _pairs(test, args.test_pairs, args.seed + 1)
    lo = np.minimum(pa_tr.min(0), pb_tr.min(0))
    hi = np.maximum(pa_tr.max(0), pb_tr.max(0))
    pa_tr, pb_tr = _normalise(pa_tr, lo, hi), _normalise(pb_tr, lo, hi)
    pa_te, pb_te = _normalise(pa_te, lo, hi), _normalise(pb_te, lo, hi)
    Ytr = pair_labels(pa_tr, pb_tr, eps=EPS)
    Ttr = pair_types(pa_tr, pb_tr, eps=EPS)
    Yte = pair_labels(pa_te, pb_te, eps=EPS)
    Tte = pair_types(pa_te, pb_te, eps=EPS)
    print(
        f"[data] train {len(pa_tr)} test {len(pa_te)} "
        f"cells {dict(zip(*np.unique(Tte, return_counts=True)))}",
        flush=True,
    )

    results: dict[str, dict[str, list[dict]]] = {
        arm: {level: [] for level in LEVELS} for arm in ARMS
    }
    calib_rows = []

    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        n = len(pa_tr)
        perm = rng.permutation(n)
        n_cal = max(1, int(round(args.cal_fraction * n)))
        cal_idx, fit_idx = perm[:n_cal], perm[n_cal:]

        pa_fit, pb_fit = pa_tr[fit_idx], pb_tr[fit_idx]
        y_fit = Ytr[fit_idx]
        pa_cal, pb_cal = pa_tr[cal_idx], pb_tr[cal_idx]
        y_cal = Ytr[cal_idx]
        pa_cal_n, pb_cal_n = _noisy(pa_cal, pb_cal, CONTRACT_SIGMA, rng)

        # Hard baselines.
        X_fit = _features(pa_fit, pb_fit)
        head_clean = _train(X_fit, y_fit, "relative", seed, args.epochs, args.device)

        X_aug = [X_fit]
        y_aug = [y_fit]
        for sigma in (0.005, 0.01, 0.02):
            qa, qb = _noisy(pa_fit, pb_fit, sigma, rng)
            X_aug.append(_features(qa, qb))
            y_aug.append(y_fit)
        head_aug = _train(
            np.concatenate(X_aug, axis=0),
            np.concatenate(y_aug, axis=0),
            "relative", seed, args.epochs, args.device,
        )

        # Soft calibration.
        tau0, tau0_score = calibrate_tau(pa_cal, pb_cal, y_cal, eps=EPS)
        tau_fixed, tau_fixed_score = calibrate_tau(
            pa_cal_n, pb_cal_n, y_cal, eps=EPS
        )
        k, k_score = calibrate_k(
            pa_cal_n, pb_cal_n, y_cal, eps=EPS, tau0=tau0,
            sigma=CONTRACT_SIGMA,
        )
        tau_contract = max(tau0, k * CONTRACT_SIGMA)
        thr, sel_acc, sel_cov = calibrate_abstain(
            pa_cal_n, pb_cal_n, y_cal, eps=EPS, tau=tau_contract,
            target_coverage=args.target_coverage,
        )
        sigma_hat = float(
            np.std(
                np.concatenate(
                    [pa_cal_n - pa_cal, pb_cal_n - pb_cal], axis=0
                )
            )
        )
        calib_rows.append(
            {
                "seed": int(seed),
                "tau0": float(tau0),
                "tau0_score": float(tau0_score),
                "tau_fixed": float(tau_fixed),
                "tau_fixed_score": float(tau_fixed_score),
                "k": float(k),
                "k_score": float(k_score),
                "tau_contract": float(tau_contract),
                "abstain_threshold": float(thr),
                "abstain_selective_acc": float(sel_acc),
                "abstain_selective_coverage": float(sel_cov),
                "sigma_hat": sigma_hat,
            }
        )
        print(
            f"[seed {seed}] tau0 {tau0:.4f} k {k:.2f} tau {tau_contract:.4f} "
            f"sigma_hat {sigma_hat:.4f} thr {thr:.2f}",
            flush=True,
        )

        for level in LEVELS:
            rng_eval = np.random.default_rng(seed + 1000)
            qa, qb = _noisy(pa_te, pb_te, level, rng_eval)
            results["hard_clean"][level].append(
                _eval_hard(head_clean, qa, qb, Yte, Tte, args.device)
            )
            results["hard_noise_aug"][level].append(
                _eval_hard(head_aug, qa, qb, Yte, Tte, args.device)
            )
            results["soft_fixed"][level].append(
                _eval_soft(qa, qb, Yte, Tte, tau_fixed)
            )
            tau_true = max(tau0, k * level)
            results["soft_oracle"][level].append(
                _eval_soft(qa, qb, Yte, Tte, tau_true)
            )
            tau_hat = max(tau0, k * sigma_hat)
            results["soft_estimated"][level].append(
                _eval_soft(qa, qb, Yte, Tte, tau_hat)
            )
            results["soft_oracle_abstain"][level].append(
                _eval_soft(qa, qb, Yte, Tte, tau_true, threshold=thr)
            )

    aggregated = {
        arm: {str(level): _agg(rows) for level, rows in by_level.items()}
        for arm, by_level in results.items()
    }

    candidate = "soft_estimated"
    primary = float(aggregated[candidate][str(CONTRACT_SIGMA)]["near_worst"]["mean"])
    clean_guard = float(aggregated[candidate]["0.0"]["near_worst"]["mean"])
    overall_contract = float(aggregated[candidate][str(CONTRACT_SIGMA)]["overall"]["mean"])
    overall_baseline = float(aggregated["hard_clean"][str(CONTRACT_SIGMA)]["overall"]["mean"])

    # Latency: soft is closed-form; hard is a tiny MLP.
    t0 = time.perf_counter()
    soft_predict(pa_te[:1000], pb_te[:1000], eps=EPS, tau=tau_contract)
    soft_ms = (time.perf_counter() - t0) * 1000.0
    t0 = time.perf_counter()
    _predict(head_clean, _features(pa_te[:1000], pb_te[:1000]), args.device)
    hard_ms = (time.perf_counter() - t0) * 1000.0

    guards = {
        "clean_boundary_worst_cell": clean_guard,
        "clean_guard_ok": clean_guard >= 0.933 - 0.02,
        "coverage_worst_cell_constant": 1.000,
        "coverage_guard_ok": True,
        "open_set_auroc_constant": 0.9915,
        "open_set_reject_constant": 0.9618,
        "open_set_guard_ok": True,
        "overall_contract": overall_contract,
        "overall_baseline": overall_baseline,
        "overall_guard_ok": overall_contract >= overall_baseline - 0.02,
        "soft_ms_per_1000": soft_ms,
        "hard_ms_per_1000": hard_ms,
        "latency_guard_ok": soft_ms <= 10.0,
    }
    hard_control = float(
        aggregated["hard_noise_aug"][str(CONTRACT_SIGMA)]["near_worst"]["mean"]
    )
    guards["hard_noise_aug_near_worst"] = hard_control
    guards["h3_margin"] = primary - hard_control
    guards["h3_ok"] = (primary - hard_control) >= args.h3_margin
    check = check_readout_reasoning_preregistration(
        {"u_boundary_soft": {"worst_cell_acc": primary}}
    )

    payload = {
        "stage": "boundary_bench",
        "clevr_root": args.clevr_root,
        "train_split": args.train_split,
        "test_split": args.test_split,
        "contract_sigma": CONTRACT_SIGMA,
        "stress_sigma": STRESS_SIGMA,
        "eps": EPS,
        "epochs": int(args.epochs),
        "seeds": list(args.seeds),
        "train_pairs": int(len(pa_tr)),
        "test_pairs": int(len(pa_te)),
        "calibration": calib_rows,
        "arms": aggregated,
        "candidate": candidate,
        "primary_contract_near_worst": primary,
        "guards": guards,
        "preregistration": check,
        "note": (
            "Primary = near-aligned worst-cell at contract noise (sigma=0.01). "
            "Coverage and open-set guards are constants from U2/U1b, not rerun. "
            "hard_noise_aug is the capacity/data control for H3."
        ),
    }
    _write_json(out / "boundary_bench.json", payload)
    (out / "BOUNDARY_REPORT.md").write_text(_render(payload), encoding="utf-8")
    return payload


def _fmt(row: dict, key: str) -> str:
    return f"{row[key]['mean']:.3f} ± {row[key]['std']:.3f}"


def _render(p: dict) -> str:
    lines = [
        "# PSS-Boundary：噪声边界基准（D1–D5）",
        "",
        f"- CLEVR: `{p['clevr_root']}` · train `{p['train_split']}` "
        f"({p['train_pairs']} pairs) · test `{p['test_split']}` ({p['test_pairs']} pairs)",
        f"- contract σ={p['contract_sigma']} · stress σ={p['stress_sigma']} · eps={p['eps']}",
        f"- epochs {p['epochs']} · seeds {p['seeds']}",
        "",
        "## 1. 主表（3 seeds，mean ± std）",
        "",
        "| arm | level | overall | horizontal | vertical | diagonal | near worst | absence |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        for level in ("0.0", str(p["contract_sigma"]), str(p["stress_sigma"])):
            r = p["arms"][arm][level]
            lines.append(
                f"| `{arm}` | σ={level} | {_fmt(r, 'overall')} | "
                f"{r['cells']['horizontal']:.3f} | {r['cells']['vertical']:.3f} | "
                f"{r['cells']['diagonal']:.3f} | {_fmt(r, 'near_worst')} | "
                f"{_fmt(r, 'absence_error')} |"
            )
    lines += [
        "",
        "## 2. 诊断：主指标与守卫",
        "",
        f"- candidate: `{p['candidate']}` · contract near-worst: "
        f"**{p['primary_contract_near_worst']:.3f}**",
        "",
        "| guard | value | ok |",
        "|---|---:|---|",
    ]
    for key, value in p["guards"].items():
        if key.endswith("_ok"):
            continue
        ok = p["guards"].get(key + "_ok", "—")
        lines.append(f"| `{key}` | {value} | {ok} |")
    lines += [
        "",
        "## 3. 校准",
        "",
        "| seed | tau0 | tau_fixed | k | tau_contract | sigma_hat | abstain thr |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in p["calibration"]:
        lines.append(
            f"| {row['seed']} | {row['tau0']:.4f} | {row['tau_fixed']:.4f} | "
            f"{row['k']:.2f} | {row['tau_contract']:.4f} | {row['sigma_hat']:.4f} | "
            f"{row['abstain_threshold']:.2f} |"
        )
    check = p["preregistration"]
    lines += [
        "",
        "## 4. 预注册检查",
        "",
        f"- 检查 arm 数: {check['checked']} · 通过: **{check['ok']}**",
    ]
    if check["mismatches"]:
        lines.append("")
        for mismatch in check["mismatches"]:
            lines.append(f"- `{mismatch}`")
    guard_ok = all(
        p["guards"].get(key, True)
        for key in (
            "clean_guard_ok", "coverage_guard_ok", "open_set_guard_ok",
            "overall_guard_ok", "latency_guard_ok", "h3_ok",
        )
    )
    go = bool(check["ok"] and guard_ok)
    lines += [
        "",
        "## 5. Go / No-Go",
        "",
        f"**{'GO' if go else 'NO-GO'}**",
        "",
        f"- primary ≥ 0.80: {p['primary_contract_near_worst'] >= 0.80}",
        f"- guards all ok: {guard_ok}",
        f"- H3 (soft − hard_noise_aug ≥ margin): {p['guards'].get('h3_ok')} "
        f"(Δ={p['guards'].get('h3_margin', 0.0):+.3f}, "
        f"control={p['guards'].get('hard_noise_aug_near_worst', 0.0):.3f})",
        f"- preregistration ok: {check['ok']}",
        "",
        "## 6. 判读与边界",
        "",
        "- 主指标是契约噪声下的 near-aligned worst-cell；overall 只是守卫，不允许用它掩盖边界。",
        "- `hard_noise_aug` 是 H3 的容量/数据负对照；如果它追平 soft，增益不能归因于不确定性。",
        "- coverage/open-set 守卫来自 U2/U1b 的常数，不重复跑；它们防的是回退，不是重新证明。",
        "- CLEVR 是渲染数据；σ=0.01 是像素坐标代理，不是真实 RGB-D 标定。",
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
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cal-fraction", type=float, default=0.10)
    ap.add_argument("--target-coverage", type=float, default=0.90)
    ap.add_argument("--h3-margin", type=float, default=0.05)
    ap.add_argument("--out", default="eval/results/boundary_bench")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    payload = run(args)
    print(json.dumps({
        "primary": payload["primary_contract_near_worst"],
        "guards": {k: v for k, v in payload["guards"].items() if k.endswith("_ok")},
        "prereg": payload["preregistration"],
    }, indent=2))


if __name__ == "__main__":
    main()
