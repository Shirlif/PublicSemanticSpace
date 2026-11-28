"""Empirical-Bayes boundary on U6 canonical 3D geometry (3 deg + 1 % / 5 deg + 2 %).

The boundary benchmark (``docs/experiments/PSS-Boundary-Report.md``) validated
the grid-Bayes boundary on **normalised pixel coordinates** and left one open
question: is that a special case of the pixel proxy, or does the same decision
rule work on the canonical 3D -> table-top projection that U6 uses?

This script closes that gap:

* ordered pairs and object indices come from ``run_upper_bound_u6._pair_indices``
  (byte-identical to U3/U6 sampling);
* world 3D centres come from ``_xyz``; the pinhole camera is the U6 DLT fit
  (``_fit_camera``); noisy table-top XY comes from ``_canonical_xy`` with the
  same pose/depth levels and the **same test RNG seeds** as U6, so the noisy
  geometry is the same geometry U6 scores;
* coordinates are min-max normalised with the **train clean** pair endpoints,
  exactly like U6 (range ~[0, 1]);
* the prior is a ``GridBayesBoundary(eps=0.02, bins=257)`` fitted on train
  clean canonical ``(dx, dy)``; the likelihood sigma is **calibrated on the
  train split** (per seed and level) from the pair-difference noise in the same
  normalised units, so no test labels or test coordinates enter calibration;
* test labels and pair types come from clean canonical geometry.  Only the
  boundary input is noisy, matching U6's deployment question.

Levels: clean, pose3_depth1 (3 deg + 1 %), pose5_depth2 (5 deg + 2 %).
Pre-registered arm: ``u_boundary_soft`` read at **pose3_depth1**, metric
``worst_cell_acc`` (here the near-aligned worst cell ``min(horizontal,
vertical)``).  Bar 0.80; a failure is kept in the report and the bar is never
relaxed.

Example
-------
python -u -m pss.eval.scripts.run_boundary_canonical \
  --out eval/results/boundary_canonical --device cpu
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
from pss.eval.scripts.run_upper_bound_u6 import (
    POSE_DEPTH_LEVELS,
    _canonical_xy,
    _fit_camera,
    _pair_indices,
    _xyz,
)
from pss.eval.upper_bound import cell_accuracy, exact4, pair_labels, pair_types
from pss.reasoning.relations.bayes_boundary import GridBayesBoundary

EPS = 0.02
BINS = 257
CONTRACT_SIGMA = 0.01  # pixel-proxy contract, for the reference arm only
LEVELS = tuple(name for name, _, _ in POSE_DEPTH_LEVELS)
PRIMARY_LEVEL = "pose3_depth1"  # 3 deg + 1 %, the pre-registered reading


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


def _normalise_pairs(xy: np.ndarray, ia: np.ndarray, ib: np.ndarray, lo, hi):
    norm = _normalise(np.asarray(xy, dtype=np.float64), lo, hi)
    return norm[ia], norm[ib]


def _calibrate_sigma(clean_diff: np.ndarray, noisy_diff: np.ndarray) -> dict:
    """Per-seed train calibration of the pair-difference noise.

    The boundary sees ``dx, dy`` between two objects, so the relevant scale is
    the noise on the pair difference, not on single-object coordinates.  Pose
    noise is shared per scene, which makes the pair difference less noisy than
    two independent object draws.  Returns per-axis std and the pooled scalar
    the ``GridBayesBoundary`` API needs.
    """

    delta = np.asarray(noisy_diff, dtype=np.float64) - np.asarray(
        clean_diff, dtype=np.float64
    )
    sx = float(np.std(delta[:, 0]))
    sy = float(np.std(delta[:, 1]))
    return {
        "sigma_x": sx,
        "sigma_y": sy,
        "sigma": float(np.sqrt(0.5 * (sx * sx + sy * sy))),
        "bias_x": float(np.mean(delta[:, 0])),
        "bias_y": float(np.mean(delta[:, 1])),
    }


def _states(d: np.ndarray, eps: float = EPS) -> np.ndarray:
    out = np.full(d.shape, 2, dtype=np.int8)
    out[d > eps] = 0
    out[d < -eps] = 1
    return out


def _kernel_bayes(
    prior: np.ndarray,
    prior_states: np.ndarray,
    observed: np.ndarray,
    sigma: float,
    chunk: int = 500,
) -> np.ndarray:
    """Exact kernel-vote Bayes on the same observations as the grid layer.

    ``run_boundary_ceiling`` uses this decision rule and is a genuine noisy-
    observation upper bound.  Comparing it with ``GridBayesBoundary`` on the
    *same* noisy pairs separates "the information is gone" from "the Gaussian
    moment approximation loses it".
    """

    two_sigma2 = 2.0 * float(sigma) ** 2
    out = []
    for start in range(0, len(observed), chunk):
        obs = observed[start : start + chunk]
        d2 = (
            (obs[:, None, 0] - prior[None, :, 0]) ** 2
            + (obs[:, None, 1] - prior[None, :, 1]) ** 2
        )
        w = np.exp(-d2 / two_sigma2)
        w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
        votes = np.zeros((len(obs), 2, 3), dtype=np.float64)
        for state in range(3):
            for axis in range(2):
                votes[:, axis, state] = (
                    w * (prior_states[None, :, axis] == state)
                ).sum(axis=1)
        decided = votes.argmax(axis=2)
        sx, sy = decided[:, 0], decided[:, 1]
        out.append(np.stack([sx == 0, sx == 1, sy == 1, sy == 0], axis=1))
    return np.concatenate(out)


def _summarise(rows: list[dict]) -> dict:
    def stat(values):
        return {"mean": float(np.mean(values)), "std": float(np.std(values))}

    return {
        "overall": stat([r["overall"] for r in rows]),
        "near_worst": stat([r["near_worst"] for r in rows]),
        "absence_error": stat([r["absence_error"] for r in rows]),
        "cells": {
            name: stat([r["cells"].get(name, 0.0) for r in rows])
            for name in ("horizontal", "vertical", "diagonal")
        },
        "n": len(rows),
    }


def _summarise_calibration(rows: list[dict]) -> dict:
    return {
        key: {
            "mean": float(np.mean([r[key] for r in rows])),
            "std": float(np.std([r[key] for r in rows])),
        }
        for key in ("sigma", "sigma_x", "sigma_y", "bias_x", "bias_y")
    }


def _load_u6_comparison(path: Path) -> dict | None:
    if not path.exists():
        return None
    u6 = json.loads(path.read_text(encoding="utf-8"))
    summary = u6.get("summary", {})
    out: dict[str, dict] = {}
    clean = summary.get("clean_canonical")
    if clean:
        out["clean"] = {
            "worst_cell_acc": float(clean["worst_cell_acc"]["mean"]),
            "overall": float(clean["overall"]["mean"]),
            "cells": {
                name: float(clean["cells"][name]["mean"])
                for name in ("horizontal", "vertical", "diagonal")
                if name in clean.get("cells", {})
            },
        }
    for name in LEVELS:
        row = summary.get("canonical", {}).get(name)
        if row:
            out[name] = {
                "worst_cell_acc": float(row["worst_cell_acc"]["mean"]),
                "overall": float(row["overall"]["mean"]),
                "cells": {
                    cell: float(row["cells"][cell]["mean"])
                    for cell in ("horizontal", "vertical", "diagonal")
                    if cell in row.get("cells", {})
                },
            }
    return out or None


def _pixel_reference(train: dict, test: dict, args) -> dict:
    """Same grid-Bayes layer on the pixel proxy, with and without test noise.

    ``run_boundary_bayes.py`` varied the lookup sigma on **clean** test
    coordinates.  The hard/soft arms in ``run_boundary_bench.py`` are scored on
    **actually noisy** coordinates.  This reference evaluates both ways with
    the identical pixel split, so the canonical result can be read against the
    right control.
    """

    pa_tr, pb_tr, _, _ = _pairs(train, args.train_pairs, args.seed)
    pa_te, pb_te, _, _ = _pairs(test, args.test_pairs, args.seed + 1)
    lo = np.minimum(pa_tr.min(0), pb_tr.min(0))
    hi = np.maximum(pa_tr.max(0), pb_tr.max(0))
    pa_tr = _normalise(pa_tr, lo, hi)
    pb_tr = _normalise(pb_tr, lo, hi)
    pa_te = _normalise(pa_te, lo, hi)
    pb_te = _normalise(pb_te, lo, hi)

    dx_tr = pb_tr[:, 0] - pa_tr[:, 0]
    dy_tr = pb_tr[:, 1] - pa_tr[:, 1]
    y = pair_labels(pa_te, pb_te, eps=EPS)
    types = pair_types(pa_te, pb_te, eps=EPS)
    boundary = GridBayesBoundary(eps=EPS, bins=args.bins).fit(dx_tr, dy_tr)

    clean_results = {}
    for sigma in (0.0, CONTRACT_SIGMA):
        bits, _ = boundary.predict(
            pb_te[:, 0] - pa_te[:, 0], pb_te[:, 1] - pa_te[:, 1], sigma=sigma
        )
        clean_results[str(sigma)] = _metrics(bits, y, types)

    noisy_rows = []
    kernel_rows = []
    kernel_prior = np.stack([dx_tr, dy_tr], axis=1)
    if args.kernel_prior_samples and len(kernel_prior) > args.kernel_prior_samples:
        rng_prior = np.random.default_rng(args.seed)
        idx = rng_prior.choice(
            len(kernel_prior), size=args.kernel_prior_samples, replace=False
        )
        kernel_prior = kernel_prior[idx]
    kernel_states = np.stack(
        [_states(kernel_prior[:, 0]), _states(kernel_prior[:, 1])], axis=1
    )
    for seed in args.seeds:
        # Same RNG stream as ``run_boundary_bench``'s evaluation noise.
        rng = np.random.default_rng(seed + 1000)
        qa = pa_te + rng.normal(0.0, CONTRACT_SIGMA, pa_te.shape)
        qb = pb_te + rng.normal(0.0, CONTRACT_SIGMA, pb_te.shape)
        observed = np.stack([qb[:, 0] - qa[:, 0], qb[:, 1] - qa[:, 1]], axis=1)
        bits, _ = boundary.predict(
            observed[:, 0], observed[:, 1], sigma=CONTRACT_SIGMA
        )
        noisy_rows.append(_metrics(bits, y, types))
        kernel_bits = _kernel_bayes(
            kernel_prior, kernel_states, observed, sigma=CONTRACT_SIGMA
        )
        kernel_rows.append(_metrics(kernel_bits, y, types))

    return {
        "contract_sigma": CONTRACT_SIGMA,
        "clean_inputs": {
            "lookup_sigma_0.0": clean_results["0.0"],
            "lookup_sigma_0.01": clean_results[str(CONTRACT_SIGMA)],
        },
        "noisy_inputs": _summarise(noisy_rows),
        "noisy_inputs_kernel": _summarise(kernel_rows),
        "note": (
            "clean_inputs reproduces run_boundary_bayes (clean test coords, "
            "noise-conditioned lookup); noisy_inputs adds the contract noise "
            "to the test coords, which is what the hard baseline faces. "
            "noisy_inputs_kernel is the exact kernel vote on the same noisy "
            "observations; the ceiling report's 0.971 is its *overall* exact-4, "
            "not its near-worst cell."
        ),
    }


def run(args) -> dict:
    root = Path(args.clevr_root)
    train = _load_cache(root, args.train_split)
    test = _load_cache(root, args.test_split)

    ia_tr, ib_tr = _pair_indices(train, args.train_pairs, args.seed)
    ia_te, ib_te = _pair_indices(test, args.test_pairs, args.seed + 1)

    xyz_tr = _xyz(train)
    xyz_te = _xyz(test)
    scene_tr = np.asarray(train["scene"])
    scene_te = np.asarray(test["scene"])
    cam = _fit_camera(train, args.camera_points, args.seed)

    # Normalisation and pair endpoints exactly as U6 does for canonical.
    lo = np.minimum(xyz_tr[ia_tr, :2].min(0), xyz_tr[ib_tr, :2].min(0))
    hi = np.maximum(xyz_tr[ia_tr, :2].max(0), xyz_tr[ib_tr, :2].max(0))

    clean_tr_xy = _normalise(xyz_tr[:, :2], lo, hi)
    clean_te_xy = _normalise(xyz_te[:, :2], lo, hi)
    d_tr = clean_tr_xy[ib_tr] - clean_tr_xy[ia_tr]
    d_te_clean = clean_te_xy[ib_te] - clean_te_xy[ia_te]
    pa_te_clean, pb_te_clean = clean_te_xy[ia_te], clean_te_xy[ib_te]

    Yte = pair_labels(pa_te_clean, pb_te_clean, eps=EPS)
    Tte = pair_types(pa_te_clean, pb_te_clean, eps=EPS)

    pa_tr_clean, pb_tr_clean = clean_tr_xy[ia_tr], clean_tr_xy[ib_tr]
    weights = None
    if args.balance_strength > 0:
        types_tr = pair_types(pa_tr_clean, pb_tr_clean, eps=EPS)
        counts = {
            kind: max(int((types_tr == kind).sum()), 1)
            for kind in ("horizontal", "vertical", "diagonal")
        }
        weights = np.asarray(
            [
                (1.0 / 3.0 / (counts[kind] / len(types_tr)))
                ** float(args.balance_strength)
                for kind in types_tr
            ],
            dtype=np.float64,
        )
        weights /= weights.sum()
        print(
            f"[balance] canonical pair counts {counts}; "
            f"strength {args.balance_strength}",
            flush=True,
        )

    boundary = GridBayesBoundary(eps=EPS, bins=args.bins).fit(
        d_tr[:, 0], d_tr[:, 1], weights=weights
    )
    clean_bits, clean_conf = boundary.predict(
        d_te_clean[:, 0], d_te_clean[:, 1], sigma=0.0
    )
    clean_metrics = _metrics(clean_bits, Yte, Tte)
    clean_metrics["mean_confidence"] = float(clean_conf.mean())

    level_rows: dict[str, list[dict]] = {name: [] for name in LEVELS}
    level_kernel: dict[str, list[dict]] = {name: [] for name in LEVELS}
    level_calib: dict[str, list[dict]] = {name: [] for name in LEVELS}

    # Kernel-Bayes reference prior: train canonical pair differences.
    kernel_prior = d_tr
    if args.kernel_prior_samples and len(kernel_prior) > args.kernel_prior_samples:
        rng_prior = np.random.default_rng(args.seed)
        idx = rng_prior.choice(
            len(kernel_prior), size=args.kernel_prior_samples, replace=False
        )
        kernel_prior = kernel_prior[idx]
    kernel_states = np.stack(
        [_states(kernel_prior[:, 0]), _states(kernel_prior[:, 1])], axis=1
    )

    print(
        f"[data] train pairs {len(ia_tr)} test pairs {len(ia_te)} "
        f"cells {dict(zip(*np.unique(Tte, return_counts=True)))}",
        flush=True,
    )
    print(
        f"[camera] rms {cam['rms_px']:.2f} px det(R) {cam['det_R']:+.3f} "
        f"C {np.round(cam['C'], 2).tolist()} · "
        f"normalisation lo {np.round(lo, 4).tolist()} hi {np.round(hi, 4).tolist()}",
        flush=True,
    )

    for seed in args.seeds:
        for j, (name, yaw_deg, depth_sigma) in enumerate(POSE_DEPTH_LEVELS):
            # Train calibration noise (distinct RNG stream from U6's test).
            rng_cal = np.random.default_rng(seed * 100000 + j + 7)
            xy_tr_noisy = _canonical_xy(
                xyz_tr,
                scene_tr,
                cam,
                yaw_deg=yaw_deg,
                pitch_deg=yaw_deg,
                depth_sigma=depth_sigma,
                rng=rng_cal,
            )
            ia_tr_n, ib_tr_n = _normalise_pairs(xy_tr_noisy, ia_tr, ib_tr, lo, hi)
            calib = _calibrate_sigma(d_tr, ib_tr_n - ia_tr_n)
            level_calib[name].append(calib)

            # Test noise: same seeds as U6's combined canonical level.
            rng_te = np.random.default_rng(seed * 100000 + j + 17)
            xy_te_noisy = _canonical_xy(
                xyz_te,
                scene_te,
                cam,
                yaw_deg=yaw_deg,
                pitch_deg=yaw_deg,
                depth_sigma=depth_sigma,
                rng=rng_te,
            )
            ia_te_n, ib_te_n = _normalise_pairs(xy_te_noisy, ia_te, ib_te, lo, hi)
            d_te_noisy = ib_te_n - ia_te_n
            bits, conf = boundary.predict(
                d_te_noisy[:, 0], d_te_noisy[:, 1], sigma=calib["sigma"]
            )
            row = _metrics(bits, Yte, Tte)
            row["mean_confidence"] = float(conf.mean())
            level_rows[name].append(row)

            kernel_bits = _kernel_bayes(
                kernel_prior,
                kernel_states,
                d_te_noisy,
                sigma=calib["sigma"],
            )
            kernel_row = _metrics(kernel_bits, Yte, Tte)
            level_kernel[name].append(kernel_row)
            print(
                f"[seed {seed}] {name} sigma {calib['sigma']:.4f} "
                f"overall {row['overall']:.3f} "
                f"h {row['cells'].get('horizontal', 0):.3f} "
                f"v {row['cells'].get('vertical', 0):.3f} "
                f"d {row['cells'].get('diagonal', 0):.3f} "
                f"near {row['near_worst']:.3f} "
                f"kernel near {kernel_row['near_worst']:.3f}",
                flush=True,
            )

    levels = {
        name: {
            "metrics": _summarise(level_rows[name]),
            "kernel": _summarise(level_kernel[name]),
            "calibration": _summarise_calibration(level_calib[name]),
        }
        for name in LEVELS
    }
    primary = float(levels[PRIMARY_LEVEL]["metrics"]["near_worst"]["mean"])
    check = check_readout_reasoning_preregistration(
        {"u_boundary_soft": {"worst_cell_acc": primary}}
    )
    comparison = _load_u6_comparison(Path(args.u6_json))

    payload = {
        "stage": "boundary_canonical",
        "clevr_root": str(root),
        "train_split": args.train_split,
        "test_split": args.test_split,
        "train_pairs": int(len(ia_tr)),
        "test_pairs": int(len(ia_te)),
        "eps": EPS,
        "bins": int(args.bins),
        "seeds": list(args.seeds),
        "pose_depth_levels": [
            {"name": name, "yaw_pitch_deg": yaw, "depth_sigma": depth_sigma}
            for name, yaw, depth_sigma in POSE_DEPTH_LEVELS
        ],
        "normalisation": {"lo": lo.tolist(), "hi": hi.tolist()},
        "camera": cam,
        "clean": {
            "metrics": clean_metrics,
            "note": "clean canonical is deterministic; evaluated once",
        },
        "levels": levels,
        "primary_level": PRIMARY_LEVEL,
        "primary_near_worst": primary,
        "balance_strength": float(args.balance_strength),
        "comparison_u6_hard": comparison,
        "pixel_reference": _pixel_reference(train, test, args),
        "preregistration": check,
        "note": (
            "canonical 3D->table-top projection; Bayes sigma is calibrated on "
            "the train split per seed/level from pair-difference noise in "
            "normalised units. Test labels/types stay clean; only the boundary "
            "input is noisy. No real RGB-D calibration."
        ),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "boundary_canonical.json", payload)
    (out / "BOUNDARY_CANONICAL_RESULTS.md").write_text(
        _render(payload), encoding="utf-8"
    )
    return payload


def _fmt(stat: dict) -> str:
    return f"{stat['mean']:.3f} ± {stat['std']:.3f}"


def _render(p: dict) -> str:
    lines = [
        "# Boundary canonical — empirical-Bayes boundary on U6 3D noise",
        "",
        f"- CLEVR: `{p['clevr_root']}` · train {p['train_pairs']} / "
        f"test {p['test_pairs']} pairs · eps={p['eps']} · bins={p['bins']}",
        f"- camera DLT rms {p['camera']['rms_px']:.2f} px · "
        f"det(R) {p['camera']['det_R']:+.3f}",
        "",
        "## 1. Main table (mean ± std over seeds)",
        "",
        "| level | overall | horizontal | vertical | diagonal | near worst | absence |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    c = p["clean"]["metrics"]
    lines.append(
        f"| clean | {c['overall']:.3f} | {c['cells'].get('horizontal', 0):.3f} | "
        f"{c['cells'].get('vertical', 0):.3f} | {c['cells'].get('diagonal', 0):.3f} | "
        f"**{c['near_worst']:.3f}** | {c['absence_error']:.3f} |"
    )
    for name, _, _ in POSE_DEPTH_LEVELS:
        row = p["levels"][name]["metrics"]
        lines.append(
            f"| {name} | {_fmt(row['overall'])} | {_fmt(row['cells']['horizontal'])} | "
            f"{_fmt(row['cells']['vertical'])} | {_fmt(row['cells']['diagonal'])} | "
            f"**{_fmt(row['near_worst'])}** | {_fmt(row['absence_error'])} |"
        )
    lines += [
        "",
        "### Same noisy observations: grid moment vs exact kernel vote",
        "",
        "| level | grid near worst | kernel near worst | grid overall | kernel overall |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, _, _ in POSE_DEPTH_LEVELS:
        grid = p["levels"][name]["metrics"]
        ker = p["levels"][name]["kernel"]
        lines.append(
            f"| {name} | {_fmt(grid['near_worst'])} | **{_fmt(ker['near_worst'])}** | "
            f"{_fmt(grid['overall'])} | {_fmt(ker['overall'])} |"
        )
    lines += [
        "",
        "### Train-calibrated sigma (normalised units)",
        "",
        "| level | sigma_x | sigma_y | pooled sigma | bias_x | bias_y |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, _, _ in POSE_DEPTH_LEVELS:
        cal = p["levels"][name]["calibration"]
        lines.append(
            f"| {name} | {cal['sigma_x']['mean']:.4f} | {cal['sigma_y']['mean']:.4f} | "
            f"{cal['sigma']['mean']:.4f} | {cal['bias_x']['mean']:+.2e} | "
            f"{cal['bias_y']['mean']:+.2e} |"
        )
    if p.get("comparison_u6_hard"):
        lines += [
            "",
            "## 2. Comparison with the U6 hard baseline",
            "",
            "| level | U6 hard worst cell | Bayes near worst | delta |",
            "|---|---:|---:|---:|",
        ]
        for name in ("clean",) + tuple(n for n, _, _ in POSE_DEPTH_LEVELS):
            hard = p["comparison_u6_hard"].get(name)
            if not hard:
                continue
            if name == "clean":
                bayes = p["clean"]["metrics"]["near_worst"]
            else:
                bayes = p["levels"][name]["metrics"]["near_worst"]["mean"]
            lines.append(
                f"| {name} | {hard['worst_cell_acc']:.3f} | {bayes:.3f} | "
                f"{bayes - hard['worst_cell_acc']:+.3f} |"
            )
    check = p["preregistration"]
    ref = p.get("pixel_reference")
    if ref:
        ci = ref["clean_inputs"]
        ni = ref["noisy_inputs"]
        nk = ref["noisy_inputs_kernel"]
        lines += [
            "",
            "## 3. Pixel reference (same Bayes layer, same pixel split)",
            "",
            "| inputs | lookup sigma | overall | near worst |",
            "|---|---:|---:|---:|",
            f"| clean | 0.00 | {ci['lookup_sigma_0.0']['overall']:.3f} | "
            f"{ci['lookup_sigma_0.0']['near_worst']:.3f} |",
            f"| clean | 0.01 | {ci['lookup_sigma_0.01']['overall']:.3f} | "
            f"**{ci['lookup_sigma_0.01']['near_worst']:.3f}** |",
            f"| noisy (sigma 0.01), grid | 0.01 | {_fmt(ni['overall'])} | "
            f"**{_fmt(ni['near_worst'])}** |",
            f"| noisy (sigma 0.01), kernel | 0.01 | {_fmt(nk['overall'])} | "
            f"**{_fmt(nk['near_worst'])}** |",
            "",
            f"- {ref['note']}",
        ]
    lines += [
        "",
        "## 4. Preregistration `u_boundary_soft`",
        "",
        f"- read at `{p['primary_level']}` · near-worst {p['primary_near_worst']:.3f} "
        f"(bar 0.80)",
        f"- checked {check['checked']} · ok: **{check['ok']}**",
    ]
    if check["mismatches"]:
        for mismatch in check["mismatches"]:
            lines.append(f"- `{mismatch}`")
    lines += [
        "",
        "## 5. Boundaries",
        "",
        "- Not a real RGB-D calibration: CLEVR is rendered geometry and the "
        "pose/depth errors are isotropic Gaussian proxies.",
        "- The Bayes sigma is a scalar; per-axis sigma_x/sigma_y are reported "
        "but the API uses the pooled value.",
        "- Test labels/types come from clean geometry, so this measures decision "
        "robustness, not noisy-label robustness.",
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
    ap.add_argument("--bins", type=int, default=BINS)
    ap.add_argument("--balance-strength", type=float, default=0.0)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--camera-points", type=int, default=4000)
    ap.add_argument(
        "--kernel-prior-samples",
        type=int,
        default=0,
        help="subsample the kernel-Bayes prior; 0 = use all train pairs",
    )
    ap.add_argument(
        "--u6-json",
        default="eval/results/upper_bound/u6/u6.json",
    )
    ap.add_argument("--out", default="eval/results/boundary_canonical")
    args = ap.parse_args()
    payload = run(args)
    print(json.dumps(payload["preregistration"], indent=2))


if __name__ == "__main__":
    main()
