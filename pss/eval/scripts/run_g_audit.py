"""G0–G4: coverage / matching audit for the g factor.

Executable form of ``docs/plans/PSS-g因子优化实验计划.md``:

  b_audit       E1 — per-cell support / error / contribution on the B line
  b_response    E2 — support -> worst-cell exact-4 curve (single vs combination)
  b_invariance  E3 — absolute / relative / normalized / scale-invariant probe
  a_audit       E4 — modality x class x view cells on the A line
  final         G6 — read the stage JSONs and report L1–L3

The B line uses the synthetic P5b grid (5 regions) and the 7x7 geometry from
``run_geometry_validation``.  The A line reuses the cached paired features from
``eval/results/readout_opt`` so the audit measures the readout, not feature
extraction.

Examples
--------
python -m pss.eval.scripts.run_g_audit --stage b_audit \
  --out eval/results/g_audit
python -m pss.eval.scripts.run_g_audit --stage b_response \
  --out eval/results/g_audit
python -m pss.eval.scripts.run_g_audit --stage b_invariance \
  --out eval/results/g_audit
python -m pss.eval.scripts.run_g_audit --stage a_audit \
  --out eval/results/g_audit
python -m pss.eval.scripts.run_g_audit --stage final \
  --out eval/results/g_audit
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pss._console import ensure_utf8
from pss.eval.g_audit import (
    coverage_cells,
    coverage_dashboard,
    mean_std,
    paired_bootstrap_ci,
    worst_cell_exact4,
)
from pss.eval.readout_reasoning_preregistration import (
    check_readout_reasoning_preregistration,
)
from pss.eval.relational_bench import NAMES, REGIONS, RELATIONS
from pss.eval.scripts.run_attribution_grid import (
    MODALITIES,
    build_bank,
    concat,
    fit_mlp,
    fused_scores,
    load_features,
    task_split,
)
from pss.eval.scripts.run_geometry_validation import (
    TYPES,
    _extra_rows,
    _grid,
    _held_out_pairs,
    _isolation,
    _labels,
    _pair_type as _grid_pair_type,
    _pairs_within,
    _predict as _grid_predict,
    _subset_names,
)
from pss.eval.scripts.run_relation_per_label import (
    _pair_type as _p5b_pair_type,
    _probs as _p5b_probs,
    _true_labels as _p5b_true_labels,
)
from pss.eval.scripts.run_typed_relations import _train_spatial_head
from pss.projection.heads import load_projection
from pss.prototypes.scorers import MahalanobisScorer, predict_classes
from pss.reasoning.relations.coverage import ordered_pairs, unseen_pairs

P5B_SUBSETS = ("BCD", "ABD", "ACG")
GRID7_SUBSETS = ("block3", "strip_x", "strip_y", "diag")
FEATURE_MODES = ("absolute", "relative", "normalized", "scale_invariant")
TASK_MODES = ("relative", "absolute", "mixed")
EPS = 1e-9


# --------------------------------------------------------------------------- #
# generic feature / label helpers (E3 needs more than the two legacy modes)
# --------------------------------------------------------------------------- #
def _pair_feature_g(pa, pb, mode: str) -> list[float]:
    pa = np.asarray(pa, dtype=np.float32)
    pb = np.asarray(pb, dtype=np.float32)
    dx = float(pb[0] - pa[0])
    dy = float(pb[1] - pa[1])
    if mode == "absolute":
        return [
            float(pa[0]), float(pa[1]), float(pb[0]), float(pb[1]),
            dx, dy, abs(dx), abs(dy),
        ]
    if mode == "relative":
        return [dx, dy, abs(dx), abs(dy)]
    if mode == "normalized":
        scale = abs(dx) + abs(dy)
        scale = scale if scale > 1e-12 else 1.0
        return [dx / scale, dy / scale, abs(dx) / scale, abs(dy) / scale]
    if mode == "scale_invariant":
        scale = max(abs(dx), abs(dy))
        scale = scale if scale > 1e-12 else 1.0
        return [dx / scale, dy / scale, abs(dx) / scale, abs(dy) / scale]
    raise ValueError(f"unknown feature mode {mode!r}")


def _absolute_labels(pa, pb, eps: float = 0.1) -> list[bool]:
    return [pa[0] < -eps, pa[0] > eps, pa[1] > eps, pa[1] < -eps]


def _labels_g(pa, pb, mode: str) -> list[bool]:
    if mode == "relative":
        return [bool(x) for x in _labels(pa, pb, "relative")]
    if mode == "absolute":
        return [bool(x) for x in _absolute_labels(pa, pb)]
    if mode == "mixed":
        return [bool(x) for x in _labels(pa, pb, "relative")] + [
            bool(x) for x in _absolute_labels(pa, pb)
        ]
    raise ValueError(f"unknown task mode {mode!r}")


def _matrix_g(pairs, coords, feature_mode, task_mode, repeats=1, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for a, b in pairs:
        for _ in range(int(repeats)):
            pa = coords[a] + rng.normal(scale=noise, size=2)
            pb = coords[b] + rng.normal(scale=noise, size=2)
            X.append(_pair_feature_g(pa, pb, feature_mode))
            y.append(_labels_g(pa, pb, task_mode))
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


class _Head(nn.Module):
    """Small MLP for the synthetic relation probes (variable in/out dim)."""

    def __init__(self, in_dim: int, out_dim: int = 4, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def _train_head(X, y, *, seed: int, epochs: int, device: str) -> _Head:
    torch.manual_seed(int(seed))
    head = _Head(int(X.shape[1]), int(y.shape[1])).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y, dtype=torch.float32, device=device)
    for _ in range(int(epochs)):
        perm = torch.randperm(len(Xt), device=device)
        for start in range(0, len(perm), 128):
            idx = perm[start:start + 128]
            loss = F.binary_cross_entropy_with_logits(head(Xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    head.eval()
    return head


def _predict_head(head, X, device: str) -> np.ndarray:
    with torch.no_grad():
        logits = head(torch.as_tensor(X, dtype=torch.float32, device=device))
        return (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(bool)


# --------------------------------------------------------------------------- #
# generic JSON helpers
# --------------------------------------------------------------------------- #
def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, bool):
        return obj
    return obj


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "n/a" if math.isnan(value) else f"{value:.{digits}f}"


# --------------------------------------------------------------------------- #
# E1: B-line coverage audit
# --------------------------------------------------------------------------- #
def _absence_key(dx: float, dy: float) -> str:
    if abs(dy) <= EPS:
        return "dy_zero"
    if abs(dx) <= EPS:
        return "dx_zero"
    return "general"


def _audit_rows(rows: list[dict]) -> dict:
    """Turn per-seed pair rows into coverage dashboards."""

    type_pred_items = []
    type_items = []
    pair_items = []
    absence_items = []
    for row in rows:
        for predicate, correct in zip(RELATIONS, row["correct"]):
            type_pred_items.append((f"{row['type']}|{predicate}", float(correct), 1))
        type_items.append((row["type"], float(row["exact"]), 1))
        pair_items.append((row["pair"], float(row["exact"]), 1))
        absence_items.append(
            (_absence_key(row["dx"], row["dy"]), float(row["exact"]), 1)
        )

    expected_type_pred = [f"{t}|{p}" for t in TYPES for p in RELATIONS]
    dashboards = {
        "by_type_predicate": coverage_dashboard(
            coverage_cells(type_pred_items), expected_cells=expected_type_pred
        ),
        "by_type": coverage_dashboard(
            coverage_cells(type_items), expected_cells=TYPES
        ),
        "by_pair": coverage_dashboard(coverage_cells(pair_items)),
        "by_absence": coverage_dashboard(
            coverage_cells(absence_items),
            expected_cells=("dy_zero", "dx_zero", "general"),
        ),
    }
    per_seed = defaultdict(list)
    for row in rows:
        per_seed[int(row["seed"])].append(float(row["exact"]))
    return {
        "overall_exact4": mean_std([np.mean(v) for v in per_seed.values()]),
        "per_seed_exact4": {
            str(seed): float(np.mean(values)) for seed, values in sorted(per_seed.items())
        },
        "worst_type_exact4": worst_cell_exact4(
            rows, lambda r: r["type"], min_support=max(5, len(per_seed))
        ),
        "dashboards": dashboards,
    }


def _p5b_rows(head, subset, seed, device, feature_mode) -> list[dict]:
    rows = []
    for a, b in unseen_pairs(NAMES, subset):
        probs = _p5b_probs(head, a, b, device, feature_mode)
        truth = _p5b_true_labels(a, b)
        pred = [p > 0.5 for p in probs]
        rows.append(
            {
                "subset": "".join(subset),
                "seed": int(seed),
                "pair": f"{a}->{b}",
                "type": _p5b_pair_type(a, b),
                "dx": float(REGIONS[b][0] - REGIONS[a][0]),
                "dy": float(REGIONS[b][1] - REGIONS[a][1]),
                "true": truth,
                "pred": pred,
                "correct": [bool(t == p) for t, p in zip(truth, pred)],
                "exact": bool(all(t == p for t, p in zip(truth, pred))),
            }
        )
    return rows


def _grid_rows(head, pairs, coords, seed, device, feature_mode="relative",
               task_mode="relative") -> list[dict]:
    X, _ = _matrix_g(pairs, coords, feature_mode, task_mode, repeats=1, noise=0.0)
    pred = _predict_head(head, X, device) if isinstance(head, _Head) else _grid_predict(
        head, X, device
    )
    truth = np.asarray(
        [_labels_g(coords[a], coords[b], task_mode) for a, b in pairs], dtype=bool
    )
    rows = []
    for i, (a, b) in enumerate(pairs):
        rows.append(
            {
                "seed": int(seed),
                "pair": f"{a}->{b}",
                "type": _grid_pair_type(coords[a], coords[b]),
                "dx": float(coords[b][0] - coords[a][0]),
                "dy": float(coords[b][1] - coords[a][1]),
                "true": truth[i].tolist(),
                "pred": pred[i].tolist(),
                "correct": (truth[i] == pred[i]).tolist(),
                "exact": bool(np.all(truth[i] == pred[i])),
            }
        )
    return rows


def _train_p5b_head(train_regions, seed, epochs, device, feature_mode="relative",
                    extra_horizontal=0, extra_vertical=0, n_per_pair=300):
    head, _, exact, per_label = _train_spatial_head(
        device,
        epochs=epochs,
        seed=seed,
        train_regions=train_regions,
        extra_horizontal=extra_horizontal,
        extra_vertical=extra_vertical,
        noise=0.03,
        n_per_pair=n_per_pair,
        feature_mode=feature_mode,
    )
    return head, exact, per_label


def _train_grid7_head(train_pairs, coords, seed, epochs, device, feature_mode="relative",
                      repeats=50):
    X, y = _matrix_g(train_pairs, coords, feature_mode, "relative",
                     repeats=repeats, noise=0.03, seed=seed)
    return _train_head(X, y, seed=seed, epochs=epochs, device=device)


def _p5b_profiles(subsets) -> dict:
    profiles = {}
    for subset in subsets:
        key = "".join(subset)
        pairs = ordered_pairs(subset)
        x_iso = y_iso = False
        positives = {p: 0 for p in RELATIONS}
        for a, b in pairs:
            truth = _p5b_true_labels(a, b)
            for predicate, label in zip(RELATIONS, truth):
                positives[predicate] += int(label)
            dx = REGIONS[b][0] - REGIONS[a][0]
            dy = REGIONS[b][1] - REGIONS[a][1]
            if abs(dx) > EPS and abs(dy) <= EPS:
                x_iso = True
            if abs(dy) > EPS and abs(dx) <= EPS:
                y_iso = True
        profiles[key] = {
            "regions": list(subset),
            "n_train_pairs": len(pairs),
            "x_isolating": x_iso,
            "y_isolating": y_iso,
            "positives": positives,
        }
    return profiles


def _grid_profiles(subset_kinds, n, max_test) -> dict:
    coords = _grid(n)
    all_names = list(coords)
    profiles = {}
    for kind in subset_kinds:
        names = _subset_names(coords, kind)
        test_pairs = _held_out_pairs(all_names, names, max_test, seed=4321)
        type_counts = {t: 0 for t in TYPES}
        for a, b in test_pairs:
            type_counts[_grid_pair_type(coords[a], coords[b])] += 1
        has_diagonal = any(
            _grid_pair_type(coords[a], coords[b]) == "diagonal"
            for a, b in _pairs_within(names)
        )
        profiles[kind] = {
            "names": names,
            "n_train_pairs": len(_pairs_within(names)),
            "n_test_pairs": len(test_pairs),
            "isolation": _isolation(names, coords),
            "test_type_counts": type_counts,
            "has_diagonal": has_diagonal,
        }
    return coords, profiles


def stage_b_audit(args) -> dict:
    out = Path(args.out)
    device = args.device
    seeds = [int(s) for s in args.seeds]
    n = int(args.grid_n)

    payload = {
        "stage": "b_audit",
        "task": "B-line coverage audit (pair_type x predicate x region_pair x absence)",
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "config": {
            "device": device,
            "seeds": seeds,
            "epochs": int(args.epochs),
            "p5b_subsets": list(args.p5b_subsets),
            "grid_subsets": list(args.grid_subsets),
            "grid_n": n,
            "n_per_pair": int(args.n_per_pair),
            "feature_mode": args.b_feature,
            "full_epochs": int(args.full_epochs),
            "full_repeats": int(args.full_repeats),
        },
        "families": {},
    }

    # -- P5b family ---------------------------------------------------------
    p5b_profiles = _p5b_profiles(args.p5b_subsets)
    p5b_rows = {}
    for subset in args.p5b_subsets:
        key = "".join(subset)
        p5b_rows[key] = []
        for seed in seeds:
            head, exact, per_label = _train_p5b_head(
                subset, seed, args.epochs, device, args.b_feature,
                n_per_pair=args.p5b_n_per_pair,
            )
            p5b_rows[key].extend(_p5b_rows(head, subset, seed, device, args.b_feature))
            print(f"[b_audit] p5b {key} seed {seed}: exact4={exact:.3f} "
                  f"per-label={per_label:.3f}", flush=True)

    # full-coverage reference, one head per seed, evaluated on every subset pool
    p5b_full = {key: [] for key in p5b_profiles}
    for seed in seeds:
        head, exact, per_label = _train_p5b_head(
            None, seed, args.full_epochs, device, args.b_feature,
            n_per_pair=args.p5b_n_per_pair,
        )
        print(f"[b_audit] p5b FULL seed {seed}: exact4={exact:.3f} "
              f"per-label={per_label:.3f}", flush=True)
        for subset in args.p5b_subsets:
            key = "".join(subset)
            rows = _p5b_rows(head, subset, seed, device, args.b_feature)
            p5b_full[key].append(np.mean([r["exact"] for r in rows]))

    p5b_family = {"subsets": {}, "profiles": p5b_profiles}
    for key, rows in p5b_rows.items():
        audit = _audit_rows(rows)
        audit["reference"] = {
            "a_current": audit["overall_exact4"]["mean"],
            "a_full": mean_std(p5b_full[key])["mean"],
            "a_ceiling": 1.0,
            "a_ceiling_note": "synthetic 5-region grid with a dead zone",
        }
        p5b_family["subsets"][key] = audit
    payload["families"]["p5b"] = p5b_family

    # -- 7x7 family ---------------------------------------------------------
    coords, grid_profiles = _grid_profiles(args.grid_subsets, n, args.max_test)
    all_names = list(coords)
    grid_rows = {kind: [] for kind in args.grid_subsets}
    for kind in args.grid_subsets:
        names = grid_profiles[kind]["names"]
        train_pairs = _pairs_within(names)
        test_pairs = _held_out_pairs(all_names, names, args.max_test, seed=4321)
        for seed in seeds:
            head = _train_grid7_head(
                train_pairs, coords, seed, args.epochs, device,
                args.b_feature, repeats=args.n_per_pair,
            )
            rows = _grid_rows(head, test_pairs, coords, seed, device, args.b_feature)
            grid_rows[kind].extend(rows)
            print(f"[b_audit] grid7 {kind} seed {seed}: "
                  f"exact4={np.mean([r['exact'] for r in rows]):.3f}", flush=True)

    all_pairs = _pairs_within(all_names)
    grid_full = {kind: [] for kind in args.grid_subsets}
    for seed in seeds:
        head = _train_grid7_head(
            all_pairs, coords, seed, args.full_epochs, device,
            args.b_feature, repeats=args.full_repeats,
        )
        for kind in args.grid_subsets:
            names = grid_profiles[kind]["names"]
            test_pairs = _held_out_pairs(all_names, names, args.max_test, seed=4321)
            rows = _grid_rows(head, test_pairs, coords, seed, device, args.b_feature)
            grid_full[kind].append(np.mean([r["exact"] for r in rows]))
            print(f"[b_audit] grid7 FULL seed {seed} -> {kind}: "
                  f"exact4={grid_full[kind][-1]:.3f}", flush=True)

    grid_family = {
        "subsets": {},
        "profiles": {
            kind: {k: v for k, v in profile.items() if k != "names"}
            for kind, profile in grid_profiles.items()
        },
    }
    for kind, rows in grid_rows.items():
        audit = _audit_rows(rows)
        audit["reference"] = {
            "a_current": audit["overall_exact4"]["mean"],
            "a_full": mean_std(grid_full[kind])["mean"],
            "a_ceiling": 1.0,
            "a_ceiling_note": "synthetic 7x7 grid, relative labels with a dead zone",
        }
        grid_family["subsets"][kind] = audit
    payload["families"]["grid7"] = grid_family

    # -- H1 summary ---------------------------------------------------------
    # A near-saturated subset has almost no residual error, so its "share"
    # is trivially 1.0 on one stray mistake.  H1 is about where the *remaining*
    # error sits, so only subsets with meaningful residual error are ranked.
    candidates = []
    for family, data in payload["families"].items():
        for subset, audit in data["subsets"].items():
            dash = audit["dashboards"]["by_type_predicate"]
            overall = audit["overall_exact4"]["mean"]
            if overall >= 0.99 or dash["total_error"] <= 0.01:
                continue
            candidates.append(
                {
                    "family": family,
                    "subset": subset,
                    "worst_cell": dash["worst_cell"],
                    "worst_cell_error": dash["worst_cell_error"],
                    "worst_contribution_share": dash["worst_contribution_share"],
                    "overall_exact4": overall,
                    "worst_type_exact4": audit["worst_type_exact4"]["value"],
                    "a_full": audit["reference"]["a_full"],
                }
            )
    candidates.sort(key=lambda c: (-c["worst_contribution_share"], c["worst_type_exact4"]))
    payload["summary"] = {
        "h1_worst_cells": candidates,
        "h1_supported": bool(
            candidates and candidates[0]["worst_contribution_share"] >= 0.30
        ),
        "h1_threshold": 0.30,
        "h1_excluded": "subsets with overall exact-4 >= 0.99 (no residual error)",
        "worst_subset": candidates[0]["subset"] if candidates else None,
        "worst_family": candidates[0]["family"] if candidates else None,
    }

    _write_json(out / "b_coverage.json", payload)
    (out / "B_COVERAGE.md").write_text(_render_b_audit(payload), encoding="utf-8")
    return payload


def _render_b_audit(p: dict) -> str:
    lines = [
        "# B 线覆盖审计（E1 / G1）",
        "",
        f"- generated: {p['generated']}",
        f"- seeds: {p['config']['seeds']} · epochs: {p['config']['epochs']} · "
        f"feature: {p['config']['feature_mode']} · grid: {p['config']['grid_n']}x"
        f"{p['config']['grid_n']}",
        "- cell 定义：`pair_type × predicate`（误差分解）、`pair_type`（exact-4）、"
        "`region_pair`、`absence`。",
        "",
        "## 1. 总体与参照",
        "",
        "| family | subset | overall exact-4 | worst type exact-4 | A_full | worst cell | "
        "contribution share |",
        "|---|---|---:|---:|---:|---|---:|",
    ]
    for family, data in p["families"].items():
        for subset, audit in data["subsets"].items():
            dash = audit["dashboards"]["by_type_predicate"]
            lines.append(
                f"| {family} | {subset} | {_fmt(audit['overall_exact4']['mean'])} ± "
                f"{_fmt(audit['overall_exact4']['std'])} | "
                f"{_fmt(audit['worst_type_exact4']['value'])} | "
                f"{_fmt(audit['reference']['a_full'])} | `{dash['worst_cell']}` | "
                f"{_fmt(dash['worst_contribution_share'])} |"
            )

    lines += [
        "",
        "近饱和子集（`overall ≈ 1.000`）的剩余误差接近零，其 contribution share "
        "不可解释；H1 排名与 §2 已排除这些子集（见 §4）。",
        "",
        "## 2. 最差 cell（按 contribution）",
        "",
        "| family | subset | cell | support | error | contribution |",
        "|---|---|---|---:|---:|---:|",
    ]
    for family, data in p["families"].items():
        for subset, audit in data["subsets"].items():
            dash = audit["dashboards"]["by_type_predicate"]
            if dash["total_error"] <= 0.01:
                continue
            by_cell = {row["cell"]: row for row in dash["cells"]}
            for cell in dash["worst_cells"]:
                row = by_cell[cell]
                lines.append(
                    f"| {family} | {subset} | `{cell}` | {row['support']} | "
                    f"{_fmt(row['error'])} | {_fmt(row['contribution'])} |"
                )

    lines += [
        "",
        "## 3. 空 cell",
        "",
    ]
    any_empty = False
    for family, data in p["families"].items():
        for subset, audit in data["subsets"].items():
            empty = audit["dashboards"]["by_type_predicate"]["empty_cells"]
            if empty:
                any_empty = True
                lines.append(f"- `{family}/{subset}`: {', '.join(f'`{c}`' for c in empty)}")
    if not any_empty:
        lines.append("- 无空 cell（每个 pair_type × predicate 组合在测试池中都有支撑）。")

    lines += [
        "",
        "## 4. H1 判读",
        "",
        f"- 阈值：最差 cell 的 contribution share ≥ "
        f"{p['summary']['h1_threshold']:.2f}。",
        f"- 结果：**{'支持 H1' if p['summary']['h1_supported'] else '不支持 H1'}**。"
        f"最集中子集 `{p['summary']['worst_family']}/{p['summary']['worst_subset']}`。",
        "",
        "| rank | family | subset | worst cell | share | worst type exact-4 |",
        "|---:|---|---|---|---:|---:|",
    ]
    for i, row in enumerate(p["summary"]["h1_worst_cells"], start=1):
        lines.append(
            f"| {i} | {row['family']} | {row['subset']} | `{row['worst_cell']}` | "
            f"{_fmt(row['worst_contribution_share'])} | "
            f"{_fmt(row['worst_type_exact4'])} |"
        )

    lines += [
        "",
        "## 5. 边界",
        "",
        "- B 线是合成网格；cell 定义换任务必须重做。",
        "- `A_full` 使用全部区域/全部 pair type 训练，但每对重复数与 epochs 少于子集臂；"
        "它衡量的是覆盖，不是同等算力。",
        "- `A_ceiling = 1.000` 只成立于带死区的合成任务。",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# E2: coverage response curve
# --------------------------------------------------------------------------- #
def _read_target(audit_path: Path, family: str | None = None,
                 subset: str | None = None) -> dict:
    data = json.loads(audit_path.read_text(encoding="utf-8"))
    candidates = data["summary"]["h1_worst_cells"]
    if family and subset:
        chosen = next(
            (c for c in candidates if c["family"] == family and c["subset"] == subset),
            None,
        )
        if chosen is None:
            chosen = {"family": family, "subset": subset}
        return chosen
    if family:
        chosen = next((c for c in candidates if c["family"] == family), None)
        if chosen:
            return chosen
    return candidates[0]


def _augment_axes_factory(family: str, subset: str, args) -> dict:
    """Describe the missing / present axes for the chosen subset."""

    if family == "grid7":
        coords, profiles = _grid_profiles((subset,), args.grid_n, args.max_test)
        profile = profiles[subset]
        missing = []
        if not profile["isolation"]["x"]:
            missing.append("x")
        if not profile["isolation"]["y"]:
            missing.append("y")
        if not profile["has_diagonal"]:
            missing.append("diagonal")
        present = [axis for axis in ("x", "y", "diagonal") if axis not in missing]
        return {
            "family": family,
            "subset": subset,
            "coords": coords,
            "profiles": profiles,
            "missing": missing,
            "present": present,
            "single": [axis for axis in missing if axis in ("x", "y")][:1],
            "combination": [axis for axis in ("x", "y", "diagonal") if axis in missing],
            "noop": ["x"] if "x" in present else (["diagonal"] if "diagonal" in present else []),
        }

    # P5b
    profile = _p5b_profiles((subset,))[subset]
    missing = []
    if not profile["x_isolating"]:
        missing.append("x")
    if not profile["y_isolating"]:
        missing.append("y")
    present = [axis for axis in ("x", "y") if axis not in missing]
    return {
        "family": family,
        "subset": subset,
        "profiles": profile,
        "missing": missing,
        "present": present,
        "single": missing[:1],
        "combination": missing[:],
        "noop": present[:1],
    }


def _split_amount(amount: int, axes: list[str]) -> dict[str, int]:
    if not axes:
        return {}
    base = int(amount) // len(axes)
    counts = {axis: base for axis in axes}
    counts[axes[0]] += int(amount) - base * len(axes)
    return counts


def _eval_grid7_arm(head, kind, coords, args, seed) -> list[dict]:
    # rebuild the same held-out pool used in b_audit
    all_names = list(coords)
    names = _subset_names(coords, kind)
    test_pairs = _held_out_pairs(all_names, names, args.max_test, seed=4321)
    return _grid_rows(head, test_pairs, coords, seed, args.device, args.b_feature)


def _eval_p5b_arm(head, subset, seed, args) -> list[dict]:
    return _p5b_rows(head, subset, seed, args.device, args.b_feature)


def _train_grid7_arm(augment: dict[str, int], kind, coords, args, seed):
    names = _subset_names(coords, kind)
    train_pairs = _pairs_within(names)
    X, y = _matrix_g(train_pairs, coords, args.b_feature, "relative",
                     repeats=args.n_per_pair, noise=0.03, seed=seed)
    if augment:
        extent = (
            float(min(c[0] for c in coords.values())),
            float(min(c[1] for c in coords.values())),
            float(max(c[0] for c in coords.values())),
            float(max(c[1] for c in coords.values())),
        )
        rng = np.random.default_rng(seed + 10_000)
        add = []
        for axis, count in augment.items():
            if count:
                add.extend(_extra_rows(rng, axis, count, args.b_feature, extent))
        if add:
            X = np.vstack([X, np.asarray([r[0] for r in add], dtype=np.float32)])
            y = np.vstack([y, np.asarray([r[1] for r in add], dtype=np.float32)])
    return _train_head(X, y, seed=seed, epochs=args.epochs, device=args.device)


def _train_p5b_arm(augment: dict[str, int], subset, args, seed):
    extra_h = int(augment.get("x", 0))
    extra_v = int(augment.get("y", 0))
    head, _, _ = _train_p5b_head(
        subset, seed, args.epochs, args.device, args.b_feature,
        extra_horizontal=extra_h,
        extra_vertical=extra_v,
        n_per_pair=args.p5b_n_per_pair,
    )
    return head


def _arm_summary(rows_by_seed: dict[int, list[dict]]) -> dict:
    all_rows = [row for rows in rows_by_seed.values() for row in rows]
    per_seed = {
        str(seed): float(np.mean([r["exact"] for r in rows]))
        for seed, rows in rows_by_seed.items()
    }
    worst = worst_cell_exact4(
        all_rows, lambda r: r["type"], min_support=max(5, len(rows_by_seed))
    )
    return {
        "overall_exact4": mean_std(list(per_seed.values())),
        "per_seed_exact4": per_seed,
        "worst_cell_acc": worst["value"],
        "worst_cell": worst["cell"],
        "by_type": {
            t: mean_std([np.mean([r["exact"] for r in rows if r["type"] == t])
                         for rows in rows_by_seed.values()])
            for t in TYPES
        },
        "n_pairs": len(all_rows),
    }


def stage_b_response(args) -> dict:
    out = Path(args.out)
    audit_path = out / "b_coverage.json"
    if not audit_path.exists():
        raise SystemExit(
            f"{audit_path} not found; run --stage b_audit first (G1 before G2)."
        )
    target = _read_target(audit_path, args.target_family, args.target_subset)
    family, subset = target["family"], target["subset"]
    axes = _augment_axes_factory(family, subset, args)
    seeds = [int(s) for s in args.seeds]
    amounts = [int(a) for a in args.amounts]

    rows: dict[str, dict[str, dict[int, list[dict]]]] = {
        str(amount): {} for amount in amounts
    }
    a_full_rows: dict[int, list[dict]] = {}

    # full-coverage reference (independent of amount)
    if family == "grid7":
        coords = axes["coords"]
        all_names = list(coords)
        all_pairs = _pairs_within(all_names)
        for seed in seeds:
            head = _train_grid7_head(
                all_pairs, coords, seed, args.full_epochs, args.device,
                args.b_feature, repeats=args.full_repeats,
            )
            a_full_rows[seed] = _eval_grid7_arm(head, subset, coords, args, seed)
    else:
        for seed in seeds:
            head, _, _ = _train_p5b_head(
                None, seed, args.full_epochs, args.device, args.b_feature,
                n_per_pair=args.p5b_n_per_pair,
            )
            a_full_rows[seed] = _eval_p5b_arm(head, subset, seed, args)

    a_full = _arm_summary(a_full_rows)

    for amount in amounts:
        rows[str(amount)] = {}
        arm_specs = {
            "baseline": {},
            "single": _split_amount(amount, axes["single"]),
            "combination": _split_amount(amount, axes["combination"]),
            "noop": _split_amount(amount, axes["noop"]),
        }
        for arm, augment in arm_specs.items():
            per_seed: dict[int, list[dict]] = {}
            for seed in seeds:
                if family == "grid7":
                    head = _train_grid7_arm(augment, subset, axes["coords"], args, seed)
                    per_seed[seed] = _eval_grid7_arm(
                        head, subset, axes["coords"], args, seed
                    )
                else:
                    head = _train_p5b_arm(augment, subset, args, seed)
                    per_seed[seed] = _eval_p5b_arm(head, subset, seed, args)
            summary = _arm_summary(per_seed)
            summary["augment"] = augment
            summary["delta_vs_full"] = (
                summary["worst_cell_acc"] - a_full["worst_cell_acc"]
            )
            rows[str(amount)][arm] = summary
            print(
                f"[b_response] {amount:>4} {arm:<12} worst={summary['worst_cell_acc']:.3f} "
                f"overall={summary['overall_exact4']['mean']:.3f} "
                f"delta_full={summary['delta_vs_full']:+.3f}",
                flush=True,
            )

    observed = {
        "g_response_combination": {
            "worst_cell_acc": rows[str(amounts[-1])]["combination"]["worst_cell_acc"],
            "delta_vs_full": rows[str(amounts[-1])]["combination"]["delta_vs_full"],
        },
        "g_response_single": {
            "worst_cell_acc": rows[str(amounts[-1])]["single"]["worst_cell_acc"],
            "delta_vs_full": rows[str(amounts[-1])]["single"]["delta_vs_full"],
        },
        "g_a_full": {
            "worst_cell_acc": a_full["worst_cell_acc"],
            "delta_vs_full": 0.0,
        },
    }
    check = check_readout_reasoning_preregistration(observed)

    payload = {
        "stage": "b_response",
        "task": "E2 coverage response curve",
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "config": {
            "device": args.device,
            "seeds": seeds,
            "epochs": int(args.epochs),
            "amounts": amounts,
            "feature_mode": args.b_feature,
            "target_override": {
                "family": args.target_family,
                "subset": args.target_subset,
            },
        },
        "target": {
            "family": family,
            "subset": subset,
            "missing_axes": axes["missing"],
            "present_axes": axes["present"],
            "engine": "audit summary" if not (args.target_family and args.target_subset)
                      else "override",
        },
        "a_full": a_full,
        "curve": rows,
        "preregistration": check,
        "note": (
            "A_full is trained once per seed and evaluated on the same test pool; "
            "it uses fewer repeats/epochs than the subset arms (coverage, not equal compute). "
            "The negative control is the single-axis arm at the largest augmentation."
        ),
    }
    _write_json(out / "b_response.json", payload)
    (out / "B_RESPONSE.md").write_text(_render_b_response(payload), encoding="utf-8")
    return payload


def _render_b_response(p: dict) -> str:
    lines = [
        "# B 线覆盖响应曲线（E2 / G2）",
        "",
        f"- generated: {p['generated']}",
        f"- target: `{p['target']['family']}/{p['target']['subset']}` · "
        f"missing axes: {p['target']['missing_axes']} · "
        f"present: {p['target']['present_axes']}",
        f"- A_full: worst-cell {_fmt(p['a_full']['worst_cell_acc'])} "
        f"({p['a_full']['worst_cell']}), overall "
        f"{_fmt(p['a_full']['overall_exact4']['mean'])}",
        "",
        "主指标是 worst-cell exact-4（pair_type 维度的最小 exact-4），不是 overall。",
        "",
        "## 1. 响应曲线",
        "",
        "| support | arm | augment | worst-cell exact-4 | overall exact-4 | Δ vs A_full |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for amount, arms in p["curve"].items():
        for arm, row in arms.items():
            lines.append(
                f"| {amount} | `{arm}` | `{row['augment']}` | "
                f"{_fmt(row['worst_cell_acc'])} | "
                f"{_fmt(row['overall_exact4']['mean'])} ± "
                f"{_fmt(row['overall_exact4']['std'])} | "
                f"{_fmt(row['delta_vs_full'], 3)} |"
            )

    lines += [
        "",
        "## 2. 每类 pair 的 exact-4",
        "",
        "| support | arm | horizontal | vertical | diagonal |",
        "|---:|---|---:|---:|---:|",
    ]
    for amount, arms in p["curve"].items():
        for arm, row in arms.items():
            lines.append(
                f"| {amount} | `{arm}` | "
                + " | ".join(
                    f"{_fmt(row['by_type'][t]['mean'])}" for t in TYPES
                )
                + " |"
            )

    check = p["preregistration"]
    lines += [
        "",
        "## 3. 预注册检查",
        "",
        f"- 检查 arm 数: {check['checked']} · 通过: **{check['ok']}**",
    ]
    if check["mismatches"]:
        lines.append("")
        for mismatch in check["mismatches"]:
            lines.append(f"- `{mismatch}`")

    combo = p["curve"][str(p["config"]["amounts"][-1])]["combination"]
    single = p["curve"][str(p["config"]["amounts"][-1])]["single"]
    noop = p["curve"][str(p["config"]["amounts"][-1])]["noop"]
    baseline = p["curve"][str(p["config"]["amounts"][-1])]["baseline"]
    lines += [
        "",
        "## 4. 判读",
        "",
        f"- 组合臂 worst-cell：{_fmt(baseline['worst_cell_acc'])} → "
        f"{_fmt(combo['worst_cell_acc'])}；单轴臂 → "
        f"{_fmt(single['worst_cell_acc'])}；no-op → {_fmt(noop['worst_cell_acc'])}。",
    ]
    if combo["worst_cell_acc"] >= 0.90 and combo["delta_vs_full"] >= -0.03:
        lines.append("- **组合臂达到预注册 bar**：worst-cell ≥ 0.90 且距 A_full ≤ 0.03。")
    else:
        lines.append("- **组合臂未达到预注册 bar**：记录失败，不调 bar。")
    if single["delta_vs_full"] <= -0.10:
        lines.append("- **单轴负对照成立**：单轴明显落后 A_full，H2 得到支持。")
    else:
        lines.append("- **单轴负对照不成立**：单轴已接近 A_full，H2 不成立，"
                     "应撤回“必须组合补”的规则。")
    if abs(noop["worst_cell_acc"] - baseline["worst_cell_acc"]) <= 1e-9:
        lines.append(
            "- no-op 对照的 worst-cell 与 baseline 相同；"
            f"overall 差值 {noop['overall_exact4']['mean'] - baseline['overall_exact4']['mean']:+.3f}。"
            "额外已覆盖数据不修复缺失 cell。"
        )

    lines += [
        "",
        "## 5. 边界",
        "",
        "- 增强样本是合成反例，只教 absence case，不增加新区域。",
        "- 曲线进入平台（连续两档增益 < 1 点或 CI 跨 0）应停止，不继续加量。",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# E3: invariance matching
# --------------------------------------------------------------------------- #
def _split_held_pairs(pairs, seed=2026):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pairs))
    half = len(pairs) // 2
    val = [pairs[int(i)] for i in order[:half]]
    test = [pairs[int(i)] for i in order[half:]]
    return val, test


def _task_pair(task_mode: str) -> tuple[str, str]:
    if task_mode == "relative":
        return "relative", "absolute"
    return "absolute", "relative"


def _train_eval_head(Xtr, ytr, Xva, yva, Xte, yte, seed, args):
    head = _train_head(Xtr, ytr, seed=seed, epochs=args.epochs, device=args.device)
    val_pred = _predict_head(head, Xva, args.device)
    test_pred = _predict_head(head, Xte, args.device)
    return {
        "head": head,
        "val_exact": float(np.mean(np.all(val_pred == yva.astype(bool), axis=1))),
        "test_exact": float(np.mean(np.all(test_pred == yte.astype(bool), axis=1))),
        "test_pred": test_pred,
        "val_pred": val_pred,
    }


def stage_b_invariance(args) -> dict:
    out = Path(args.out)
    n = int(args.grid_n)
    coords = _grid(n)
    all_names = list(coords)
    names = _subset_names(coords, args.inv_subset)
    train_pairs = _pairs_within(names)
    held = _held_out_pairs(all_names, names, args.max_test, seed=4321)
    val_pairs, test_pairs = _split_held_pairs(held, seed=2026)
    seeds = [int(s) for s in args.seeds]

    # test type matrix (for worst-cell exact-4)
    test_types = np.asarray([_grid_pair_type(coords[a], coords[b]) for a, b in test_pairs])

    payload = {
        "stage": "b_invariance",
        "task": "E3 invariance matching probe",
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "config": {
            "device": args.device,
            "seeds": seeds,
            "epochs": int(args.epochs),
            "grid_n": n,
            "train_subset": args.inv_subset,
            "n_train_pairs": len(train_pairs),
            "n_val_pairs": len(val_pairs),
            "n_test_pairs": len(test_pairs),
            "feature_modes": list(FEATURE_MODES),
            "task_modes": list(TASK_MODES),
        },
        "tasks": {},
    }

    for task_mode in TASK_MODES:
        matched_feature, mismatched_feature = _task_pair(task_mode)
        feature_results: dict[str, dict] = {}
        per_seed_preds: dict[str, dict[int, np.ndarray]] = defaultdict(dict)

        for feature in FEATURE_MODES:
            Xtr, ytr = _matrix_g(train_pairs, coords, feature, task_mode,
                                 repeats=args.n_per_pair, noise=0.03, seed=0)
            Xva, yva = _matrix_g(val_pairs, coords, feature, task_mode,
                                 repeats=1, noise=0.0)
            Xte, yte = _matrix_g(test_pairs, coords, feature, task_mode,
                                 repeats=1, noise=0.0)
            runs = []
            for seed in seeds:
                result = _train_eval_head(Xtr, ytr, Xva, yva, Xte, yte, seed, args)
                runs.append(result)
                per_seed_preds[feature][seed] = result["test_pred"]
                print(
                    f"[b_invariance] {task_mode:<8} {feature:<15} seed {seed}: "
                    f"val={result['val_exact']:.3f} test={result['test_exact']:.3f}",
                    flush=True,
                )
            feature_results[feature] = {
                "val_exact": mean_std([r["val_exact"] for r in runs]),
                "test_exact": mean_std([r["test_exact"] for r in runs]),
                "worst_type_exact4": worst_cell_exact4(
                    [
                        {
                            "type": str(test_types[i]),
                            "exact": bool(
                                np.all(per_seed_preds[feature][seed][i] == yte[i])
                            ),
                        }
                        for seed in seeds
                        for i in range(len(test_pairs))
                    ],
                    lambda r: r["type"],
                    min_support=max(5, len(seeds)),
                )["value"],
            }

        # val-only selection (exploratory; does not define matched/mismatched)
        selected = max(
            FEATURE_MODES,
            key=lambda f: feature_results[f]["val_exact"]["mean"],
        )

        matched_correct = []
        mismatched_correct = []
        for i, truth in enumerate(yte.astype(bool)):
            for seed in seeds:
                matched_correct.append(
                    float(np.all(per_seed_preds[matched_feature][seed][i] == truth))
                )
                mismatched_correct.append(
                    float(np.all(per_seed_preds[mismatched_feature][seed][i] == truth))
                )
        diffs = np.asarray(matched_correct) - np.asarray(mismatched_correct)
        gap = paired_bootstrap_ci(diffs, n_boot=args.bootstrap, seed=0)

        per_seed_gaps = []
        for seed in seeds:
            seed_diffs = []
            for i, truth in enumerate(yte.astype(bool)):
                seed_diffs.append(
                    float(np.all(per_seed_preds[matched_feature][seed][i] == truth))
                    - float(np.all(per_seed_preds[mismatched_feature][seed][i] == truth))
                )
            per_seed_gaps.append(paired_bootstrap_ci(seed_diffs, n_boot=args.bootstrap,
                                                     seed=seed))

        payload["tasks"][task_mode] = {
            "matched_feature": matched_feature,
            "mismatched_feature": mismatched_feature,
            "val_selected_feature": selected,
            "sensitive": bool(gap["mean"] >= 0.20),
            "m_gap": {
                "mean": gap["mean"],
                "lo": gap["lo"],
                "hi": gap["hi"],
                "per_seed": per_seed_gaps,
            },
            "features": feature_results,
        }

    # robustness to coordinate perturbation, relative task, no retraining
    robustness = {}
    for feature in ("relative", "absolute", "scale_invariant"):
        Xtr, ytr = _matrix_g(train_pairs, coords, feature, "relative",
                             repeats=args.n_per_pair, noise=0.03, seed=0)
        heads = {
            seed: _train_head(Xtr, ytr, seed=seed, epochs=args.epochs,
                              device=args.device)
            for seed in seeds
        }
        per_noise = {}
        for noise in (0.03, 0.10, 0.20):
            values = []
            for seed, head in heads.items():
                Xn, yn = _matrix_g(test_pairs, coords, feature, "relative",
                                   repeats=1, noise=noise, seed=seed + 999)
                pred = _predict_head(head, Xn, args.device)
                values.append(float(np.mean(np.all(pred == yn.astype(bool), axis=1))))
            per_noise[str(noise)] = mean_std(values)
        robustness[feature] = per_noise

    payload["robustness"] = robustness

    # pre-registered bars are read on the absolute-position split only
    absolute = payload["tasks"]["absolute"]
    matched_worst = absolute["features"][absolute["matched_feature"]]["worst_type_exact4"]
    mismatched_worst = absolute["features"][absolute["mismatched_feature"]]["worst_type_exact4"]
    observed = {
        "g_invariance_matched": {"worst_cell_acc": matched_worst},
        "g_invariance_mismatched": {"worst_cell_acc": mismatched_worst},
    }
    payload["preregistration"] = check_readout_reasoning_preregistration(observed)
    payload["verdict"] = {
        "h3_supported_on_sensitive_split": bool(
            absolute["m_gap"]["mean"] >= 0.20 and absolute["val_selected_feature"]
            == absolute["matched_feature"]
        ),
        "sensitive_split": "absolute",
        "relative_split_is_sensitive": payload["tasks"]["relative"]["sensitive"],
    }

    _write_json(out / "b_invariance.json", payload)
    (out / "B_INVARIANCE.md").write_text(_render_b_invariance(payload), encoding="utf-8")
    return payload


def _render_b_invariance(p: dict) -> str:
    lines = [
        "# B 线不变性匹配探针（E3 / G3）",
        "",
        f"- generated: {p['generated']}",
        f"- grid: {p['config']['grid_n']}x{p['config']['grid_n']} · "
        f"train subset: {p['config']['train_subset']} · "
        f"train/val/test pairs: {p['config']['n_train_pairs']}/"
        f"{p['config']['n_val_pairs']}/{p['config']['n_test_pairs']}",
        "- 选择只用 val；test 只报一次。matched/mismatched 是预注册的，不由 val 改。",
        "",
        "## 1. 主表：task × feature 的 test exact-4",
        "",
        "| task | feature | val exact-4 | test exact-4 | worst type exact-4 |",
        "|---|---|---:|---:|---:|",
    ]
    for task, data in p["tasks"].items():
        for feature, row in data["features"].items():
            lines.append(
                f"| {task} | `{feature}` | {_fmt(row['val_exact']['mean'])} | "
                f"{_fmt(row['test_exact']['mean'])} ± {_fmt(row['test_exact']['std'])} | "
                f"{_fmt(row['worst_type_exact4'])} |"
            )

    lines += [
        "",
        "## 2. 匹配 / 错配",
        "",
        "| task | matched | mismatched | matched test | mismatched test | M_gap [95% CI] | 敏感? |",
        "|---|---|---|---:|---:|---|---|",
    ]
    for task, data in p["tasks"].items():
        gap = data["m_gap"]
        lines.append(
            f"| {task} | `{data['matched_feature']}` | "
            f"`{data['mismatched_feature']}` | "
            f"{_fmt(data['features'][data['matched_feature']]['test_exact']['mean'])} | "
            f"{_fmt(data['features'][data['mismatched_feature']]['test_exact']['mean'])} | "
            f"{gap['mean']:+.3f} [{gap['lo']:+.3f}, {gap['hi']:+.3f}] | "
            f"{'是' if data['sensitive'] else '否'} |"
        )

    lines += [
        "",
        "## 3. 坐标扰动鲁棒性（相对任务，不重训）",
        "",
        "| feature | noise 0.03 | noise 0.10 | noise 0.20 |",
        "|---|---:|---:|---:|",
    ]
    for feature, row in p["robustness"].items():
        lines.append(
            f"| `{feature}` | "
            + " | ".join(
                f"{_fmt(row[str(noise)]['mean'])} ± {_fmt(row[str(noise)]['std'])}"
                for noise in (0.03, 0.10, 0.20)
            )
            + " |"
        )

    check = p["preregistration"]
    lines += [
        "",
        "## 4. 预注册检查（绝对值任务 = 敏感性 split）",
        "",
        f"- 检查 arm 数: {check['checked']} · 通过: **{check['ok']}**",
    ]
    if check["mismatches"]:
        lines.append("")
        for mismatch in check["mismatches"]:
            lines.append(f"- `{mismatch}`")

    verdict = p["verdict"]
    lines += [
        "",
        "## 5. 判读",
        "",
        f"- 预注册敏感性 split：`{verdict['sensitive_split']}`；"
        f"相对 split 是否敏感：{'是' if verdict['relative_split_is_sensitive'] else '否'}。",
        f"- H3（matched − mismatched ≥ 0.20 且 val 选中 matched）：**"
        f"{'支持' if verdict['h3_supported_on_sensitive_split'] else '不支持'}**。",
        "- 相对任务上 absolute 也能学好，是因为它包含 delta 项；"
        "真正的不变性错配只在绝对位置任务上暴露。",
        "- normalized / scale_invariant 是探索臂，不进入主校正。",
        "",
        "## 6. 边界",
        "",
        "- 7x7 合成几何；绝对任务只测 subject 的半平面归属。",
        "- val/test 是 held-out pair 的随机对半切分；seed 级 CI 见 JSON。",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# E4: A-line modality x class x view audit
# --------------------------------------------------------------------------- #
def _subset_columns(rep, rows, modalities) -> np.ndarray:
    return np.concatenate([rep[m][rows] for m in modalities], axis=1)


def _reject_at(seen: np.ndarray, acceptance: float = 0.95) -> float:
    return float(np.quantile(np.asarray(seen), 1.0 - acceptance))


def _class_auroc(score_c: np.ndarray, positive: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if positive.sum() == 0 or (~positive).sum() == 0:
        return float("nan")
    return float(roc_auc_score(positive.astype(int), score_c))


def _load_a_audit_rep(args):
    """Return ``(feats, task, projected_rep)`` for the A-line audit.

    ``--backbone base`` reuses the cached instance-level features and the
    production projection.  ``--backbone dinov2`` swaps the vision encoder for
    the frozen DINOv2 expert and the projection trained with it; this is the
    closest available *real pretrained backbone* proxy.  DINOv2 is not a VLM,
    so the result is labelled a partial E5, not a VLM replication.
    """

    if args.backbone == "base":
        feats = load_features(args.data, args.experts, Path(args.feature_cache))
        task = task_split(feats, args.regime)
        raw = {m: np.asarray(feats[m], dtype=np.float32) for m in MODALITIES}
        heads = {
            m: load_projection(Path(args.projections) / f"{m}.pt", device=args.device)
            for m in MODALITIES
        }
        rep: dict[str, np.ndarray] = {}
        with torch.no_grad():
            for m in MODALITIES:
                heads[m].eval()
                rep[m] = heads[m](
                    torch.as_tensor(raw[m]).to(args.device)
                ).cpu().numpy().astype(np.float64)
        return feats, task, rep

    if args.backbone != "dinov2":
        raise ValueError(f"unknown backbone {args.backbone!r}")

    from pss.eval.scripts.run_encoder_priority import BY_NAME, build_encoders
    from pss.prototypes.build import extract_projected_features

    variant = BY_NAME["dinov2"]
    vision, tactile = build_encoders(variant, args.device)
    feats_obj, projected = extract_projected_features(
        args.real_data,
        args.real_experts,
        variant.projection_dir,
        device=args.device,
        vision_expert=vision,
        tactile_expert=tactile,
    )
    feats = {
        "vision": np.asarray(feats_obj.vision, dtype=np.float32),
        "tactile": np.asarray(feats_obj.tactile, dtype=np.float32),
        "geometry": np.asarray(feats_obj.geometry, dtype=np.float32),
        "attrs": np.asarray(feats_obj.attrs, dtype=np.float32),
        "categories": np.asarray([str(x) for x in feats_obj.categories]),
        "instances": np.asarray([str(x) for x in feats_obj.instances]),
        "splits": np.asarray([str(x) for x in feats_obj.splits]),
        "views": np.asarray(feats_obj.views),
        "sample_ids": np.asarray([str(x) for x in feats_obj.sample_ids]),
    }
    rep = {m: np.asarray(projected[m], dtype=np.float64) for m in MODALITIES}
    task = task_split(feats, "category")
    return feats, task, rep


def stage_a_audit(args) -> dict:
    out = Path(args.out)
    device = args.device
    feats, task, rep = _load_a_audit_rep(args)
    print("[a_audit] projected features ready", flush=True)

    Xtr = _subset_columns(rep, task["train_idx"], MODALITIES)
    Xte = _subset_columns(rep, task["test_idx"], MODALITIES)
    Xhe = _subset_columns(rep, task["held_idx"], MODALITIES)
    lda = MahalanobisScorer().fit(Xtr, task["ytr"])
    lda_pred = predict_classes(lda, Xte)
    a_full = float(np.mean(lda_pred == task["yte"]))

    bank = build_bank(rep, task["train_idx"], task["labels"], task["classes"])
    proto_scores = fused_scores(rep, bank, task["test_idx"])
    a_current = float(np.mean(proto_scores.argmax(1) == task["yte"]))

    # strong learned-head proxy (3 seeds); no bar, only a reference
    mlp_probs = [
        fit_mlp(Xtr, task["ytr"], _subset_columns(rep, task["val_idx"], MODALITIES),
                task["yva"], Xte, len(task["classes"]), seed=seed, device=device)[0]
        for seed in args.seeds
    ]
    mlp_acc = float(np.mean(np.mean(mlp_probs, axis=0).argmax(1) == task["yte"]))
    print(f"[a_audit] a_current={a_current:.3f} a_full={a_full:.3f} mlp={mlp_acc:.3f}",
          flush=True)

    labels = np.asarray(task["labels"])
    views = np.asarray(feats["views"])
    splits = np.asarray(feats["splits"])
    seen_mask = np.isin(splits, ["train", "val", "test"])
    cls_to_i = task["cls_to_i"]
    classes = task["classes"]
    all_views = sorted(set(int(v) for v in views))
    vmax = max(all_views) + 1
    train_views = [v for v in all_views if v <= vmax - 3]
    eval_views = [v for v in all_views if v >= vmax - 2]

    # -- class x view cells (held-out views only; in-view rows are in-sample) --
    class_view_rows = []
    train_support = defaultdict(int)
    view_accuracy = {}
    for view in eval_views:
        rows = np.where(seen_mask & (views == view))[0]
        if len(rows) == 0:
            continue
        lda_pred_view = predict_classes(lda, _subset_columns(rep, rows, MODALITIES))
        proto_pred_view = np.asarray(classes)[
            fused_scores(rep, bank, rows).argmax(axis=1)
        ]
        lda_ok, proto_ok = [], []
        for row, lp, pp in zip(rows, lda_pred_view, proto_pred_view):
            label = str(labels[row])
            if label not in cls_to_i:
                continue
            lda_correct = bool(int(lp) == int(cls_to_i[label]))
            proto_correct = bool(pp == label)
            lda_ok.append(lda_correct)
            proto_ok.append(proto_correct)
            class_view_rows.append(
                {
                    "class": label,
                    "view": view,
                    "lda_correct": lda_correct,
                    "proto_correct": proto_correct,
                }
            )
        if lda_ok:
            view_accuracy[view] = {
                "support": len(lda_ok),
                "lda_acc": float(np.mean(lda_ok)),
                "proto_acc": float(np.mean(proto_ok)),
            }
    for row in task["train_idx"]:
        label = str(labels[row])
        if label in cls_to_i:
            train_support[(label, int(views[row]))] += 1

    class_view_cells = []
    buckets = defaultdict(list)
    for row in class_view_rows:
        buckets[(row["class"], int(row["view"]))].append(row)
    for (cls, view), group in buckets.items():
        class_view_cells.append(
            {
                "cell": f"{cls}|v{view}",
                "class": cls,
                "view": view,
                "support": len(group),
                "lda_acc": float(np.mean([r["lda_correct"] for r in group])),
                "proto_acc": float(np.mean([r["proto_correct"] for r in group])),
            }
        )
    class_view_cells.sort(key=lambda r: (r["lda_acc"], -r["support"], r["cell"]))
    expected_cv = [f"{cls}|v{view}" for cls in classes for view in eval_views]

    # -- modality subset x class cells -------------------------------------
    modality_class_cells = []
    subset_overall = {}
    for subset in (
        ("vision",),
        ("tactile",),
        ("geometry",),
        ("vision", "tactile"),
        ("vision", "geometry"),
        ("tactile", "geometry"),
        MODALITIES,
    ):
        key = "+".join(subset)
        Xtr_s = _subset_columns(rep, task["train_idx"], subset)
        Xte_s = _subset_columns(rep, task["test_idx"], subset)
        scorer = MahalanobisScorer().fit(Xtr_s, task["ytr"])
        pred = predict_classes(scorer, Xte_s)
        subset_overall[key] = float(np.mean(pred == task["yte"]))
        bucket = defaultdict(list)
        for idx, ok in zip(task["yte"], pred == task["yte"]):
            bucket[str(task["classes"][int(idx)])].append(float(ok))
        for cls, values in bucket.items():
            modality_class_cells.append(
                {
                    "cell": f"{key}|{cls}",
                    "modality_mask": key,
                    "class": cls,
                    "support": len(values),
                    "accuracy": float(np.mean(values)),
                }
            )

    # -- structural empty cells -------------------------------------------
    structural = []
    # A held-out view (>= vmax-2) is absent from training by construction, so
    # only a *training* view with no support is a structural gap.
    for cls in classes:
        for view in train_views:
            if train_support.get((cls, view), 0) == 0:
                structural.append(
                    {
                        "cell": f"{cls}|v{view}",
                        "kind": "class_view_never_trained",
                        "eval_support": 0,
                    }
                )

    # modality availability: a modality is structurally absent for a class if
    # its mean feature norm is negligible relative to that modality's median.
    modality_scale = {
        m: float(np.median(np.linalg.norm(rep[m][task["train_idx"]], axis=1)))
        for m in MODALITIES
    }
    for cls in classes:
        cls_rows = task["train_idx"][
            np.asarray([str(labels[i]) for i in task["train_idx"]]) == cls
        ]
        if len(cls_rows) == 0:
            continue
        for m in MODALITIES:
            norm = float(np.median(np.linalg.norm(rep[m][cls_rows], axis=1)))
            if norm < 1e-6 * max(modality_scale[m], 1e-12):
                structural.append(
                    {
                        "cell": f"{cls}|{m}",
                        "kind": "modality_absent_in_train",
                        "median_norm": norm,
                        "scale": modality_scale[m],
                    }
                )

    # -- per-class operating point (AUROC / reject) on the full LDA --------
    scores_te = lda.scores(Xte)
    scores_he = lda.scores(Xhe)
    seen_max = scores_te.max(axis=1)
    threshold = _reject_at(seen_max)
    per_class = []
    for idx, cls in enumerate(classes):
        mask = task["yte"] == idx
        if not np.any(mask):
            continue
        positive = np.concatenate([mask, np.zeros(len(scores_he), dtype=bool)])
        score_c = np.concatenate([scores_te[:, idx], scores_he[:, idx]])
        per_class.append(
            {
                "class": str(cls),
                "support": int(mask.sum()),
                "acc": float(np.mean(np.argmax(scores_te[mask], axis=1) == idx)),
                "auroc_vs_unseen": _class_auroc(score_c, positive),
                "reject_at_95": float(np.mean(seen_max[mask] < threshold)),
            }
        )
    per_class.sort(key=lambda r: (r["acc"], -r["support"], r["class"]))

    class_view_dashboard = {
        "cells": class_view_cells,
        "worst_cells": [
            {"cell": r["cell"], "support": r["support"], "lda_acc": r["lda_acc"]}
            for r in class_view_cells[:5]
        ],
        "empty_cells": [
            cell for cell in expected_cv
            if cell not in {r["cell"] for r in class_view_cells}
        ],
        "min_support": 1,
    }
    modality_dashboard = {
        "cells": modality_class_cells,
        "worst_cells": [
            {
                "cell": r["cell"],
                "support": r["support"],
                "accuracy": r["accuracy"],
            }
            for r in sorted(
                modality_class_cells,
                key=lambda r: (r["accuracy"], -r["support"], r["cell"]),
            )[:10]
        ],
        "empty_cells": [],
        "min_support": 1,
    }

    payload = {
        "stage": "a_audit",
        "task": "E4 A-line modality x class x view audit",
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "config": {
            "data": args.data if args.backbone == "base" else args.real_data,
            "experts": args.experts if args.backbone == "base" else args.real_experts,
            "projections": args.projections if args.backbone == "base"
                          else "checkpoints/encoder_priority/dinov2/projection",
            "backbone": args.backbone,
            "regime": args.regime if args.backbone == "base" else "category",
            "classes": len(classes),
            "seeds": [int(s) for s in args.seeds],
            "train_views": train_views,
            "eval_views": eval_views,
        },
        "reference": {
            "a_current": a_current,
            "a_full": a_full,
            "a_ceiling": None,
            "a_ceiling_note": (
                "no oracle for this synthetic category task; the MLP reference "
                f"({mlp_acc:.4f}) is a strong-head proxy, not a ceiling"
            ),
            "mlp_proxy": mlp_acc,
        },
        "coverage": {
            "class_view": class_view_dashboard,
            "modality_class": modality_dashboard,
            "modality_mask_overall": subset_overall,
            "view_accuracy": view_accuracy,
        },
        "structural_empty_cells": structural,
        "per_class": per_class,
        "notes": [
            "a_current is the deployed fused prototype readout; a_full is the "
            "full-modality pooled-covariance LDA.",
            "class x view cells use every seen split, so the test view (the last "
            "one) is one column, not the whole table.",
            "reject_at_95 uses the seen-test score quantile; it is an operating "
            "point, not a generalisation claim.",
            "backbone=dinov2 is a real pretrained vision backbone, not a VLM; "
            "it is a partial E5 proxy, not a VLM replication.",
        ],
    }
    _write_json(out / "a_coverage.json", payload)
    (out / "A_COVERAGE.md").write_text(_render_a_audit(payload), encoding="utf-8")
    return payload


def _render_a_audit(p: dict) -> str:
    ref = p["reference"]
    lines = [
        "# A 线模态 × 类别 × 视角审计（E4 / G4）",
        "",
        f"- generated: {p['generated']}",
        f"- backbone: `{p['config'].get('backbone', 'base')}` · "
        f"regime: {p['config']['regime']} · classes: {p['config']['classes']}",
        f"- `A_current`（部署原型 cosine）: **{_fmt(ref['a_current'])}**",
        f"- `A_full`（全模态 LDA）: **{_fmt(ref['a_full'])}**",
        f"- `A_ceiling`: n/a — {ref['a_ceiling_note']}",
        f"- MLP strong-head proxy: {_fmt(ref['mlp_proxy'])}",
        "",
        "## 1. 模态子集整体",
        "",
        "| modality mask | LDA acc |",
        "|---|---:|",
    ]
    for key, value in p["coverage"]["modality_mask_overall"].items():
        lines.append(f"| `{key}` | {_fmt(value)} |")

    lines += [
        "",
        "## 2. 每个 held-out 视图",
        "",
        "| view | support | LDA acc | proto acc |",
        "|---|---:|---:|---:|",
    ]
    for view, row in p["coverage"]["view_accuracy"].items():
        lines.append(
            f"| v{view} | {row['support']} | {_fmt(row['lda_acc'])} | "
            f"{_fmt(row['proto_acc'])} |"
        )

    lines += [
        "",
        "## 3. 最差 class × view（LDA）",
        "",
        "每个 held-out view 上每类的 support 可能很小（base 线为 1/类）；"
        "本表只作诊断，主指标是 §1 的模态子集整体与 §5 的结构性空 cell。",
        "",
        "| cell | support | LDA acc | proto acc |",
        "|---|---:|---:|---:|",
    ]
    cv = {row["cell"]: row for row in p["coverage"]["class_view"]["cells"]}
    for row in p["coverage"]["class_view"]["worst_cells"]:
        full = cv[row["cell"]]
        lines.append(
            f"| `{row['cell']}` | {row['support']} | {_fmt(row['lda_acc'])} | "
            f"{_fmt(full['proto_acc'])} |"
        )

    lines += [
        "",
        "## 4. 最差 modality × class（LDA）",
        "",
        "| cell | support | acc |",
        "|---|---:|---:|",
    ]
    for row in p["coverage"]["modality_class"]["worst_cells"]:
        lines.append(f"| `{row['cell']}` | {row['support']} | {_fmt(row['accuracy'])} |")

    lines += [
        "",
        "## 5. 结构性空 cell",
        "",
    ]
    if p["structural_empty_cells"]:
        for cell in p["structural_empty_cells"][:40]:
            lines.append(f"- `{cell['cell']}`: {cell['kind']}")
    else:
        lines.append("- 未发现结构性空 cell：每个类别在训练视图中都有全部模态的支撑。")

    worst = p["per_class"][:8]
    lines += [
        "",
        "## 6. 最差类别（按闭集 acc）",
        "",
        "| class | support | acc | AUROC vs unseen | reject @95 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in worst:
        lines.append(
            f"| `{row['class']}` | {row['support']} | {_fmt(row['acc'])} | "
            f"{_fmt(row['auroc_vs_unseen'])} | {_fmt(row['reject_at_95'])} |"
        )

    lines += [
        "",
        "## 7. 判读",
        "",
        f"- `A_current` 与 `A_full` 差距 {_fmt(ref['a_full'] - ref['a_current'])}；"
        "若差距主要来自某个模态子集，瓶颈在融合/读出，不在覆盖。",
        "- class × view 的空 cell 若只出现在最后一个视图，是任务设计（held-out view），"
        "不是数据缺口；真正要补的是训练视图内某类只有单一模态的情况。",
        "- A 线没有 oracle，`A_ceiling` 不能填 1.000；L3 在本线不可判定。",
        "",
        "## 8. 边界",
        "",
        "- 合成代理数据；结论需在真实 VLM 骨架上重跑（E5）。",
        "- 池化协方差 LDA 是闭集读出，不包含搜索/代数。",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# G6: final matrix
# --------------------------------------------------------------------------- #
def _load_optional(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def stage_final(args) -> dict:
    out = Path(args.out)
    b_audit = _load_optional(out / "b_coverage.json")
    b_response = _load_optional(out / "b_response.json")
    b_invariance = _load_optional(out / "b_invariance.json")
    a_audit = _load_optional(out / "a_coverage.json")

    grading = {}
    if b_audit:
        candidates = b_audit["summary"]["h1_worst_cells"]
        worst_gap = (
            min(candidates, key=lambda c: c["overall_exact4"] - c["a_full"])
            if candidates
            else None
        )
        grading["L1_b_line"] = {
            "a_current": worst_gap["overall_exact4"] if worst_gap else None,
            "a_full": worst_gap["a_full"] if worst_gap else None,
            "worst_subset": worst_gap["subset"] if worst_gap else None,
            "worst_family": worst_gap["family"] if worst_gap else None,
            "worst_by_share_subset": b_audit["summary"]["worst_subset"],
            "h1_supported": b_audit["summary"]["h1_supported"],
            "satisfied": bool(
                worst_gap
                and worst_gap["overall_exact4"] - worst_gap["a_full"] >= -0.03
            ),
            "evidence": (
                "a_current - a_full on the largest-gap audit subset; "
                "a positive value means the current subset already matches full coverage"
            ),
        }
    if b_response:
        last = str(b_response["config"]["amounts"][-1])
        combo = b_response["curve"][last]["combination"]
        single = b_response["curve"][last]["single"]
        grading["L1_response"] = {
            "combination_worst_cell": combo["worst_cell_acc"],
            "single_worst_cell": single["worst_cell_acc"],
            "combination_delta_vs_full": combo["delta_vs_full"],
            "single_delta_vs_full": single["delta_vs_full"],
            "satisfied": bool(
                combo["worst_cell_acc"] >= 0.90
                and combo["delta_vs_full"] >= -0.03
            ),
            "h2_supported": bool(single["delta_vs_full"] <= -0.10),
        }
    if b_invariance:
        abs_task = b_invariance["tasks"]["absolute"]
        rel_task = b_invariance["tasks"]["relative"]
        grading["L2_invariance"] = {
            "sensitive_split": "absolute",
            "m_gap": abs_task["m_gap"]["mean"],
            "m_gap_ci": [abs_task["m_gap"]["lo"], abs_task["m_gap"]["hi"]],
            "val_selected": abs_task["val_selected_feature"],
            "matched": abs_task["matched_feature"],
            "relative_split_gap": rel_task["m_gap"]["mean"],
            "satisfied": bool(
                abs_task["m_gap"]["mean"] >= 0.20
                and abs_task["val_selected_feature"] == abs_task["matched_feature"]
            ),
            "aux_worst_cell_bar_ok": bool(
                b_invariance.get("preregistration", {}).get("ok")
            ),
            "aux_mismatches": b_invariance.get("preregistration", {}).get(
                "mismatches", []
            ),
        }
    if a_audit:
        grading["L3_a_line"] = {
            "a_current": a_audit["reference"]["a_current"],
            "a_full": a_audit["reference"]["a_full"],
            "a_ceiling": a_audit["reference"]["a_ceiling"],
            "satisfied": None,
            "reason": "no oracle ceiling on the A line; L3 cannot be assessed",
        }
        grading["A_line_coverage"] = {
            "structural_empty_cells": len(a_audit.get("structural_empty_cells", [])),
            "satisfied_coverage": len(a_audit.get("structural_empty_cells", [])) == 0,
            "a_current": a_audit["reference"]["a_current"],
            "a_full": a_audit["reference"]["a_full"],
            "note": (
                "no structural empty cell; the current-vs-full gap is the readout "
                "(f), not coverage (g)"
            ),
        }
    if b_audit:
        ceilings = []
        for _family, data in b_audit["families"].items():
            for _subset, audit in data["subsets"].items():
                ref = audit["reference"]
                if ref.get("a_full") is not None and ref.get("a_ceiling") is not None:
                    ceilings.append(abs(ref["a_full"] - ref["a_ceiling"]))
        grading["L3_b_line"] = {
            "a_full_vs_ceiling_max_gap": max(ceilings) if ceilings else None,
            "satisfied": bool(ceilings and max(ceilings) <= 0.03),
            "evidence": "synthetic A_full = A_ceiling = 1.000",
        }

    b_line_levels = {
        "L1": bool(
            grading.get("L1_response", {}).get("satisfied")
            or grading.get("L1_b_line", {}).get("satisfied")
        ),
        "L2": bool(grading.get("L2_invariance", {}).get("satisfied")),
        "L3": bool(grading.get("L3_b_line", {}).get("satisfied")),
    }
    a_line_levels = {
        "L1": bool(grading.get("A_line_coverage", {}).get("satisfied_coverage")),
        "L3": None,
    }
    levels = {
        "L1_coverage_saturated": b_line_levels["L1"],
        "L2_matching_correct": b_line_levels["L2"],
        "L3_ceiling_aligned": b_line_levels["L3"],
    }
    payload = {
        "stage": "final",
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "sources": {
            "b_audit": bool(b_audit),
            "b_response": bool(b_response),
            "b_invariance": bool(b_invariance),
            "a_audit": bool(a_audit),
        },
        "grading": grading,
        "levels": levels,
        "levels_by_task": {
            "b_line": b_line_levels,
            "a_line": a_line_levels,
        },
        "all_three_b_line": all(b_line_levels.values()),
        "all_three": all(levels.values()),
        "caveat": (
            "All B-line measurements are synthetic (P5b / 7x7 grid). The A line is "
            "a synthetic 384-class proxy; E5 on a real VLM backbone has not been run "
            "(the DINOv2 run is a real vision-backbone proxy on a different task). "
            "No L1–L3 claim may be extended beyond these tasks."
        ),
    }
    _write_json(out / "final_matrix.json", payload)
    (out / "G_FINAL_REPORT.md").write_text(_render_final(payload), encoding="utf-8")
    return payload


def _render_final(p: dict) -> str:
    lines = [
        "# g 因子最终验证矩阵（G6）",
        "",
        f"- generated: {p['generated']}",
        f"- sources: {p['sources']}",
        "",
        "## 1. 分级",
        "",
        "| task | L1 覆盖饱和 | L2 匹配正确 | L3 上限对齐 | 三级同时 |",
        "|---|---|---|---|---|",
        f"| B line（合成） | "
        f"{'✅' if p['levels_by_task']['b_line']['L1'] else '❌'} | "
        f"{'✅' if p['levels_by_task']['b_line']['L2'] else '❌'} | "
        f"{'✅' if p['levels_by_task']['b_line']['L3'] else '❌'} | "
        f"{'是' if p['all_three_b_line'] else '否'} |",
        f"| A line（合成代理） | "
        f"{'✅ 无结构性空 cell' if p['levels_by_task']['a_line']['L1'] else '❌'} | "
        f"n/a | 不可判定（无 oracle） | 否 |",
        "",
        "全局口径（只用可判定的 B 线三级）：",
        "",
        "| 级别 | 条件 | 结果 |",
        "|---|---|---|",
        f"| L1 覆盖饱和 | `A_current ≈ A_full` / 组合臂达标 | "
        f"{'✅' if p['levels']['L1_coverage_saturated'] else '❌'} |",
        f"| L2 匹配正确 | matched − mismatched ≥ 0.20 且选中 matched | "
        f"{'✅' if p['levels']['L2_matching_correct'] else '❌'} |",
        f"| L3 上限对齐 | `A_full ≈ A_ceiling` | "
        f"{'✅' if p['levels']['L3_ceiling_aligned'] else '❌/不可判定'} |",
        "",
        f"**B 线三级同时满足：{'是' if p['all_three_b_line'] else '否'}**。"
        " A 线因 L3 无 oracle，不能声称达到上界；其覆盖审计只支持 L1。",
        "",
        "## 2. 证据",
        "",
    ]
    for key, value in p["grading"].items():
        lines.append(f"### {key}")
        lines.append("")
        for field, field_value in value.items():
            lines.append(f"- `{field}`: {field_value}")
        lines.append("")
    lines += [
        "## 3. 边界",
        "",
        f"- {p['caveat']}",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def run_stage(args) -> dict:
    if args.stage == "b_audit":
        return stage_b_audit(args)
    if args.stage == "b_response":
        return stage_b_response(args)
    if args.stage == "b_invariance":
        return stage_b_invariance(args)
    if args.stage == "a_audit":
        return stage_a_audit(args)
    if args.stage == "final":
        return stage_final(args)
    raise ValueError(f"unknown stage {args.stage!r}")


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--stage",
        required=True,
        choices=("b_audit", "b_response", "b_invariance", "a_audit", "final"),
    )
    ap.add_argument("--out", default="eval/results/g_audit")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--full-epochs", type=int, default=300)
    ap.add_argument("--n-per-pair", type=int, default=50)
    ap.add_argument("--p5b-n-per-pair", type=int, default=300)
    ap.add_argument("--max-test", type=int, default=800)
    ap.add_argument("--grid-n", type=int, default=7)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--b-feature", choices=("absolute", "relative"), default="relative")
    ap.add_argument("--p5b-subsets", nargs="+", default=list(P5B_SUBSETS))
    ap.add_argument("--grid-subsets", nargs="+", default=list(GRID7_SUBSETS))
    ap.add_argument("--full-repeats", type=int, default=5)
    ap.add_argument("--inv-subset", default="block3")
    ap.add_argument("--amounts", nargs="+", type=int, default=[300, 500, 1000])
    ap.add_argument("--target-family", choices=("p5b", "grid7"), default=None)
    ap.add_argument("--target-subset", default=None)
    ap.add_argument("--data", default="data/category_pretrain_instance8")
    ap.add_argument(
        "--experts", default="checkpoints/category_pretrain_instance8_experts"
    )
    ap.add_argument(
        "--projections", default="checkpoints/category_pretrain_instance8_projection"
    )
    ap.add_argument("--feature-cache", default="eval/results/readout_opt")
    ap.add_argument("--regime", choices=["category", "view-instance"], default="view-instance")
    ap.add_argument("--backbone", choices=("base", "dinov2"), default="base")
    ap.add_argument("--real-data", default="data/category_pretrain")
    ap.add_argument(
        "--real-experts", default="checkpoints/category_pretrain_experts"
    )
    args = ap.parse_args()

    payload = run_stage(args)
    print(f"output: {Path(args.out)}")
    if args.stage != "final":
        print(f"stage {args.stage} done")


if __name__ == "__main__":
    main()
