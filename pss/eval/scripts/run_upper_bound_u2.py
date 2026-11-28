"""U2: real-geometry coverage audit (CLEVR-CoGenT object layouts).

Executable form of ``docs/plans/PSS-架构上界复核与下一阶段.md`` §6 (U2).

The B line's coverage rule was established on a synthetic 5-region grid and a
7x7 grid.  U2 repeats the *same* manipulation on real rendered object layouts:
objects from CLEVR scenes are normalised to the unit square, every ordered pair
is turned into a ``horizontal`` (``dy=0``), ``vertical`` (``dx=0``) or
``diagonal`` pair, and a spatial relation head is trained with one or more pair
types withheld.

Arms (same as G2):

  baseline      only horizontal pairs (missing vertical + diagonal)
  single        + vertical                       (missing diagonal)
  combination   + vertical + diagonal            (full pair-type coverage)
  noop          + more horizontal                (covered type cannot fix it)
  full          all three types                  (reference)

The primary metric is ``worst_cell_acc``: the minimum exact-4 over the three
pair types.  ``delta_vs_full`` is the signed gap to ``full``.

Example
-------
python -u -m pss.eval.scripts.run_upper_bound_u2 \
  --out eval/results/upper_bound/u2 --device cuda
"""

from __future__ import annotations

import argparse
import json
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
from pss.eval.upper_bound import (
    PAIR_TYPES,
    cell_accuracy,
    pair_labels,
    pair_types,
    worst_cell_accuracy,
)

SCENE_FILES = {
    "traina": "CLEVR_trainA_scenes.json",
    "vala": "CLEVR_valA_scenes.json",
    "valb": "CLEVR_valB_scenes.json",
    "testa": "CLEVR_testA_scenes.json",
    "testb": "CLEVR_testB_scenes.json",
}

ARMS = ("baseline", "single", "combination", "noop", "full")
ARM_TYPES = {
    "baseline": ("horizontal",),
    "single": ("horizontal", "vertical"),
    "combination": ("horizontal", "vertical", "diagonal"),
    "noop": ("horizontal",),
    "full": tuple(PAIR_TYPES),
}


def _scene_path(root: Path, split: str) -> Path:
    return root / "scenes" / SCENE_FILES[split]


def _scenes(root: Path, split: str, limit: int) -> list[dict]:
    payload = json.loads(_scene_path(root, split).read_text(encoding="utf-8"))
    return payload.get("scenes", [])[: int(limit)]


def _scene_pairs(root: Path, split: str, limit: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """All ordered object pairs in pixel coordinates (raw, un-normalised)."""

    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for scene in _scenes(root, split, limit):
        pts = [
            np.asarray(obj.get("pixel_coords", [0.0, 0.0, 0.0])[:2], dtype=np.float32)
            for obj in scene.get("objects", [])
        ]
        for i, pa in enumerate(pts):
            for j, pb in enumerate(pts):
                if i != j:
                    pairs.append((pa.copy(), pb.copy()))
    return pairs


def _fit_scale(pairs: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    pts = np.concatenate([np.stack([a, b]) for a, b in pairs], axis=0)
    return pts.min(axis=0), pts.max(axis=0)


def _normalise(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    lo: np.ndarray,
    hi: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    span = np.where(hi > lo, hi - lo, 1.0)
    return [((a - lo) / span, (b - lo) / span) for a, b in pairs]


def _variant_pairs(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    kinds: tuple[str, ...],
    *,
    min_offset: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build rows for the requested pair types with exact absence axes."""

    X, Y, T = [], [], []
    for pa, pb in pairs:
        dx = float(pb[0] - pa[0])
        dy = float(pb[1] - pa[1])
        for kind in kinds:
            if kind == "horizontal":
                if abs(dx) < min_offset:
                    continue
                qa, qb = pa.copy(), pb.copy()
                qb[1] = qa[1]
            elif kind == "vertical":
                if abs(dy) < min_offset:
                    continue
                qa, qb = pa.copy(), pb.copy()
                qb[0] = qa[0]
            else:
                if abs(dx) < min_offset or abs(dy) < min_offset:
                    continue
                qa, qb = pa, pb
            X.append(_pair_feature(qa, qb, "relative"))
            Y.append(pair_labels(qa, qb, eps=0.0))
            T.append(kind)
    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(Y, dtype=np.float32),
        np.asarray(T, dtype=object),
    )


def _take(
    X: np.ndarray, Y: np.ndarray, T: np.ndarray, n: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n >= len(X):
        return X, Y, T
    idx = rng.choice(len(X), size=n, replace=False)
    return X[idx], Y[idx], T[idx]


def _concat(parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]]):
    return (
        np.concatenate([p[0] for p in parts], axis=0),
        np.concatenate([p[1] for p in parts], axis=0),
        np.concatenate([p[2] for p in parts], axis=0),
    )


def _balanced_test(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    *,
    per_type: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Equal numbers of horizontal / vertical / diagonal pairs from held-out scenes."""

    rng = np.random.default_rng(seed)
    X, Y, T = _variant_pairs(pairs, tuple(PAIR_TYPES))
    rows = []
    for kind in PAIR_TYPES:
        idx = np.where(T == kind)[0]
        if len(idx) > per_type:
            idx = rng.choice(idx, size=per_type, replace=False)
        rows.append(idx)
    rows = np.concatenate(rows)
    rng.shuffle(rows)
    return X[rows], Y[rows], T[rows]


def _train_arm(
    X: np.ndarray,
    Y: np.ndarray,
    Xte: np.ndarray,
    Yte: np.ndarray,
    Tte: np.ndarray,
    *,
    seeds: tuple[int, ...],
    epochs: int,
    device: str,
) -> dict:
    per_seed = []
    for seed in seeds:
        head = _train(X, Y, "relative", seed, epochs, device)
        pred = _predict(head, Xte, device)
        per_seed.append(
            {
                "overall": float(np.all(pred == Yte.astype(bool), axis=1).mean()),
                "cells": cell_accuracy(Tte, Yte, pred),
                "worst_cell_acc": worst_cell_accuracy(Tte, Yte, pred),
            }
        )
    return {
        "overall": float(np.mean([r["overall"] for r in per_seed])),
        "overall_std": float(np.std([r["overall"] for r in per_seed])),
        "worst_cell_acc": float(np.mean([r["worst_cell_acc"] for r in per_seed])),
        "worst_cell_std": float(np.std([r["worst_cell_acc"] for r in per_seed])),
        "cells": {
            k: float(np.mean([r["cells"].get(k, 0.0) for r in per_seed]))
            for k in PAIR_TYPES
        },
        "seeds": list(seeds),
    }


def run(args) -> dict:
    root = Path(args.clevr_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_pairs = _scene_pairs(root, args.train_split, args.train_scenes)
    test_pairs = _scene_pairs(root, args.test_split, args.test_scenes)
    lo, hi = _fit_scale(train_pairs)
    train_pairs = _normalise(train_pairs, lo, hi)
    test_pairs = _normalise(test_pairs, lo, hi)
    print(
        f"[data] train pairs {len(train_pairs)} test pairs {len(test_pairs)}",
        flush=True,
    )

    Xte, Yte, Tte = _balanced_test(test_pairs, per_type=args.per_type, seed=args.seed)
    print(f"[test] rows {len(Xte)} cells {dict(zip(*np.unique(Tte, return_counts=True)))}", flush=True)

    rng = np.random.default_rng(args.seed)
    parts = {kind: _variant_pairs(train_pairs, (kind,)) for kind in PAIR_TYPES}
    base = _take(*parts["horizontal"], args.baseline_rows, rng)
    single_aug = _take(*parts["vertical"], args.augment, rng)
    combo_v = _take(*parts["vertical"], args.augment // 2, rng)
    combo_d = _take(*parts["diagonal"], args.augment // 2, rng)
    noop_aug = _take(*parts["horizontal"], args.augment, rng)
    full = _concat(
        [
            _take(*parts["horizontal"], args.full_rows // 3, rng),
            _take(*parts["vertical"], args.full_rows // 3, rng),
            _take(*parts["diagonal"], args.full_rows // 3, rng),
        ]
    )
    arm_data = {
        "baseline": base,
        "single": _concat([base, single_aug]),
        "combination": _concat([base, combo_v, combo_d]),
        "noop": _concat([base, noop_aug]),
        "full": full,
    }

    results: dict[str, dict] = {}
    for arm in ARMS:
        X, Y, _ = arm_data[arm]
        print(
            f"[arm {arm}] train rows {len(X)} "
            f"(base {len(base[0])}, augment {len(X) - len(base[0])})",
            flush=True,
        )
        results[arm] = _train_arm(
            X, Y, Xte, Yte, Tte,
            seeds=tuple(args.seeds),
            epochs=args.epochs,
            device=args.device,
        )
        results[arm]["n_train"] = int(len(X))
        print(
            f"[arm {arm}] worst {results[arm]['worst_cell_acc']:.3f} "
            f"overall {results[arm]['overall']:.3f}",
            flush=True,
        )

    full_worst = results["full"]["worst_cell_acc"]
    for arm, row in results.items():
        row["delta_vs_full"] = float(row["worst_cell_acc"] - full_worst)

    observed = {
        f"u_geo_{arm}": {
            "worst_cell_acc": results[arm]["worst_cell_acc"],
            "delta_vs_full": results[arm]["delta_vs_full"],
            "test_acc": results[arm]["overall"],
        }
        for arm in ARMS
    }
    check = check_readout_reasoning_preregistration(observed)

    payload = {
        "stage": "u2",
        "clevr_root": str(root),
        "train_split": args.train_split,
        "test_split": args.test_split,
        "train_scenes": int(args.train_scenes),
        "test_scenes": int(args.test_scenes),
        "per_type": int(args.per_type),
        "baseline_rows": int(args.baseline_rows),
        "augment": int(args.augment),
        "full_rows": int(args.full_rows),
        "epochs": int(args.epochs),
        "seeds": list(args.seeds),
        "arms": results,
        "observed": observed,
        "preregistration": check,
        "note": (
            "real CLEVR object layouts normalised to the unit square; "
            "horizontal/vertical pairs are made exact by snapping one axis; "
            "test is balanced over pair types and uses held-out scenes. "
            "All arms use the same row budget (~baseline_rows + augment), so "
            "delta_vs_full isolates coverage, not sample size."
        ),
    }
    _write_json(out / "u2.json", payload)
    (out / "U2_REPORT.md").write_text(_render(payload), encoding="utf-8")
    return payload


def _render(p: dict) -> str:
    lines = [
        "# U2：真实几何（CLEVR）上的覆盖审计",
        "",
        f"- CLEVR: `{p['clevr_root']}` · train `{p['train_split']}` "
        f"({p['train_scenes']} scenes) · test `{p['test_split']}` "
        f"({p['test_scenes']} scenes)",
        f"- test rows/type: {p['per_type']} · epochs {p['epochs']} · seeds {p['seeds']}",
        "",
        "## 1. 主表（worst-cell = 三个 pair type 的最小 exact-4）",
        "",
        "| arm | train rows | horizontal | vertical | diagonal | worst cell | overall | Δ vs full |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        r = p["arms"][arm]
        c = r["cells"]
        lines.append(
            f"| `{arm}` | {r['n_train']} | {c['horizontal']:.3f} | "
            f"{c['vertical']:.3f} | {c['diagonal']:.3f} | "
            f"{r['worst_cell_acc']:.3f} | {r['overall']:.3f} | "
            f"{r['delta_vs_full']:+.3f} |"
        )
    check = p["preregistration"]
    lines += [
        "",
        "## 2. 预注册检查",
        "",
        f"- 检查 arm 数: {check['checked']} · 通过: **{check['ok']}**",
    ]
    if check["mismatches"]:
        lines.append("")
        for mismatch in check["mismatches"]:
            lines.append(f"- `{mismatch}`")
    lines += [
        "",
        "## 3. 判读",
        "",
        "- 若 `single` 仍显著落后 `combination`（Δ ≤ −0.10），G2 的“单轴不够、"
        "必须补缺失 pair type”在真实几何上复现。",
        "- 若 `noop` 的 worst cell 与 `baseline` 同量级，说明多给已覆盖数据不能修复缺失 cell。",
        "- 若真实几何没有 absence-cell 效应（single ≈ combination），覆盖规则只适用于合成网格。",
        "",
        "## 4. 边界",
        "",
        "- CLEVR 是渲染数据，不是机器人；结论写“真实几何/渲染”，不写 sim2real。",
        "- `horizontal`/`vertical` 由 snapping 一条轴构造，另一轴保留真实 CLEVR 布局；"
        "`diagonal` 用原始对象对。",
        "- 测试用 held-out scenes，不是 held-out 坐标；域偏移只到 valA，valB 未作为主表。",
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
    ap.add_argument("--train-split", choices=sorted(SCENE_FILES), default="traina")
    ap.add_argument("--test-split", choices=sorted(SCENE_FILES), default="vala")
    ap.add_argument("--train-scenes", type=int, default=2000)
    ap.add_argument("--test-scenes", type=int, default=500)
    ap.add_argument("--per-type", type=int, default=5000)
    ap.add_argument("--baseline-rows", type=int, default=5000)
    ap.add_argument("--augment", type=int, default=1000)
    ap.add_argument("--full-rows", type=int, default=6000)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="eval/results/upper_bound/u2")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    payload = run(args)
    for arm in ARMS:
        r = payload["arms"][arm]
        print(f"  {arm:<12} worst {r['worst_cell_acc']:.3f} overall {r['overall']:.3f}")
    check = payload["preregistration"]
    print(f"preregistration ok={check['ok']} checked={check['checked']}")
    for mismatch in check["mismatches"]:
        print(f"  - {mismatch}")


if __name__ == "__main__":
    main()
