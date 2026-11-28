"""Scene-model benchmark: observations -> 3D state -> relation queries.

D1 entry point.  Additional arms register themselves from
``pss.scene.arms_3d``, ``arms_constraints`` and ``arms_uncertainty``; the
benchmark itself does not need to change.

Example
-------
python -u -m pss.eval.scripts.run_scene_bench \
  --test-scenes 100 --n-views 3 --out eval/results/scene_bench
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pss._console import ensure_utf8
from pss.scene.bench_api import (
    ARMS,
    BenchConfig,
    build_examples,
    evaluate_arm,
    load_optional_arms,
    run_prepare,
)
from pss.scene.clevr import load_scenes


def _aggregate(seed_rows: list[dict]) -> dict:
    """Mean/std across noise seeds for every metric's mean."""

    if not seed_rows:
        return {}
    keys = set()
    for row in seed_rows:
        keys.update(row)
    out = {}
    for key in sorted(keys):
        values = [
            float(row[key]["mean"])
            for row in seed_rows
            if isinstance(row.get(key), dict) and "mean" in row[key]
        ]
        if not values:
            continue
        out[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "n": len(values),
        }
    out["near_support"] = int(np.mean([r.get("near_support", 0) for r in seed_rows]))
    out["n_pairs"] = int(np.mean([r.get("n_pairs", 0) for r in seed_rows]))
    out["n_scenes"] = int(np.mean([r.get("n_scenes", 0) for r in seed_rows]))
    return out


def run(args) -> dict:
    root = Path(args.clevr_root)
    train = load_scenes(
        root, args.train_split, max_scenes=args.train_scenes, seed=args.seed
    )
    test = load_scenes(
        root, args.test_split, max_scenes=args.test_scenes, seed=args.seed + 1
    )
    load_optional_arms()
    arms = list(args.arms) if args.arms else sorted(ARMS)
    missing = [arm for arm in arms if arm not in ARMS]
    if missing:
        raise KeyError(f"unknown arms {missing}; known: {sorted(ARMS)}")
    print(f"[data] train {len(train)} test {len(test)} scenes; arms {arms}", flush=True)

    per_arm: dict[str, list[dict]] = {arm: [] for arm in arms}
    for seed in args.seeds:
        config = BenchConfig(
            n_views=args.n_views,
            sigma_px=args.sigma_px,
            sigma_depth=args.sigma_depth,
            sigma_pose_yaw=args.sigma_pose_yaw,
            sigma_pose_pitch=args.sigma_pose_pitch,
            outlier_frac=args.outlier_frac,
            eps_xy=args.eps_xy,
            eps_z=args.eps_z,
            eps_c=args.eps_c,
            eps_s=args.eps_s,
            table_z=args.table_z,
            seed=seed,
        )
        run_prepare(train, config)
        examples = build_examples(test, config, seed_offset=seed * 1000)
        for arm in arms:
            metrics = evaluate_arm(arm, examples, config)
            per_arm[arm].append(metrics)
            print(
                f"[seed {seed}] {arm:<24} near {metrics['near_aligned']['mean']:.3f} "
                f"overall {metrics['overall']['mean']:.3f} "
                f"pose_med {metrics['median']['mean']:.3f}",
                flush=True,
            )

    aggregated = {arm: _aggregate(rows) for arm, rows in per_arm.items()}
    payload = {
        "stage": "scene_bench",
        "clevr_root": str(root),
        "train_split": args.train_split,
        "test_split": args.test_split,
        "train_scenes": len(train),
        "test_scenes": len(test),
        "config": {
            "n_views": args.n_views,
            "sigma_px": args.sigma_px,
            "sigma_depth": args.sigma_depth,
            "sigma_pose_yaw": args.sigma_pose_yaw,
            "sigma_pose_pitch": args.sigma_pose_pitch,
            "outlier_frac": args.outlier_frac,
            "eps_xy": args.eps_xy,
            "eps_z": args.eps_z,
            "eps_c": args.eps_c,
            "eps_s": args.eps_s,
            "seeds": list(args.seeds),
        },
        "arms": aggregated,
        "note": (
            "3D scene benchmark: observations are virtual multi-view pinhole "
            "projections of CLEVR 3d_coords; labels are computed on true 3D "
            "poses; near_aligned is the accuracy on pairs whose true |dx| or "
            "|dy| is within [0.5,1.5]*eps_xy."
        ),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "scene_bench.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "SCENE_BENCH_REPORT.md").write_text(
        _render(payload), encoding="utf-8"
    )
    return payload


def _fmt(arm: dict, key: str) -> str:
    row = arm.get(key)
    if not row:
        return "—"
    return f"{row['mean']:.3f} ± {row['std']:.3f}"


def _render(p: dict) -> str:
    lines = [
        "# PSS 三维场景建模基准（D1）",
        "",
        f"- CLEVR: `{p['clevr_root']}` · train {p['train_scenes']} / test {p['test_scenes']} scenes",
        f"- n_views {p['config']['n_views']} · σ_px {p['config']['sigma_px']} · "
        f"σ_depth {p['config']['sigma_depth']} · outlier {p['config']['outlier_frac']}",
        f"- eps_xy {p['config']['eps_xy']} · eps_z {p['config']['eps_z']} · "
        f"eps_c {p['config']['eps_c']}",
        "",
        "## 1. 主表",
        "",
        "| arm | near-aligned | overall | pose median | pose p90 | same_level | contact | collision | support |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, row in p["arms"].items():
        lines.append(
            f"| `{arm}` | {_fmt(row, 'near_aligned')} | {_fmt(row, 'overall')} | "
            f"{_fmt(row, 'median')} | {_fmt(row, 'p90')} | "
            f"{_fmt(row, 'same_level')} | {_fmt(row, 'contact')} | "
            f"{_fmt(row, 'collision')} | {_fmt(row, 'support')} |"
        )
    lines += [
        "",
        "## 2. 口径",
        "",
        "- 标签在**真实 3D pose** 上计算；估计在**观测恢复的 3D pose** 上查询。",
        "- `near_aligned`：真实 `|dx|` 或 `|dy|` 落在 `[0.5,1.5]·eps_xy` 的对。",
        "- `same_level` / `contact` / `collision` / `support` 是三维独有查询。",
        "- 多视角是虚拟针孔相机，不是完整渲染；这是估计器测试，不是感知前端测试。",
        "",
        "## 3. 边界",
        "",
        "- CLEVR `3d_coords` 是标签；观测由已知相机投影 + 噪声生成。",
        "- 物体用球形近似半径（small 0.35 / large 0.70），不是网格碰撞。",
        "- 关联默认已知；outlier 只是像素/深度离群，不是错误关联。",
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
    ap.add_argument("--train-scenes", type=int, default=200)
    ap.add_argument("--test-scenes", type=int, default=100)
    ap.add_argument("--n-views", type=int, default=3)
    ap.add_argument("--sigma-px", type=float, default=2.0)
    ap.add_argument("--sigma-depth", type=float, default=0.01)
    ap.add_argument("--sigma-pose-yaw", type=float, default=0.0)
    ap.add_argument("--sigma-pose-pitch", type=float, default=0.0)
    ap.add_argument("--outlier-frac", type=float, default=0.0)
    ap.add_argument("--eps-xy", type=float, default=0.12)
    ap.add_argument("--eps-z", type=float, default=0.05)
    ap.add_argument("--eps-c", type=float, default=0.02)
    ap.add_argument("--eps-s", type=float, default=0.05)
    ap.add_argument("--table-z", type=float, default=0.0)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", nargs="+", default=None)
    ap.add_argument("--out", default="eval/results/scene_bench")
    args = ap.parse_args()
    payload = run(args)
    for arm, row in payload["arms"].items():
        print(
            f"{arm:<24} near {row['near_aligned']['mean']:.3f} "
            f"overall {row['overall']['mean']:.3f}"
        )


if __name__ == "__main__":
    main()
