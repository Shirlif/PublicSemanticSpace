"""B0: how much planner OOD does region coverage actually buy?

The P5b result says the relation head is the gate: exact-4 1.000 -> OOD 1.000,
exact-4 0.571 -> OOD 0.682.  This script turns that single comparison into a
curve by training the same head on every subset of a given size and measuring,
for each subset:

  - relation-probe exact-4 on the pairs it did **not** see (held-out regions),
  - PSS planner OOD success on the fixed held-out relation-pair task.

The pre-registered decision (``coverage_decision``) is taken at full coverage.
The useful output is the *shape*: if OOD saturates well before full coverage,
the next lever is B2/B3 (predicate quality, search); if it only clears the bar
at full coverage, B1 (labelling the missing cells) is the lever.

Example
-------
python -m pss.eval.scripts.run_relation_coverage \
  --sizes 2 3 4 5 --epochs 400 --out eval/results/relation_coverage
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pss._console import ensure_utf8
from pss.eval.readout_reasoning_preregistration import coverage_decision
from pss.eval.relational_bench import NAMES, REGIONS, TEST_PAIRS
from pss.eval.scripts.run_typed_relations import _learned_rel, _train_spatial_head
from pss.eval.scripts.run_version_b_paired import build_episodes, evaluate_pss
from pss.reasoning.relations.coverage import (
    coverage_features,
    coverage_report,
    pairs_within,
    subsets_by_size,
)

OOD_EPISODE_SEED = 4321

FEATURE_NAMES = (
    "n_regions",
    "axis_isolation",
    "sign_pattern_seen",
    "directions",
    "extremes",
    "hull_area",
    "n_outside_hull",
    "max_hull_distance",
    "sum_hull_distance",
    "x_span",
    "y_span",
    "x_isolating",
    "y_isolating",
)


def _aggregate(rows: list[dict]) -> dict:
    exact = np.asarray([r["exact4"] for r in rows], dtype=np.float64)
    ood = np.asarray([r["ood_success"] for r in rows], dtype=np.float64)
    return {
        "n_subsets": len(rows),
        "exact4": {
            "mean": float(exact.mean()),
            "std": float(exact.std()),
            "min": float(exact.min()),
            "max": float(exact.max()),
        },
        "ood_success": {
            "mean": float(ood.mean()),
            "std": float(ood.std()),
            "min": float(ood.min()),
            "max": float(ood.max()),
        },
    }


def run(device="cuda", sizes=(2, 3, 4, 5), epochs=400, head_seeds=(0,),
        n_test_per_pair=50, out_dir="eval/results/relation_coverage") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    episodes = build_episodes(TEST_PAIRS, n_test_per_pair, seed=OOD_EPISODE_SEED)
    lattice = subsets_by_size(NAMES, sizes)

    rows: list[dict] = []
    for size in sizes:
        for subset in lattice[int(size)]:
            # ``train_regions=None`` means "all regions"; with an explicit
            # subset the probe is evaluated on pairs involving an unseen
            # region, which is exactly the held-out-region setting.
            train_regions = None if len(subset) == len(NAMES) else subset
            coverage = coverage_report(
                NAMES, pairs_within(NAMES if train_regions is None else subset)
            )
            for head_seed in head_seeds:
                head, pairs_te, exact4, per_label = _train_spatial_head(
                    device, epochs=epochs, seed=int(head_seed),
                    train_regions=train_regions,
                )
                ood = float(evaluate_pss(_learned_rel(head, device), episodes).mean())
                rows.append(
                    {
                        "size": int(size),
                        "head_seed": int(head_seed),
                        "regions": list(subset),
                        "region_fraction": coverage.region_fraction,
                        "pair_fraction": coverage.fraction,
                        "n_test_pairs": len(pairs_te),
                        "exact4": float(exact4),
                        "per_label": float(per_label),
                        "ood_success": ood,
                    }
                )
                print(
                    f"size {size} {''.join(subset)} seed {head_seed}: "
                    f"exact4={exact4:.3f} ood={ood:.3f}",
                    flush=True,
                )

    summary = {
        int(size): _aggregate([r for r in rows if r["size"] == int(size)])
        for size in sizes
    }
    ordered = sorted(summary)
    first_at_95 = next(
        (k for k in ordered if summary[k]["ood_success"]["mean"] >= 0.95), None
    )
    payload = {
        "task": "P5b relation-composition OOD (region-coverage sweep)",
        "regions": list(NAMES),
        "sizes": [int(s) for s in sizes],
        "head_seeds": [int(s) for s in head_seeds],
        "epochs": int(epochs),
        "n_test_per_pair": int(n_test_per_pair),
        "episode_seed": OOD_EPISODE_SEED,
        "summary": {str(k): v for k, v in summary.items()},
        "first_size_at_95_ood": first_at_95,
        "decision_at_full_coverage": coverage_decision(
            summary[ordered[-1]]["ood_success"]["mean"]
        ),
        "rows": rows,
    }
    (out / "coverage.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "COVERAGE_REPORT.md").write_text(_render(payload), encoding="utf-8")
    return payload


def _fmt(value, digits=3) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _render(p: dict) -> str:
    lines = [
        "# B0：关系头区域覆盖曲线",
        "",
        f"- task: {p['task']}",
        f"- regions: {', '.join(p['regions'])} · sizes: {p['sizes']} · "
        f"head seeds: {p['head_seeds']}",
        f"- epochs: {p['epochs']} · n/pair: {p['n_test_per_pair']} · "
        f"episode seed: {p['episode_seed']}",
        "- 脚本：`pss/eval/scripts/run_relation_coverage.py`",
        "",
        "---",
        "",
        "## 1. 按覆盖大小汇总",
        "",
        "| 覆盖区域数 | 区域覆盖 | 子集数 | held-out exact-4 | OOD success |",
        "|---:|---:|---:|---:|---:|",
    ]
    for size in p["sizes"]:
        s = p["summary"][str(size)]
        lines.append(
            f"| {size} | {size}/{len(p['regions'])} | {s['n_subsets']} | "
            f"{_fmt(s['exact4']['mean'])} ± {_fmt(s['exact4']['std'])} "
            f"({_fmt(s['exact4']['min'])}–{_fmt(s['exact4']['max'])}) | "
            f"{_fmt(s['ood_success']['mean'])} ± {_fmt(s['ood_success']['std'])} "
            f"({_fmt(s['ood_success']['min'])}–{_fmt(s['ood_success']['max'])}) |"
        )

    lines += [
        "",
        "## 2. 判读",
        "",
        f"- 第一个 OOD 均值 ≥ 0.95 的覆盖大小：**{p['first_size_at_95_ood']}**",
        f"- 满覆盖时的预注册判定：**{p['decision_at_full_coverage']}**",
        "",
        "曲线形状比单点重要：",
        "",
        "- 若在满覆盖之前就饱和到 0.95，B1（补覆盖）的优先级低于 B2/B3（谓词质量、搜索）。",
        "- 若只在满覆盖才达标，B1 是主杠杆；此时用覆盖矩阵挑单元补标注的收益最大。",
        "- 每个大小的子集集合固定，均值已控制\"具体留下哪些区域\"的差异；",
        "  查看 `coverage.json` 的 `rows` 可定位坏子集。",
        "",
        "## 3. 边界",
        "",
        "- 符号、小规模任务；head seed 数有限时，std 只反映 seed 波动的一部分。",
        "- held-out exact-4 的评测对每个子集不同（各自的未见区域对），",
        "  这正是要测的量，但跨大小比较时要记住这一点。",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# B1: offline re-analysis of an existing sweep
# --------------------------------------------------------------------------- #
def _ranks(values) -> np.ndarray:
    """Average ranks, so ties (booleans, zero distances) do not bias rho."""

    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=np.float64)
    ranks[order] = np.arange(len(arr), dtype=np.float64)
    sorted_arr = arr[order]
    i = 0
    while i < len(arr):
        j = i
        while j + 1 < len(arr) and sorted_arr[j + 1] == sorted_arr[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def _spearman(x, y) -> float:
    rx = _ranks(x)
    ry = _ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def reanalyze(path, out_dir=None) -> dict:
    """Re-rank an existing coverage sweep by coordinate geometry.

    Nothing is retrained: each row's ``regions`` list is turned into the B1
    features in :func:`pss.reasoning.relations.coverage.coverage_features`,
    and the rank correlation with the observed ``exact4`` / ``ood_success`` is
    reported.  The point is to test whether "coverage direction" predicts OOD
    better than "coverage count" before spending labels on B1.
    """

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    regions = list(payload["regions"])
    missing = [r for r in regions if r not in REGIONS]
    if missing:
        raise ValueError(
            f"coverage sweep regions {missing} have no coordinates in REGIONS; "
            "B1 re-analysis currently supports the P5b grid"
        )
    coords = {r: REGIONS[r] for r in regions}
    rows = []
    for row in payload["rows"]:
        row = dict(row)
        row["features"] = coverage_features(row["regions"], regions, coords)
        rows.append(row)

    exact4 = [r["exact4"] for r in rows]
    ood = [r["ood_success"] for r in rows]
    correlations = {
        name: {
            "spearman_exact4": _spearman([r["features"][name] for r in rows], exact4),
            "spearman_ood": _spearman([r["features"][name] for r in rows], ood),
        }
        for name in FEATURE_NAMES
    }
    correlations_by_seed = {}
    seeds = sorted({r["head_seed"] for r in rows if "head_seed" in r})
    for seed in seeds:
        group = [r for r in rows if r.get("head_seed") == seed]
        correlations_by_seed[str(seed)] = {
            name: {
                "spearman_exact4": _spearman(
                    [r["features"][name] for r in group], [r["exact4"] for r in group]
                ),
                "spearman_ood": _spearman(
                    [r["features"][name] for r in group],
                    [r["ood_success"] for r in group],
                ),
            }
            for name in FEATURE_NAMES
        }
    result = {
        "source": str(source),
        "task": payload.get("task"),
        "regions": regions,
        "n_rows": len(rows),
        "head_seeds": seeds,
        "features": list(FEATURE_NAMES),
        "correlations": correlations,
        "correlations_by_seed": correlations_by_seed,
        "rows": rows,
    }
    out = Path(out_dir) if out_dir is not None else source.parent
    out.mkdir(parents=True, exist_ok=True)
    (out / "coverage_direction.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "COVERAGE_DIRECTION.md").write_text(
        _render_direction(result), encoding="utf-8"
    )
    return result


def _render_direction(p: dict) -> str:
    ranked = sorted(
        p["correlations"].items(), key=lambda kv: -abs(kv[1]["spearman_ood"])
    )
    best_name, best = ranked[0]
    lines = [
        "# B1：覆盖方向重分析（离线，不重训）",
        "",
        f"- source: `{p['source']}` · rows: {p['n_rows']}",
        "- 指标：每个子集的坐标几何特征 vs 观测 exact-4 / OOD 的 Spearman ρ。",
        "- 说明：正相关表示该特征越大、效果越好；hull 距离类指标相反（越大越差）。",
        "",
        "## 1. 特征排序（按 |ρ(OOD)|）",
        "",
        "| 特征 | ρ(exact-4) | ρ(OOD) |",
        "|---|---:|---:|",
    ]
    for name, c in ranked:
        lines.append(
            f"| `{name}` | {c['spearman_exact4']:+.3f} | {c['spearman_ood']:+.3f} |"
        )

    by_seed = p.get("correlations_by_seed") or {}
    if by_seed:
        seeds = sorted(by_seed, key=int)
        lines += [
            "",
            "## 1b. 按 seed 的 ρ(OOD)（防单 seed 偶然）",
            "",
            "| 特征 | " + " | ".join(f"seed {s}" for s in seeds) + " | mean ± std |",
            "|---|" + "---:|" * len(seeds) + "---:|",
        ]
        for name in FEATURE_NAMES:
            vals = [by_seed[s][name]["spearman_ood"] for s in seeds]
            mean = float(np.mean(vals))
            std = float(np.std(vals))
            lines.append(
                f"| `{name}` | "
                + " | ".join(f"{v:+.3f}" for v in vals)
                + f" | {mean:+.3f} ± {std:.3f} |"
            )

    rows3 = [r for r in p["rows"] if r["size"] == 3]
    groups3: dict[str, list[dict]] = {}
    agg3: list[dict] = []
    if rows3:
        for r in rows3:
            groups3.setdefault("".join(r["regions"]), []).append(r)
        for subset, group in groups3.items():
            features = group[0]["features"]
            agg3.append(
                {
                    "subset": subset,
                    "n": len(group),
                    "axis_isolation": features["axis_isolation"],
                    "sign_pattern_seen": features["sign_pattern_seen"],
                    "n_outside_hull": features["n_outside_hull"],
                    "exact4": float(np.mean([r["exact4"] for r in group])),
                    "ood": float(np.mean([r["ood_success"] for r in group])),
                }
            )
        lines += [
            "",
            "## 2. 同尺寸对照（3/5 子集，按子集聚合 seeds）",
            "",
            "| 子集 | seeds | axis_isolation | sign_pattern_seen | hull 外点数 | "
            "exact-4 (mean) | OOD (mean) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in sorted(agg3, key=lambda r: -r["ood"]):
            lines.append(
                f"| {row['subset']} | {row['n']} | {row['axis_isolation']} | "
                f"{row['sign_pattern_seen']:.3f} | {row['n_outside_hull']} | "
                f"{row['exact4']:.3f} | {row['ood']:.3f} |"
            )

    lines += [
        "",
        "## 3. 判读",
        "",
        f"- 与 OOD 相关性最强的特征是 **`{best_name}`** "
        f"(ρ={best['spearman_ood']:+.3f})；区域计数 `n_regions` 的 ρ="
        f"{p['correlations']['n_regions']['spearman_ood']:+.3f}，"
        f"`axis_isolation` 的 ρ={p['correlations']['axis_isolation']['spearman_ood']:+.3f}。",
        f"- 样本：{p['n_rows']} 行 / {len(p.get('head_seeds', []))} 个 head seed；"
        "1b 的按 seed ρ 是保守口径，主表是合并口径。",
    ]
    if agg3:
        worst = min(agg3, key=lambda r: r["ood"])
        best_row = max(agg3, key=lambda r: r["ood"])
        lines.append(
            f"- 同为 3/5，最差 **{worst['subset']}**（OOD 均值 {worst['ood']:.3f}，"
            f"{worst['n']} seeds）的 axis_isolation={worst['axis_isolation']}、"
            f"hull 外点={worst['n_outside_hull']}；最好 **{best_row['subset']}**"
            f"（OOD 均值 {best_row['ood']:.3f}，{best_row['n']} seeds）的 "
            f"axis_isolation={best_row['axis_isolation']}、"
            f"hull 外点={best_row['n_outside_hull']}。"
        )
        if worst["n_outside_hull"] > best_row["n_outside_hull"]:
            lines.append(
                "- 这一对上凸包外点数方向一致（最差子集外点更多），"
                "但它的整体 ρ 仍低于 axis_isolation。"
            )
        elif worst["n_outside_hull"] == best_row["n_outside_hull"]:
            lines.append(
                "- **凸包假设不能解释这一对**：最差与最好子集的 hull 外点数相同。"
            )
        else:
            lines.append(
                "- **凸包假设被这一对反证**：最差子集的 hull 外点数反而更少，"
                "所以“训练点在凸包外”不能解释它的失败。"
            )
        if worst["axis_isolation"] < best_row["axis_isolation"]:
            lines.append(
                "- **轴隔离假设与这一对一致**：最好子集同时含 x/y 隔离对，"
                "最差子集缺至少一条轴。"
            )
        else:
            lines.append(
                "- **轴隔离假设未完全分离**：最差与最好子集的 axis_isolation 相同，"
                "需要 per-label 诊断或更多 seed 才能定位。"
            )
        better_same = [
            r
            for r in agg3
            if r["axis_isolation"] == worst["axis_isolation"]
            and r["subset"] != worst["subset"]
            and r["ood"] > worst["ood"] + 0.05
        ]
        if better_same:
            top = sorted(better_same, key=lambda r: -r["ood"])[:2]
            lines.append(
                f"- **但轴隔离不能单独解释最差子集**：同为 axis_isolation="
                f"{worst['axis_isolation']} 的 "
                + "、".join(f"{r['subset']}({r['ood']:.3f})" for r in top)
                + " 明显更好；最差子集是异常点，需要 per-label 诊断，"
                "而不是只补隔离对。"
            )

    partial = [r for r in p["rows"] if r["size"] < len(p["regions"])]
    high = [r for r in partial if r["features"]["axis_isolation"] >= 2]
    low = [r for r in partial if r["features"]["axis_isolation"] <= 1]
    if high and low:
        high_min = min(r["ood_success"] for r in high)
        low_max = max(r["ood_success"] for r in low)
        if high_min < low_max:
            lines.append(
                f"- **只是相关，不是必要/充分条件**：非满覆盖里，axis_isolation=2 的"
                f"子集最低 OOD {high_min:.3f}，axis_isolation≤1 的子集最高 OOD "
                f"{low_max:.3f}；两侧都有重叠。"
            )
        else:
            lines.append(
                f"- **分离良好**：axis_isolation=2 的子集最低 OOD {high_min:.3f} ≥ "
                f"axis_isolation≤1 的子集最高 OOD {low_max:.3f}。"
            )

    lines += [
        "",
        "## 4. 边界",
        "",
        f"- 这是对既有 B0 扫描的离线重排：{len(p.get('head_seeds', [])) or 1} 个 head seed；"
        "ρ 给的是点估计与按 seed 的离散度，不是置信区间。",
        "- feature 之间高度相关（n_regions / axis_isolation / sign_pattern_seen），",
        "  ρ 排序只能提示方向，不能当作因果归因。",
        "- 坐标只对 P5b 网格有效；换任务需要提供新的 `region -> (x, y)` 表。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", nargs="+", type=int, default=[2, 3, 4, 5])
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--head-seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--n-test", type=int, default=50)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="eval/results/relation_coverage")
    ap.add_argument(
        "--reanalyze",
        default=None,
        help="path to an existing coverage.json; skip training, write B1 direction report",
    )
    args = ap.parse_args()
    if args.reanalyze:
        result = reanalyze(args.reanalyze, out_dir=args.out)
        ranked = sorted(
            result["correlations"].items(), key=lambda kv: -abs(kv[1]["spearman_ood"])
        )
        print("B1 direction re-analysis (|rho OOD| order):")
        for name, c in ranked:
            print(
                f"  {name:<20} rho_exact4={c['spearman_exact4']:+.3f} "
                f"rho_ood={c['spearman_ood']:+.3f}"
            )
        print(f"output: {Path(args.out) / 'COVERAGE_DIRECTION.md'}")
        return
    payload = run(
        device=args.device,
        sizes=tuple(args.sizes),
        epochs=args.epochs,
        head_seeds=tuple(args.head_seeds),
        n_test_per_pair=args.n_test,
        out_dir=args.out,
    )
    print(
        f"first size at OOD>=0.95: {payload['first_size_at_95_ood']} · "
        f"decision at full coverage: {payload['decision_at_full_coverage']}"
    )


if __name__ == "__main__":
    main()
