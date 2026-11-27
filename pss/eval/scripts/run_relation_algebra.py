"""B2: does algebraic consistency improve held-out relation prediction?

The relation head is trained with plain BCE on ordered pairs, so nothing forces
``left(a, b)`` and ``right(b, a)`` to agree.  This ablation adds the
antisymmetry + exclusivity penalty from
:mod:`pss.reasoning.relations.algebra` inside ``_train_spatial_head`` and
measures, per region subset and per head seed:

  - held-out exact-4 / per-label accuracy,
  - planner OOD success with the learned predicate,
  - the number of algebraic violations the learned head produces on the
    held-out region pairs.

The zero-weight arm is the control and must reproduce the P1 numbers.

Example
-------
python -m pss.eval.scripts.run_relation_algebra \
  --subsets ACG BCD BDG --weights 0.0 0.1 --seeds 0 1 2 --epochs 400
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pss._console import ensure_utf8
from pss.eval.relational_bench import NAMES, RELATIONS, TEST_PAIRS
from pss.eval.scripts.run_typed_relations import _learned_rel, _train_spatial_head
from pss.eval.scripts.run_version_b_paired import build_episodes, evaluate_pss
from pss.reasoning.relations.algebra import consistency
from pss.reasoning.relations.coverage import ordered_pairs, unseen_pairs

OOD_EPISODE_SEED = 4321
DEFAULT_SUBSETS = ("ABCDG", "ACG", "BCG", "BCD", "BDG", "ABC", "ACD")


def _facts_for(rel_fn, pairs) -> set[tuple[str, str, str]]:
    facts = set()
    for a, b in pairs:
        for predicate in RELATIONS:
            if rel_fn(a, b, predicate):
                facts.add((a, predicate, b))
    return facts


def _held_out_pairs(subset) -> list[tuple[str, str]]:
    if set(subset) == set(NAMES):
        return ordered_pairs(NAMES)
    return unseen_pairs(NAMES, subset)


def run(
    subsets=DEFAULT_SUBSETS,
    weights=(0.0, 0.1),
    seeds=(0, 1, 2),
    epochs: int = 400,
    n_test_per_pair: int = 50,
    device: str = "cpu",
    out_dir: str = "eval/results/relation_algebra",
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    episodes = build_episodes(TEST_PAIRS, n_test_per_pair, seed=OOD_EPISODE_SEED)

    rows: list[dict] = []
    for subset in subsets:
        subset = tuple(subset)
        if any(r not in NAMES for r in subset):
            raise ValueError(f"unknown regions in {subset}; expected {NAMES}")
        train_regions = None if set(subset) == set(NAMES) else subset
        pairs_ext = _held_out_pairs(subset)
        for weight in weights:
            for seed in seeds:
                head, _pairs_te, exact4, per_label = _train_spatial_head(
                    device,
                    epochs=epochs,
                    seed=int(seed),
                    train_regions=train_regions,
                    algebra_weight=float(weight),
                )
                rel_fn = _learned_rel(head, device)
                ood = float(evaluate_pss(rel_fn, episodes).mean())
                check = consistency(_facts_for(rel_fn, pairs_ext))
                row = {
                    "subset": "".join(subset),
                    "weight": float(weight),
                    "seed": int(seed),
                    "exact4": float(exact4),
                    "per_label": float(per_label),
                    "ood_success": ood,
                    "n_facts": int(check["n_facts"]),
                    "n_violations": int(
                        check["exclusivity"]
                        + check["antisymmetry"]
                        + check["inverse_gap"]
                        + check["transitive_gap"]
                    ),
                    "consistency_score": float(check["score"]),
                    "exclusivity": int(check["exclusivity"]),
                    "antisymmetry": int(check["antisymmetry"]),
                }
                rows.append(row)
                print(
                    f"{row['subset']:<6} w={weight:<4} seed={seed} "
                    f"exact4={exact4:.3f} per={per_label:.3f} ood={ood:.3f} "
                    f"viol={row['n_violations']} score={check['score']:.3f}",
                    flush=True,
                )

    summary = {}
    for subset in dict.fromkeys(r["subset"] for r in rows):
        summary[subset] = {}
        for weight in sorted({r["weight"] for r in rows}):
            group = [r for r in rows if r["subset"] == subset and r["weight"] == weight]
            if not group:
                continue
            summary[subset][f"{weight:g}"] = {
                key: {
                    "mean": float(np.mean([r[key] for r in group])),
                    "std": float(np.std([r[key] for r in group])),
                }
                for key in ("exact4", "per_label", "ood_success", "n_violations")
            }

    deltas = {}
    for subset in summary:
        keys = sorted(summary[subset])
        if not keys:
            continue
        base = keys[0]
        deltas[subset] = {
            key: {
                "exact4": summary[subset][key]["exact4"]["mean"]
                - summary[subset][base]["exact4"]["mean"],
                "ood_success": summary[subset][key]["ood_success"]["mean"]
                - summary[subset][base]["ood_success"]["mean"],
                "n_violations": summary[subset][key]["n_violations"]["mean"]
                - summary[subset][base]["n_violations"]["mean"],
            }
            for key in keys[1:]
        }

    payload = {
        "task": "P5b relation algebra ablation",
        "subsets": ["".join(s) for s in subsets],
        "weights": [float(w) for w in weights],
        "seeds": [int(s) for s in seeds],
        "epochs": int(epochs),
        "n_test_per_pair": int(n_test_per_pair),
        "episode_seed": OOD_EPISODE_SEED,
        "rows": rows,
        "summary": summary,
        "deltas_vs_zero_weight": deltas,
    }
    (out / "relation_algebra.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "RELATION_ALGEBRA_REPORT.md").write_text(_render(payload), encoding="utf-8")
    return payload


def _render(p: dict) -> str:
    weights = sorted({r["weight"] for r in p["rows"]})
    lines = [
        "# B2：关系代数一致性消融",
        "",
        f"- subsets: {', '.join(p['subsets'])} · weights: {weights} · "
        f"seeds: {p['seeds']} · epochs: {p['epochs']}",
        f"- OOD episodes: {p['n_test_per_pair']}/pair · seed {p['episode_seed']}",
        "",
        "## 1. 按子集 / 权重汇总（3 seeds 均值 ± std）",
        "",
        "| 子集 | weight | exact-4 | per-label | OOD | violations |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for subset, by_weight in p["summary"].items():
        for weight, s in by_weight.items():
            lines.append(
                f"| {subset} | {weight} | {s['exact4']['mean']:.3f} ± {s['exact4']['std']:.3f} | "
                f"{s['per_label']['mean']:.3f} | {s['ood_success']['mean']:.3f} ± "
                f"{s['ood_success']['std']:.3f} | {s['n_violations']['mean']:.1f} |"
            )

    lines += [
        "",
        "## 2. 相对零权重臂的增量",
        "",
        "| 子集 | weight | Δexact-4 | ΔOOD | Δviolations |",
        "|---|---:|---:|---:|---:|",
    ]
    for subset, by_weight in p["deltas_vs_zero_weight"].items():
        for weight, d in by_weight.items():
            lines.append(
                f"| {subset} | {weight} | {d['exact4']:+.3f} | "
                f"{d['ood_success']:+.3f} | {d['n_violations']:+.1f} |"
            )

    lines += [
        "",
        "## 3. 判读",
        "",
    ]
    flat = [
        (subset, weight, d)
        for subset, by in p["deltas_vs_zero_weight"].items()
        for weight, d in by.items()
    ]
    if flat:
        best = max(flat, key=lambda x: x[2]["ood_success"])
        lines.append(
            f"- 提升最大的子集：**{best[0]}**（weight {best[1]}，"
            f"Δexact-4 {best[2]['exact4']:+.3f}，ΔOOD {best[2]['ood_success']:+.3f}）。"
        )
        unchanged = [
            x
            for x in flat
            if abs(x[2]["exact4"]) < 1e-9 and abs(x[2]["ood_success"]) < 1e-9
        ]
        if unchanged:
            lines.append(
                "- **代数损失没有改变**："
                + "、".join(x[0] for x in unchanged)
                + " 的 Δexact-4 与 ΔOOD 都为 0。"
            )
        worse = [x for x in flat if x[2]["n_violations"] > 0]
        if worse:
            lines.append(
                "- **violations 不是单调下降**："
                + "、".join(f"{x[0]}({x[2]['n_violations']:+.1f})" for x in worse)
                + "；该计数包含 inverse/transitive gap，不等于矛盾数。"
            )
        worst = min(
            p["summary"],
            key=lambda s: p["summary"][s][min(p["summary"][s])]["ood_success"]["mean"],
        )
        zero_worst = all(
            abs(d["exact4"]) < 1e-9 and abs(d["ood_success"]) < 1e-9
            for d in p["deltas_vs_zero_weight"].get(worst, {}).values()
        )
        if zero_worst:
            lines.append(
                f"- **最差子集不变**：`{worst}` 在所有非零权重下 Δ 为 0，"
                "说明代数约束修不了它的失败；需要 per-label 诊断（哪个谓词坏），"
                "而不是继续加 algebra 权重。"
            )

    lines += [
        "",
        "## 4. 边界",
        "",
        "- 只有一个 grid（5 区域）和一个小 head；结论不外推到其它几何。",
        "- `violations` 统计在 held-out 区域对上，包含 inverse/transitive gap；",
        "  这些 gap 可能来自只存储单向事实，不全是矛盾。",
        "- 权重网格很粗（0.0 vs 0.1），只回答“有没有用”，不回答最优权重。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subsets", nargs="+", default=list(DEFAULT_SUBSETS))
    ap.add_argument("--weights", nargs="+", type=float, default=[0.0, 0.1])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=50)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="eval/results/relation_algebra")
    args = ap.parse_args()
    payload = run(
        subsets=args.subsets,
        weights=tuple(args.weights),
        seeds=tuple(args.seeds),
        epochs=args.epochs,
        n_test_per_pair=args.n_test,
        device=args.device,
        out_dir=args.out,
    )
    print(f"output: {Path(args.out) / 'RELATION_ALGEBRA_REPORT.md'}")
    print(json.dumps(payload["deltas_vs_zero_weight"], ensure_ascii=False))


if __name__ == "__main__":
    main()
