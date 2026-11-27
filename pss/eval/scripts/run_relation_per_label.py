"""BCD per-label diagnosis: which predicate fails, and does augmentation fix it?

B1/B2 left one concrete failure: the BCD head (trained on B/C/D) is 0.335 OOD
while ABD/ACD with ``axis_isolation=1`` reach 0.855/0.793.  The per-label
diagnosis showed why: BCD has no ``dy = 0, dx != 0`` pair, so it never learns
that a zero vertical delta suppresses ``above/below``; on horizontal held-out
pairs those predicates fire as false positives.  ABD is the mirror image
(no ``dx = 0, dy != 0`` pair, so ``left/right`` false-fire on vertical pairs).

This script runs two arms:

  baseline   train on the subset's region pairs only
  augmented  add synthetic counterexamples on the missing isolation axis
             (BCD gets horizontal pairs, ABD gets vertical pairs)

and reports, per arm x subset x seed:

  - exact-4 overall and split by pair type (horizontal / vertical / diagonal),
  - per-predicate precision / recall / F1,
  - a per-pair table with the true label vector, mean probability and the
    number of seeds that got the vector right.

The augmentation is applied automatically based on which isolation direction
the subset lacks; ACG has both, so its augmented arm is a no-op control.

Example
-------
python -m pss.eval.scripts.run_relation_per_label \
  --subsets BCD ABD ACG --seeds 0 1 2 --epochs 400 --augment 300 \
  --out eval/results/bcd_diagnosis_aug
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pss._console import ensure_utf8
from pss.eval.relational_bench import NAMES, REGIONS, RELATIONS
from pss.eval.scripts.run_typed_relations import (
    _pair_feature,
    _pair_labels,
    _train_spatial_head,
)
from pss.reasoning.relations.coverage import ordered_pairs, unseen_pairs

TYPES = ("horizontal", "vertical", "diagonal")
DEFAULT_SUBSETS = ("BCD", "ABD", "ACG")
EPS = 1e-9


def _pair_type(a: str, b: str) -> str:
    dx = REGIONS[b][0] - REGIONS[a][0]
    dy = REGIONS[b][1] - REGIONS[a][1]
    if abs(dy) <= EPS:
        return "horizontal"
    if abs(dx) <= EPS:
        return "vertical"
    return "diagonal"


def _true_labels(a: str, b: str) -> list[bool]:
    return [bool(x) for x in _pair_labels(REGIONS[a], REGIONS[b])]


def _probs(head, a: str, b: str, device: str, feature_mode: str = "absolute") -> list[float]:
    x = torch.tensor(
        [_pair_feature(REGIONS[a], REGIONS[b], feature_mode)],
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        values = torch.sigmoid(head.net(x))[0].cpu().numpy()
    return [float(v) for v in values]


def _exact(rows: list[dict]) -> float:
    if not rows:
        return float("nan")
    return float(
        np.mean([all(t == p for t, p in zip(r["true"], r["pred"])) for r in rows])
    )


def _prf(rows: list[dict], index: int) -> dict:
    tp = fp = fn = tn = 0
    for r in rows:
        truth = bool(r["true"][index])
        pred = bool(r["pred"][index])
        if truth and pred:
            tp += 1
        elif truth:
            fn += 1
        elif pred:
            fp += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": (tp + tn) / max(len(rows), 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _mean_std(values) -> dict:
    values = [v for v in values if not np.isnan(v)]
    if not values:
        return {"mean": float("nan"), "std": float("nan")}
    return {"mean": float(np.mean(values)), "std": float(np.std(values))}


def _aggregate(rows: list[dict], arm: str, subset: str, seeds) -> dict:
    per_seed = {}
    for seed in seeds:
        group = [
            r
            for r in rows
            if r["arm"] == arm and r["subset"] == subset and r["seed"] == seed
        ]
        per_seed[str(seed)] = {
            "exact4": _exact(group),
            "by_type": {t: _exact([r for r in group if r["type"] == t]) for t in TYPES},
            "by_predicate": {
                predicate: _prf(group, i) for i, predicate in enumerate(RELATIONS)
            },
        }
    return {
        "per_seed": per_seed,
        "mean": {
            "exact4": _mean_std([v["exact4"] for v in per_seed.values()]),
            "by_type": {
                t: _mean_std([v["by_type"][t] for v in per_seed.values()])
                for t in TYPES
            },
            "by_predicate": {
                predicate: {
                    key: _mean_std(
                        [v["by_predicate"][predicate][key] for v in per_seed.values()]
                    )
                    for key in ("accuracy", "precision", "recall", "f1")
                }
                for predicate in RELATIONS
            },
        },
    }


def _training_profile(subset) -> dict:
    pairs = ordered_pairs(subset)
    positives = {predicate: 0 for predicate in RELATIONS}
    x_isolating = y_isolating = False
    for a, b in pairs:
        labels = _true_labels(a, b)
        for predicate, label in zip(RELATIONS, labels):
            positives[predicate] += int(label)
        dx = REGIONS[b][0] - REGIONS[a][0]
        dy = REGIONS[b][1] - REGIONS[a][1]
        if abs(dx) > EPS and abs(dy) <= EPS:
            x_isolating = True
        if abs(dy) > EPS and abs(dx) <= EPS:
            y_isolating = True
    return {
        "n_pairs": len(pairs),
        "x_isolating": x_isolating,
        "y_isolating": y_isolating,
        "positives": positives,
    }


def _extra_counts(profile: dict, augment: int) -> dict:
    """Add counterexamples only on the missing isolation axis."""

    return {
        "horizontal": int(augment) if augment > 0 and not profile["x_isolating"] else 0,
        "vertical": int(augment) if augment > 0 and not profile["y_isolating"] else 0,
    }


def run(
    subsets=DEFAULT_SUBSETS,
    seeds=(0, 1, 2),
    epochs: int = 400,
    device: str = "cpu",
    out_dir: str = "eval/results/bcd_diagnosis",
    augment: int = 0,
    noise: float = 0.03,
    n_per_pair: int = 300,
    feature_mode: str = "absolute",
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    arm_specs = [("baseline", 0)]
    if augment > 0:
        arm_specs.append(("augmented", int(augment)))

    profiles: dict[str, dict] = {}
    extras: dict[str, dict] = {}
    for subset in subsets:
        key = "".join(subset)
        profiles[key] = _training_profile(subset)
        extras[key] = _extra_counts(profiles[key], augment)

    rows: list[dict] = []
    for arm, _amount in arm_specs:
        for subset in subsets:
            key = "".join(subset)
            train_regions = None if set(subset) == set(NAMES) else tuple(subset)
            pairs = (
                unseen_pairs(NAMES, subset)
                if train_regions is not None
                else ordered_pairs(NAMES)
            )
            extra_h = 0 if arm == "baseline" else extras[key]["horizontal"]
            extra_v = 0 if arm == "baseline" else extras[key]["vertical"]
            for seed in seeds:
                head, _pairs_te, exact4, per_label = _train_spatial_head(
                    device,
                    epochs=epochs,
                    seed=int(seed),
                    train_regions=train_regions,
                    extra_horizontal=extra_h,
                    extra_vertical=extra_v,
                    noise=noise,
                    n_per_pair=n_per_pair,
                    feature_mode=feature_mode,
                )
                for a, b in pairs:
                    probs = _probs(head, a, b, device, feature_mode)
                    rows.append(
                        {
                            "arm": arm,
                            "subset": key,
                            "seed": int(seed),
                            "pair": f"{a}->{b}",
                            "type": _pair_type(a, b),
                            "dx": float(REGIONS[b][0] - REGIONS[a][0]),
                            "dy": float(REGIONS[b][1] - REGIONS[a][1]),
                            "true": _true_labels(a, b),
                            "prob": probs,
                            "pred": [p > 0.5 for p in probs],
                        }
                    )
                print(
                    f"{arm:<9} {key:<5} seed {seed}: exact4={exact4:.3f} "
                    f"per-label={per_label:.3f}",
                    flush=True,
                )

    summary = {
        arm: {
            key: _aggregate(rows, arm, key, seeds)
            for key in profiles
        }
        for arm, _ in arm_specs
    }

    pair_table: dict[str, dict[str, list[dict]]] = {}
    for arm, _ in arm_specs:
        pair_table[arm] = {}
        for key in profiles:
            grouped: dict[str, list[dict]] = {}
            for r in rows:
                if r["arm"] == arm and r["subset"] == key:
                    grouped.setdefault(r["pair"], []).append(r)
            pair_table[arm][key] = []
            for pair, group in grouped.items():
                correct = sum(
                    all(t == p for t, p in zip(r["true"], r["pred"])) for r in group
                )
                pair_table[arm][key].append(
                    {
                        "pair": pair,
                        "type": group[0]["type"],
                        "dx": group[0]["dx"],
                        "dy": group[0]["dy"],
                        "true": group[0]["true"],
                        "mean_prob": [
                            float(np.mean([r["prob"][i] for r in group]))
                            for i in range(len(RELATIONS))
                        ],
                        "mean_pred": [
                            bool(np.mean([r["pred"][i] for r in group]) >= 0.5)
                            for i in range(len(RELATIONS))
                        ],
                        "seeds_correct": correct,
                        "n_seeds": len(group),
                    }
                )

    payload = {
        "task": "BCD per-label diagnosis with axis augmentation",
        "subsets": ["".join(s) for s in subsets],
        "seeds": [int(s) for s in seeds],
        "epochs": int(epochs),
        "augment": int(augment),
        "noise": float(noise),
        "n_per_pair": int(n_per_pair),
        "feature_mode": feature_mode,
        "predicates": list(RELATIONS),
        "pair_types": list(TYPES),
        "training_profiles": profiles,
        "extras": extras,
        "summary": summary,
        "pair_table": pair_table,
        "rows": rows,
    }
    (out / "bcd_diagnosis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "BCD_DIAGNOSIS.md").write_text(_render(payload), encoding="utf-8")
    return payload


def _fmt(value: float, digits: int = 3) -> str:
    return "n/a" if value is None or np.isnan(value) else f"{value:.{digits}f}"


def _delta(aug: float, base: float) -> float:
    return aug - base


def _pred_cell(summary: dict, predicate: str) -> str:
    m = summary["mean"]["by_predicate"][predicate]
    return (
        f"{_fmt(m['precision']['mean'])}/{_fmt(m['recall']['mean'])}/"
        f"{_fmt(m['f1']['mean'])}"
    )


def _render(p: dict) -> str:
    arms = list(p["summary"])
    lines = [
        "# BCD per-label 诊断（含单轴增强）",
        "",
        f"- subsets: {', '.join(p['subsets'])} · seeds: {p['seeds']} · "
        f"epochs: {p['epochs']} · augment: {p['augment']} · "
        f"feature: {p.get('feature_mode', 'absolute')}",
        "- baseline = 只用子集内区域对；augmented = 在缺失的隔离方向补合成反例；"
        "ACG 两个方向都有，augmented 是空对照。",
        "",
        "## 1. 训练覆盖与增强",
        "",
        "| 子集 | x 隔离对 | y 隔离对 | left+ | right+ | above+ | below+ | "
        "extra horizontal | extra vertical |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for subset, profile in p["training_profiles"].items():
        pos = profile["positives"]
        extra = p["extras"][subset]
        lines.append(
            f"| {subset} | {'是' if profile['x_isolating'] else '否'} | "
            f"{'是' if profile['y_isolating'] else '否'} | "
            f"{pos['left']} | {pos['right']} | {pos['above']} | {pos['below']} | "
            f"{extra['horizontal']} | {extra['vertical']} |"
        )

    lines += [
        "",
        "## 2. Held-out exact-4（按对类型，3 seeds 均值 ± std）",
        "",
        "| arm | 子集 | overall | horizontal | vertical | diagonal |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for arm in arms:
        for subset, s in p["summary"][arm].items():
            mean = s["mean"]
            lines.append(
                f"| {arm} | {subset} | {_fmt(mean['exact4']['mean'])} ± "
                f"{_fmt(mean['exact4']['std'])} | "
                + " | ".join(
                    f"{_fmt(mean['by_type'][t]['mean'])} ± "
                    f"{_fmt(mean['by_type'][t]['std'])}"
                    for t in TYPES
                )
                + " |"
            )

    lines += [
        "",
        "## 3. 四谓词 precision / recall / F1（3 seeds 均值）",
        "",
        "| arm | 子集 | left | right | above | below |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for arm in arms:
        for subset, s in p["summary"][arm].items():
            cells = [_pred_cell(s, predicate) for predicate in p["predicates"]]
            lines.append(f"| {arm} | {subset} | " + " | ".join(cells) + " |")

    base_bcd = {r["pair"]: r for r in p["pair_table"]["baseline"].get("BCD", [])}
    aug_bcd = {
        r["pair"]: r for r in p["pair_table"].get("augmented", {}).get("BCD", [])
    }
    if base_bcd:
        lines += [
            "",
            "## 4. BCD 逐对明细（mean prob，L/R/A/B）",
            "",
        ]
        if aug_bcd:
            lines += [
                "| pair | type | dx,dy | true | base prob | base pred | "
                "aug prob | aug pred | correct base→aug |",
                "|---|---|---|---|---|---|---|---|---:|",
            ]
            for pair, row in base_bcd.items():
                aug = aug_bcd.get(pair)
                true = "".join("1" if x else "0" for x in row["true"])
                base_prob = " ".join(f"{x:.2f}" for x in row["mean_prob"])
                base_pred = "".join("1" if x else "0" for x in row["mean_pred"])
                if aug:
                    aug_prob = " ".join(f"{x:.2f}" for x in aug["mean_prob"])
                    aug_pred = "".join("1" if x else "0" for x in aug["mean_pred"])
                    correct = f"{row['seeds_correct']}→{aug['seeds_correct']}"
                else:
                    aug_prob = aug_pred = "n/a"
                    correct = f"{row['seeds_correct']}"
                lines.append(
                    f"| {pair} | {row['type']} | ({row['dx']:+.0f},{row['dy']:+.0f}) | "
                    f"{true} | {base_prob} | {base_pred} | {aug_prob} | {aug_pred} | "
                    f"{correct} |"
                )
        else:
            lines += [
                "| pair | type | dx,dy | true | mean prob | mean pred | seeds correct |",
                "|---|---|---|---|---|---|---:|",
            ]
            for row in base_bcd.values():
                true = "".join("1" if x else "0" for x in row["true"])
                prob = " ".join(f"{x:.2f}" for x in row["mean_prob"])
                pred = "".join("1" if x else "0" for x in row["mean_pred"])
                lines.append(
                    f"| {row['pair']} | {row['type']} | "
                    f"({row['dx']:+.0f},{row['dy']:+.0f}) | {true} | {prob} | "
                    f"{pred} | {row['seeds_correct']}/{row['n_seeds']} |"
                )

    lines += [
        "",
        "## 5. 增强前后对比",
        "",
    ]
    if len(arms) < 2:
        lines.append("- 只跑了 baseline；加 `--augment N` 才会生成 augmented 臂。")
    else:
        lines += [
            "| 子集 | Δoverall | Δhorizontal | Δvertical | Δdiagonal | "
            "Δabove P | Δbelow P | Δleft P | Δright P |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for subset in p["summary"]["baseline"]:
            base = p["summary"]["baseline"][subset]["mean"]
            aug = p["summary"]["augmented"][subset]["mean"]
            precision_deltas = [
                _delta(
                    aug["by_predicate"][q]["precision"]["mean"],
                    base["by_predicate"][q]["precision"]["mean"],
                )
                for q in ("above", "below", "left", "right")
            ]
            lines.append(
                f"| {subset} | {_delta(aug['exact4']['mean'], base['exact4']['mean']):+.3f} | "
                + " | ".join(
                    f"{_delta(aug['by_type'][t]['mean'], base['by_type'][t]['mean']):+.3f}"
                    for t in TYPES
                )
                + " | "
                + " | ".join(f"{v:+.3f}" for v in precision_deltas)
                + " |"
            )

        bcd_base = p["summary"]["baseline"].get("BCD")
        bcd_aug = p["summary"]["augmented"].get("BCD")
        if bcd_base and bcd_aug:
            h_base = bcd_base["mean"]["by_type"]["horizontal"]["mean"]
            h_aug = bcd_aug["mean"]["by_type"]["horizontal"]["mean"]
            p_above_base = bcd_base["mean"]["by_predicate"]["above"]["precision"]["mean"]
            p_above_aug = bcd_aug["mean"]["by_predicate"]["above"]["precision"]["mean"]
            p_below_base = bcd_base["mean"]["by_predicate"]["below"]["precision"]["mean"]
            p_below_aug = bcd_aug["mean"]["by_predicate"]["below"]["precision"]["mean"]
            lines.append("")
            lines.append(
                f"- **BCD horizontal exact-4**：{_fmt(h_base)} → {_fmt(h_aug)} "
                f"（Δ{_delta(h_aug, h_base):+.3f}）。"
            )
            lines.append(
                f"- **BCD above/below precision**：{_fmt(p_above_base)}/"
                f"{_fmt(p_below_base)} → {_fmt(p_above_aug)}/"
                f"{_fmt(p_below_aug)}。"
            )
            if h_aug >= 0.99:
                lines.append(
                    "- **增强修复了 absence case**：水平对的 exact-4 到达 1.000，"
                    "above/below 的假阳性被压掉。"
                )
            elif h_aug > h_base + 0.2:
                lines.append(
                    "- **增强显著改善但未完全修复**：水平对仍低于 1.000，"
                    "下一步建议去掉绝对坐标/做坐标归一化，或加几何 absence 约束。"
                )
            else:
                lines.append(
                    "- **增强没有修复**：只补反例不足以让头泛化到 held-out 点；"
                    "下一步必须改特征（去绝对坐标）或加显式 absence 约束。"
                )

        acg_base = p["summary"]["baseline"].get("ACG")
        acg_aug = p["summary"]["augmented"].get("ACG")
        if acg_base and acg_aug:
            acg_delta = _delta(
                acg_aug["mean"]["exact4"]["mean"],
                acg_base["mean"]["exact4"]["mean"],
            )
            if abs(acg_delta) < 1e-9:
                lines.append(
                    "- 对照 ACG：没有可补的隔离方向，augmented 与 baseline 完全一致"
                    "（Δ=0），说明增强逻辑只作用在缺失轴上。"
                )

    lines += [
        "",
        "## 6. 边界",
        "",
        "- 每个子集 3 个 head seed；pair 级明细是跨 seed 的平均。",
        "- 增强样本是 y=0（或 x=0）上的合成对，不是新的真实区域；只教 absence case。",
        "- 只测 5 区域 P5b 网格；结论不外推到其它几何。",
        "- 训练使用 `_pair_labels` 的 eps=0.1 死区；held-out 的 dx/dy 都是精确网格值。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subsets", nargs="+", default=list(DEFAULT_SUBSETS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--augment", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.03)
    ap.add_argument("--n-per-pair", type=int, default=300)
    ap.add_argument(
        "--feature-mode", choices=["absolute", "relative"], default="absolute"
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="eval/results/bcd_diagnosis")
    args = ap.parse_args()
    run(
        subsets=args.subsets,
        seeds=tuple(args.seeds),
        epochs=args.epochs,
        device=args.device,
        out_dir=args.out,
        augment=args.augment,
        noise=args.noise,
        n_per_pair=args.n_per_pair,
        feature_mode=args.feature_mode,
    )
    print(f"output: {Path(args.out) / 'BCD_DIAGNOSIS.md'}")


if __name__ == "__main__":
    main()
