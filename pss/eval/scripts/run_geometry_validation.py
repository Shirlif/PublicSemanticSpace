"""Larger-geometry and absolute-position validation.

P5b is a 5-point grid and its four predicates are translation-invariant, so
relative features are a *task-matched* prior.  This script tests the two
remaining caveats from the report:

1. **Absolute position carries meaning.**  A task whose labels depend on the
   subject's absolute zone (left/right/above/below half-plane) must defeat a
   relative-only feature.  If it does, "relative is better" is not universal.
2. **Larger geometry.**  A 7x7 grid (49 regions) with several training subsets:
   a 3x3 block (both isolation directions present), a horizontal strip (only
   x-isolation), a vertical strip (only y-isolation), and the diagonal
   (neither).  The P5b augmentation rule is applied where a direction is
   missing, and the absolute/relative comparison is repeated.

Example
-------
python -m pss.eval.scripts.run_geometry_validation --n 7 --seeds 0 1 2 \
  --epochs 400 --extra 500 --out eval/results/geometry_validation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pss._console import ensure_utf8
from pss.eval.scripts.run_typed_relations import _pair_feature, _pair_labels
from pss.memory.typed_relations import SpatialRelationHead

TYPES = ("horizontal", "vertical", "diagonal")
RELATIVE_PREDICATES = ("left", "right", "above", "below")
ABSOLUTE_PREDICATES = ("subject_left", "subject_right", "subject_above", "subject_below")
EPS = 1e-9

ALL_EXPERIMENTS = (
    {"subset": "block3", "label": "relative", "augment": ()},
    {"subset": "block3", "label": "absolute", "augment": ()},
    {"subset": "strip_x", "label": "relative", "augment": ("y",)},
    {"subset": "strip_y", "label": "relative", "augment": ("x",)},
    {"subset": "diag", "label": "relative", "augment": ("x", "y")},
)

# Focused follow-up: is the missing piece the axis, or the missing pair type?
PAIR_TYPE_EXPERIMENTS = (
    {"subset": "strip_x", "label": "relative", "augment": ()},
    {"subset": "strip_x", "label": "relative", "augment": ("y",)},
    {"subset": "strip_x", "label": "relative", "augment": ("y", "diagonal")},
    {"subset": "strip_x", "label": "relative", "augment": ("diagonal",)},
)


def _grid(n: int) -> dict[str, np.ndarray]:
    half = (n - 1) / 2.0
    return {
        f"{i}{j}": np.array([(i - half), (j - half)], dtype=np.float32)
        for i in range(n)
        for j in range(n)
    }


def _absolute_labels(pa, pb, eps: float = 0.1) -> list[bool]:
    """Subject's absolute half-plane membership; ``pb`` is unused."""

    return [pa[0] < -eps, pa[0] > eps, pa[1] > eps, pa[1] < -eps]


def _labels(pa, pb, mode: str) -> list[bool]:
    if mode == "relative":
        return [bool(x) for x in _pair_labels(pa, pb)]
    if mode == "absolute":
        return [bool(x) for x in _absolute_labels(pa, pb)]
    raise ValueError(f"unknown label mode {mode!r}")


def _pair_type(pa, pb) -> str:
    dx = pb[0] - pa[0]
    dy = pb[1] - pa[1]
    if abs(dy) <= EPS:
        return "horizontal"
    if abs(dx) <= EPS:
        return "vertical"
    return "diagonal"


def _subset_names(coords: dict[str, np.ndarray], kind: str) -> list[str]:
    names = list(coords)
    if kind == "block3":
        return [n for n in names if 2 <= int(n[0]) <= 4 and 2 <= int(n[1]) <= 4]
    if kind == "strip_x":
        return [n for n in names if int(n[1]) == 3]
    if kind == "strip_y":
        return [n for n in names if int(n[0]) == 3]
    if kind == "diag":
        return [n for n in names if int(n[0]) == int(n[1])]
    raise ValueError(f"unknown subset {kind!r}")


def _isolation(names, coords) -> dict:
    x_isolating = y_isolating = False
    for a in names:
        for b in names:
            if a == b:
                continue
            dx = coords[b][0] - coords[a][0]
            dy = coords[b][1] - coords[a][1]
            if abs(dx) > EPS and abs(dy) <= EPS:
                x_isolating = True
            if abs(dy) > EPS and abs(dx) <= EPS:
                y_isolating = True
    return {"x": x_isolating, "y": y_isolating}


def _pairs_within(names) -> list[tuple[str, str]]:
    return [(a, b) for a in names for b in names if a != b]


def _held_out_pairs(all_names, train_names, max_pairs: int, seed: int):
    train = set(train_names)
    pairs = [
        (a, b)
        for a in all_names
        for b in all_names
        if a != b and (a not in train or b not in train)
    ]
    if len(pairs) > max_pairs:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[int(i)] for i in idx]
    return pairs


def _matrix(
    pairs,
    coords,
    feature_mode: str,
    label_mode: str,
    repeats: int = 1,
    noise: float = 0.0,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for a, b in pairs:
        for _ in range(int(repeats)):
            pa = coords[a] + rng.normal(scale=noise, size=2)
            pb = coords[b] + rng.normal(scale=noise, size=2)
            X.append(_pair_feature(pa, pb, feature_mode))
            y.append(_labels(pa, pb, label_mode))
    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.float32),
    )


def _extra_rows(rng, axis: str, count: int, feature_mode: str, extent, noise: float = 0.02):
    """Synthetic pairs with the other axis exact (absence-case supervision)."""

    xmin, ymin, xmax, ymax = extent
    rows = []
    for _ in range(int(count)):
        if axis == "diagonal":
            dx = float(rng.choice([-2.0, -1.0, 1.0, 2.0]))
            dy = float(rng.choice([-2.0, -1.0, 1.0, 2.0]))
            x1 = float(
                rng.uniform(xmin, xmax - dx) if dx > 0 else rng.uniform(xmin - dx, xmax)
            )
            y1 = float(
                rng.uniform(ymin, ymax - dy) if dy > 0 else rng.uniform(ymin - dy, ymax)
            )
            pa = np.array([x1, y1], dtype=np.float32)
            pb = np.array([x1 + dx, y1 + dy], dtype=np.float32)
            pa = pa + rng.normal(scale=noise, size=2)
            pb = pb + rng.normal(scale=noise, size=2)
            rows.append((_pair_feature(pa, pb, feature_mode), _pair_labels(pa, pb)))
            continue

        offset = float(rng.choice([-2.0, -1.0, 1.0, 2.0]))
        if axis == "x":
            y = float(rng.uniform(ymin, ymax))
            if offset > 0:
                x1 = float(rng.uniform(xmin, xmax - offset))
            else:
                x1 = float(rng.uniform(xmin - offset, xmax))
            pa = np.array([x1, y], dtype=np.float32)
            pb = np.array([x1 + offset, y], dtype=np.float32)
        else:
            x = float(rng.uniform(xmin, xmax))
            if offset > 0:
                y1 = float(rng.uniform(ymin, ymax - offset))
            else:
                y1 = float(rng.uniform(ymin - offset, ymax))
            pa = np.array([x, y1], dtype=np.float32)
            pb = np.array([x, y1 + offset], dtype=np.float32)
        pa = pa + rng.normal(scale=noise, size=2)
        pb = pb + rng.normal(scale=noise, size=2)
        rows.append((_pair_feature(pa, pb, feature_mode), _pair_labels(pa, pb)))
    return rows


def _train(X, y, feature_mode: str, seed: int, epochs: int, device: str):
    torch.manual_seed(seed)
    head = SpatialRelationHead(feature_mode=feature_mode).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    y = torch.as_tensor(y, dtype=torch.float32, device=device)
    for _ in range(epochs):
        perm = torch.randperm(len(X), device=device)
        for start in range(0, len(perm), 128):
            idx = perm[start : start + 128]
            loss = F.binary_cross_entropy_with_logits(head.net(X[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    head.eval()
    return head


def _predict(head, X, device: str) -> np.ndarray:
    with torch.no_grad():
        logits = head.net(torch.as_tensor(X, dtype=torch.float32, device=device))
        return (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(bool)


def _exact(Y, P) -> float:
    return float(np.mean(np.all(Y.astype(bool) == P, axis=1)))


def _prf(Y, P, index: int) -> dict:
    y = Y[:, index].astype(bool)
    p = P[:, index]
    tp = int(np.sum(y & p))
    fp = int(np.sum(~y & p))
    fn = int(np.sum(y & ~p))
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def run(
    n: int = 7,
    seeds=(0, 1, 2),
    epochs: int = 400,
    extra: int = 500,
    max_test: int = 800,
    n_per_pair: int = 50,
    device: str = "cpu",
    out_dir: str = "eval/results/geometry_validation",
    focus: str = "all",
) -> dict:
    coords = _grid(n)
    all_names = list(coords)
    values = np.stack(list(coords.values()))
    extent = (
        float(values[:, 0].min()),
        float(values[:, 1].min()),
        float(values[:, 0].max()),
        float(values[:, 1].max()),
    )

    experiments = PAIR_TYPE_EXPERIMENTS if focus == "pair_type" else ALL_EXPERIMENTS

    rows: list[dict] = []
    profiles: dict[str, dict] = {}
    for exp in experiments:
        subset = exp["subset"]
        if subset in profiles:
            continue
        names = _subset_names(coords, subset)
        train_pairs = _pairs_within(names)
        test_pairs = _held_out_pairs(all_names, names, max_test, seed=4321)
        isolation = _isolation(names, coords)
        type_counts = {t: 0 for t in TYPES}
        for a, b in test_pairs:
            type_counts[_pair_type(coords[a], coords[b])] += 1
        profiles[subset] = {
            "names": names,
            "n_train_pairs": len(train_pairs),
            "n_test_pairs": len(test_pairs),
            "isolation": isolation,
            "test_type_counts": type_counts,
        }

    for exp in experiments:
        subset = exp["subset"]
        label_mode = exp["label"]
        names = profiles[subset]["names"]
        isolation = profiles[subset]["isolation"]
        train_pairs = _pairs_within(names)
        test_pairs = _held_out_pairs(all_names, names, max_test, seed=4321)
        Y_test = np.array(
            [_labels(coords[a], coords[b], label_mode) for a, b in test_pairs],
            dtype=np.float32,
        )
        test_types = np.array([_pair_type(coords[a], coords[b]) for a, b in test_pairs])

        for feature_mode in ("absolute", "relative"):
            axes = tuple(exp["augment"])
            arms = [("baseline", 0)]
            if axes:
                arms.append(("augmented:" + "+".join(axes), int(extra)))
            for arm, amount in arms:
                for seed in seeds:
                    X_train, y_train = _matrix(
                        train_pairs,
                        coords,
                        feature_mode,
                        label_mode,
                        repeats=n_per_pair,
                        noise=0.03,
                        seed=seed,
                    )
                    if arm != "baseline":
                        rng = np.random.default_rng(seed + 10_000)
                        base = amount // len(axes)
                        counts = {axis: base for axis in axes}
                        counts[axes[0]] += amount - base * len(axes)
                        add = []
                        for axis in axes:
                            if counts[axis]:
                                add.extend(
                                    _extra_rows(
                                        rng, axis, counts[axis], feature_mode, extent
                                    )
                                )
                        if add:
                            X_train = np.vstack([X_train, np.array([r[0] for r in add])])
                            y_train = np.vstack([y_train, np.array([r[1] for r in add])])

                    head = _train(
                        X_train, y_train, feature_mode, int(seed), epochs, device
                    )
                    X_test, _ = _matrix(test_pairs, coords, feature_mode, label_mode)
                    pred = _predict(head, X_test, device)
                    by_type = {
                        t: _exact(Y_test[test_types == t], pred[test_types == t])
                        for t in TYPES
                    }
                    pred_names = (
                        RELATIVE_PREDICATES
                        if label_mode == "relative"
                        else ABSOLUTE_PREDICATES
                    )
                    rows.append(
                        {
                            "subset": subset,
                            "label": label_mode,
                            "feature": feature_mode,
                            "arm": arm,
                            "seed": int(seed),
                            "exact4": _exact(Y_test, pred),
                            "by_type": by_type,
                            "by_predicate": {
                                name: _prf(Y_test, pred, i)
                                for i, name in enumerate(pred_names)
                            },
                        }
                    )
                    print(
                        f"{subset:<8} {label_mode:<8} {feature_mode:<8} {arm:<9} "
                        f"seed {seed}: exact4={rows[-1]['exact4']:.3f} "
                        f"h/v/d={by_type['horizontal']:.2f}/"
                        f"{by_type['vertical']:.2f}/{by_type['diagonal']:.2f}",
                        flush=True,
                    )

    summary = {}
    for row in rows:
        key = f"{row['subset']}|{row['label']}|{row['feature']}|{row['arm']}"
        summary.setdefault(key, []).append(row)

    aggregated = {}
    for key, group in summary.items():
        aggregated[key] = {
            "subset": group[0]["subset"],
            "label": group[0]["label"],
            "feature": group[0]["feature"],
            "arm": group[0]["arm"],
            "n_seeds": len(group),
            "exact4": {
                "mean": float(np.mean([r["exact4"] for r in group])),
                "std": float(np.std([r["exact4"] for r in group])),
            },
            "by_type": {
                t: {
                    "mean": float(np.mean([r["by_type"][t] for r in group])),
                    "std": float(np.std([r["by_type"][t] for r in group])),
                }
                for t in TYPES
            },
            "by_predicate": {
                name: {
                    key2: float(np.mean([r["by_predicate"][name][key2] for r in group]))
                    for key2 in ("precision", "recall", "f1")
                }
                for name in group[0]["by_predicate"]
            },
        }

    payload = {
        "task": "geometry scaling + absolute-position validation",
        "grid": n,
        "seeds": [int(s) for s in seeds],
        "epochs": int(epochs),
        "extra": int(extra),
        "max_test": int(max_test),
        "n_per_pair": int(n_per_pair),
        "focus": focus,
        "profiles": {
            subset: {k: v for k, v in profile.items() if k != "names"}
            for subset, profile in profiles.items()
        },
        "results": aggregated,
        "rows": rows,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "geometry_validation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "GEOMETRY_VALIDATION.md").write_text(_render(payload), encoding="utf-8")
    return payload


def _fmt(value: float, digits: int = 3) -> str:
    return "n/a" if value is None or np.isnan(value) else f"{value:.{digits}f}"


def _render(p: dict) -> str:
    lines = [
        "# 大几何与绝对位置验证",
        "",
        f"- grid: {p['grid']}x{p['grid']} · seeds: {p['seeds']} · "
        f"epochs: {p['epochs']} · extra: {p['extra']} · max test pairs: {p['max_test']}",
        "",
        "## 1. 子集与覆盖",
        "",
        "| 子集 | 训练对 | 测试对 | x 隔离 | y 隔离 | test h/v/d |",
        "|---|---:|---:|---|---|---|",
    ]
    for subset, profile in p["profiles"].items():
        counts = profile["test_type_counts"]
        lines.append(
            f"| {subset} | {profile['n_train_pairs']} | {profile['n_test_pairs']} | "
            f"{'是' if profile['isolation']['x'] else '否'} | "
            f"{'是' if profile['isolation']['y'] else '否'} | "
            f"{counts['horizontal']}/{counts['vertical']}/{counts['diagonal']} |"
        )

    lines += [
        "",
        "## 2. Held-out exact-4（3 seeds 均值 ± std）",
        "",
        "| 子集 | label | feature | arm | overall | horizontal | vertical | diagonal |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for result in p["results"].values():
        lines.append(
            f"| {result['subset']} | {result['label']} | {result['feature']} | "
            f"{result['arm']} | {_fmt(result['exact4']['mean'])} ± "
            f"{_fmt(result['exact4']['std'])} | "
            + " | ".join(
                f"{_fmt(result['by_type'][t]['mean'])} ± "
                f"{_fmt(result['by_type'][t]['std'])}"
                for t in TYPES
            )
            + " |"
        )

    lines += [
        "",
        "## 3. 判读",
        "",
    ]
    abs_abs = p["results"].get("block3|absolute|absolute|baseline")
    abs_rel = p["results"].get("block3|absolute|relative|baseline")
    if abs_abs and abs_rel:
        lines.append(
            f"- **绝对位置任务**：absolute 特征 exact-4 "
            f"{_fmt(abs_abs['exact4']['mean'])}，relative 特征 "
            f"{_fmt(abs_rel['exact4']['mean'])}。"
            + (
                " → 相对特征按构造失败，说明它只是平移不变任务的匹配先验，不是通用优势。"
                if abs_abs["exact4"]["mean"] > abs_rel["exact4"]["mean"] + 0.3
                else " → 两者差距没有预期大，需要检查任务/训练覆盖。"
            )
        )
    rel_abs = p["results"].get("block3|relative|absolute|baseline")
    rel_rel = p["results"].get("block3|relative|relative|baseline")
    if rel_abs and rel_rel:
        lines.append(
            f"- **相对位置任务**：absolute 特征 exact-4 "
            f"{_fmt(rel_abs['exact4']['mean'])}，relative 特征 "
            f"{_fmt(rel_rel['exact4']['mean'])}。"
        )

    for subset in ("strip_x", "strip_y", "diag"):
        group = [
            result
            for result in p["results"].values()
            if result["subset"] == subset and result["label"] == "relative"
        ]
        if not group:
            continue
        lines.append(f"- **{subset}**：")
        for result in sorted(group, key=lambda x: (x["feature"], x["arm"])):
            lines.append(
                f"  - {result['feature']} / {result['arm']}: "
                f"exact-4 {_fmt(result['exact4']['mean'])}"
                f"（h/v/d {_fmt(result['by_type']['horizontal']['mean'])}/"
                f"{_fmt(result['by_type']['vertical']['mean'])}/"
                f"{_fmt(result['by_type']['diagonal']['mean'])}）"
            )
        base = p["results"].get(f"{subset}|relative|relative|baseline")
        for arm in (
            "augmented:y",
            "augmented:y+diagonal",
            "augmented:diagonal",
            "augmented:x",
            "augmented:x+diagonal",
            "augmented:x+y",
        ):
            result = p["results"].get(f"{subset}|relative|relative|{arm}")
            if result and base:
                delta = result["exact4"]["mean"] - base["exact4"]["mean"]
                lines.append(
                    f"  - {arm} 相对 baseline：{delta:+.3f}，"
                    f"diagonal {_fmt(result['by_type']['diagonal']['mean'])}。"
                )

    base = p["results"].get("strip_x|relative|relative|baseline")
    single = p["results"].get("strip_x|relative|relative|augmented:y")
    both = p["results"].get("strip_x|relative|relative|augmented:y+diagonal")
    diag_only = p["results"].get("strip_x|relative|relative|augmented:diagonal")
    if base and single and both:
        d_base = base["by_type"]["diagonal"]["mean"]
        d_single = single["by_type"]["diagonal"]["mean"]
        d_both = both["by_type"]["diagonal"]["mean"]
        lines.append(
            f"- **pair-type 覆盖对照（strip_x, relative）**：baseline "
            f"{_fmt(base['exact4']['mean'])}（diagonal {_fmt(d_base)}）；+vertical "
            f"{_fmt(single['exact4']['mean'])}（diagonal {_fmt(d_single)}）；"
            f"+vertical+diagonal {_fmt(both['exact4']['mean'])}"
            f"（diagonal {_fmt(d_both)}）。"
        )
        if diag_only:
            lines.append(
                f"  - +diagonal-only：{_fmt(diag_only['exact4']['mean'])}"
                f"（diagonal {_fmt(diag_only['by_type']['diagonal']['mean'])}）。"
            )
        best_control = max(base["exact4"]["mean"], single["exact4"]["mean"])
        if both["exact4"]["mean"] > best_control + 0.05 and d_both > d_single + 0.05:
            lines.append(
                "  - **pair-type 覆盖假设得到支持**：补 diagonal 后 diagonal exact-4 "
                "恢复，overall 超过 baseline 与单轴增强。"
            )
        else:
            lines.append(
                "  - **pair-type 覆盖假设未完全支持**：补 diagonal 没有同时恢复 "
                "diagonal 与 overall；需要检查增强量、头容量或几何覆盖。"
            )

    lines += [
        "",
        "## 4. 边界",
        "",
        "- 7x7 网格仍是合成几何；区域命名/坐标是规则的。",
        "- 绝对标签任务只测 subject 的半平面归属，不代表所有绝对位置语义。",
        "- 测试对随机采样到固定上限；类型计数见 §1。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=7)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--extra", type=int, default=500)
    ap.add_argument("--max-test", type=int, default=800)
    ap.add_argument("--n-per-pair", type=int, default=50)
    ap.add_argument("--focus", choices=["all", "pair_type"], default="all")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="eval/results/geometry_validation")
    args = ap.parse_args()
    run(
        n=args.n,
        seeds=tuple(args.seeds),
        epochs=args.epochs,
        extra=args.extra,
        max_test=args.max_test,
        n_per_pair=args.n_per_pair,
        focus=args.focus,
        device=args.device,
        out_dir=args.out,
    )
    print(f"output: {Path(args.out) / 'GEOMETRY_VALIDATION.md'}")


if __name__ == "__main__":
    main()
