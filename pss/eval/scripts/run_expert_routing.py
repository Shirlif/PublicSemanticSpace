"""Active-set routing over the frozen expert pool.

The fusion audit asks whether the full pool is complementary.  This script asks
the runtime question: can a cheap router invoke only the experts a sample needs
and keep most of the fusion gain?

Protocol (384-class cross-view instance task by default):

- fit one pooled-covariance LDA per non-empty expert subset on train;
- route each test sample to a subset by a cheap signal (vision's predicted
  class, or per-expert confidence), then use that subset's own model;
- report accuracy against the mean number of experts invoked.

The oracle rows are upper bounds with perfect subset selection; they are not
deployable and exist only to show how much headroom routing leaves.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

from pss._console import ensure_utf8
from pss.encoders.scripts.bench_expert_zoo import _fit_lda, _standardize_blocks
from pss.eval.scripts.run_attribution_grid import load_features, task_split
from pss.prototypes.scorers import predict_classes

MODALITIES = ("vision", "tactile", "geometry")


def _subsets(names: tuple[str, ...]):
    for size in range(1, len(names) + 1):
        for subset in combinations(names, size):
            yield tuple(subset)


def _acc(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(pred == y)) if len(y) else float("nan")


def _fit_models(X, y, train_idx, val_idx, test_idx, names):
    """Fit one LDA per expert subset; keep val/test predictions."""
    models: dict[tuple[str, ...], dict] = {}
    for subset in _subsets(names):
        Xtr = np.concatenate([X[name][train_idx] for name in subset], axis=1)
        Xva = np.concatenate([X[name][val_idx] for name in subset], axis=1)
        Xte = np.concatenate([X[name][test_idx] for name in subset], axis=1)
        scorer = _fit_lda(Xtr, y[train_idx])
        val_pred = predict_classes(scorer, Xva)
        test_pred = predict_classes(scorer, Xte)
        models[subset] = {
            "val_pred": val_pred,
            "test_pred": test_pred,
            "val_acc": _acc(val_pred, y[val_idx]),
            "test_acc": _acc(test_pred, y[test_idx]),
        }
    return models


def _route_confidence(models, X, y, train_idx, val_idx, test_idx, names, k: int):
    """Top-k experts by calibrated per-expert confidence, then subset model."""
    conf_test = []
    for name in names:
        val_scores = _scores(X[name], y, train_idx, val_idx)
        test_scores = _scores(X[name], y, train_idx, test_idx)
        mu = val_scores.mean(axis=0, keepdims=True)
        sigma = val_scores.std(axis=0, keepdims=True) + 1e-9
        conf_test.append(((test_scores - mu) / sigma).max(axis=1))
    conf = np.stack(conf_test, axis=1)
    chosen = np.argsort(-conf, axis=1)[:, :k]
    out = np.empty(len(test_idx), dtype=np.int64)
    used = []
    for row in range(len(test_idx)):
        chosen_names = {names[int(i)] for i in chosen[row]}
        subset = tuple(name for name in names if name in chosen_names)
        models[subset]  # subset of size k always exists
        out[row] = models[subset]["test_pred"][row]
        used.append(len(subset))
    return out, float(np.mean(used))


def _scores(X, y, train_idx, eval_idx):
    scorer = _fit_lda(X[train_idx], y[train_idx])
    return np.asarray(scorer.scores(X[eval_idx]))


def _route_class(models, y, val_idx, test_idx, names, classes):
    """Choose the best subset per class on val, route with vision's prediction."""
    vision = models[("vision",)]
    class_subset: dict[int, tuple[str, ...]] = {}
    for cls in range(len(classes)):
        mask = y[val_idx] == cls
        if not mask.any():
            class_subset[cls] = tuple(names)
            continue
        scored = {
            subset: _acc(models[subset]["val_pred"][mask], y[val_idx][mask])
            for subset in models
        }
        class_subset[cls] = max(
            scored, key=lambda subset: (scored[subset], -len(subset), subset)
        )

    out = np.empty(len(test_idx), dtype=np.int64)
    used = []
    for row, cls in enumerate(vision["test_pred"]):
        subset = class_subset[int(cls)]
        out[row] = models[subset]["test_pred"][row]
        used.append(len(set(subset) | {"vision"}))
    return out, float(np.mean(used)), class_subset


def _oracle(models, y, test_idx, names):
    """Perfect per-sample subset selection; smallest correct subset wins."""
    n = len(test_idx)
    best_size = np.full(n, len(names) + 1, dtype=np.int64)
    best_correct = np.zeros(n, dtype=bool)
    for subset in _subsets(names):
        correct = models[subset]["test_pred"] == y[test_idx]
        better = correct & (len(subset) < best_size)
        best_size[better] = len(subset)
        best_correct |= correct
    sizes = np.where(best_correct, best_size, len(names))
    return float(best_correct.mean()), float(sizes.mean()), Counter(sizes.tolist())


def _cascade(models, X, y, train_idx, val_idx, test_idx, names, coverages):
    """Deployable cascade: run vision first, escalate only when uncertain.

    The threshold is calibrated on val so that ``coverage`` of val samples are
    accepted by vision alone.  Escalated samples pay for the full pool.
    """
    val_scores = _scores(X["vision"], y, train_idx, val_idx)
    test_scores = _scores(X["vision"], y, train_idx, test_idx)
    mu = val_scores.mean(axis=0, keepdims=True)
    sigma = val_scores.std(axis=0, keepdims=True) + 1e-9
    val_conf = ((val_scores - mu) / sigma).max(axis=1)
    test_conf = ((test_scores - mu) / sigma).max(axis=1)
    vision_test = models[("vision",)]["test_pred"]
    all_test = models[tuple(names)]["test_pred"]

    rows = []
    for coverage in coverages:
        threshold = float(np.quantile(val_conf, coverage))
        accept = test_conf >= threshold
        pred = np.where(accept, vision_test, all_test)
        escalation = float(1.0 - accept.mean())
        rows.append(
            {
                "router": f"cascade coverage {coverage:.0%}",
                "acc": _acc(pred, y[test_idx]),
                "mean_experts": 1.0 + (len(names) - 1) * escalation,
                "escalation_rate": escalation,
                "threshold": threshold,
            }
        )
    return rows


def _fmt(value, digits=3):
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def main() -> None:
    ensure_utf8()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/category_pretrain_instance8")
    parser.add_argument(
        "--experts", default="checkpoints/category_pretrain_instance8_experts"
    )
    parser.add_argument("--cache", default="eval/results/readout_opt")
    parser.add_argument("--out", default="eval/results/expert_routing")
    args = parser.parse_args()

    feats = load_features(args.data, args.experts, Path(args.cache))
    task = task_split(feats, "view-instance")
    names = tuple(n for n in MODALITIES if n in feats)
    X = {n: np.asarray(feats[n], dtype=np.float64) for n in names}
    n_rows = len(feats["instances"])
    split_arr = np.array(["ignore"] * n_rows, dtype=object)
    split_arr[task["train_idx"]] = "train"
    split_arr[task["val_idx"]] = "val"
    split_arr[task["test_idx"]] = "test"

    # Standardize each expert on train, then fit every subset model.
    X = _standardize_blocks(X, task={
        "train": task["train_idx"], "val": task["val_idx"], "test": task["test_idx"]
    })
    y = np.asarray(
        [
            task["cls_to_i"].get(str(feats["instances"][i]), -1)
            for i in range(n_rows)
        ],
        dtype=np.int64,
    )
    models = _fit_models(
        X, y, task["train_idx"], task["val_idx"], task["test_idx"], names
    )

    rows = []
    for subset, model in sorted(models.items(), key=lambda kv: (len(kv[0]), kv[0])):
        rows.append(
            {
                "route": "+".join(subset),
                "experts": len(subset),
                "val_acc": model["val_acc"],
                "test_acc": model["test_acc"],
            }
        )

    all_subset = tuple(names)
    best_single = max(
        ((name,), models[(name,)]["test_acc"]) for name in names
    )
    route_rows = []
    for k in range(1, len(names) + 1):
        pred, mean_used = _route_confidence(
            models, X, y, task["train_idx"], task["val_idx"], task["test_idx"], names, k
        )
        route_rows.append(
            {
                "router": f"confidence top-{k}",
                "acc": _acc(pred, y[task["test_idx"]]),
                "mean_experts": mean_used,
            }
        )

    cls_pred, cls_used, class_subset = _route_class(
        models, y, task["val_idx"], task["test_idx"], names, task["classes"]
    )
    route_rows.append(
        {
            "router": "class-conditional (vision class)",
            "acc": _acc(cls_pred, y[task["test_idx"]]),
            "mean_experts": cls_used,
        }
    )
    route_rows.extend(
        _cascade(
            models, X, y, task["train_idx"], task["val_idx"], task["test_idx"],
            names, coverages=(0.3, 0.5, 0.7, 0.9),
        )
    )

    oracle_acc, oracle_used, oracle_sizes = _oracle(
        models, y, task["test_idx"], names
    )

    payload = {
        "data": args.data,
        "classes": len(task["classes"]),
        "splits": {
            "train": int(len(task["train_idx"])),
            "val": int(len(task["val_idx"])),
            "test": int(len(task["test_idx"])),
        },
        "subset_models": rows,
        "all_experts": {
            "acc": models[all_subset]["test_acc"],
            "experts": len(all_subset),
        },
        "best_single": {
            "route": "+".join(best_single[0]),
            "acc": best_single[1],
            "experts": 1,
        },
        "routers": route_rows,
        "oracle": {
            "acc": oracle_acc,
            "mean_experts": oracle_used,
            "minimal_size_counts": dict(sorted(oracle_sizes.items())),
        },
        "class_subset": {
            str(cls): "+".join(subset)
            for cls, subset in sorted(class_subset.items())
        },
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "expert_routing.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# Active-set expert routing",
        "",
        f"> data: `{args.data}`; classes: {payload['classes']}; "
        f"train/val/test = {payload['splits']['train']}/"
        f"{payload['splits']['val']}/{payload['splits']['test']}",
        "",
        "## Subset models (LDA on train, test accuracy)",
        "",
        "| subset | experts | val | test |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['route']}` | {row['experts']} | "
            f"{_fmt(row['val_acc'])} | {_fmt(row['test_acc'])} |"
        )
    lines += [
        "",
        "## Routers",
        "",
        "| router | acc | mean experts |",
        "|---|---:|---:|",
        f"| all experts | {_fmt(payload['all_experts']['acc'])} | "
        f"{payload['all_experts']['experts']} |",
        f"| best single | {_fmt(payload['best_single']['acc'])} | 1 |",
    ]
    for row in route_rows:
        lines.append(
            f"| {row['router']} | {_fmt(row['acc'])} | {_fmt(row['mean_experts'])} |"
        )
    lines += [
        f"| oracle subset (upper bound) | {_fmt(oracle_acc)} | "
        f"{_fmt(oracle_used)} |",
        "",
        "> `confidence top-k` uses every expert's score, so it is a post-hoc "
        "router and does not save compute. The deployable rows are `cascade`: "
        "vision runs first, and the remaining experts run only on escalated "
        "samples. Oracle is perfect subset selection, not deployable.",
        "",
        "## Per-class subset (selected on val)",
        "",
        "| class | subset |",
        "|---|---|",
    ]
    for cls, subset in sorted(class_subset.items()):
        lines.append(f"| {cls} | `{'+'.join(subset)}` |")
    lines.append("")
    (out / "EXPERT_ROUTING.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2)[:4000])
    print(f"Wrote {out / 'EXPERT_ROUTING.md'}")


if __name__ == "__main__":
    main()
