"""P0-1: DINOv2 instance token for PSS re-ID.

Four token variants share the same PSS entity representation:

- base: vision projection + geometry + pose + category;
- dino: base + frozen DINOv2 CLS;
- tactile: base + tactile projection;
- both: base + tactile + DINOv2.

The graph matcher is trained separately for each variant. A missing-modality
test then zeroes one token block of the best ``both`` model.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pss._console import ensure_utf8
from pss.eval.baselines.entity_transformer import build_pairs
from pss.memory.graph_matcher import evaluate_graph, save_graph, train_graph
from pss.memory.learned_metric import (
    DINO_DIM,
    TACTILE_DIM,
    TOKEN_DIM,
    token_dim,
    token_layout,
)

SUBSETS = ("mixed", "samecat", "near_identical")
VARIANTS = (
    ("base", False, False),
    ("dino", False, True),
    ("tactile", True, False),
    ("both", True, True),
)


def _mean_std(values):
    return (
        float(np.mean(values)) if values else 0.0,
        float(np.std(values)) if values else 0.0,
    )


def _agg(runs):
    overall = [r["reid_accuracy"] for r in runs]
    mean, std = _mean_std(overall)
    per = {}
    for subset in SUBSETS:
        m, s = _mean_std([r["per_subset"][subset]["reid_accuracy"] for r in runs])
        per[subset] = {"mean": m, "std": s}
    return {"overall_mean": mean, "overall_std": std, "per_subset": per}


def _select(tokens: np.ndarray, include_tactile: bool, include_dino: bool) -> np.ndarray:
    """Slice a full (both) token array down to one variant."""
    base = tokens[:, :TOKEN_DIM]
    if not include_tactile and not include_dino:
        return base
    parts = [base]
    if include_tactile:
        start = TOKEN_DIM
        parts.append(tokens[:, start : start + TACTILE_DIM])
    if include_dino:
        start = TOKEN_DIM + TACTILE_DIM
        parts.append(tokens[:, start : start + DINO_DIM])
    return np.concatenate(parts, axis=1)


def select_pairs(pairs, include_tactile: bool, include_dino: bool):
    return [
        replace(
            pair,
            pre=_select(pair.pre, include_tactile, include_dino),
            post=_select(pair.post, include_tactile, include_dino),
        )
        for pair in pairs
    ]


def zero_modality(pairs, modality: str, include_tactile=True, include_dino=True):
    layout = token_layout(include_tactile, include_dino)
    start, end = layout[modality]
    out = []
    for pair in pairs:
        pre = pair.pre.copy()
        post = pair.post.copy()
        pre[:, start:end] = 0.0
        post[:, start:end] = 0.0
        out.append(replace(pair, pre=pre, post=post))
    return out


def run(
    data_dir="data/tracking_bench_v2",
    expert_dir="checkpoints/category_pretrain_experts",
    projection_dir="checkpoints/phase45_projection",
    out_dir="eval/results/pss_vs_transformer_v2",
    seeds=5,
    epochs=200,
    device="cpu",
    transformer_json="eval/results/pss_vs_transformer_v2/p1_tracking.json",
):
    full_train = build_pairs(
        data_dir,
        expert_dir,
        projection_dir,
        "train",
        device,
        include_tactile=True,
        include_dino=True,
    )
    full_val = build_pairs(
        data_dir,
        expert_dir,
        projection_dir,
        "val",
        device,
        include_tactile=True,
        include_dino=True,
    )
    full_test = build_pairs(
        data_dir,
        expert_dir,
        projection_dir,
        "test",
        device,
        include_tactile=True,
        include_dino=True,
    )
    if full_train:
        assert full_train[0].pre.shape[1] == token_dim(True, True)
    train_variants = {
        name: select_pairs(full_train, t, d) for name, t, d in VARIANTS
    }
    val_variants = {name: select_pairs(full_val, t, d) for name, t, d in VARIANTS}
    test_variants = {name: select_pairs(full_test, t, d) for name, t, d in VARIANTS}

    results, best_models = {}, {}
    for name, include_tactile, include_dino in VARIANTS:
        dim = token_dim(include_tactile, include_dino)
        runs, best_val, best_model = [], -1.0, None
        for seed in range(seeds):
            model, info = train_graph(
                train_variants[name],
                val_variants[name],
                epochs=epochs,
                seed=seed,
                device=device,
                token_dim=dim,
            )
            runs.append(evaluate_graph(model, test_variants[name], device))
            if info["best_val_reid"] > best_val:
                best_val = info["best_val_reid"]
                best_model = model
        best_models[name] = best_model
        results[name] = {
            "token_dim": dim,
            "include_tactile": include_tactile,
            "include_dino": include_dino,
            "runs": runs,
            "aggregate": _agg(runs),
            "best_val_reid": best_val,
        }
        save_graph(best_model, Path(out_dir) / f"graph_matcher_{name}.pt")

    both_test = test_variants["both"]
    ablations = {
        "full": _agg([evaluate_graph(best_models["both"], both_test, device)]),
        "no_tactile": _agg(
            [evaluate_graph(best_models["both"], zero_modality(both_test, "tactile"), device)]
        ),
        "no_dino": _agg(
            [evaluate_graph(best_models["both"], zero_modality(both_test, "dino"), device)]
        ),
        "no_vision": _agg(
            [evaluate_graph(best_models["both"], zero_modality(both_test, "vision"), device)]
        ),
        "no_tactile_no_dino": _agg(
            [
                evaluate_graph(
                    best_models["both"],
                    zero_modality(zero_modality(both_test, "tactile"), "dino"),
                    device,
                )
            ]
        ),
    }

    transformer = None
    if Path(transformer_json).exists():
        payload = json.loads(Path(transformer_json).read_text(encoding="utf-8"))
        transformer = payload.get("transformer", {}).get("aggregate")

    base = results["base"]["aggregate"]
    metrics = {
        "data_dir": str(data_dir),
        "n_train": len(full_train),
        "n_val": len(full_val),
        "n_test": len(full_test),
        "variants": results,
        "ablation_both": ablations,
        "transformer": transformer,
        "delta_vs_base": {
            name: results[name]["aggregate"]["overall_mean"] - base["overall_mean"]
            for name in results
        },
        "delta_vs_transformer": {
            name: results[name]["aggregate"]["overall_mean"]
            - transformer["overall_mean"]
            if transformer
            else None
            for name in results
        },
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "p0_dino.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _report(out / "p0_dino.md", metrics)
    return metrics


def _report(path, metrics):
    lines = [
        "# P0-1: DINOv2 instance token",
        "",
        f"- data: `{metrics['data_dir']}`",
        f"- train / val / test: {metrics['n_train']} / {metrics['n_val']} / {metrics['n_test']}",
        "",
        "| variant | token dim | overall | mixed | samecat | near-identical | delta vs base |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, _t, _d in VARIANTS:
        a = metrics["variants"][name]["aggregate"]
        d = metrics["delta_vs_base"][name]
        lines.append(
            f"| {name} | {metrics['variants'][name]['token_dim']} | "
            f"{a['overall_mean']:.3f} ± {a['overall_std']:.3f} | "
            f"{a['per_subset']['mixed']['mean']:.3f} | "
            f"{a['per_subset']['samecat']['mean']:.3f} | "
            f"{a['per_subset']['near_identical']['mean']:.3f} | {d:+.3f} |"
        )
    if metrics["transformer"]:
        t = metrics["transformer"]
        lines.append(
            f"| Transformer | - | {t['overall_mean']:.3f} ± {t['overall_std']:.3f} | "
            f"{t['per_subset']['mixed']['mean']:.3f} | "
            f"{t['per_subset']['samecat']['mean']:.3f} | "
            f"{t['per_subset']['near_identical']['mean']:.3f} | - |"
        )
    lines += [
        "",
        "## Missing-modality ablation (both model)",
        "",
        "| condition | overall | mixed | samecat | near-identical |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, a in metrics["ablation_both"].items():
        lines.append(
            f"| {name} | {a['overall_mean']:.3f} | {a['per_subset']['mixed']['mean']:.3f} | "
            f"{a['per_subset']['samecat']['mean']:.3f} | "
            f"{a['per_subset']['near_identical']['mean']:.3f} |"
        )
    lines += [
        "",
        "All variants share vision/geometry/pose/category; only the identity blocks change.",
        "Ablation zeroes one block at test time on the best-val model trained with both blocks.",
        "The ablation 'full' row is that single checkpoint, so it can differ from the 5-seed mean above.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tracking_bench_v2")
    ap.add_argument("--experts", default="checkpoints/category_pretrain_experts")
    ap.add_argument("--projections", default="checkpoints/phase45_projection")
    ap.add_argument("--out", default="eval/results/pss_vs_transformer_v2")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    metrics = run(
        args.data,
        args.experts,
        args.projections,
        args.out,
        seeds=args.seeds,
        epochs=args.epochs,
        device=args.device,
    )
    for name, _t, _d in VARIANTS:
        a = metrics["variants"][name]["aggregate"]
        print(
            f"{name}: {a['overall_mean']:.3f} ± {a['overall_std']:.3f} "
            f"(near {a['per_subset']['near_identical']['mean']:.3f})"
        )
    print(f"output: {args.out}")


if __name__ == "__main__":
    main()
