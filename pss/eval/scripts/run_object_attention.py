"""Minimal object-centric attention matcher on tracking_bench_v2."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pss._console import ensure_utf8
from pss.eval.baselines.entity_transformer import build_pairs
from pss.memory.learned_metric import TACTILE_TOKEN_DIM
from pss.memory.object_attention import (
    ObjectAttentionMatcher,
    evaluate_object_attention,
    train_object_attention,
)

SUBSETS = ("mixed", "samecat", "near_identical")


def _mean_std(values):
    return float(np.mean(values)), float(np.std(values))


def _agg(runs):
    m, s = _mean_std([r["reid_accuracy"] for r in runs])
    per = {}
    for subset in SUBSETS:
        a, b = _mean_std([r["per_subset"][subset]["reid_accuracy"] for r in runs])
        per[subset] = {"mean": a, "std": b}
    return {"overall_mean": m, "overall_std": s, "per_subset": per}


def run(
    data_dir="data/tracking_bench_v2",
    out_dir="eval/results/pss_vs_transformer_v2",
    seeds=5,
    epochs=200,
    device="cpu",
):
    expert_dir = "checkpoints/category_pretrain_experts"
    projection_dir = "checkpoints/phase45_projection"
    train = build_pairs(
        data_dir, expert_dir, projection_dir, "train", device, include_tactile=True
    )
    val = build_pairs(
        data_dir, expert_dir, projection_dir, "val", device, include_tactile=True
    )
    test = build_pairs(
        data_dir, expert_dir, projection_dir, "test", device, include_tactile=True
    )
    runs = []
    best_model, best_val = None, -1.0
    for seed in range(seeds):
        model, info = train_object_attention(
            train,
            val,
            TACTILE_TOKEN_DIM,
            epochs=epochs,
            seed=seed,
            device=device,
        )
        runs.append(evaluate_object_attention(model, test, device))
        if info["best_val_reid"] > best_val:
            best_val = info["best_val_reid"]
            best_model = model
    metrics = {
        "data_dir": str(data_dir),
        "seeds": seeds,
        "aggregate": _agg(runs),
        "runs": runs,
        "params": sum(p.numel() for p in best_model.parameters()),
        "best_val_reid": best_val,
        "references": {
            "graph_tactile": 0.989,
            "sparse_k1": 0.996,
        },
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "object_attention.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _report(out / "object_attention.md", metrics)
    return metrics


def _report(path, metrics):
    a = metrics["aggregate"]
    lines = [
        "# Minimal object-centric attention matcher",
        "",
        f"- data: `{metrics['data_dir']}`",
        f"- params: {metrics['params']}, seeds: {metrics['seeds']}",
        "",
        "| method | overall | mixed | samecat | near-identical |",
        "|---|---:|---:|---:|---:|",
        f"| Object attention | {a['overall_mean']:.3f} ± {a['overall_std']:.3f} | "
        f"{a['per_subset']['mixed']['mean']:.3f} | "
        f"{a['per_subset']['samecat']['mean']:.3f} | "
        f"{a['per_subset']['near_identical']['mean']:.3f} |",
        f"| PSS graph+tactile (ref) | 0.989 | 1.000 | 1.000 | 0.956 |",
        f"| sparse k=1 transformer (ref) | 0.996 | 0.996 | 1.000 | 0.991 |",
        "",
        "Object attention uses typed spatial/semantic/tactile biases and active-set masks.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tracking_bench_v2")
    ap.add_argument("--out", default="eval/results/pss_vs_transformer_v2")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=200)
    args = ap.parse_args()
    m = run(args.data, args.out, args.seeds, args.epochs)
    a = m["aggregate"]
    print(f"object attention: {a['overall_mean']:.3f} ± {a['overall_std']:.3f}; near {a['per_subset']['near_identical']['mean']:.3f}")
    print(f"output: {args.out}")


if __name__ == "__main__":
    main()
