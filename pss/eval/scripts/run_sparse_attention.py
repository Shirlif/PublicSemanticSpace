"""Sparse/retrieval attention transformer baselines.

The transformer attends only to each token's top-k nearest pose tokens. This
is a sparse-attention baseline, not a compute optimization: the mask is still
materialized densely, so latency is reported only for accuracy context.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pss._console import ensure_utf8
from pss.eval.baselines.entity_transformer import (
    build_pairs,
    evaluate_transformer,
    train_transformer,
)
from pss.memory.learned_metric import TACTILE_TOKEN_DIM

SUBSETS = ("mixed", "samecat", "near_identical")
VARIANTS = (("full", None), ("sparse_k1", 1), ("sparse_k2", 2), ("sparse_k4", 4))


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
    tactile=True,
):
    expert_dir = "checkpoints/category_pretrain_experts"
    projection_dir = "checkpoints/phase45_projection"
    train = build_pairs(
        data_dir, expert_dir, projection_dir, "train", device, include_tactile=tactile
    )
    val = build_pairs(
        data_dir, expert_dir, projection_dir, "val", device, include_tactile=tactile
    )
    test = build_pairs(
        data_dir, expert_dir, projection_dir, "test", device, include_tactile=tactile
    )
    token_dim = TACTILE_TOKEN_DIM if tactile else 279
    results = {}
    for name, topk in VARIANTS:
        runs = []
        for seed in range(seeds):
            model, _ = train_transformer(
                train,
                val,
                epochs=epochs,
                seed=seed,
                device=device,
                token_dim=token_dim,
                sparse_topk=topk,
            )
            runs.append(evaluate_transformer(model, test, device))
        results[name] = {"sparse_topk": topk, "runs": runs, "aggregate": _agg(runs)}
    metrics = {
        "data_dir": str(data_dir),
        "tactile": bool(tactile),
        "seeds": seeds,
        "variants": results,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sparse_attention.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _report(out / "sparse_attention.md", metrics)
    return metrics


def _report(path, metrics):
    lines = [
        "# Sparse/retrieval attention transformer",
        "",
        f"- data: `{metrics['data_dir']}`, tactile: {metrics['tactile']}",
        "",
        "| variant | top-k | overall | mixed | samecat | near-identical |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, r in metrics["variants"].items():
        a = r["aggregate"]
        lines.append(
            f"| {name} | {r['sparse_topk'] or '-'} | "
            f"{a['overall_mean']:.3f} ± {a['overall_std']:.3f} | "
            f"{a['per_subset']['mixed']['mean']:.3f} | "
            f"{a['per_subset']['samecat']['mean']:.3f} | "
            f"{a['per_subset']['near_identical']['mean']:.3f} |"
        )
    lines += [
        "",
        "Sparse attention keeps each token's top-k nearest pose tokens; the mask is dense, so this is not a latency optimization.",
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
    for name, r in m["variants"].items():
        a = r["aggregate"]
        print(f"{name}: {a['overall_mean']:.3f} ± {a['overall_std']:.3f}")
    print(f"output: {args.out}")


if __name__ == "__main__":
    main()
