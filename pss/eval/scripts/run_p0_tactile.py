"""P0.3: tactile-augmented graph matcher for instance-level re-ID."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pss._console import ensure_utf8
from pss.eval.baselines.entity_transformer import build_pairs
from pss.memory.graph_matcher import GraphMatcher, evaluate_graph, save_graph, train_graph
from pss.memory.learned_metric import TACTILE_TOKEN_DIM

SUBSETS = ("mixed", "samecat", "near_identical")


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


def run(
    data_dir,
    expert_dir,
    projection_dir,
    out_dir,
    seeds=5,
    epochs=200,
    device="cpu",
    base_json="eval/results/pss_vs_transformer_v2/p0_graph_matcher.json",
    transformer_json="eval/results/pss_vs_transformer_v2/p1_tracking.json",
):
    train = build_pairs(data_dir, expert_dir, projection_dir, "train", device, include_tactile=True)
    val = build_pairs(data_dir, expert_dir, projection_dir, "val", device, include_tactile=True)
    test = build_pairs(data_dir, expert_dir, projection_dir, "test", device, include_tactile=True)

    runs, training = [], []
    best_model, best_val = None, -1.0
    for seed in range(seeds):
        model, info = train_graph(
            train,
            val,
            epochs=epochs,
            seed=seed,
            device=device,
            token_dim=TACTILE_TOKEN_DIM,
        )
        result = evaluate_graph(model, test, device)
        runs.append(result)
        training.append({"seed": seed, "best_val_reid": info["best_val_reid"]})
        if info["best_val_reid"] > best_val:
            best_val = info["best_val_reid"]
            best_model = model
    if best_model is not None:
        save_graph(best_model, Path(out_dir) / "graph_matcher_tactile.pt")
    agg = _agg(runs)
    base = json.loads(Path(base_json).read_text(encoding="utf-8"))["pss_graph"]["aggregate"] if Path(base_json).exists() else None
    transformer = json.loads(Path(transformer_json).read_text(encoding="utf-8"))["transformer"]["aggregate"] if Path(transformer_json).exists() else None
    metrics = {
        "data_dir": str(data_dir),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "token_dim": TACTILE_TOKEN_DIM,
        "tactile_graph": {
            "seeds": seeds,
            "runs": runs,
            "training": training,
            "aggregate": agg,
        },
        "base_graph": base,
        "transformer": transformer,
        "delta_vs_base": agg["overall_mean"] - base["overall_mean"] if base else None,
        "delta_vs_transformer": agg["overall_mean"] - transformer["overall_mean"] if transformer else None,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "p0_tactile.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _report(out / "p0_tactile.md", metrics)
    return metrics


def _report(path, metrics):
    g = metrics["tactile_graph"]["aggregate"]
    lines = [
        "# P0.3: tactile-augmented graph matcher",
        "",
        f"- data: `{metrics['data_dir']}`",
        f"- token dim: {metrics['token_dim']}",
        "",
        "| method | overall | mixed | samecat | near-identical |",
        "|---|---:|---:|---:|---:|",
        f"| PSS + graph + tactile | {g['overall_mean']:.3f} ± {g['overall_std']:.3f} | "
        f"{g['per_subset']['mixed']['mean']:.3f} | {g['per_subset']['samecat']['mean']:.3f} | "
        f"{g['per_subset']['near_identical']['mean']:.3f} |",
    ]
    if metrics["base_graph"]:
        b = metrics["base_graph"]
        lines.append(
            f"| PSS + graph (vision/geometry) | {b['overall_mean']:.3f} | "
            f"{b['per_subset']['mixed']['mean']:.3f} | {b['per_subset']['samecat']['mean']:.3f} | "
            f"{b['per_subset']['near_identical']['mean']:.3f} |"
        )
    if metrics["transformer"]:
        t = metrics["transformer"]
        lines.append(
            f"| Transformer | {t['overall_mean']:.3f} | {t['per_subset']['mixed']['mean']:.3f} | "
            f"{t['per_subset']['samecat']['mean']:.3f} | {t['per_subset']['near_identical']['mean']:.3f} |"
        )
    lines += [
        "",
        f"delta vs base graph: {metrics['delta_vs_base']:+.3f}" if metrics['delta_vs_base'] is not None else "delta vs base graph: n/a",
        f"delta vs transformer: {metrics['delta_vs_transformer']:+.3f}" if metrics['delta_vs_transformer'] is not None else "delta vs transformer: n/a",
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
    ap.add_argument("--base-json", default="eval/results/pss_vs_transformer_v2/p0_graph_matcher.json")
    ap.add_argument("--transformer-json", default="eval/results/pss_vs_transformer_v2/p1_tracking.json")
    args = ap.parse_args()
    metrics = run(
        args.data, args.experts, args.projections, args.out,
        seeds=args.seeds, epochs=args.epochs, device=args.device,
        base_json=args.base_json, transformer_json=args.transformer_json,
    )
    g = metrics["tactile_graph"]["aggregate"]
    print(
        f"P0.3 tactile graph: {g['overall_mean']:.3f} ± {g['overall_std']:.3f}; "
        f"near={g['per_subset']['near_identical']['mean']:.3f}; "
        f"delta vs base {metrics['delta_vs_base']:+.3f}" if metrics['delta_vs_base'] is not None else "delta vs base n/a"
    )
    print(f"output: {args.out}")


if __name__ == "__main__":
    main()
