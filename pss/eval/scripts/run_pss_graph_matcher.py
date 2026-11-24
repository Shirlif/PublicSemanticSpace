"""P0.1: PSS set-level graph matcher vs transformer on occlusion re-ID."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pss._console import ensure_utf8
from pss.eval.baselines.entity_transformer import build_pairs
from pss.memory.graph_matcher import evaluate_graph, save_graph, train_graph

SUBSETS = ("mixed", "samecat", "near_identical")


def _mean_std(values):
    return (
        float(np.mean(values)) if values else 0.0,
        float(np.std(values)) if values else 0.0,
    )


def _agg(runs):
    overall = [r["reid_accuracy"] for r in runs]
    mean, std = _mean_std(overall)
    per_subset = {}
    for subset in SUBSETS:
        vals = [r["per_subset"][subset]["reid_accuracy"] for r in runs]
        m, s = _mean_std(vals)
        per_subset[subset] = {"mean": m, "std": s}
    return {"overall_mean": mean, "overall_std": std, "per_subset": per_subset}


def run_graph(
    data_dir,
    expert_dir,
    projection_dir,
    out_dir,
    seeds=5,
    epochs=200,
    device="cpu",
    transformer_json="eval/results/pss_vs_transformer_v2/p1_tracking.json",
    metric_json="eval/results/pss_vs_transformer_v2/p1c_learned_metric.json",
):
    train = build_pairs(data_dir, expert_dir, projection_dir, "train", device)
    val = build_pairs(data_dir, expert_dir, projection_dir, "val", device)
    test = build_pairs(data_dir, expert_dir, projection_dir, "test", device)

    runs = []
    training = []
    best_model = None
    best_val = -1.0
    for seed in range(seeds):
        model, info = train_graph(train, val, epochs=epochs, seed=seed, device=device)
        result = evaluate_graph(model, test, device)
        runs.append(result)
        training.append({"seed": seed, "best_val_reid": info["best_val_reid"]})
        if info["best_val_reid"] > best_val:
            best_val = info["best_val_reid"]
            best_model = model
    if best_model is not None:
        save_graph(best_model, Path(out_dir) / "graph_matcher.pt")
    agg = _agg(runs)
    transformer = None
    metric = None
    tpath = Path(transformer_json)
    if tpath.exists():
        transformer = json.loads(tpath.read_text(encoding="utf-8"))["transformer"]["aggregate"]
    mpath = Path(metric_json)
    if mpath.exists():
        metric = json.loads(mpath.read_text(encoding="utf-8"))["pss_learned_metric"]["aggregate"]
    delta = agg["overall_mean"] - transformer["overall_mean"] if transformer else None
    metrics = {
        "data_dir": str(data_dir),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "pss_graph": {
            "seeds": seeds,
            "runs": runs,
            "training": training,
            "aggregate": agg,
            "params": sum(p.numel() for p in best_model.parameters()) if best_model is not None else 0,
        },
        "pss_metric_reference": metric,
        "transformer_reference": transformer,
        "h0": {
            "delta_vs_transformer": delta,
            "threshold": 0.0,
            "pass": bool(delta is not None and delta >= 0.0),
        },
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "p0_graph_matcher.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_report(out / "p0_graph_matcher.md", metrics)
    return metrics


def _write_report(path, metrics):
    g = metrics["pss_graph"]["aggregate"]
    tr = metrics["transformer_reference"]
    m = metrics["pss_metric_reference"]
    lines = [
        "# P0.1: PSS set-level graph matcher",
        "",
        f"- data: `{metrics['data_dir']}`",
        f"- train / val / test: {metrics['n_train']} / {metrics['n_val']} / {metrics['n_test']}",
        f"- graph params: {metrics['pss_graph']['params']}",
        "",
        "| method | overall | mixed | samecat | near-identical |",
        "|---|---:|---:|---:|---:|",
        f"| PSS + graph matcher | {g['overall_mean']:.3f} ± {g['overall_std']:.3f} | "
        f"{g['per_subset']['mixed']['mean']:.3f} | {g['per_subset']['samecat']['mean']:.3f} | "
        f"{g['per_subset']['near_identical']['mean']:.3f} |",
    ]
    if m:
        lines.append(
            f"| PSS + learned metric | {m['overall_mean']:.3f} | "
            f"{m['per_subset']['mixed']['mean']:.3f} | {m['per_subset']['samecat']['mean']:.3f} | "
            f"{m['per_subset']['near_identical']['mean']:.3f} |"
        )
    if tr:
        lines.append(
            f"| Transformer | {tr['overall_mean']:.3f} ± {tr['overall_std']:.3f} | "
            f"{tr['per_subset']['mixed']['mean']:.3f} | {tr['per_subset']['samecat']['mean']:.3f} | "
            f"{tr['per_subset']['near_identical']['mean']:.3f} |"
        )
    delta = metrics["h0"]["delta_vs_transformer"]
    delta_text = f"{delta:+.3f}" if delta is not None else "n/a"
    lines += [
        "",
        f"**H0**: graph - transformer = {delta_text} -> "
        f"**{'PASS' if metrics['h0']['pass'] else 'FAIL'}**",
        "",
        "Graph matcher uses within-set pose/category relations before scoring pairs.",
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
    ap.add_argument("--transformer-json", default="eval/results/pss_vs_transformer_v2/p1_tracking.json")
    ap.add_argument("--metric-json", default="eval/results/pss_vs_transformer_v2/p1c_learned_metric.json")
    args = ap.parse_args()
    metrics = run_graph(
        args.data,
        args.experts,
        args.projections,
        args.out,
        seeds=args.seeds,
        epochs=args.epochs,
        device=args.device,
        transformer_json=args.transformer_json,
        metric_json=args.metric_json,
    )
    g = metrics["pss_graph"]["aggregate"]
    delta = metrics["h0"]["delta_vs_transformer"]
    delta_text = f"{delta:+.3f}" if delta is not None else "n/a"
    print(
        f"P0.1 graph: {g['overall_mean']:.3f} ± {g['overall_std']:.3f}; "
        f"H0 delta {delta_text}"
    )
    print(f"output: {args.out}")


if __name__ == "__main__":
    main()
