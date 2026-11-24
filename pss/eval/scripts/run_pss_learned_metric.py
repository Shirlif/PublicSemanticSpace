"""P1c: PSS with a learned re-ID metric vs the entity-token transformer.

This keeps the scene graph and only replaces the hand-designed association
cost with a Siamese metric embedding.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pss._console import ensure_utf8
from pss.eval.baselines.entity_transformer import build_pairs
from pss.eval.baselines.entity_transformer import evaluate_pss_tracker
from pss.eval.baselines.entity_transformer import EntityTransformer
from pss.memory.learned_metric import (
    evaluate_metric,
    save_metric,
    train_metric,
)

SUBSETS = ("mixed", "samecat", "near_identical")


def _mean_std(values):
    return (
        float(np.mean(values)) if values else 0.0,
        float(np.std(values)) if values else 0.0,
    )


def _aggregate(runs):
    overall = [r["reid_accuracy"] for r in runs]
    mean, std = _mean_std(overall)
    per_subset = {}
    for subset in SUBSETS:
        values = [r["per_subset"][subset]["reid_accuracy"] for r in runs]
        m, s = _mean_std(values)
        per_subset[subset] = {"mean": m, "std": s}
    return {"overall_mean": mean, "overall_std": std, "per_subset": per_subset}


def run_p1c(
    data_dir: str | Path,
    expert_dir: str | Path,
    projection_dir: str | Path,
    out_dir: str | Path,
    seeds: int = 5,
    epochs: int = 200,
    device: str = "cpu",
    transformer_json: str | Path = "eval/results/pss_vs_transformer/p1_tracking.json",
    metric_hidden: int = 410,
    assignment_weight: float = 0.0,
) -> dict:
    train = build_pairs(data_dir, expert_dir, projection_dir, "train", device)
    val = build_pairs(data_dir, expert_dir, projection_dir, "val", device)
    test = build_pairs(data_dir, expert_dir, projection_dir, "test", device)

    runs = []
    training = []
    best_model = None
    best_val = -1.0
    for seed in range(seeds):
        model, info = train_metric(
            train,
            val,
            epochs=epochs,
            seed=seed,
            device=device,
            hidden=metric_hidden,
            assignment_weight=assignment_weight,
        )
        result = evaluate_metric(model, test, device)
        runs.append(result)
        training.append(
            {
                "seed": seed,
                "best_val_reid": info["best_val_reid"],
                "final_loss": info["history"][-1]["loss"] if info["history"] else 0.0,
            }
        )
        if info["best_val_reid"] > best_val:
            best_val = info["best_val_reid"]
            best_model = model
    if best_model is not None:
        save_metric(best_model, Path(out_dir) / "learned_metric.pt")

    hand_pss = evaluate_pss_tracker(
        data_dir, expert_dir, projection_dir, "test", device
    )
    transformer_path = Path(transformer_json)
    transformer = None
    if transformer_path.exists():
        transformer = json.loads(transformer_path.read_text(encoding="utf-8"))[
            "transformer"
        ]["aggregate"]

    metric_agg = _aggregate(runs)
    transformer_params = sum(p.numel() for p in EntityTransformer().parameters())
    metric_params = (
        sum(p.numel() for p in best_model.parameters()) if best_model is not None else 0
    )
    delta = (
        metric_agg["overall_mean"] - transformer["overall_mean"]
        if transformer is not None
        else None
    )
    metrics = {
        "data_dir": str(data_dir),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "hand_pss": hand_pss,
        "pss_learned_metric": {
            "seeds": seeds,
            "hidden": metric_hidden,
            "assignment_weight": assignment_weight,
            "params": metric_params,
            "transformer_params": transformer_params,
            "runs": runs,
            "training": training,
            "aggregate": metric_agg,
        },
        "transformer_reference": transformer,
        "h1c": {
            "metric": "reid_accuracy",
            "pss_learned": metric_agg["overall_mean"],
            "transformer_mean": (
                transformer["overall_mean"] if transformer is not None else None
            ),
            "delta_vs_transformer": delta,
            "threshold": 0.0,
            "pass": bool(delta is not None and delta >= 0.0),
        },
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "p1c_learned_metric.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_report(out / "p1c_learned_metric.md", metrics)
    return metrics


def _write_report(path: Path, metrics: dict) -> None:
    hand = metrics["hand_pss"]
    learned = metrics["pss_learned_metric"]["aggregate"]
    tr = metrics["transformer_reference"]
    h1c = metrics["h1c"]
    lines = [
        "# P1c: PSS with learned re-ID metric",
        "",
        f"- data: `{metrics['data_dir']}`",
        f"- train / val / test: {metrics['n_train']} / {metrics['n_val']} / {metrics['n_test']}",
        f"- metric params: {metrics['pss_learned_metric']['params']} "
        f"(hidden={metrics['pss_learned_metric']['hidden']}); "
        f"transformer params: {metrics['pss_learned_metric']['transformer_params']}",
        "",
        "| method | overall re-ID | mixed | samecat | near-identical |",
        "|---|---:|---:|---:|---:|",
        f"| PSS hand cost | {hand['reid_accuracy']:.3f} | "
        f"{hand['per_subset']['mixed']['reid_accuracy']:.3f} | "
        f"{hand['per_subset']['samecat']['reid_accuracy']:.3f} | "
        f"{hand['per_subset']['near_identical']['reid_accuracy']:.3f} |",
        f"| **PSS + learned metric** | **{learned['overall_mean']:.3f} ± "
        f"{learned['overall_std']:.3f}** | "
        f"{learned['per_subset']['mixed']['mean']:.3f} | "
        f"{learned['per_subset']['samecat']['mean']:.3f} | "
        f"{learned['per_subset']['near_identical']['mean']:.3f} |",
    ]
    if tr is not None:
        lines.append(
            f"| Transformer reference | {tr['overall_mean']:.3f} ± "
            f"{tr['overall_std']:.3f} | "
            f"{tr['per_subset']['mixed']['mean']:.3f} | "
            f"{tr['per_subset']['samecat']['mean']:.3f} | "
            f"{tr['per_subset']['near_identical']['mean']:.3f} |"
        )
    lines += [
        "",
        f"**H1c** (learned metric vs transformer): delta = "
        f"{h1c['delta_vs_transformer']:+.3f} -> "
        f"**{'PASS' if h1c['pass'] else 'FAIL'}**",
        "",
        "PSS keeps the scene graph; only the hand-designed re-ID cost is replaced by a learned Siamese embedding.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tracking_bench_v1")
    ap.add_argument("--experts", default="checkpoints/category_pretrain_experts")
    ap.add_argument("--projections", default="checkpoints/phase45_projection")
    ap.add_argument("--out", default="eval/results/pss_vs_transformer")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--transformer-json", default="eval/results/pss_vs_transformer/p1_tracking.json")
    ap.add_argument("--metric-hidden", type=int, default=410)
    ap.add_argument("--assignment-weight", type=float, default=0.0)
    args = ap.parse_args()
    metrics = run_p1c(
        args.data,
        args.experts,
        args.projections,
        args.out,
        seeds=args.seeds,
        epochs=args.epochs,
        device=args.device,
        transformer_json=args.transformer_json,
        metric_hidden=args.metric_hidden,
        assignment_weight=args.assignment_weight,
    )
    learned = metrics["pss_learned_metric"]["aggregate"]
    print("P1c complete:")
    print(f"  PSS + learned metric: {learned['overall_mean']:.3f} ± {learned['overall_std']:.3f}")
    if metrics["transformer_reference"] is not None:
        print(f"  Transformer: {metrics['transformer_reference']['overall_mean']:.3f}")
    print(f"  H1c delta: {metrics['h1c']['delta_vs_transformer']:+.3f} "
          f"({'PASS' if metrics['h1c']['pass'] else 'FAIL'})")
    print(f"  output: {args.out}")


if __name__ == "__main__":
    main()
