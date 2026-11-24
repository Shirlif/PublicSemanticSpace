"""P1: PSS vs entity-token transformer on occlusion re-ID.

Example:
    python -m pss.eval.scripts.run_pss_vs_transformer \
        --data data/tracking_bench_v1 \
        --experts checkpoints/category_pretrain_experts \
        --projections checkpoints/phase45_projection \
        --out eval/results/pss_vs_transformer --seeds 5
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
    evaluate_pss_tracker,
    evaluate_transformer,
    train_transformer,
)
from pss.memory.tracking import TrackerConfig

SUBSETS = ("mixed", "samecat", "near_identical")


def _mean_std(values: list[float]) -> tuple[float, float]:
    return float(np.mean(values)) if values else 0.0, float(np.std(values)) if values else 0.0


def _aggregate_seeds(results: list[dict]) -> dict:
    overall = [r["reid_accuracy"] for r in results]
    mean, std = _mean_std(overall)
    per_subset = {}
    for subset in SUBSETS:
        values = [r["per_subset"][subset]["reid_accuracy"] for r in results]
        m, s = _mean_std(values)
        per_subset[subset] = {"mean": m, "std": s}
    return {"overall_mean": mean, "overall_std": std, "per_subset": per_subset}


def run_p1(
    data_dir: str | Path,
    expert_dir: str | Path,
    projection_dir: str | Path,
    out_dir: str | Path,
    seeds: int = 5,
    epochs: int = 200,
    device: str = "cpu",
) -> dict:
    train = build_pairs(data_dir, expert_dir, projection_dir, "train", device)
    val = build_pairs(data_dir, expert_dir, projection_dir, "val", device)
    test = build_pairs(data_dir, expert_dir, projection_dir, "test", device)

    pss_grid = []
    best = None
    for gate in (0.3, 0.5, 0.7, 1.0):
        cfg = TrackerConfig(reid_pose_gate=gate)
        result = evaluate_pss_tracker(
            data_dir, expert_dir, projection_dir, "val", device, cfg
        )
        pss_grid.append({"gate": gate, "reid_accuracy": result["reid_accuracy"]})
        if best is None or result["reid_accuracy"] > best[1]["reid_accuracy"]:
            best = (gate, result)
    best_gate, _ = best
    pss_test = evaluate_pss_tracker(
        data_dir,
        expert_dir,
        projection_dir,
        "test",
        device,
        TrackerConfig(reid_pose_gate=best_gate),
    )

    transformer_runs = []
    training = []
    for seed in range(seeds):
        model, train_info = train_transformer(
            train, val, epochs=epochs, seed=seed, device=device
        )
        result = evaluate_transformer(model, test, device)
        transformer_runs.append(result)
        training.append(
            {
                "seed": seed,
                "best_val_reid": train_info["best_val_reid"],
                "final_epoch": train_info["history"][-1]["epoch"] if train_info["history"] else 0,
                "final_loss": train_info["history"][-1]["loss"] if train_info["history"] else 0.0,
            }
        )
    transformer_agg = _aggregate_seeds(transformer_runs)
    delta = pss_test["reid_accuracy"] - transformer_agg["overall_mean"]
    metrics = {
        "data_dir": str(data_dir),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "pss": {
            **pss_test,
            "val_gate_grid": pss_grid,
            "selected_gate": best_gate,
        },
        "transformer": {
            "seeds": seeds,
            "runs": transformer_runs,
            "training": training,
            "aggregate": transformer_agg,
        },
        "h1": {
            "metric": "reid_accuracy",
            "pss": pss_test["reid_accuracy"],
            "transformer_mean": transformer_agg["overall_mean"],
            "delta": delta,
            "threshold": 0.10,
            "pass": bool(delta >= 0.10),
        },
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "p1_tracking.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_report(out / "p1_tracking.md", metrics)
    return metrics


def _write_report(path: Path, metrics: dict) -> None:
    pss = metrics["pss"]
    tr = metrics["transformer"]["aggregate"]
    h1 = metrics["h1"]
    lines = [
        "# P1: PSS vs entity-token transformer",
        "",
        f"- data: `{metrics['data_dir']}`",
        f"- train / val / test sequences: {metrics['n_train']} / {metrics['n_val']} / {metrics['n_test']}",
        f"- PSS selected pose gate: {pss['selected_gate']}",
        "",
        "## Overall re-ID accuracy",
        "",
        "| method | re-ID accuracy |",
        "|---|---:|",
        f"| PSS | {pss['reid_accuracy']:.3f} |",
        f"| Transformer | {tr['overall_mean']:.3f} ± {tr['overall_std']:.3f} |",
        "",
        f"**H1**: delta = {h1['delta']:+.3f} (threshold {h1['threshold']:.2f}) -> "
        f"**{'PASS' if h1['pass'] else 'FAIL'}**",
        "",
        "## Per subset",
        "",
        "| subset | PSS | Transformer |",
        "|---|---:|---:|",
    ]
    for subset in SUBSETS:
        lines.append(
            f"| {subset} | {pss['per_subset'][subset]['reid_accuracy']:.3f} | "
            f"{tr['per_subset'][subset]['mean']:.3f} ± {tr['per_subset'][subset]['std']:.3f} |"
        )
    lines += [
        "",
        "## PSS pose-gate validation grid",
        "",
        "| gate | val re-ID |",
        "|---|---:|",
    ]
    for row in pss["val_gate_grid"]:
        lines.append(f"| {row['gate']} | {row['reid_accuracy']:.3f} |")
    lines += [
        "",
        "## Notes",
        "",
        "- L1 architecture-only: GT categories and frozen vision/geometry features; both methods get the same tokens.",
        "- P1 evaluates occlusion re-ID on the last pre-occlusion frame vs first post-occlusion frame.",
        "- Full online tracker ID retention and Transformer B are follow-ups.",
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
    args = ap.parse_args()
    metrics = run_p1(
        args.data,
        args.experts,
        args.projections,
        args.out,
        seeds=args.seeds,
        epochs=args.epochs,
        device=args.device,
    )
    print("P1 complete:")
    print(f"  PSS re-ID: {metrics['pss']['reid_accuracy']:.3f}")
    print(f"  Transformer: {metrics['transformer']['aggregate']['overall_mean']:.3f} "
          f"± {metrics['transformer']['aggregate']['overall_std']:.3f}")
    print(f"  H1 delta: {metrics['h1']['delta']:+.3f} "
          f"({'PASS' if metrics['h1']['pass'] else 'FAIL'})")
    print(f"  output: {args.out}")


if __name__ == "__main__":
    main()
