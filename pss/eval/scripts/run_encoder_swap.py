"""Encoder swap experiment: frozen DINOv2 vision + stronger projections.

Stages:
- dino: train DINOv2 projection heads and rebuild prototypes/open-set;
- reid: train tactile graph matcher on DINO-projected perception;
- p5b: run the perceived relational benchmark with DINO perception.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pss._console import ensure_utf8
from pss.encoders.tactile import load_tactile_expert
from pss.encoders.vision.dinov2 import load_dinov2_encoder
from pss.eval.baselines.entity_transformer import build_pairs
from pss.memory.graph_matcher import evaluate_graph, save_graph, train_graph
from pss.memory.learned_metric import TACTILE_TOKEN_DIM
from pss.projection.train import train_projections
from pss.prototypes.build import run_phase5

SUBSETS = ("mixed", "samecat", "near_identical")


def _agg(runs):
    def ms(key, subset=None):
        vals = [r[key] if subset is None else r["per_subset"][subset]["reid_accuracy"] for r in runs]
        return float(np.mean(vals)), float(np.std(vals))

    m, s = ms("reid_accuracy")
    per = {}
    for subset in SUBSETS:
        a, b = ms("reid_accuracy", subset)
        per[subset] = {"mean": a, "std": b}
    return {"overall_mean": m, "overall_std": s, "per_subset": per}


def stage_dino(out_dir="eval/results/pss_vs_transformer"):
    dino = load_dinov2_encoder("cpu")
    projection_dir = Path("checkpoints/phase45_projection_dinov2")
    train_projections(
        "data/category_pretrain",
        "checkpoints/category_pretrain_experts",
        projection_dir,
        epochs=60,
        batch_size=64,
        seed=0,
        vision_expert=dino,
    )
    prototype_dir = Path("checkpoints/phase45_prototypes_dinov2")
    metrics = run_phase5(
        "data/category_pretrain",
        "checkpoints/category_pretrain_experts",
        projection_dir,
        prototype_dir,
        seed=0,
        vision_expert=dino,
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "encoder_swap_dino.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _report_dino(out / "encoder_swap_dino.md", metrics)
    return metrics


def stage_reid(out_dir="eval/results/pss_vs_transformer"):
    dino = load_dinov2_encoder("cpu")
    tactile = load_tactile_expert(
        "checkpoints/category_pretrain_experts/tactile.pt", device="cpu"
    )
    projection_dir = "checkpoints/phase45_projection_dinov2"
    train = build_pairs(
        "data/tracking_bench_v2",
        "checkpoints/category_pretrain_experts",
        projection_dir,
        "train",
        "cpu",
        include_tactile=True,
        vision_expert=dino,
        tactile_expert=tactile,
    )
    val = build_pairs(
        "data/tracking_bench_v2",
        "checkpoints/category_pretrain_experts",
        projection_dir,
        "val",
        "cpu",
        include_tactile=True,
        vision_expert=dino,
        tactile_expert=tactile,
    )
    test = build_pairs(
        "data/tracking_bench_v2",
        "checkpoints/category_pretrain_experts",
        projection_dir,
        "test",
        "cpu",
        include_tactile=True,
        vision_expert=dino,
        tactile_expert=tactile,
    )
    runs = []
    best_model, best_val = None, -1.0
    for seed in range(5):
        model, info = train_graph(
            train, val, epochs=200, seed=seed, device="cpu",
            token_dim=TACTILE_TOKEN_DIM, assignment_weight=1.0,
        )
        runs.append(evaluate_graph(model, test, device="cpu"))
        if info["best_val_reid"] > best_val:
            best_val = info["best_val_reid"]
            best_model = model
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_graph(best_model, out / "graph_matcher_dino_vision.pt")
    metrics = {
        "aggregate": _agg(runs),
        "runs": runs,
        "best_val_reid": best_val,
        "params": sum(p.numel() for p in best_model.parameters()),
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    (out / "encoder_swap_reid.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _report_reid(out / "encoder_swap_reid.md", metrics)
    return metrics


def _report_dino(path, metrics):
    fusion = metrics["fusion"]
    open_set = metrics["open_set"]
    lines = [
        "# Encoder swap: frozen DINOv2 vision",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| closed-set fusion val | {fusion['val_acc']:.3f} |",
        f"| closed-set fusion test | {fusion['test_acc']:.3f} |",
        f"| open-set AUROC | {open_set['auroc']:.3f} |",
        f"| unseen reject | {open_set['unseen_reject_rate']:.3f} |",
        f"| seen false reject | {open_set['seen_false_reject_rate']:.3f} |",
        "",
        "Vision expert: frozen DINOv2-small CLS; projection heads retrained on category_pretrain.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _report_reid(path, metrics):
    a = metrics["aggregate"]
    lines = [
        "# Encoder swap: DINOv2 vision + tactile re-ID",
        "",
        "| overall | mixed | samecat | near-identical | params |",
        "|---:|---:|---:|---:|---:|",
        f"| {a['overall_mean']:.3f} ± {a['overall_std']:.3f} | "
        f"{a['per_subset']['mixed']['mean']:.3f} | "
        f"{a['per_subset']['samecat']['mean']:.3f} | "
        f"{a['per_subset']['near_identical']['mean']:.3f} | {metrics['params']} |",
        "",
        "Reference: original SmallCNN + tactile graph = 0.989 / near 0.956.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("dino", "reid"), required=True)
    ap.add_argument("--out", default="eval/results/pss_vs_transformer")
    args = ap.parse_args()
    if args.stage == "dino":
        m = stage_dino(args.out)
        print(f"DINO fusion test {m['fusion']['test_acc']:.3f}; open-set AUROC {m['open_set']['auroc']:.3f}")
    else:
        m = stage_reid(args.out)
        a = m["aggregate"]
        print(f"DINO-vision re-ID {a['overall_mean']:.3f} ± {a['overall_std']:.3f}; near {a['per_subset']['near_identical']['mean']:.3f}")
    print(f"output: {args.out}")


if __name__ == "__main__":
    main()
