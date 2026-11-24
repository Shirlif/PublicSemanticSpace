"""P0-4: stronger transformer re-ID baselines.

Compares:
- base entity-token transformer (vision+geometry+pose+category);
- frozen DINOv2 CLS + the same transformer head.

Both receive the same PSS entity state; only the visual token changes.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pss._console import ensure_utf8
from pss.eval.baselines.entity_transformer import (
    EntityTransformer,
    build_pairs,
    evaluate_transformer,
    train_transformer,
)
from pss.memory.learned_metric import DINO_TOKEN_DIM, TOKEN_DIM

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


def _run_variant(train, val, test, token_dim, seeds, epochs, device):
    runs = []
    for seed in range(seeds):
        model, _ = train_transformer(
            train,
            val,
            epochs=epochs,
            seed=seed,
            device=device,
            token_dim=token_dim,
        )
        runs.append(evaluate_transformer(model, test, device))
    return runs


def run(
    data_dir="data/tracking_bench_v2",
    out_dir="eval/results/pss_vs_transformer_v2",
    seeds=5,
    epochs=200,
    device="cpu",
):
    expert_dir = "checkpoints/category_pretrain_experts"
    projection_dir = "checkpoints/phase45_projection"
    train = build_pairs(data_dir, expert_dir, projection_dir, "train", device)
    val = build_pairs(data_dir, expert_dir, projection_dir, "val", device)
    test = build_pairs(data_dir, expert_dir, projection_dir, "test", device)

    base_runs = _run_variant(train, val, test, TOKEN_DIM, seeds, epochs, device)
    dino_train = build_pairs(
        data_dir, expert_dir, projection_dir, "train", device, include_dino=True
    )
    dino_val = build_pairs(
        data_dir, expert_dir, projection_dir, "val", device, include_dino=True
    )
    dino_test = build_pairs(
        data_dir, expert_dir, projection_dir, "test", device, include_dino=True
    )
    dino_runs = _run_variant(
        dino_train, dino_val, dino_test, DINO_TOKEN_DIM, seeds, epochs, device
    )

    metrics = {
        "data_dir": str(data_dir),
        "n_test": len(test),
        "base": {"token_dim": TOKEN_DIM, "runs": base_runs, "aggregate": _agg(base_runs)},
        "dino": {
            "token_dim": DINO_TOKEN_DIM,
            "runs": dino_runs,
            "aggregate": _agg(dino_runs),
        },
        "params": {
            "base": int(sum(p.numel() for p in EntityTransformer(token_dim=TOKEN_DIM).parameters())),
            "dino": int(sum(p.numel() for p in EntityTransformer(token_dim=DINO_TOKEN_DIM).parameters())),
        },
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "p04_strong_baselines.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _report(out / "p04_strong_baselines.md", metrics)
    return metrics


def _report(path, metrics):
    lines = [
        "# P0-4: stronger transformer baselines",
        "",
        f"- data: `{metrics['data_dir']}`, test pairs: {metrics['n_test']}",
        "",
        "| baseline | token dim | params | overall | mixed | samecat | near-identical |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("base", "dino"):
        a = metrics[name]["aggregate"]
        lines.append(
            f"| {name} | {metrics[name]['token_dim']} | {metrics['params'][name]} | "
            f"{a['overall_mean']:.3f} ± {a['overall_std']:.3f} | "
            f"{a['per_subset']['mixed']['mean']:.3f} | "
            f"{a['per_subset']['samecat']['mean']:.3f} | "
            f"{a['per_subset']['near_identical']['mean']:.3f} |"
        )
    lines += [
        "",
        "DINO baseline uses frozen DINOv2-small CLS (384-d) as an extra visual token;",
        "the transformer head and training budget are the same as the base baseline.",
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
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    m = run(args.data, args.out, args.seeds, args.epochs, args.device)
    for name in ("base", "dino"):
        a = m[name]["aggregate"]
        print(f"{name}: {a['overall_mean']:.3f} ± {a['overall_std']:.3f}")
    print(f"output: {args.out}")


if __name__ == "__main__":
    main()
