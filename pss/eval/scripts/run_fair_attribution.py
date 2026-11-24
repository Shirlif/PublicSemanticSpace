"""Fair attribution: give the transformer the same tactile token and Hungarian.

Compares four cells:
- base tokens: PSS graph vs entity transformer;
- tactile tokens: PSS graph vs entity transformer.

Both methods see the same tokens and use Hungarian assignment at evaluation.
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
from pss.memory.graph_matcher import GraphMatcher, evaluate_graph, train_graph
from pss.memory.learned_metric import TACTILE_TOKEN_DIM, TOKEN_DIM

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


def _run_cell(train, val, test, token_dim, seeds, epochs, device):
    graph_runs, tr_runs = [], []
    for seed in range(seeds):
        graph, _ = train_graph(
            train, val, epochs=epochs, seed=seed, device=device, token_dim=token_dim
        )
        graph_runs.append(evaluate_graph(graph, test, device))
        tr, _ = train_transformer(
            train, val, epochs=epochs, seed=seed, device=device, token_dim=token_dim
        )
        tr_runs.append(evaluate_transformer(tr, test, device))
    return {
        "token_dim": token_dim,
        "pss": _agg(graph_runs),
        "transformer": _agg(tr_runs),
        "delta": _agg(graph_runs)["overall_mean"] - _agg(tr_runs)["overall_mean"],
    }


def run(
    data_dir="data/tracking_bench_v2",
    out_dir="eval/results/pss_vs_transformer_v2",
    seeds=5,
    epochs=200,
    device="cpu",
):
    expert_dir = "checkpoints/category_pretrain_experts"
    projection_dir = "checkpoints/phase45_projection"
    base_train = build_pairs(data_dir, expert_dir, projection_dir, "train", device)
    base_val = build_pairs(data_dir, expert_dir, projection_dir, "val", device)
    base_test = build_pairs(data_dir, expert_dir, projection_dir, "test", device)
    tac_train = build_pairs(
        data_dir, expert_dir, projection_dir, "train", device, include_tactile=True
    )
    tac_val = build_pairs(
        data_dir, expert_dir, projection_dir, "val", device, include_tactile=True
    )
    tac_test = build_pairs(
        data_dir, expert_dir, projection_dir, "test", device, include_tactile=True
    )

    cells = {
        "base": _run_cell(base_train, base_val, base_test, TOKEN_DIM, seeds, epochs, device),
        "tactile": _run_cell(
            tac_train, tac_val, tac_test, TACTILE_TOKEN_DIM, seeds, epochs, device
        ),
    }
    metrics = {
        "data_dir": str(data_dir),
        "seeds": seeds,
        "cells": cells,
        "params": {
            "graph_base": int(sum(p.numel() for p in GraphMatcher(token_dim=TOKEN_DIM).parameters())),
            "graph_tactile": int(sum(p.numel() for p in GraphMatcher(token_dim=TACTILE_TOKEN_DIM).parameters())),
            "transformer_base": int(sum(p.numel() for p in EntityTransformer(token_dim=TOKEN_DIM).parameters())),
            "transformer_tactile": int(sum(p.numel() for p in EntityTransformer(token_dim=TACTILE_TOKEN_DIM).parameters())),
        },
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "fair_attribution.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _report(out / "fair_attribution.md", metrics)
    return metrics


def _report(path, metrics):
    lines = [
        "# Fair attribution: same tactile token, same Hungarian",
        "",
        "| token set | PSS graph | transformer | delta |",
        "|---|---:|---:|---:|",
    ]
    for name, c in metrics["cells"].items():
        p, t = c["pss"], c["transformer"]
        lines.append(
            f"| {name} | {p['overall_mean']:.3f} ± {p['overall_std']:.3f} | "
            f"{t['overall_mean']:.3f} ± {t['overall_std']:.3f} | "
            f"{c['delta']:+.3f} |"
        )
    lines += [
        "",
        "Both methods receive identical tokens and use Hungarian assignment at evaluation.",
        "If the tactile transformer closes the gap, the advantage is input-level; if not, it is architectural.",
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
    for name, c in m["cells"].items():
        print(
            f"{name}: PSS {c['pss']['overall_mean']:.3f} vs TR {c['transformer']['overall_mean']:.3f} "
            f"(delta {c['delta']:+.3f})"
        )
    print(f"output: {args.out}")


if __name__ == "__main__":
    main()
