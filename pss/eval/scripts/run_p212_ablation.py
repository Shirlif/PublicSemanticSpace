"""P2-12: graph matcher ablations and paired bootstrap.

Ablations on tracking_bench_v2:
- full graph (edges + assignment-aware loss + Hungarian);
- no within-set edge messages;
- no assignment-aware loss;
- greedy matching instead of Hungarian;
- tactile and pairwise references from earlier runs.
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
from pss.memory.graph_matcher import evaluate_graph, train_graph
from pss.memory.learned_metric import TOKEN_DIM

SUBSETS = ("mixed", "samecat", "near_identical")


def _mean_std(values):
    return float(np.mean(values)), float(np.std(values))


def _agg(runs):
    overall = [r["reid_accuracy"] for r in runs]
    m, s = _mean_std(overall)
    per = {}
    for subset in SUBSETS:
        a, b = _mean_std([r["per_subset"][subset]["reid_accuracy"] for r in runs])
        per[subset] = {"mean": a, "std": b}
    return {"overall_mean": m, "overall_std": s, "per_subset": per}


def _rows(run):
    return [r["reid_accuracy"] for r in run["rows"]]


def _bootstrap_delta(pss_runs, tr_runs, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    deltas = []
    for a, b in zip(pss_runs, tr_runs):
        if len(a["rows"]) != len(b["rows"]):
            continue
        deltas.extend(
            [x["reid_accuracy"] - y["reid_accuracy"] for x, y in zip(a["rows"], b["rows"])]
        )
    if not deltas:
        return None
    deltas = np.asarray(deltas, dtype=float)
    stats = []
    for _ in range(n_boot):
        sample = rng.choice(deltas, size=len(deltas), replace=True)
        stats.append(float(np.mean(sample)))
    stats = np.asarray(stats)
    return {
        "mean_delta": float(np.mean(deltas)),
        "ci95_low": float(np.percentile(stats, 2.5)),
        "ci95_high": float(np.percentile(stats, 97.5)),
        "p_delta_le_0": float(np.mean(stats <= 0.0)),
        "n_pairs": int(len(deltas)),
    }


def run(
    data_dir="data/tracking_bench_v2",
    out_dir="eval/results/pss_vs_transformer_v2",
    seeds=5,
    epochs=200,
    device="cpu",
):
    train = build_pairs(data_dir, "checkpoints/category_pretrain_experts",
                        "checkpoints/phase45_projection", "train", device)
    val = build_pairs(data_dir, "checkpoints/category_pretrain_experts",
                      "checkpoints/phase45_projection", "val", device)
    test = build_pairs(data_dir, "checkpoints/category_pretrain_experts",
                       "checkpoints/phase45_projection", "test", device)

    variants = {}
    full_runs = None
    configs = {
        "full": {"use_edges": True, "assignment_weight": 1.0},
        "no_edges": {"use_edges": False, "assignment_weight": 1.0},
        "no_assignment": {"use_edges": True, "assignment_weight": 0.0},
    }
    for name, cfg in configs.items():
        runs = []
        for seed in range(seeds):
            model, _ = train_graph(
                train,
                val,
                epochs=epochs,
                seed=seed,
                device=device,
                token_dim=TOKEN_DIM,
                **cfg,
            )
            runs.append(evaluate_graph(model, test, device))
        variants[name] = {"config": cfg, "runs": runs, "aggregate": _agg(runs)}
        if name == "full":
            full_runs = runs

    # Greedy matching on the full model's seed-0 run.
    model0, _ = train_graph(
        train, val, epochs=epochs, seed=0, device=device, token_dim=TOKEN_DIM
    )
    greedy = evaluate_graph(model0, test, device, matching="greedy")
    variants["greedy"] = {
        "config": {"matching": "greedy"},
        "runs": [greedy],
        "aggregate": _agg([greedy]),
    }

    tr_runs = []
    for seed in range(seeds):
        model, _ = train_transformer(train, val, epochs=epochs, seed=seed, device=device)
        tr_runs.append(evaluate_transformer(model, test, device))
    transformer = {"runs": tr_runs, "aggregate": _agg(tr_runs)}

    bootstrap = _bootstrap_delta(full_runs, tr_runs)
    tactile = None
    tpath = Path(out_dir) / "p0_tactile.json"
    if tpath.exists():
        payload = json.loads(tpath.read_text(encoding="utf-8"))
        tactile = payload["tactile_graph"]["aggregate"]

    metrics = {
        "data_dir": str(data_dir),
        "n_test": len(test),
        "variants": variants,
        "transformer": transformer,
        "bootstrap_full_vs_transformer": bootstrap,
        "tactile_reference": tactile,
        "params": {
            "graph": int(sum(p.numel() for p in model0.parameters())),
            "transformer": int(sum(p.numel() for p in model.parameters())),
        },
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "p212_ablation.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _report(out / "p212_ablation.md", metrics)
    return metrics


def _report(path, metrics):
    lines = [
        "# P2-12: graph matcher ablations and paired bootstrap",
        "",
        f"- data: `{metrics['data_dir']}`, test pairs: {metrics['n_test']}",
        f"- params: graph {metrics['params']['graph']}, transformer {metrics['params']['transformer']}",
        "",
        "| variant | overall | mixed | samecat | near-identical |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("full", "no_edges", "no_assignment", "greedy"):
        a = metrics["variants"][name]["aggregate"]
        lines.append(
            f"| {name} | {a['overall_mean']:.3f} ± {a['overall_std']:.3f} | "
            f"{a['per_subset']['mixed']['mean']:.3f} | "
            f"{a['per_subset']['samecat']['mean']:.3f} | "
            f"{a['per_subset']['near_identical']['mean']:.3f} |"
        )
    t = metrics["transformer"]["aggregate"]
    lines.append(
        f"| transformer | {t['overall_mean']:.3f} ± {t['overall_std']:.3f} | "
        f"{t['per_subset']['mixed']['mean']:.3f} | "
        f"{t['per_subset']['samecat']['mean']:.3f} | "
        f"{t['per_subset']['near_identical']['mean']:.3f} |"
    )
    if metrics["tactile_reference"]:
        a = metrics["tactile_reference"]
        lines.append(
            f"| + tactile (ref) | {a['overall_mean']:.3f} ± {a['overall_std']:.3f} | "
            f"{a['per_subset']['mixed']['mean']:.3f} | "
            f"{a['per_subset']['samecat']['mean']:.3f} | "
            f"{a['per_subset']['near_identical']['mean']:.3f} |"
        )
    b = metrics["bootstrap_full_vs_transformer"]
    lines += [
        "",
        "## Paired bootstrap: full graph - transformer",
        "",
        f"- mean delta: {b['mean_delta']:+.3f}" if b else "- no bootstrap",
        f"- 95% CI: [{b['ci95_low']:+.3f}, {b['ci95_high']:+.3f}]" if b else "",
        f"- P(delta <= 0): {b['p_delta_le_0']:.4f}" if b else "",
        f"- paired observations: {b['n_pairs']}" if b else "",
        "",
        "Greedy uses the full seed-0 model and replaces Hungarian with score-greedy assignment.",
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
    for name in ("full", "no_edges", "no_assignment", "greedy"):
        a = m["variants"][name]["aggregate"]
        print(f"{name}: {a['overall_mean']:.3f} ± {a['overall_std']:.3f}")
    print(f"bootstrap: {m['bootstrap_full_vs_transformer']}")
    print(f"output: {args.out}")


if __name__ == "__main__":
    main()
