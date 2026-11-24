"""P0-2: cross-benchmark re-ID on tracking_bench_v2 and a harder v3.

v3 adds more objects, 40-frame sequences, a 30-frame occlusion, and larger
renders. The key test is training on v2 and evaluating on v3 without seeing
v3 labels, plus training on v3 to measure the in-domain ceiling.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
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
from pss.memory.learned_metric import token_layout

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


def _zero(pairs, block):
    layout = token_layout(False, False)
    start, end = layout[block]
    out = []
    for pair in pairs:
        pre = pair.pre.copy()
        post = pair.post.copy()
        pre[:, start:end] = 0.0
        post[:, start:end] = 0.0
        out.append(replace(pair, pre=pre, post=post))
    return out


def _train_and_eval_graph(train_dir, val_dir, test_pairs, seeds, epochs, device):
    train = build_pairs(train_dir, "checkpoints/category_pretrain_experts",
                        "checkpoints/phase45_projection", "train", device)
    val = build_pairs(val_dir, "checkpoints/category_pretrain_experts",
                      "checkpoints/phase45_projection", "val", device)
    runs = []
    for seed in range(seeds):
        model, _ = train_graph(train, val, epochs=epochs, seed=seed, device=device)
        runs.append(evaluate_graph(model, test_pairs, device))
    return _agg(runs)


def _train_and_eval_transformer(train_dir, val_dir, test_pairs, seeds, epochs, device):
    train = build_pairs(train_dir, "checkpoints/category_pretrain_experts",
                        "checkpoints/phase45_projection", "train", device)
    val = build_pairs(val_dir, "checkpoints/category_pretrain_experts",
                      "checkpoints/phase45_projection", "val", device)
    runs = []
    for seed in range(seeds):
        model, _ = train_transformer(train, val, epochs=epochs, seed=seed, device=device)
        runs.append(evaluate_transformer(model, test_pairs, device))
    return _agg(runs)


def run(
    v2_dir="data/tracking_bench_v2",
    v3_dir="data/tracking_bench_v3",
    out_dir="eval/results/pss_vs_transformer_v2",
    seeds=5,
    epochs=200,
    device="cpu",
):
    expert_dir = "checkpoints/category_pretrain_experts"
    projection_dir = "checkpoints/phase45_projection"
    v2_test = build_pairs(v2_dir, expert_dir, projection_dir, "test", device)
    v3_test = build_pairs(v3_dir, expert_dir, projection_dir, "test", device)

    results = {}
    for name, train_dir, val_dir, test_sets in (
        ("v2_to_v2", v2_dir, v2_dir, {"v2": v2_test}),
        ("v2_to_v3", v2_dir, v2_dir, {"v3": v3_test}),
        ("v3_to_v3", v3_dir, v3_dir, {"v3": v3_test}),
    ):
        for test_name, test_pairs in test_sets.items():
            graph = _train_and_eval_graph(
                train_dir, val_dir, test_pairs, seeds, epochs, device
            )
            tr = _train_and_eval_transformer(
                train_dir, val_dir, test_pairs, seeds, epochs, device
            )
            results[f"{name}:{test_name}"] = {"pss_graph": graph, "transformer": tr}

    # Reuse the v2-trained PSS model? Retraining is cheap enough here; instead
    # measure modality loss on the v3-trained model, which is the cleanest
    # in-domain test of missing sensors.
    modality = {}
    v3_train = build_pairs(v3_dir, expert_dir, projection_dir, "train", device)
    v3_val = build_pairs(v3_dir, expert_dir, projection_dir, "val", device)
    graph_model, _ = train_graph(v3_train, v3_val, epochs=epochs, seed=0, device=device)
    for block in ("vision", "geometry"):
        subset = _zero(v3_test, block)
        modality[f"no_{block}"] = _agg(
            [evaluate_graph(graph_model, subset, device)]
        )
    modality["full"] = _agg([evaluate_graph(graph_model, v3_test, device)])

    metrics = {
        "v2_dir": str(v2_dir),
        "v3_dir": str(v3_dir),
        "n_v2_test": len(v2_test),
        "n_v3_test": len(v3_test),
        "n_objects_v2": int(v2_test[0].labels.shape[0]) if v2_test else 0,
        "n_objects_v3": int(v3_test[0].labels.shape[0]) if v3_test else 0,
        "results": results,
        "v3_modality_drop": modality,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "p02_cross_bench.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _report(out / "p02_cross_bench.md", metrics)
    return metrics


def _report(path, metrics):
    lines = [
        "# P0-2: cross-benchmark re-ID (v2 -> v3)",
        "",
        f"- v2 test: {metrics['n_v2_test']} sequences, {metrics['n_objects_v2']} objects",
        f"- v3 test: {metrics['n_v3_test']} sequences, {metrics['n_objects_v3']} objects",
        "",
        "| train -> test | PSS graph | transformer | delta |",
        "|---|---:|---:|---:|",
    ]
    for key, value in metrics["results"].items():
        p = value["pss_graph"]
        t = value["transformer"]
        lines.append(
            f"| {key} | {p['overall_mean']:.3f} ± {p['overall_std']:.3f} | "
            f"{t['overall_mean']:.3f} ± {t['overall_std']:.3f} | "
            f"{p['overall_mean'] - t['overall_mean']:+.3f} |"
        )
    lines += [
        "",
        "## v3 missing-modality (PSS graph trained on v3, seed 0)",
        "",
        "| condition | overall | mixed | samecat | near-identical |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, a in metrics["v3_modality_drop"].items():
        lines.append(
            f"| {name} | {a['overall_mean']:.3f} | {a['per_subset']['mixed']['mean']:.3f} | "
            f"{a['per_subset']['samecat']['mean']:.3f} | "
            f"{a['per_subset']['near_identical']['mean']:.3f} |"
        )
    lines += [
        "",
        "v3 uses 5 objects, 40 frames, 30-frame occlusion, 160x160 renders, random camera/light.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", default="data/tracking_bench_v2")
    ap.add_argument("--v3", default="data/tracking_bench_v3")
    ap.add_argument("--out", default="eval/results/pss_vs_transformer_v2")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    m = run(args.v2, args.v3, args.out, args.seeds, args.epochs, args.device)
    for key, value in m["results"].items():
        p, t = value["pss_graph"], value["transformer"]
        print(f"{key}: PSS {p['overall_mean']:.3f} vs TR {t['overall_mean']:.3f}")
    print(f"output: {args.out}")


if __name__ == "__main__":
    main()
