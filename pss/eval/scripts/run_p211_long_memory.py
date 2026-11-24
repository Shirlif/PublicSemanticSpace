"""P2-11: long-absence re-ID.

The long benchmark has 80-frame sequences and a 70-frame full occlusion.
Evaluates the v2-trained graph matcher and a transformer re-ID baseline.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pss._console import ensure_utf8
from pss.encoders.data import load_index
from pss.eval.baselines.entity_transformer import build_pairs, train_transformer
from pss.eval.scripts.run_p1b_tracking import _run_tracker
from pss.memory import TrackerConfig
from pss.memory.graph_matcher import GraphMatcherReID, load_graph
from pss.memory.learned_metric import token_flags
from pss.memory.perception import load_perception_models
from pss.memory.transformer_reid import TransformerReID
from pss.sim.dataset import load_sample


def _by_subset(rows):
    out = {}
    for row in rows:
        out.setdefault(row.get("subset", "unknown"), []).append(row)
    return out


def run(
    data_dir="data/tracking_bench_long",
    graph_model="eval/results/pss_vs_transformer_v2/graph_matcher_tactile.pt",
    out_dir="eval/results/pss_vs_transformer_v2",
    seed=0,
    epochs=200,
    device="cpu",
):
    data_dir = Path(data_dir)
    index = load_index(data_dir)
    entries = [e for e in index["entries"] if e.get("split") == "test"]
    samples = [(e, load_sample(data_dir / e["path"])) for e in entries]
    graph = load_graph(graph_model, device)
    tactile, dino = token_flags(int(graph.node_proj.in_features))
    models = load_perception_models(
        "checkpoints/category_pretrain_experts",
        "checkpoints/phase45_projection",
        device=device,
        dino=dino,
    )
    graph_reid = GraphMatcherReID(
        graph, device=device, include_tactile=tactile, include_dino=dino
    )
    train = build_pairs("data/tracking_bench_v2", "checkpoints/category_pretrain_experts",
                        "checkpoints/phase45_projection", "train", device)
    val = build_pairs("data/tracking_bench_v2", "checkpoints/category_pretrain_experts",
                      "checkpoints/phase45_projection", "val", device)
    transformer, _ = train_transformer(train, val, epochs=epochs, seed=seed, device=device)
    tr_reid = TransformerReID(transformer, device=device)

    graph_run = _run_tracker(
        samples, models, TrackerConfig(metric=graph_reid, metric_max_cost=10.0), seed=seed
    )
    tr_run = _run_tracker(
        samples, models, TrackerConfig(metric=tr_reid, metric_max_cost=10.0), seed=seed
    )

    def summary(run_result):
        rows = run_result["rows"]
        per = {}
        for subset, items in _by_subset(rows).items():
            per[subset] = {
                "n": len(items),
                "id_retention": float(np.mean([r["id_retention"] for r in items])),
                "switches": float(np.mean([r["id_switches"] for r in items])),
                "reappearance": float(np.mean([r["reappearance"] for r in items])),
            }
        return {
            "id_retention": run_result["id_retention"],
            "switches": run_result["id_switches"],
            "reappearance": run_result["reappearance"],
            "per_subset": per,
        }

    metrics = {
        "data_dir": str(data_dir),
        "n_sequences": len(samples),
        "absence_frames": int(entries[0]["frames"]) - 10 if entries else 0,
        "graph": summary(graph_run),
        "transformer": summary(tr_run),
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "p211_long_memory.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _report(out / "p211_long_memory.md", metrics)
    return metrics


def _report(path, metrics):
    lines = [
        "# P2-11: long-absence re-ID",
        "",
        f"- sequences: {metrics['n_sequences']}, full occlusion: {metrics['absence_frames']} frames",
        "",
        "| method | ID retention | switches | reappearance |",
        "|---|---:|---:|---:|",
        f"| PSS graph+tactile | {metrics['graph']['id_retention']:.3f} | "
        f"{metrics['graph']['switches']:.2f} | {metrics['graph']['reappearance']:.3f} |",
        f"| Transformer re-ID | {metrics['transformer']['id_retention']:.3f} | "
        f"{metrics['transformer']['switches']:.2f} | {metrics['transformer']['reappearance']:.3f} |",
        "",
        "| subset | PSS retention | TR retention |",
        "|---|---:|---:|",
    ]
    for subset in ("mixed", "samecat", "near_identical"):
        g = metrics["graph"]["per_subset"].get(subset)
        t = metrics["transformer"]["per_subset"].get(subset)
        if g and t:
            lines.append(f"| {subset} | {g['id_retention']:.3f} | {t['id_retention']:.3f} |")
    lines += ["", "Tracker re-ID age window: 150 frames.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tracking_bench_long")
    ap.add_argument("--graph-model", default="eval/results/pss_vs_transformer_v2/graph_matcher_tactile.pt")
    ap.add_argument("--out", default="eval/results/pss_vs_transformer_v2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=200)
    args = ap.parse_args()
    m = run(args.data, args.graph_model, args.out, args.seed, args.epochs)
    print(f"graph: {m['graph']['id_retention']:.3f}; transformer: {m['transformer']['id_retention']:.3f}")
    print(f"output: {args.out}")


if __name__ == "__main__":
    main()
