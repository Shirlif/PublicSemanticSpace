"""P1b fair tracking baseline: transformer + tactile vs PSS + tactile.

Both methods receive the same tactile token. This closes the fair-attribution
gap left by the original P1b, whose transformer baseline had no tactile.
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
from pss.memory.learned_metric import TACTILE_TOKEN_DIM, token_flags
from pss.memory.perception import load_perception_models
from pss.memory.transformer_reid import TransformerReID
from pss.sim.dataset import load_sample


def run(
    data_dir="data/tracking_bench_v2",
    graph_model="eval/results/pss_vs_transformer_v2/graph_matcher_tactile.pt",
    out_dir="eval/results/pss_vs_transformer_v2",
    seeds=5,
    epochs=200,
    device="cpu",
):
    data_dir = Path(data_dir)
    index = load_index(data_dir)
    entries = [e for e in index["entries"] if e.get("split") == "test"]
    samples = [(e, load_sample(data_dir / e["path"])) for e in entries]
    models = load_perception_models(
        "checkpoints/category_pretrain_experts",
        "checkpoints/phase45_projection",
        device=device,
    )
    graph = load_graph(graph_model, device)
    tactile, dino = token_flags(int(graph.node_proj.in_features))
    graph_reid = GraphMatcherReID(
        graph, device=device, include_tactile=tactile, include_dino=dino
    )

    train = build_pairs(
        data_dir, "checkpoints/category_pretrain_experts",
        "checkpoints/phase45_projection", "train", device, include_tactile=True,
    )
    val = build_pairs(
        data_dir, "checkpoints/category_pretrain_experts",
        "checkpoints/phase45_projection", "val", device, include_tactile=True,
    )

    hand = _run_tracker(samples, models, TrackerConfig(), seed=0)
    pss = _run_tracker(
        samples, models, TrackerConfig(metric=graph_reid, metric_max_cost=10.0), seed=0
    )

    tr_runs = []
    for seed in range(seeds):
        transformer, _ = train_transformer(
            train,
            val,
            epochs=epochs,
            seed=seed,
            device=device,
            token_dim=TACTILE_TOKEN_DIM,
        )
        tr_reid = TransformerReID(
            transformer, device=device, include_tactile=True, max_cost=10.0
        )
        tr_runs.append(
            _run_tracker(
                samples,
                models,
                TrackerConfig(metric=tr_reid, metric_max_cost=10.0),
                seed=seed,
            )
        )

    def agg(runs):
        return {
            "id_retention_mean": float(np.mean([r["id_retention"] for r in runs])),
            "id_retention_std": float(np.std([r["id_retention"] for r in runs])),
            "switches_mean": float(np.mean([r["id_switches"] for r in runs])),
            "reappearance_mean": float(np.mean([r["reappearance"] for r in runs])),
            "reappearance_std": float(np.std([r["reappearance"] for r in runs])),
        }

    metrics = {
        "data_dir": str(data_dir),
        "n_sequences": len(samples),
        "tactile": bool(tactile),
        "hand": {
            "id_retention": hand["id_retention"],
            "id_switches": hand["id_switches"],
            "reappearance": hand["reappearance"],
        },
        "pss_graph": {
            "id_retention": pss["id_retention"],
            "id_switches": pss["id_switches"],
            "reappearance": pss["reappearance"],
        },
        "transformer_tactile": agg(tr_runs),
        "transformer_runs": tr_runs,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "p1b_transformer_tactile.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _report(out / "p1b_transformer_tactile.md", metrics)
    return metrics


def _report(path, metrics):
    p = metrics["pss_graph"]
    t = metrics["transformer_tactile"]
    lines = [
        "# P1b fair tracking: transformer + tactile",
        "",
        f"- sequences: {metrics['n_sequences']}",
        "",
        "| method | ID retention | switches | reappearance |",
        "|---|---:|---:|---:|",
        f"| Hand PSS | {metrics['hand']['id_retention']:.3f} | "
        f"{metrics['hand']['id_switches']:.2f} | {metrics['hand']['reappearance']:.3f} |",
        f"| PSS + graph + tactile | **{p['id_retention']:.3f}** | **{p['id_switches']:.2f}** | **{p['reappearance']:.3f}** |",
        f"| Transformer + tactile | {t['id_retention_mean']:.3f} ± {t['id_retention_std']:.3f} | "
        f"{t['switches_mean']:.2f} | {t['reappearance_mean']:.3f} ± {t['reappearance_std']:.3f} |",
        "",
        "Both PSS and transformer receive the same tactile token and Hungarian assignment.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tracking_bench_v2")
    ap.add_argument("--graph-model", default="eval/results/pss_vs_transformer_v2/graph_matcher_tactile.pt")
    ap.add_argument("--out", default="eval/results/pss_vs_transformer_v2")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=200)
    args = ap.parse_args()
    m = run(args.data, args.graph_model, args.out, args.seeds, args.epochs)
    print(
        f"PSS {m['pss_graph']['id_retention']:.3f}; "
        f"TR+tactile {m['transformer_tactile']['id_retention_mean']:.3f} ± "
        f"{m['transformer_tactile']['id_retention_std']:.3f}"
    )
    print(f"output: {args.out}")


if __name__ == "__main__":
    main()
