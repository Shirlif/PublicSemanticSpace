"""P1b: full online tracking, PSS graph matcher vs transformer re-ID."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pss._console import ensure_utf8
from pss.encoders.data import load_index
from pss.eval.baselines.entity_transformer import (
    EntityTransformer,
    build_pairs,
    train_transformer,
)
from pss.memory import SceneGraph, Tracker, TrackerConfig
from pss.memory.graph_matcher import (
    GraphMatcherReID,
    load_graph,
)
from pss.memory.learned_metric import token_flags
from pss.memory.perception import PoseNoise, extract_frame, load_perception_models
from pss.memory.transformer_reid import TransformerReID
from pss.sim.dataset import load_sample


def _id_retention(gt_tracks):
    retentions, switches, reappear = [], [], []
    for seq in gt_tracks.values():
        seq = sorted(seq)
        ids = [tid for _, tid in seq]
        if not ids:
            continue
        from collections import Counter

        counts = Counter(ids)
        retentions.append(max(counts.values()) / len(ids))
        switches.append(sum(1 for a, b in zip(ids, ids[1:]) if a != b))
        for (t0, id0), (t1, id1) in zip(seq, seq[1:]):
            if t1 - t0 > 1:
                reappear.append(id1 == id0)
    return {
        "id_retention": float(np.mean(retentions)) if retentions else 0.0,
        "id_switches": float(np.mean(switches)) if switches else 0.0,
        "reappearance": float(np.mean(reappear)) if reappear else 1.0,
    }


def _run_tracker(samples, models, config, seed=0):
    rows = []
    for i, (entry, sample) in enumerate(samples):
        tracker = Tracker(config, SceneGraph())
        gt_tracks = {}
        for t in range(sample["rgb"].shape[0]):
            obs = extract_frame(sample, t, models, PoseNoise(seed=seed + i, enabled=True))
            events = tracker.step(obs, t)
            for key in ("matched", "reid", "new"):
                for tid, gid in events.get(key, []):
                    gt_tracks.setdefault(gid, []).append((t, tid))
        rows.append({**entry, **_id_retention(gt_tracks)})
    return {
        "id_retention": float(np.mean([r["id_retention"] for r in rows])),
        "id_switches": float(np.mean([r["id_switches"] for r in rows])),
        "reappearance": float(np.mean([r["reappearance"] for r in rows])),
        "rows": rows,
    }


def run_p1b(
    data_dir,
    expert_dir,
    projection_dir,
    out_dir,
    graph_model="eval/results/pss_vs_transformer_v2/graph_matcher.pt",
    seeds=3,
    epochs=200,
    device="cpu",
):
    data_dir = Path(data_dir)
    index = load_index(data_dir)
    entries = [e for e in index["entries"] if e.get("split") == "test"]
    samples = [(e, load_sample(data_dir / e["path"])) for e in entries]
    graph = load_graph(graph_model, device)
    include_tactile, include_dino = token_flags(int(graph.node_proj.in_features))
    models = load_perception_models(
        expert_dir,
        projection_dir,
        prototype_dir=None,
        device=device,
        dino=include_dino,
    )

    train = build_pairs(data_dir, expert_dir, projection_dir, "train", device)
    val = build_pairs(data_dir, expert_dir, projection_dir, "val", device)
    graph_reid = GraphMatcherReID(
        graph,
        device=device,
        include_tactile=include_tactile,
        include_dino=include_dino,
    )
    transformer, info = train_transformer(
        train, val, epochs=epochs, seed=0, device=device
    )
    transformer_reid = TransformerReID(transformer, device=device)

    hand = _run_tracker(samples, models, TrackerConfig(), seed=0)
    pss_graph = _run_tracker(
        samples,
        models,
        TrackerConfig(metric=graph_reid, metric_max_cost=10.0),
        seed=0,
    )
    tr = _run_tracker(
        samples,
        models,
        TrackerConfig(metric=transformer_reid, metric_max_cost=10.0),
        seed=0,
    )
    metrics = {
        "data_dir": str(data_dir),
        "n_sequences": len(samples),
        "include_tactile": include_tactile,
        "include_dino": include_dino,
        "hand": hand,
        "pss_graph": pss_graph,
        "transformer": tr,
        "graph_val": info["best_val_reid"],
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    suffix = ("_tactile" if include_tactile else "") + ("_dino" if include_dino else "")
    (out / f"p1b_tracking{suffix}.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_report(out / f"p1b_tracking{suffix}.md", metrics)
    return metrics


def _write_report(path, metrics):
    lines = [
        "# P1b: full online tracking",
        "",
        f"- data: `{metrics['data_dir']}`",
        f"- sequences: {metrics['n_sequences']}",
        "",
        "| method | ID retention | switches | reappearance |",
        "|---|---:|---:|---:|",
        f"| Hand PSS tracker | {metrics['hand']['id_retention']:.3f} | "
        f"{metrics['hand']['id_switches']:.2f} | {metrics['hand']['reappearance']:.3f} |",
        f"| PSS + graph matcher{'+tactile' if metrics.get('include_tactile') else ''}{'+dino' if metrics.get('include_dino') else ''} | {metrics['pss_graph']['id_retention']:.3f} | "
        f"{metrics['pss_graph']['id_switches']:.2f} | {metrics['pss_graph']['reappearance']:.3f} |",
        f"| Transformer re-ID | {metrics['transformer']['id_retention']:.3f} | "
        f"{metrics['transformer']['id_switches']:.2f} | {metrics['transformer']['reappearance']:.3f} |",
        "",
        "All trackers see the same frames and simulated pose noise.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tracking_bench_v2")
    ap.add_argument("--experts", default="checkpoints/category_pretrain_experts")
    ap.add_argument("--projections", default="checkpoints/phase45_projection")
    ap.add_argument("--graph-model", default="eval/results/pss_vs_transformer_v2/graph_matcher.pt")
    ap.add_argument("--out", default="eval/results/pss_vs_transformer_v2")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    metrics = run_p1b(
        args.data,
        args.experts,
        args.projections,
        args.out,
        graph_model=args.graph_model,
        epochs=args.epochs,
        device=args.device,
    )
    print("P1b complete:")
    for name in ("hand", "pss_graph", "transformer"):
        r = metrics[name]
        print(
            f"  {name}: retention={r['id_retention']:.3f} "
            f"switches={r['id_switches']:.2f} reappear={r['reappearance']:.3f}"
        )
    print(f"  output: {args.out}")


if __name__ == "__main__":
    main()
