"""P2-13: end-to-end latency and memory profile.

Measures per-frame perception, tracker, and total latency on tracking_bench_v2
with the dense and pruned graph matchers.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pss._console import ensure_utf8
from pss.encoders.data import load_index
from pss.memory import SceneGraph, Tracker, TrackerConfig
from pss.memory.graph_matcher import GraphMatcherReID, PrunedGraphMatcherReID, load_graph
from pss.memory.learned_metric import token_flags
from pss.memory.perception import PoseNoise, extract_frame, load_perception_models
from pss.sim.dataset import load_sample


def _percentiles(values):
    return {
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "mean_ms": float(np.mean(values)),
    }


def _profile(samples, models, config, seed=0):
    perception, tracking = [], []
    for idx, (entry, sample) in enumerate(samples):
        tracker = Tracker(config, SceneGraph())
        for t in range(sample["rgb"].shape[0]):
            t0 = time.perf_counter()
            obs = extract_frame(sample, t, models, PoseNoise(seed=seed + idx, enabled=True))
            t1 = time.perf_counter()
            tracker.step(obs, t)
            t2 = time.perf_counter()
            perception.append((t1 - t0) * 1000.0)
            tracking.append((t2 - t1) * 1000.0)
    total = [a + b for a, b in zip(perception, tracking)]
    return {
        "perception": _percentiles(perception),
        "tracking": _percentiles(tracking),
        "total": _percentiles(total),
        "n_frames": len(total),
    }


def run(
    data_dir="data/tracking_bench_v2",
    graph_model="eval/results/pss_vs_transformer_v2/graph_matcher_tactile.pt",
    out_dir="eval/results/pss_vs_transformer_v2",
    device="cpu",
):
    data_dir = Path(data_dir)
    index = load_index(data_dir)
    entries = [e for e in index["entries"] if e.get("split") == "test"]
    samples = [(e, load_sample(data_dir / e["path"])) for e in entries]
    t0 = time.perf_counter()
    graph = load_graph(graph_model, device)
    load_ms = (time.perf_counter() - t0) * 1000.0
    tactile, dino = token_flags(int(graph.node_proj.in_features))
    t0 = time.perf_counter()
    models = load_perception_models(
        "checkpoints/category_pretrain_experts",
        "checkpoints/phase45_projection",
        device=device,
        dino=dino,
    )
    model_load_ms = (time.perf_counter() - t0) * 1000.0

    dense = GraphMatcherReID(
        graph, device=device, include_tactile=tactile, include_dino=dino
    )
    pruned = PrunedGraphMatcherReID(
        graph,
        device=device,
        include_tactile=tactile,
        include_dino=dino,
        pose_gate=0.5,
        top_k=16,
    )

    dense_profile = _profile(
        samples, models, TrackerConfig(metric=dense, metric_max_cost=10.0)
    )
    pruned_profile = _profile(
        samples, models, TrackerConfig(metric=pruned, metric_max_cost=10.0)
    )

    rss_mb = None
    try:
        import psutil

        rss_mb = float(psutil.Process().memory_info().rss) / (1024 * 1024)
    except Exception:
        pass

    metrics = {
        "data_dir": str(data_dir),
        "n_sequences": len(samples),
        "graph_load_ms": load_ms,
        "model_load_ms": model_load_ms,
        "rss_mb": rss_mb,
        "dense": dense_profile,
        "pruned": pruned_profile,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "p213_engineering.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _report(out / "p213_engineering.md", metrics)
    return metrics


def _report(path, metrics):
    lines = [
        "# P2-13: end-to-end latency and memory",
        "",
        f"- sequences: {metrics['n_sequences']}",
        f"- graph load: {metrics['graph_load_ms']:.1f} ms; perception load: {metrics['model_load_ms']:.1f} ms",
        f"- RSS: {metrics['rss_mb']:.1f} MB" if metrics["rss_mb"] else "- RSS: n/a",
        "",
        "| matcher | perception p50/p95 | tracking p50/p95 | total p50/p95 |",
        "|---|---:|---:|---:|",
    ]
    for name in ("dense", "pruned"):
        r = metrics[name]
        p, t = r["perception"], r["tracking"]
        tot = r["total"]
        lines.append(
            f"| {name} | {p['p50_ms']:.2f} / {p['p95_ms']:.2f} ms | "
            f"{t['p50_ms']:.2f} / {t['p95_ms']:.2f} ms | "
            f"{tot['p50_ms']:.2f} / {tot['p95_ms']:.2f} ms |"
        )
    lines += [
        "",
        "Per-frame latency; includes segmentation crop, three experts, tracker update, and re-ID.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tracking_bench_v2")
    ap.add_argument("--graph-model", default="eval/results/pss_vs_transformer_v2/graph_matcher_tactile.pt")
    ap.add_argument("--out", default="eval/results/pss_vs_transformer_v2")
    args = ap.parse_args()
    m = run(args.data, args.graph_model, args.out)
    print(f"dense total p50 {m['dense']['total']['p50_ms']:.2f} ms")
    print(f"pruned total p50 {m['pruned']['total']['p50_ms']:.2f} ms")
    print(f"output: {args.out}")


if __name__ == "__main__":
    main()
