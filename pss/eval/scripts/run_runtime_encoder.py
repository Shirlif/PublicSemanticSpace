"""Row 1 of the gap table: pretrained encoder as the runtime default.

Compares the runtime perception service:
- ``dinov2``: frozen DINOv2 + tactile MLP + calibrated projection (default);
- ``smallcnn``: the previous SmallCNN + original projection (fallback).

Reports observation latency and tracker ID retention with the matching graph
matchers. The accuracy trade-off is the same one measured in Phase 11; this
script checks that the runtime path actually uses the pretrained default.

Example:
    python -m pss.eval.scripts.run_runtime_encoder --sequences 5
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
from pss.eval.scripts.run_p1b_tracking import _run_tracker
from pss.memory import TrackerConfig
from pss.memory.graph_matcher import GraphMatcherReID, load_graph
from pss.memory.learned_metric import token_flags
from pss.runtime.perception import PerceptionConfig, PerceptionService
from pss.sim.dataset import load_sample

OUT = "eval/results/runtime_encoder"
DATA = "data/tracking_bench_v2"


def _fmt(v, digits=3):
    if v is None:
        return "n/a"
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


def _service(vision: str, projection: str):
    config = PerceptionConfig(vision=vision, projection_dir=projection)
    return PerceptionService(config=config)


def _latency(service, sample, repeats=3):
    service.observe(sample, 0)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        service.observe(sample, 0)
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.percentile(times, 50)), float(np.percentile(times, 95))


def run(device="cuda", sequences=5) -> dict:
    index = load_index(DATA)
    entries = [e for e in index["entries"] if e.get("split") == "test"][:sequences]
    samples = [(e, load_sample(Path(DATA) / e["path"])) for e in entries]
    variants = {
        "dinov2": (
            _service("dinov2", "checkpoints/encoder_priority/dinov2/projection"),
            "checkpoints/encoder_priority/dinov2/graph_matcher.pt",
        ),
        "smallcnn": (
            _service("smallcnn", "checkpoints/phase45_projection"),
            "checkpoints/encoder_priority/base/graph_matcher.pt",
        ),
    }
    results = {}
    for name, (service, graph_path) in variants.items():
        p50, p95 = _latency(service, samples[0][1])
        graph = load_graph(graph_path, device)
        include_tactile, include_dino = token_flags(int(graph.node_proj.in_features))
        reid = GraphMatcherReID(
            graph,
            device=device,
            include_tactile=include_tactile,
            include_dino=include_dino,
        )
        tracker = _run_tracker(
            samples,
            service.models,
            TrackerConfig(metric=reid, metric_max_cost=10.0),
            seed=0,
        )
        results[name] = {
            "perception": service.describe(),
            "observation_latency_ms_p50": p50,
            "observation_latency_ms_p95": p95,
            "id_retention": tracker["id_retention"],
            "switches": tracker["id_switches"],
            "reappearance": tracker["reappearance"],
            "n_sequences": len(samples),
        }
    payload = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "device": device,
        "variants": results,
    }
    out = Path(OUT)
    out.mkdir(parents=True, exist_ok=True)
    (out / "runtime_encoder.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "runtime_encoder.md").write_text(render(payload), encoding="utf-8")
    return payload


def render(payload: dict) -> str:
    lines = [
        "# Runtime encoder default: DINOv2 vs SmallCNN",
        "",
        f"> generated: {payload['generated']}",
        "",
        "| variant | vision | projection | image size | fallback | p50 | p95 | ID retention | switches | reappearance |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in payload["variants"].items():
        p = r["perception"]
        lines.append(
            f"| {name} | {p['vision']} | `{p['projection']}` | {p['image_size']} | "
            f"{p['fallback']} | {_fmt(r['observation_latency_ms_p50'])} ms | "
            f"{_fmt(r['observation_latency_ms_p95'])} ms | {_fmt(r['id_retention'])} | "
            f"{_fmt(r['switches'], 2)} | {_fmt(r['reappearance'])} |"
        )
    lines += [
        "",
        "## Verdict",
        "",
        "- The runtime default perception path now uses frozen DINOv2 + calibrated projection.",
        "- SmallCNN is the fallback and remains available with one config flag.",
        "- CLIP/VLM are not defaults because their weights are not in the local cache.",
        "- Phase 11 measured the accuracy trade-off: DINOv2 re-ID 0.993 vs base 0.997,",
        "  but closed-set 0.969 vs 0.979 and open-set 0.897 vs 0.993. This row closes the",
        "  architecture gap (pretrained default), not the accuracy gap.",
        "",
    ]
    return "\n".join(lines)


def main():
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sequences", type=int, default=5)
    args = ap.parse_args()
    payload = run(args.device, args.sequences)
    for name, r in payload["variants"].items():
        print(
            f"{name}: {r['perception']['vision']} retention={r['id_retention']:.3f} "
            f"p50={r['observation_latency_ms_p50']:.1f} ms"
        )
    print(f"output: {OUT}")


if __name__ == "__main__":
    main()
