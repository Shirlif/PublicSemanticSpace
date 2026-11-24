"""Row 3 of the gap table: object attention as a runtime binding option.

Trains the object-centric attention matcher on tracking_bench_v2, compares
tracker association against the hand cost and the graph matcher, and wires the
matcher into ``EntityLayer``/``PSSRuntime`` as the ``associator``.

Example:
    python -m pss.eval.scripts.run_runtime_binding --device cuda --sequences 8
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
from pss.eval.baselines.entity_transformer import build_pairs
from pss.eval.scripts.run_p1b_tracking import _run_tracker
from pss.memory import TrackerConfig
from pss.memory.graph_matcher import GraphMatcherReID, evaluate_graph, load_graph
from pss.memory.learned_metric import TACTILE_TOKEN_DIM, token_flags
from pss.memory.object_attention import (
    ObjectAttentionReID,
    evaluate_object_attention,
    load_object_attention,
    save_object_attention,
    train_object_attention,
)
from pss.memory.perception import load_perception_models
from pss.runtime import EntityLayer, PSSRuntime
from pss.runtime.perception import PerceptionConfig, PerceptionService
from pss.sim.dataset import load_sample

OUT = "eval/results/runtime_binding"
DATA = "data/tracking_bench_v2"
DATA_LONG = "data/tracking_bench_long"
EXPERTS = "checkpoints/category_pretrain_experts"
PROJECTIONS = "checkpoints/phase45_projection"
CKPT = "checkpoints/object_attention/object_attention.pt"


def _fmt(v, digits=3):
    if v is None:
        return "n/a"
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


def _train(device: str, epochs: int = 200):
    path = Path(CKPT)
    if path.exists():
        return load_object_attention(path, device=device), {"loaded": True, "test_accuracy": None}
    train = build_pairs(DATA, EXPERTS, PROJECTIONS, "train", device, include_tactile=True)
    val = build_pairs(DATA, EXPERTS, PROJECTIONS, "val", device, include_tactile=True)
    test = build_pairs(DATA, EXPERTS, PROJECTIONS, "test", device, include_tactile=True)
    model, info = train_object_attention(
        train, val, TACTILE_TOKEN_DIM, epochs=epochs, seed=0, device=device
    )
    save_object_attention(model, path)
    metrics = evaluate_object_attention(model, test, device)
    return model, {
        "loaded": False,
        "best_val_reid": info["best_val_reid"],
        "test_accuracy": metrics["reid_accuracy"],
        "test_near": metrics["per_subset"]["near_identical"]["reid_accuracy"],
    }


def _configs(model, device: str):
    graph = load_graph("checkpoints/encoder_priority/base/graph_matcher.pt", device)
    include_tactile, include_dino = token_flags(int(graph.node_proj.in_features))
    graph_reid = GraphMatcherReID(
        graph, device=device, include_tactile=include_tactile, include_dino=include_dino
    )
    object_reid = ObjectAttentionReID(model, device=device, max_cost=0.9)
    return {
        "hand": TrackerConfig(),
        "graph_matcher": TrackerConfig(metric=graph_reid, metric_max_cost=10.0),
        "object_attention": TrackerConfig(metric=object_reid, metric_max_cost=10.0),
    }


def _run_variant(name, samples, models, config, device):
    t0 = time.perf_counter()
    result = _run_tracker(samples, models, config, seed=0)
    elapsed = time.perf_counter() - t0
    frames = sum(sample["rgb"].shape[0] for _, sample in samples)
    return {
        "variant": name,
        "id_retention": result["id_retention"],
        "switches": result["id_switches"],
        "reappearance": result["reappearance"],
        "latency_ms_per_frame": elapsed * 1000.0 / max(1, frames),
        "n_sequences": len(samples),
    }


def run(device="cuda", sequences=8, epochs=200) -> dict:
    models = load_perception_models(EXPERTS, PROJECTIONS, device=device)
    model, train_info = _train(device, epochs)
    configs = _configs(model, device)

    def _samples(root: str):
        index = load_index(root)
        entries = [e for e in index["entries"] if e.get("split") == "test"][:sequences]
        return [(e, load_sample(Path(root) / e["path"])) for e in entries]

    datasets = {"v2": _samples(DATA), "long": _samples(DATA_LONG)}
    results = {
        dataset: {
            name: _run_variant(name, samples, models, config, device)
            for name, config in configs.items()
        }
        for dataset, samples in datasets.items()
    }
    test_pairs = build_pairs(DATA, EXPERTS, PROJECTIONS, "test", device, include_tactile=True)
    graph_eval = evaluate_graph(
        load_graph("checkpoints/encoder_priority/base/graph_matcher.pt", device),
        test_pairs,
        device,
    )
    object_eval = evaluate_object_attention(model, test_pairs, device)

    # Runtime integration: the object attention matcher becomes the entity
    # layer associator; summary() reports it.
    service = PerceptionService(
        config=PerceptionConfig(vision="smallcnn", projection_dir=PROJECTIONS),
        device=device,
    )
    runtime = PSSRuntime(
        entity_layer=EntityLayer(associator=ObjectAttentionReID(model, device=device)),
        perception=service,
    )
    runtime.ingest_sample(datasets["v2"][0][1], 0)
    components = runtime.summary()["components"]

    payload = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "device": device,
        "object_attention_train": train_info,
        "pair_reid": {
            "graph_matcher": graph_eval["reid_accuracy"],
            "object_attention": object_eval["reid_accuracy"],
            "graph_near": graph_eval["per_subset"]["near_identical"]["reid_accuracy"],
            "object_near": object_eval["per_subset"]["near_identical"]["reid_accuracy"],
        },
        "results": results,
        "runtime_components": components,
    }
    out = Path(OUT)
    out.mkdir(parents=True, exist_ok=True)
    (out / "runtime_binding.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "runtime_binding.md").write_text(render(payload), encoding="utf-8")
    return payload


def render(payload: dict) -> str:
    lines = [
        "# Row 3 — object attention as a runtime binding option",
        "",
        f"> generated: {payload['generated']}",
        "",
        f"- object attention train: {payload['object_attention_train']}",
        f"- pair re-ID overall: graph {_fmt(payload['pair_reid']['graph_matcher'])} vs "
        f"object attention {_fmt(payload['pair_reid']['object_attention'])}",
        f"- pair re-ID near-identical: graph {_fmt(payload['pair_reid']['graph_near'])} vs "
        f"object attention {_fmt(payload['pair_reid']['object_near'])}",
        "",
    ]
    for dataset, rows in payload["results"].items():
        lines += [
            f"### {dataset}",
            "",
            "| associator | ID retention | switches | reappearance | ms/frame |",
            "|---|---:|---:|---:|---:|",
        ]
        for name, r in rows.items():
            lines.append(
                f"| {name} | {_fmt(r['id_retention'])} | {_fmt(r['switches'], 2)} | "
                f"{_fmt(r['reappearance'])} | {_fmt(r['latency_ms_per_frame'], 2)} |"
            )
        lines.append("")
    lines += [
        "",
        f"- runtime components: {payload['runtime_components']}",
        "",
        "## Verdict",
        "",
        "- Object attention is now a first-class `EntityLayer` associator; the",
        "  hand cost and graph matcher remain selectable.",
        "- On this benchmark it is expected to tie the graph matcher (prior P1-15:",
        "  0.987 vs 0.989), not beat it; the row closes the binding-option gap.",
        "- Persistent slots were already present in `EntityLayer`; this adds the",
        "  object-attention association path to the runtime.",
        "",
    ]
    return "\n".join(lines)


def main():
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sequences", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=200)
    args = ap.parse_args()
    payload = run(args.device, args.sequences, args.epochs)
    for dataset, rows in payload["results"].items():
        for name, r in rows.items():
            print(
                f"{dataset}/{name}: retention={r['id_retention']:.3f} "
                f"switches={r['switches']:.2f} {r['latency_ms_per_frame']:.1f} ms/frame"
            )
    print(f"output: {OUT}")


if __name__ == "__main__":
    main()
