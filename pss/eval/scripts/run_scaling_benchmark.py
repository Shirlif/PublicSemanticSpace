"""P4: PSS active-set relation computation vs transformer full-token attention.

PSS keeps N entities in the scene graph but only computes relations among the
k active entities. The transformer baseline attends over all N tokens.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from pss._console import ensure_utf8
from pss.memory import SceneGraph, Track


def _pss_latency(n_entities: int, active_k: int = 4, repeats: int = 100) -> dict:
    graph = SceneGraph(cell_size=0.1)
    rng = np.random.default_rng(0)
    for i in range(n_entities):
        graph.add(
            Track(
                id=f"t{i:04d}",
                pos=rng.uniform(-0.5, 0.5, size=3),
                quat=np.array([1.0, 0.0, 0.0, 0.0]),
            )
        )
    active = graph.all()[:active_k]
    latencies = []
    for _ in range(repeats):
        start = time.perf_counter()
        for track in active:
            graph.query_near(track.pos, 0.15)
        # Active-set relation computation: O(k^2), independent of N.
        for i, a in enumerate(active):
            for b in active[i + 1 :]:
                _ = float(np.linalg.norm(a.pos - b.pos))
        latencies.append((time.perf_counter() - start) * 1000.0)
    return {
        "n_entities": n_entities,
        "active_k": len(active),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
    }


def _transformer_latency(
    n_tokens: int,
    d_model: int = 128,
    nhead: int = 4,
    layers: int = 2,
    repeats: int = 20,
) -> dict:
    torch.manual_seed(0)
    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=d_model * 2,
        batch_first=True,
        activation="gelu",
    )
    model = nn.TransformerEncoder(layer, num_layers=layers).eval()
    x = torch.randn(1, n_tokens, d_model)
    latencies = []
    with torch.no_grad():
        for _ in range(repeats):
            start = time.perf_counter()
            model(x)
            latencies.append((time.perf_counter() - start) * 1000.0)
    return {
        "n_tokens": n_tokens,
        "d_model": d_model,
        "layers": layers,
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
    }


def run_scaling(
    out_dir: str | Path,
    sizes: tuple[int, ...] = (4, 20, 50, 200, 1000),
    active_k: int = 4,
) -> dict:
    pss = [_pss_latency(n, active_k) for n in sizes]
    transformer = [_transformer_latency(n) for n in sizes]
    ratios = []
    for a, b in zip(pss, transformer):
        ratios.append(
            {
                "n": a["n_entities"],
                "pss_ms": a["p50_ms"],
                "transformer_ms": b["p50_ms"],
                "speedup": b["p50_ms"] / max(a["p50_ms"], 1e-9),
            }
        )
    target = next(r for r in ratios if r["n"] == 200)
    metrics = {
        "pss": pss,
        "transformer": transformer,
        "ratios": ratios,
        "p4": {
            "n": 200,
            "pss_ms": target["pss_ms"],
            "transformer_ms": target["transformer_ms"],
            "speedup": target["speedup"],
            "threshold": 10.0,
            "pass": bool(target["speedup"] >= 10.0),
        },
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "p4_scaling.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_report(out / "p4_scaling.md", metrics)
    return metrics


def _write_report(path: Path, metrics: dict) -> None:
    lines = [
        "# P4: PSS active-set vs transformer full-token attention",
        "",
        "| N | PSS p50 (ms) | Transformer p50 (ms) | speedup |",
        "|---:|---:|---:|---:|",
    ]
    for r in metrics["ratios"]:
        lines.append(
            f"| {r['n']} | {r['pss_ms']:.4f} | {r['transformer_ms']:.4f} | "
            f"{r['speedup']:.1f}x |"
        )
    p4 = metrics["p4"]
    lines += [
        "",
        f"**P4**: N=200, PSS {p4['pss_ms']:.4f} ms vs transformer "
        f"{p4['transformer_ms']:.4f} ms, speedup {p4['speedup']:.1f}x -> "
        f"**{'PASS' if p4['pass'] else 'FAIL'}**",
        "",
        "PSS computes relations only among active entities; transformer attends over all N tokens.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/results/pss_vs_transformer")
    ap.add_argument("--active-k", type=int, default=4)
    args = ap.parse_args()
    metrics = run_scaling(args.out, active_k=args.active_k)
    print("P4 complete:")
    for r in metrics["ratios"]:
        print(
            f"  N={r['n']}: PSS {r['pss_ms']:.4f} ms, "
            f"Transformer {r['transformer_ms']:.4f} ms, {r['speedup']:.1f}x"
        )
    print(f"  output: {args.out}")


if __name__ == "__main__":
    main()
