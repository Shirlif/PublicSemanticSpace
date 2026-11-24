"""Step 3: approximate active-set retrieval.

Keys are vectors; queries are noisy. Exact key lookup fails under noise.
The active set is k=8; an ANN top-1 over the active set is compared with
scanning the full history.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pss._console import ensure_utf8

D = 32
K = 8
N_VALUES = 16


def _unit(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-9)


def _accuracy(keys, values, query, sigma, rng, active_only=True):
    noise = rng.normal(scale=sigma, size=query.shape)
    q = _unit(query + noise)
    if active_only:
        sims = keys @ q
    else:
        sims = keys @ q
    idx = int(np.argmax(sims))
    return float(values[idx] == values[0])  # target is index 0 in the active set


def _latency(fn, repeats=20):
    for _ in range(2):
        fn()
    vals = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        vals.append((time.perf_counter() - t0) * 1000.0)
    return float(np.percentile(vals, 50)), float(np.percentile(vals, 95))


def run(out_dir="eval/results/pss_vs_transformer", seed=0):
    rng = np.random.default_rng(seed)
    active_keys = _unit(rng.normal(size=(K, D)))
    active_values = rng.integers(0, N_VALUES, size=K)
    history_keys = _unit(rng.normal(size=(1000, D)))
    history_values = rng.integers(0, N_VALUES, size=1000)
    query = active_keys[0].copy()

    rows = []
    for sigma in (0.0, 0.1, 0.3, 0.5):
        active_acc = np.mean(
            [_accuracy(active_keys, active_values, query, sigma, rng) for _ in range(200)]
        )
        full_keys = np.vstack([active_keys, history_keys])
        full_values = np.concatenate([active_values, history_values])
        full_acc = np.mean(
            [
                _accuracy(full_keys, full_values, query, sigma, rng, active_only=False)
                for _ in range(200)
            ]
        )
        p50, p95 = _latency(
            lambda: active_keys @ _unit(query + rng.normal(scale=sigma, size=D))
        )
        fp50, fp95 = _latency(
            lambda: full_keys @ _unit(query + rng.normal(scale=sigma, size=D))
        )
        rows.append(
            {
                "sigma": sigma,
                "active_ann_accuracy": float(active_acc),
                "full_history_ann_accuracy": float(full_acc),
                "active_p50_ms": p50,
                "active_p95_ms": p95,
                "full_p50_ms": fp50,
                "full_p95_ms": fp95,
            }
        )

    metrics = {
        "dim": D,
        "active_k": K,
        "history": 1000,
        "rows": rows,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ann_active_set.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _report(out / "ann_active_set.md", metrics)
    return metrics


def _report(path, metrics):
    lines = [
        "# Approximate active-set retrieval",
        "",
        f"- dim: {metrics['dim']}, active k: {metrics['active_k']}, history: {metrics['history']}",
        "",
        "| query noise sigma | active ANN acc | full-history ANN acc | active p50 | full p50 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for r in metrics["rows"]:
        lines.append(
            f"| {r['sigma']:.1f} | {r['active_ann_accuracy']:.3f} | "
            f"{r['full_history_ann_accuracy']:.3f} | {r['active_p50_ms']:.4f} ms | "
            f"{r['full_p50_ms']:.4f} ms |"
        )
    lines += [
        "",
        "Active ANN only searches k=8 entries; full-history ANN scans 1000 distractors.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/results/pss_vs_transformer")
    args = ap.parse_args()
    m = run(args.out)
    for r in m["rows"]:
        print(
            f"sigma {r['sigma']}: active {r['active_ann_accuracy']:.3f} "
            f"{r['active_p50_ms']:.4f} ms; full {r['full_history_ann_accuracy']:.3f} "
            f"{r['full_p50_ms']:.4f} ms"
        )
    print(f"output: {args.out}")


if __name__ == "__main__":
    main()
