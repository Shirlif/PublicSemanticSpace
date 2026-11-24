"""P0-3: graph matcher scaling and candidate pruning.

Measures the dense graph matcher, a pruned component-wise variant, and the
dense transformer re-ID baseline on synthetic N-entity sets. Also checks
pruned-vs-dense ID retention on tracking_bench_v2.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.optimize import linear_sum_assignment

from pss._console import ensure_utf8
from pss.encoders.data import load_index
from pss.eval.baselines.entity_transformer import EntityTransformer
from pss.eval.scripts.run_p1b_tracking import _run_tracker
from pss.memory import SceneGraph, TrackerConfig
from pss.memory.graph_matcher import (
    GraphMatcherReID,
    PrunedGraphMatcherReID,
    load_graph,
)
from pss.memory.learned_metric import TOKEN_DIM
from pss.memory.perception import load_perception_models
from pss.memory.transformer_reid import TransformerReID
from pss.sim.dataset import load_sample
from pss.sim.scene import CATEGORY_SPECS

CATEGORIES = sorted(CATEGORY_SPECS)


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def make_case(n: int, seed: int = 0, area_scale: float = 0.12):
    """N true track/observation pairs plus local distractors."""
    rng = np.random.default_rng(seed)
    side = max(1.0, float(np.sqrt(n) * area_scale))
    tracks, observations = [], []
    for i in range(n):
        pos = rng.uniform([-side / 2, -side / 2, 0.0], [side / 2, side / 2, 0.0])
        category = CATEGORIES[i % len(CATEGORIES)]
        vision = _unit(rng.normal(size=128))
        geometry = _unit(rng.normal(size=128))
        obs_vision = _unit(vision + rng.normal(scale=0.04, size=128))
        obs_geometry = _unit(geometry + rng.normal(scale=0.04, size=128))
        tracks.append(
            SimpleNamespace(
                pos=pos.copy(),
                category=category,
                appearance=vision,
                raw={"vision": vision, "geometry": geometry},
            )
        )
        observations.append(
            SimpleNamespace(
                pos=pos + rng.normal(scale=0.01, size=3),
                category=category,
                appearance=obs_vision,
                raw={"vision": obs_vision, "geometry": obs_geometry},
            )
        )
    return tracks, observations


def _dense_matches(adapter, tracks, observations, max_cost):
    cost = adapter.cost_matrix(tracks, observations)
    rows, cols = linear_sum_assignment(cost)
    return [(int(r), int(c), float(cost[r, c])) for r, c in zip(rows, cols) if cost[r, c] <= max_cost]


def _timed(fn, warmup=1, repeats=5):
    for _ in range(warmup):
        fn()
    values = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        values.append((time.perf_counter() - t0) * 1000.0)
    return {
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "runs": values,
    }


def _retention(matches, n):
    true = {(i, j) for i, j, _ in matches if i == j}
    matched = {i for i, _j, _ in matches}
    return {
        "n": n,
        "matched": len(matches),
        "true_matches": len(true),
        "true_retention": len(true) / n,
        "coverage": len(matched) / n,
    }


def scaling_case(n, graph, device, repeats, dense_enabled=True, pose_gate=0.08):
    tracks, observations = make_case(n, seed=n)
    dense = GraphMatcherReID(graph, device=device, max_cost=10.0)
    pruned = PrunedGraphMatcherReID(
        graph, device=device, max_cost=10.0, pose_gate=pose_gate, top_k=16
    )
    transformer = TransformerReID(EntityTransformer(token_dim=TOKEN_DIM), device=device, max_cost=10.0)

    row = {"n": n}
    if dense_enabled:
        row["dense"] = _timed(
            lambda: _dense_matches(dense, tracks, observations, 10.0),
            repeats=repeats,
        )
        dense_matches = _dense_matches(dense, tracks, observations, 10.0)
        row["dense_retention"] = _retention(dense_matches, n)
    else:
        row["dense"] = None

    row["pruned"] = _timed(
        lambda: pruned.match(tracks, observations), repeats=repeats
    )
    pruned_matches = pruned.match(tracks, observations)
    row["pruned_retention"] = _retention(pruned_matches, n)

    row["transformer"] = _timed(
        lambda: _dense_matches(transformer, tracks, observations, 10.0),
        repeats=repeats,
    )
    if row["dense"] and row["pruned"]:
        row["speedup_pruned_vs_dense"] = (
            row["dense"]["p50_ms"] / row["pruned"]["p50_ms"]
            if row["pruned"]["p50_ms"] > 0
            else None
        )
    row["speedup_pruned_vs_transformer"] = (
        row["transformer"]["p50_ms"] / row["pruned"]["p50_ms"]
        if row["pruned"]["p50_ms"] > 0
        else None
    )
    row["pruned_true_retention"] = (
        row["pruned_retention"]["true_matches"] / n if n else 1.0
    )
    if row.get("dense_retention"):
        d = row["dense_retention"]["true_matches"]
        p = row["pruned_retention"]["true_matches"]
        row["dense_true_retention"] = d / n if n else 1.0
        row["relative_retention"] = min(1.0, p / d) if d else 1.0
    return row


def v2_retention(data_dir, expert_dir, projection_dir, graph, device):
    data_dir = Path(data_dir)
    index = load_index(data_dir)
    entries = [e for e in index["entries"] if e.get("split") == "test"]
    samples = [(e, load_sample(data_dir / e["path"])) for e in entries]
    models = load_perception_models(
        expert_dir, projection_dir, prototype_dir=None, device=device
    )
    dense = GraphMatcherReID(graph, device=device, max_cost=10.0)
    pruned = PrunedGraphMatcherReID(
        graph, device=device, max_cost=10.0, pose_gate=0.5, top_k=16, category_gate=False
    )
    dense_run = _run_tracker(
        samples,
        models,
        TrackerConfig(metric=dense, metric_max_cost=10.0),
        seed=0,
    )
    pruned_run = _run_tracker(
        samples,
        models,
        TrackerConfig(metric=pruned, metric_max_cost=10.0),
        seed=0,
    )
    return {
        "n_sequences": len(samples),
        "dense": {
            "id_retention": dense_run["id_retention"],
            "switches": dense_run["id_switches"],
            "reappearance": dense_run["reappearance"],
        },
        "pruned": {
            "id_retention": pruned_run["id_retention"],
            "switches": pruned_run["id_switches"],
            "reappearance": pruned_run["reappearance"],
        },
    }


def run(
    out_dir="eval/results/pss_vs_transformer_v2",
    graph_model="eval/results/pss_vs_transformer_v2/graph_matcher.pt",
    data_dir="data/tracking_bench_v2",
    expert_dir="checkpoints/category_pretrain_experts",
    projection_dir="checkpoints/phase45_projection",
    ns=(50, 200, 1000),
    repeats=5,
    device="cpu",
    dense_max=200,
    skip_v2=False,
):
    graph = load_graph(graph_model, device)
    rows = []
    for n in ns:
        rows.append(
            scaling_case(
                n,
                graph,
                device,
                repeats=repeats,
                dense_enabled=n <= dense_max,
            )
        )
    v2 = None if skip_v2 else v2_retention(
        data_dir, expert_dir, projection_dir, graph, device
    )
    metrics = {
        "graph_model": str(graph_model),
        "device": device,
        "rows": rows,
        "v2": v2,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "p0_scaling.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _report(out / "p0_scaling.md", metrics)
    return metrics


def _fmt(value):
    return f"{value:.2f}" if value is not None else "n/a"


def _report(path, metrics):
    lines = [
        "# P0-3: graph matcher scaling",
        "",
        f"- graph model: `{metrics['graph_model']}`",
        f"- dense timed up to N={max((r['n'] for r in metrics['rows'] if r['dense']), default=0)}",
        "",
        "| N | dense p50/p95 ms | pruned p50/p95 ms | transformer p50/p95 ms | pruned vs transformer | pruned true retention | relative vs dense |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in metrics["rows"]:
        d = r["dense"]
        p = r["pruned"]
        t = r["transformer"]
        lines.append(
            f"| {r['n']} | {_fmt(d['p50_ms']) if d else 'n/a'} / "
            f"{_fmt(d['p95_ms']) if d else 'n/a'} | "
            f"{p['p50_ms']:.2f} / {p['p95_ms']:.2f} | "
            f"{t['p50_ms']:.2f} / {t['p95_ms']:.2f} | "
            f"{_fmt(r.get('speedup_pruned_vs_transformer'))}x | "
            f"{_fmt(r.get('pruned_true_retention'))} | "
            f"{_fmt(r.get('relative_retention'))} |"
        )
    lines += [
        "",
        "Candidate pruning: pose gate + category gate + appearance top-16, then component-wise GraphMatcher + Hungarian.",
        "Synthetic cases use N true pairs with local distractors; true-retention is reported in JSON.",
        "relative vs dense is capped at 1.0; see dense_true_retention in JSON for the raw comparison.",
        "",
    ]
    if metrics["v2"]:
        v = metrics["v2"]
        lines += [
            "## tracking_bench_v2 retention",
            "",
            "| matcher | ID retention | switches | reappearance |",
            "|---|---:|---:|---:|",
            f"| dense graph | {v['dense']['id_retention']:.3f} | {v['dense']['switches']:.2f} | {v['dense']['reappearance']:.3f} |",
            f"| pruned graph | {v['pruned']['id_retention']:.3f} | {v['pruned']['switches']:.2f} | {v['pruned']['reappearance']:.3f} |",
            "",
            f"- relative ID retention: {v['pruned']['id_retention'] / v['dense']['id_retention']:.3f}",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/results/pss_vs_transformer_v2")
    ap.add_argument("--graph-model", default="eval/results/pss_vs_transformer_v2/graph_matcher.pt")
    ap.add_argument("--data", default="data/tracking_bench_v2")
    ap.add_argument("--experts", default="checkpoints/category_pretrain_experts")
    ap.add_argument("--projections", default="checkpoints/phase45_projection")
    ap.add_argument("--ns", default="50,200,1000")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--dense-max", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--skip-v2", action="store_true")
    args = ap.parse_args()
    ns = tuple(int(x) for x in args.ns.split(",") if x.strip())
    metrics = run(
        out_dir=args.out,
        graph_model=args.graph_model,
        data_dir=args.data,
        expert_dir=args.experts,
        projection_dir=args.projections,
        ns=ns,
        repeats=args.repeats,
        device=args.device,
        dense_max=args.dense_max,
        skip_v2=args.skip_v2,
    )
    for r in metrics["rows"]:
        print(
            f"N={r['n']}: pruned {r['pruned']['p50_ms']:.2f} ms, "
            f"transformer {r['transformer']['p50_ms']:.2f} ms, "
            f"speedup {r.get('speedup_pruned_vs_transformer')}x, "
            f"true retention {r.get('pruned_true_retention')}"
        )
    print(f"output: {args.out}")


if __name__ == "__main__":
    main()
