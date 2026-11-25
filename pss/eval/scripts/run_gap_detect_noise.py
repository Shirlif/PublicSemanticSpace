"""Claim-gap E7: binding under non-oracle detection noise.

A non-oracle detector misses objects and jitters poses.  This experiment takes
the same long-occlusion sequences and graph+tactile tracker, then corrupts the
observation stream:

- ``miss_rate``: drop each observation independently;
- ``pose_sigma``: add Gaussian noise to the reported position.

It is a detector-level stress test, not a new detector.  The question is
whether the state layer keeps identity when the input is no longer oracle.
"""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pss._console import ensure_utf8
from pss.encoders.data import load_index
from pss.eval.scripts.run_gap_tactile_binding import (
    SUBSETS,
    _retention,
    _run_tracker,
)
from pss.memory import TrackerConfig
from pss.memory.graph_matcher import GraphMatcherReID, load_graph
from pss.memory.learned_metric import token_flags
from pss.memory.perception import PoseNoise, extract_frame, load_perception_models
from pss.sim.dataset import load_sample

CONDITIONS = (
    ("oracle", 0.0, 0.0),
    ("miss10_sigma2cm", 0.10, 0.02),
    ("miss20_sigma5cm", 0.20, 0.05),
    ("miss30_sigma8cm", 0.30, 0.08),
)


def _corrupt(frames: list[list], miss_rate: float, pose_sigma: float, seed: int) -> list[list]:
    out = copy.deepcopy(frames)
    rng = np.random.default_rng(seed)
    for frame in out:
        kept = []
        for obs in frame:
            if rng.random() < miss_rate:
                continue
            obs.pos = np.asarray(obs.pos, dtype=float) + rng.normal(
                scale=pose_sigma, size=np.asarray(obs.pos).shape
            )
            kept.append(obs)
        frame[:] = kept
    return out


def run(
    data_dir: str = "data/tracking_bench_long",
    expert_dir: str = "checkpoints/category_pretrain_experts",
    projection_dir: str = "checkpoints/phase45_projection",
    graph_model: str = "eval/results/pss_vs_transformer_v2/graph_matcher_tactile.pt",
    sequences: int = 60,
    seed: int = 0,
    device: str = "cpu",
) -> dict:
    data_dir = Path(data_dir)
    index = load_index(data_dir)
    all_entries = [e for e in index["entries"] if e.get("kind") == "occlusion"]
    by_subset: dict[str, list] = {}
    for e in all_entries:
        by_subset.setdefault(e.get("subset", "unknown"), []).append(e)
    order = [s for s in SUBSETS if s in by_subset] + [s for s in by_subset if s not in SUBSETS]
    per_subset = max(1, sequences // max(1, len(order)))
    entries = []
    for subset in order:
        entries.extend(by_subset[subset][:per_subset])
    if len(entries) < sequences:
        seen = {e["id"] for e in entries}
        for subset in order:
            for e in by_subset[subset]:
                if len(entries) >= sequences:
                    break
                if e["id"] not in seen:
                    entries.append(e)
                    seen.add(e["id"])
    entries = entries[:sequences]

    graph = load_graph(graph_model, device)
    include_tactile, include_dino = token_flags(int(graph.node_proj.in_features))
    models = load_perception_models(
        expert_dir, projection_dir, device=device, dino=include_dino
    )
    graph_reid = GraphMatcherReID(
        graph, device=device, include_tactile=include_tactile, include_dino=include_dino
    )
    config = TrackerConfig(metric=graph_reid, metric_max_cost=10.0)

    rows = []
    for si, entry in enumerate(entries):
        sample = load_sample(data_dir / entry["path"])
        frames_by_t = []
        noise = PoseNoise(seed=seed * 1000 + si, enabled=True)
        for t in range(sample["rgb"].shape[0]):
            frames_by_t.append(extract_frame(sample, t, models, noise))
        for name, miss, sigma in CONDITIONS:
            corrupted = _corrupt(frames_by_t, miss, sigma, seed=seed * 1000 + si)
            metrics = _run_tracker(corrupted, config)
            rows.append(
                {
                    "sample": entry["id"],
                    "subset": entry.get("subset", "unknown"),
                    "condition": name,
                    **metrics,
                }
            )
        print(f"[{si + 1}/{len(entries)}] {entry['id']}")

    results = {}
    for name, _, _ in CONDITIONS:
        per = [r for r in rows if r["condition"] == name]
        results[name] = {
            "overall": _retention_from_rows(per),
            "per_subset": {
                subset: _retention_from_rows([r for r in per if r["subset"] == subset])
                for subset in SUBSETS
            },
        }
    return {
        "data_dir": str(data_dir),
        "sequences": len(entries),
        "conditions": [{"name": n, "miss_rate": m, "pose_sigma": s} for n, m, s in CONDITIONS],
        "results": results,
        "rows": rows,
    }


def _retention_from_rows(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0, "id_retention": 0.0, "id_switches": 0.0, "reappearance": 0.0}
    return {
        "n": len(rows),
        "id_retention": float(np.mean([r["id_retention"] for r in rows])),
        "id_switches": float(np.mean([r["id_switches"] for r in rows])),
        "reappearance": float(np.mean([r["reappearance"] for r in rows])),
    }


def _report(m: dict) -> str:
    lines = [
        "# Claim-gap E7: binding under non-oracle detection noise",
        "",
        f"> data: `{m['data_dir']}`, sequences: {m['sequences']}",
        "",
        "| condition | miss rate | pose sigma | ID retention | switches | reappearance |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    by_name = {c["name"]: c for c in m["conditions"]}
    for name, row in m["results"].items():
        c = by_name[name]
        o = row["overall"]
        lines.append(
            f"| {name} | {c['miss_rate']:.2f} | {c['pose_sigma']:.3f} | "
            f"{o['id_retention']:.3f} | {o['id_switches']:.2f} | {o['reappearance']:.3f} |"
        )
    lines += [
        "",
        "## Per subset",
        "",
        "| condition | mixed | samecat | near-identical |",
        "|---|---:|---:|---:|",
    ]
    for name, row in m["results"].items():
        per = row["per_subset"]
        lines.append(
            f"| {name} | {per['mixed']['id_retention']:.3f} | "
            f"{per['samecat']['id_retention']:.3f} | "
            f"{per['near_identical']['id_retention']:.3f} |"
        )
    lines += [
        "",
        "## How to read this",
        "",
        "This is a stress test of the state layer's input, not a detector claim.",
        "A large drop at high miss rates means the binding contract is oracle-dependent;",
        "a flat curve means the tracker tolerates detector noise.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tracking_bench_long")
    ap.add_argument("--experts", default="checkpoints/category_pretrain_experts")
    ap.add_argument("--projections", default="checkpoints/phase45_projection")
    ap.add_argument("--graph", default="eval/results/pss_vs_transformer_v2/graph_matcher_tactile.pt")
    ap.add_argument("--sequences", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="reports/claim_gaps/detect_noise")
    args = ap.parse_args()
    metrics = run(
        data_dir=args.data,
        expert_dir=args.experts,
        projection_dir=args.projections,
        graph_model=args.graph,
        sequences=args.sequences,
        seed=args.seed,
        device=args.device,
    )
    metrics["generated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "REPORT.md").write_text(_report(metrics), encoding="utf-8")
    for name, row in metrics["results"].items():
        o = row["overall"]
        print(f"{name:20s} retention={o['id_retention']:.3f} switches={o['id_switches']:.2f}")
    print(f"output: {out}")


if __name__ == "__main__":
    main()
