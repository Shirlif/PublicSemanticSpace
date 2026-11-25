"""Claim-gap E2: tactile binding ablation.

C1 says the entity state layer preserves identity across occlusion. The
fair-attribution result showed that most of the re-ID gain comes from tactile,
not from the graph/state layer. This experiment measures that contribution
directly on the long-absence benchmark:

- graph+tactile: normal observations;
- graph+tactile_zero: tactile features zeroed (sensor absent);
- graph+vision_zero: vision features zeroed (control);
- simple_tracker: vision appearance only, no graph matcher;
- text_state_proxy: no persistent state.

Observations are extracted once per sequence and copied, so all variants see
the same poses and the same noise realisation.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pss._console import ensure_utf8
from pss.encoders.data import load_index
from pss.eval.baselines import TextStateTracker
from pss.memory import SceneGraph, Tracker, TrackerConfig
from pss.memory.graph_matcher import GraphMatcherReID, load_graph
from pss.memory.learned_metric import token_flags
from pss.memory.perception import PoseNoise, extract_frame, load_perception_models
from pss.sim.dataset import load_sample

METHODS = (
    "graph_tactile",
    "graph_tactile_zero",
    "graph_vision_zero",
    "simple_tracker",
    "text_state_proxy",
)
SUBSETS = ("mixed", "samecat", "near_identical")


def _retention(gt_tracks: dict) -> dict:
    retentions, switches, reappear = [], [], []
    for seq in gt_tracks.values():
        seq = sorted(seq)
        ids = [tid for _, tid in seq]
        if not ids:
            continue
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


def _zero_modality(frames: list[list], modality: str) -> list[list]:
    out = copy.deepcopy(frames)
    for observations in out:
        for obs in observations:
            if modality in obs.raw:
                obs.raw[modality] = np.zeros_like(obs.raw[modality])
            if modality == "vision":
                # The tracker's appearance cost reads ``obs.appearance``, not
                # ``obs.raw["vision"]``; zero both to model an absent vision
                # sensor rather than only blinding the graph matcher.
                obs.appearance = np.zeros_like(obs.appearance)
    return out


def _run_tracker(frames_by_t: list[list], config) -> dict:
    tracker = Tracker(config, SceneGraph())
    gt_tracks: dict[str, list[tuple[int, str]]] = {}
    for t, observations in enumerate(frames_by_t):
        events = tracker.step(observations, t)
        for key in ("matched", "reid", "new"):
            for track_id, gt_id in events.get(key, []):
                gt_tracks.setdefault(gt_id, []).append((t, track_id))
    return _retention(gt_tracks)


def _run_text_proxy(frames_by_t: list[list]) -> dict:
    tracker = TextStateTracker()
    gt_tracks: dict[str, list[tuple[int, str]]] = {}
    for t, observations in enumerate(frames_by_t):
        events = tracker.step(observations, t)
        for key in ("matched", "new"):
            for track_id, gt_id in events.get(key, []):
                gt_tracks.setdefault(gt_id, []).append((t, track_id))
    return _retention(gt_tracks)


def _aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0, "id_retention": 0.0, "id_switches": 0.0, "reappearance": 0.0}
    return {
        "n": len(rows),
        "id_retention": float(np.mean([r["id_retention"] for r in rows])),
        "id_switches": float(np.mean([r["id_switches"] for r in rows])),
        "reappearance": float(np.mean([r["reappearance"] for r in rows])),
    }


def run(
    data_dir: str = "data/tracking_bench_long",
    expert_dir: str = "checkpoints/category_pretrain_experts",
    projection_dir: str = "checkpoints/phase45_projection",
    graph_model: str = "eval/results/pss_vs_transformer_v2/graph_matcher_tactile.pt",
    sequences: int = 12,
    seeds: int = 2,
    device: str = "cpu",
) -> dict:
    data_dir = Path(data_dir)
    index = load_index(data_dir)
    all_entries = [e for e in index["entries"] if e.get("kind") == "occlusion"]
    if not all_entries:
        raise ValueError(f"no occlusion entries in {data_dir}")
    # Balanced subset sampling: the index is grouped by subset, so taking the
    # first N would evaluate only "mixed". Sample evenly across subsets.
    by_subset: dict[str, list] = {}
    for e in all_entries:
        by_subset.setdefault(e.get("subset", "unknown"), []).append(e)
    order = [s for s in SUBSETS if s in by_subset] + [
        s for s in by_subset if s not in SUBSETS
    ]
    per_subset = max(1, sequences // max(1, len(order)))
    entries = []
    for subset in order:
        entries.extend(by_subset[subset][:per_subset])
    # Fill the remainder from the subsets that still have entries (the
    # balanced pass under-samples the larger subset when a smaller subset
    # runs out).
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
    samples = [(e, load_sample(data_dir / e["path"])) for e in entries]

    graph = load_graph(graph_model, device)
    include_tactile, include_dino = token_flags(int(graph.node_proj.in_features))
    models = load_perception_models(
        expert_dir, projection_dir, device=device, dino=include_dino
    )
    graph_reid = GraphMatcherReID(
        graph, device=device, include_tactile=include_tactile, include_dino=include_dino
    )
    graph_config = TrackerConfig(metric=graph_reid, metric_max_cost=10.0)
    simple_config = TrackerConfig()

    rows: list[dict] = []
    for si, (entry, sample) in enumerate(samples):
        for seed in range(seeds):
            frames_by_t = []
            noise = PoseNoise(seed=seed * 1000 + si, enabled=True)
            for t in range(sample["rgb"].shape[0]):
                frames_by_t.append(extract_frame(sample, t, models, noise))

            variants = {
                "graph_tactile": frames_by_t,
                "graph_tactile_zero": _zero_modality(frames_by_t, "tactile"),
                "graph_vision_zero": _zero_modality(frames_by_t, "vision"),
                "simple_tracker": frames_by_t,
                "text_state_proxy": frames_by_t,
            }
            for name in METHODS:
                frames = variants[name]
                if name == "simple_tracker":
                    metrics = _run_tracker(frames, simple_config)
                elif name == "text_state_proxy":
                    metrics = _run_text_proxy(frames)
                else:
                    metrics = _run_tracker(frames, graph_config)
                rows.append(
                    {
                        "sample": entry["id"],
                        "subset": entry.get("subset", "unknown"),
                        "seed": seed,
                        "method": name,
                        **metrics,
                    }
                )
        print(f"[{si + 1}/{len(samples)}] {entry['id']}")

    results = {}
    for method in METHODS:
        per = [r for r in rows if r["method"] == method]
        results[method] = {
            "overall": _aggregate(per),
            "per_subset": {
                subset: _aggregate([r for r in per if r["subset"] == subset])
                for subset in SUBSETS
            },
        }

    tactile = results["graph_tactile"]["overall"]["id_retention"]
    return {
        "data_dir": str(data_dir),
        "sequences": len(samples),
        "seeds": seeds,
        "include_tactile": include_tactile,
        "include_dino": include_dino,
        "results": results,
        "rows": rows,
        "deltas": {
            "tactile_contribution": tactile
            - results["graph_tactile_zero"]["overall"]["id_retention"],
            "vision_contribution": tactile
            - results["graph_vision_zero"]["overall"]["id_retention"],
            "graph_contribution": tactile
            - results["simple_tracker"]["overall"]["id_retention"],
            "state_contribution": tactile
            - results["text_state_proxy"]["overall"]["id_retention"],
        },
    }


def _report(m: dict) -> str:
    lines = [
        "# Claim-gap E2: tactile binding ablation",
        "",
        f"> data: `{m['data_dir']}`, sequences: {m['sequences']}, seeds: {m['seeds']}",
        f"> graph includes tactile={m['include_tactile']}, dino={m['include_dino']}",
        "",
        "| variant | ID retention | switches | reappearance |",
        "|---|---:|---:|---:|",
    ]
    for name in METHODS:
        row = m["results"][name]["overall"]
        lines.append(
            f"| {name} | {row['id_retention']:.3f} | {row['id_switches']:.2f} | "
            f"{row['reappearance']:.3f} |"
        )
    lines += [
        "",
        "## Contribution deltas (ID retention)",
        "",
        "| delta | value |",
        "|---|---:|",
    ]
    for name, value in m["deltas"].items():
        lines.append(f"| {name} | {value:+.3f} |")
    lines += [
        "",
        "## Per subset",
        "",
        "| variant | mixed | samecat | near-identical |",
        "|---|---:|---:|---:|",
    ]
    for name in METHODS:
        per = m["results"][name]["per_subset"]
        lines.append(
            f"| {name} | {per['mixed']['id_retention']:.3f} | "
            f"{per['samecat']['id_retention']:.3f} | "
            f"{per['near_identical']['id_retention']:.3f} |"
        )
    lines += [
        "",
        "## How to read this",
        "",
        "`tactile_contribution` is the drop when the tactile channel is zeroed with the",
        "same tracker. If it is large, C1 must say: the state layer provides the persistence",
        "mechanism, but the identity signal is carried by tactile. `state_contribution` is",
        "the gain over the one-frame text proxy; it isolates persistence from perception.",
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
    ap.add_argument("--sequences", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="reports/claim_gaps/tactile_binding")
    args = ap.parse_args()
    metrics = run(
        data_dir=args.data,
        expert_dir=args.experts,
        projection_dir=args.projections,
        graph_model=args.graph,
        sequences=args.sequences,
        seeds=args.seeds,
        device=args.device,
    )
    metrics["generated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "REPORT.md").write_text(_report(metrics), encoding="utf-8")
    for name in METHODS:
        row = metrics["results"][name]["overall"]
        print(f"{name:22s} retention={row['id_retention']:.3f} switches={row['id_switches']:.2f}")
    for name, value in metrics["deltas"].items():
        print(f"{name:22s} {value:+.3f}")
    print(f"output: {out}")


if __name__ == "__main__":
    main()
