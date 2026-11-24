"""Phase 6 evaluation: tracking metrics, working memory, scaling.

The primary benchmark is ID retention on occlusion sequences. Episodes are a
stress test because scripted lift/place actions teleport objects.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pss.encoders.data import load_index
from pss.sim.dataset import load_sample

from .perception import PoseNoise, extract_frame, load_perception_models
from .scene_graph import SceneGraph, Track
from .tracking import Tracker, TrackerConfig
from .working import AttentionScheduler, WorkingMemory

METRIC_KEYS = (
    "id_retention",
    "id_switches",
    "reappearance_success",
    "wake_latency_frames",
    "id_inflation",
)


def evaluate_sample(sample: dict, models, noise: PoseNoise, config: TrackerConfig | None = None) -> dict:
    tracker = Tracker(config or TrackerConfig(), SceneGraph())
    gt_tracks: dict[str, list[tuple[int, str]]] = {}
    new_tracks = 0
    frames = int(sample["rgb"].shape[0])
    recognition = {"total": 0, "correct": 0, "unknown": 0, "conflict": 0}

    for t in range(frames):
        observations = extract_frame(sample, t, models, noise)
        for obs in observations:
            recognition["total"] += 1
            if obs.gt_category is not None:
                if obs.unknown or obs.category is None:
                    recognition["unknown"] += 1
                elif obs.category == obs.gt_category:
                    recognition["correct"] += 1
            recognition["conflict"] += int(obs.conflict)
        events = tracker.step(observations, t)
        new_tracks += len(events["new"])
        for track_id, gt_id in events["matched"] + events["reid"]:
            gt_tracks.setdefault(gt_id, []).append((t, track_id))

    retentions, switches, reappear, latencies = [], [], [], []
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
                latencies.append(t1 - t0 - 1)

    return {
        "frames": frames,
        "objects": len(gt_tracks),
        "id_retention": float(np.mean(retentions)) if retentions else 0.0,
        "id_switches": float(np.mean(switches)) if switches else 0.0,
        "reappearance_success": float(np.mean(reappear)) if reappear else 1.0,
        "wake_latency_frames": float(np.mean(latencies)) if latencies else 0.0,
        "new_tracks": int(new_tracks),
        "id_inflation": new_tracks / max(1, len(gt_tracks)),
        "recognition": recognition,
        "graph": tracker.graph,
    }


def aggregate_recognition(results: list[dict]) -> dict:
    keys = ("total", "correct", "unknown", "conflict")
    totals = {
        key: int(sum(r["recognition"][key] for r in results)) for key in keys
    }
    total = max(1, totals["total"])
    known = max(0, totals["total"] - totals["unknown"])
    return {
        **totals,
        "accuracy": totals["correct"] / total,
        "known_accuracy": totals["correct"] / known if known else 0.0,
        "unknown_rate": totals["unknown"] / total,
        "conflict_rate": totals["conflict"] / total,
    }


def aggregate(results: list[dict]) -> dict:
    return {
        key: float(np.mean([r[key] for r in results])) if results else 0.0
        for key in METRIC_KEYS
    }


def _unit(rng: np.random.Generator, d: int = 128) -> np.ndarray:
    v = rng.normal(size=d)
    return v / np.linalg.norm(v)


def _synthetic_tracks(n: int, seed: int = 0) -> list[Track]:
    rng = np.random.default_rng(seed)
    tracks = []
    for i in range(n):
        raw = {
            "vision": _unit(rng),
            "tactile": _unit(rng),
            "geometry": _unit(rng),
        }
        tracks.append(
            Track(
                id=f"syn_{i:03d}",
                pos=np.zeros(3),
                quat=np.array([1.0, 0.0, 0.0, 0.0]),
                appearance=raw["vision"],
                raw=raw,
                category=None,
                confidence=1.0,
            )
        )
    return tracks


def working_memory_benchmark(tracks: list[Track], capacity: int = 4, seed: int = 0) -> dict:
    usable = [t for t in tracks if t.raw]
    if len(usable) < capacity + 2:
        usable = usable + _synthetic_tracks(capacity + 2 - len(usable), seed=seed)
    wm = WorkingMemory(AttentionScheduler(capacity=capacity))
    recalls, latencies = [], []
    for target in usable:
        task_scores = {t.id: (1.0 if t.id == target.id else 0.0) for t in usable}
        start = time.perf_counter()
        selected = wm.set_scene(usable, task_scores=task_scores)
        latencies.append((time.perf_counter() - start) * 1000.0)
        if target.id in selected:
            recalls.append(wm.recall(target.id, "vision"))
    return {
        "capacity": capacity,
        "n_tracks": len(usable),
        "focus_switch_recall": float(np.mean([c > 0.999 for c in recalls])) if recalls else 0.0,
        "mean_recall_cosine": float(np.mean(recalls)) if recalls else 0.0,
        "latency_ms_p50": float(np.percentile(latencies, 50)) if latencies else 0.0,
        "latency_ms_p95": float(np.percentile(latencies, 95)) if latencies else 0.0,
    }


def scaling_benchmark(n: int = 200, active_k: int = 4, repeats: int = 50, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    graph = SceneGraph()
    for i in range(n):
        graph.add(
            Track(
                id=f"t{i:04d}",
                pos=rng.uniform(-0.5, 0.5, size=3),
                quat=np.array([1.0, 0.0, 0.0, 0.0]),
                confidence=1.0,
            )
        )
    active = graph.all()[:active_k]
    start = time.perf_counter()
    for _ in range(repeats):
        for track in active:
            graph.query_near(track.pos, 0.15)
    elapsed = (time.perf_counter() - start) / max(1, repeats)
    return {
        "n_entities": n,
        "active_k": active_k,
        "pairs_active": active_k * (active_k - 1) // 2,
        "pairs_all": n * (n - 1) // 2,
        "query_ms_per_frame": elapsed * 1000.0,
    }


def _write_report(path: Path, metrics: dict) -> None:
    occ = metrics["tracking"]["occlusion"]
    eps = metrics["tracking"]["episodes"]
    agg = metrics["tracking"]["aggregate"]
    wm = metrics["working_memory"]
    sc = metrics["scaling"]

    def rows(items):
        out = []
        for r in items:
            out.append(
                f"| {r['sample']} | {r['objects']} | {r['frames']} | "
                f"{r['id_retention']:.3f} | {r['id_switches']:.2f} | "
                f"{r['reappearance_success']:.3f} | {r['wake_latency_frames']:.1f} | "
                f"{r['id_inflation']:.2f} |"
            )
        return out

    lines = [
        "# Phase 6 report: scene graph, tracking, working memory",
        "",
        f"- data: `{metrics['data_dir']}`",
        f"- split: {metrics.get('split') or 'first N entries'}",
        f"- recognition weights: {metrics.get('recognition_weights_mode', 'n/a')}",
        f"- pose noise: simulated SLAM noise (white + random-walk drift)",
        "",
        "## Tracking",
        "",
        "| split | sample | objects | frames | ID retention | switches | reappear ok | wake latency | ID inflation |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines += rows(occ)
    lines += rows(eps)
    lines += [
        "",
        "Aggregate: "
        f"ID retention **{agg['id_retention']:.3f}**, "
        f"switches {agg['id_switches']:.2f}, "
        f"reappearance {agg['reappearance_success']:.3f}, "
        f"wake latency {agg['wake_latency_frames']:.1f} frames, "
        f"ID inflation {agg['id_inflation']:.2f}.",
        "",
    ]

    rec = metrics.get("recognition")
    if rec:
        if metrics.get("prototypes"):
            lines += [
                "## Recognition (Phase 5 prototype bank)",
                "",
                f"- prototype bank: `{metrics['prototypes']}`",
                f"- observations: {rec['total']}",
                f"- top-1 accuracy: **{rec['accuracy']:.3f}** "
                f"(known-only {rec['known_accuracy']:.3f})",
                f"- unknown rate: {rec['unknown_rate']:.3f}",
                f"- conflict rate: {rec['conflict_rate']:.3f}",
                "",
            ]
        else:
            lines += [
                "## Recognition",
                "",
                "Ground-truth category (no prototype bank loaded).",
                "",
            ]

    lines += [
        "## Working memory",
        "",
        f"- capacity: {wm['capacity']}, tracks: {wm['n_tracks']}",
        f"- focus switch recall: **{wm['focus_switch_recall']:.3f}** "
        f"(mean cosine {wm['mean_recall_cosine']:.4f})",
        f"- switch latency: p50 {wm['latency_ms_p50']:.3f} ms, "
        f"p95 {wm['latency_ms_p95']:.3f} ms",
        "",
        "## Scaling",
        "",
        f"- entities: {sc['n_entities']}, active: {sc['active_k']}",
        f"- pair counts: active {sc['pairs_active']} vs all {sc['pairs_all']}",
        f"- spatial query: {sc['query_ms_per_frame']:.3f} ms per frame",
        "",
        "## Caveats",
        "",
        "- Pose noise is simulated; there is no real SLAM, loop closure, or re-localization.",
        "- Appearance comes from the frozen small CNN trained on primitive categories.",
        "- Recognition uses the frozen Phase 5 prototype bank when one is supplied;",
        "  otherwise observations carry the ground-truth category for tracking only.",
        "- Occlusion sequences have static objects; episodes include teleport-like lift/place,",
        "  so episode ID retention is a stress test, not a realistic manipulation benchmark.",
        "- Fragment/decomposition hypotheses live in `pss/memory/decomposition.py`;",
        "  the tracking data used here has no fracture events, so that rule is",
        "  covered by its own demo/tests rather than by these sequences.",
        "- Working-memory recall is exact unbind of the stored raw feature; it does not measure semantic recognition.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_phase6(
    data_dir: str | Path,
    expert_dir: str | Path,
    projection_dir: str | Path,
    out_dir: str | Path,
    prototype_dir: str | Path | None = None,
    max_occlusion: int = 4,
    max_episodes: int = 2,
    seed: int = 0,
    split: str | None = None,
    weights_mode: str = "reliability",
) -> dict:
    data_dir = Path(data_dir)
    models = load_perception_models(
        expert_dir,
        projection_dir,
        prototype_dir,
        weights_mode=weights_mode,
    )
    index = load_index(data_dir)
    if split is None:
        occ_entries = [e for e in index["entries"] if e["kind"] == "occlusion"]
        ep_entries = [e for e in index["entries"] if e["kind"] == "episode"]
    else:
        occ_entries = [
            e
            for e in index["entries"]
            if e["kind"] == "occlusion" and e.get("split") == split
        ]
        ep_entries = [
            e
            for e in index["entries"]
            if e["kind"] == "episode" and e.get("split") == split
        ]
    occ_entries = occ_entries[:max_occlusion]
    ep_entries = ep_entries[:max_episodes]

    occ_results, ep_results = [], []
    last_graph = SceneGraph()
    for i, entry in enumerate(occ_entries):
        sample = load_sample(data_dir / entry["path"])
        result = evaluate_sample(sample, models, PoseNoise(seed=seed + i))
        occ_results.append({**result, "sample": entry["id"]})
        last_graph = result["graph"]
    for i, entry in enumerate(ep_entries):
        sample = load_sample(data_dir / entry["path"])
        result = evaluate_sample(sample, models, PoseNoise(seed=seed + 1000 + i))
        ep_results.append({**result, "sample": entry["id"]})
        last_graph = result["graph"]

    all_results = occ_results + ep_results
    metrics = {
        "data_dir": str(data_dir),
        "experts": str(expert_dir),
        "projections": str(projection_dir),
        "prototypes": str(prototype_dir) if prototype_dir is not None else None,
        "recognition_weights_mode": weights_mode,
        "split": split,
        "recognition": aggregate_recognition(all_results),
        "tracking": {
            "occlusion": [{k: v for k, v in r.items() if k != "graph"} for r in occ_results],
            "episodes": [{k: v for k, v in r.items() if k != "graph"} for r in ep_results],
            "aggregate": aggregate(all_results),
        },
        "working_memory": working_memory_benchmark(last_graph.all(), capacity=4, seed=seed),
        "scaling": scaling_benchmark(n=200, active_k=4, repeats=50, seed=seed),
        "note": "Simulated pose noise; primitive-proxy data; smoke test.",
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    last_graph.save(out / "scene_snapshot.json")
    (out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_report(out / "REPORT.md", metrics)
    return metrics
