"""Offline PSS replay over a bundle's detection table (T17).

The PSS process consumes only ``obs_detections.parquet``; it never opens
``gt.parquet``.  The replay writes an entity-assignment trace and the per-call
``pss_trace`` table that the metrics and cost layers read.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pss.isaac.bundle import (
    ENTITY_TRACE_SCHEMA,
    PRED_STATES_SCHEMA,
    RELATION_TRACE_SCHEMA,
    TRACE_SCHEMA,
)
from pss.memory.perception import Observation
from pss.memory.scene_graph import SceneGraph
from pss.memory.tracking import Tracker, TrackerConfig
from pss.relations import RelObject, compute_relations
from pss.runtime.deadline import DeadlineGuard
from pss.runtime.layers import EntityLayer
from pss.runtime.runtime import PSSRuntime
from pss.runtime.variants import build_runtime


class _AppearanceMetric:
    """Oracle-tier identity metric: exact one-hot appearance, no pose weight."""

    @staticmethod
    def _distance(track, obs) -> float:
        a = np.asarray(track.appearance, dtype=float)
        b = np.asarray(obs.appearance, dtype=float)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na <= 1e-12 or nb <= 1e-12:
            return 1.0
        return float(1.0 - np.dot(a, b) / (na * nb))

    def cost(self, track, obs) -> float:
        return self._distance(track, obs)

    def cost_matrix(self, tracks, observations) -> np.ndarray:
        matrix = np.ones((len(tracks), len(observations)), dtype=float)
        for i, track in enumerate(tracks):
            for j, obs in enumerate(observations):
                matrix[i, j] = self._distance(track, obs)
        return matrix


def _relation_config(bundle_dir: Path) -> tuple[list[str], float]:
    import json

    from pss.spec import DEFAULT_REGISTRY, load_spec

    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    spec_id = manifest.get("spec", {}).get("id")
    spec_hash = manifest.get("spec", {}).get("hash")
    if DEFAULT_REGISTRY.exists():
        for line in DEFAULT_REGISTRY.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            if entry.get("spec_id") != spec_id or entry.get("sha256") != spec_hash:
                continue
            path = Path(entry["path"])
            if not path.is_absolute():
                path = DEFAULT_REGISTRY.parents[1] / path
            if path.exists():
                spec = load_spec(path)
                cfg = spec.relations or {}
                return (
                    list(cfg.get("predicates", [])),
                    float(cfg.get("tolerance_m", 0.02)),
                )
    return [], 0.02


def _real_runtime(*, audit: bool = False) -> PSSRuntime:
    """Pose+appearance association for learned embeddings (frozen_real)."""
    config = TrackerConfig(
        gate_radius=0.25,
        max_cost=1.0,
        w_pose=0.6,
        w_app=0.3,
        w_time=0.1,
        w_category=0.05,
        metric=_AppearanceMetric(),
        metric_max_cost=0.7,
    )
    graph = SceneGraph()
    tracker = Tracker(config, graph)
    entity_layer = EntityLayer(
        graph=graph,
        tracker=tracker,
        use_default_relations=False,
        use_default_active=False,
    )
    return PSSRuntime(
        entity_layer=entity_layer,
        audit=audit,
        uncertainty_enabled=False,
        use_default_relations=False,
        use_default_active=False,
        use_default_write=False,
    )


def _oracle_runtime(*, audit: bool = False) -> PSSRuntime:
    config = TrackerConfig(
        gate_radius=0.6,
        max_cost=0.5,
        w_pose=0.05,
        w_app=0.9,
        w_time=0.05,
        w_category=0.0,
        metric=_AppearanceMetric(),
        metric_max_cost=0.15,
        reid_pose_gate=0.6,
        reid_app_min=0.5,
        reid_w_pose=0.0,
        reid_w_app=1.0,
    )
    graph = SceneGraph()
    tracker = Tracker(config, graph)
    entity_layer = EntityLayer(
        graph=graph,
        tracker=tracker,
        use_default_relations=False,
        use_default_active=False,
    )
    return PSSRuntime(
        entity_layer=entity_layer,
        audit=audit,
        uncertainty_enabled=False,
        use_default_relations=False,
        use_default_active=False,
        use_default_write=False,
    )


def _one_hot(index: int, size: int) -> np.ndarray:
    vector = np.zeros(size, dtype=float)
    vector[index] = 1.0
    return vector


def replay_bundle(
    bundle_dir: str | Path,
    *,
    write: bool = True,
    variant: str = "full",
    shuffle_embeddings: bool = False,
    deadline_ms: float | None = None,
    deadline_reserve_ms: float = 1.5,
    slow_stage_ms: float = 0.0,
    slow_burn_ms: float = 0.0,
    slow_every: int = 1,
    appearance_provider: Callable[[int, int, pd.DataFrame], list[np.ndarray]] | None = None,
    frame_hook: Callable[[int, int], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay a bundle through the runtime.

    ``deadline_ms`` enables the deadline guard: the caller's wait is bounded,
    a timed-out frame returns empty events, and the worker is joined before
    the next frame.  When enabled, the returned ``pss`` frame carries
    ``degraded`` and ``frame_ms`` columns and ``pss.attrs["deadline_guard"]``
    holds the guard summary.  The written parquet schema is unchanged.

    ``slow_stage_ms`` / ``slow_burn_ms`` / ``slow_every`` are deterministic test
    hooks for the guard; normal replay leaves them at zero.  ``slow_stage_ms``
    sleeps (releases the GIL), ``slow_burn_ms`` busy-loops (holds the GIL).

    ``frame_hook`` is called once per frame, after any timed-out worker has
    been joined and before observations are built; it can swap the active
    perception expert for that frame.  ``appearance_provider`` then supplies
    the embeddings actually fed to PSS for that frame (one row per detection,
    in detection-table order).  When ``appearance_provider`` is ``None`` the
    bundle's ``emb`` column (or the oracle one-hot fallback) is used, exactly
    as before.  Neither hook may read ``gt.parquet``.
    """
    root = Path(bundle_dir)
    detections = pd.read_parquet(root / "obs_detections.parquet")
    if detections.empty:
        empty = pd.DataFrame(columns=["step", "t_s", "gt_id", "pred_id"])
        return empty, pd.DataFrame(columns=list(TRACE_SCHEMA.names))

    identity_ids = sorted(str(v) for v in detections["emb_ref"].dropna().unique())
    index = {identity: i for i, identity in enumerate(identity_ids)}

    tier = (
        "frozen_real"
        if "emb" in detections.columns and detections["emb"].notna().any()
        else "oracle"
    )
    runtime = build_runtime(variant, tier=tier)
    guard = (
        DeadlineGuard(deadline_ms, reserve_ms=deadline_reserve_ms)
        if deadline_ms is not None
        else None
    )
    base_ingest = runtime.ingest
    if slow_stage_ms > 0 or slow_burn_ms > 0:

        def ingest_fn(observations, frame):
            slow_frame = slow_every <= 1 or frame % slow_every == 0
            if slow_frame and slow_stage_ms > 0:
                time.sleep(float(slow_stage_ms) / 1000.0)
            if slow_frame and slow_burn_ms > 0:
                burn_until = time.perf_counter() + float(slow_burn_ms) / 1000.0
                while time.perf_counter() < burn_until:
                    pass
            return base_ingest(observations, frame)
    else:
        ingest_fn = base_ingest
    last_version = 0
    last_entities = 0
    last_facts = 0
    relations_skipped = 0
    rng = np.random.default_rng(0)

    trace_rows: list[dict[str, Any]] = []
    pss_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    relation_rows: list[dict[str, Any]] = []
    predicates, tolerance = _relation_config(root)
    frames = list(detections.groupby("step", sort=True))
    for frame_idx, (step, group) in enumerate(frames):
        step = int(step)
        t_s = float(group.iloc[0]["t_s"])
        frame_start = time.perf_counter()
        if guard is not None:
            guard.wait_pending()
        if frame_hook is not None:
            frame_hook(frame_idx, step)
        frame_rows = list(group.iterrows())
        if appearance_provider is not None:
            provided = appearance_provider(frame_idx, step, group)
            if len(provided) != len(frame_rows):
                raise ValueError(
                    f"appearance_provider returned {len(provided)} rows for "
                    f"{len(frame_rows)} detections at frame {frame_idx}"
                )
            appearances = [np.asarray(vector, dtype=float) for vector in provided]
        else:
            appearances = []
            for _, row in frame_rows:
                identity = str(row["emb_ref"])
                embedding = row.get("emb")
                if embedding is not None and len(embedding) > 0:
                    appearances.append(np.asarray(embedding, dtype=float))
                else:
                    appearances.append(_one_hot(index[identity], len(identity_ids)))
        if shuffle_embeddings and len(appearances) > 1:
            appearances = list(rng.permutation(np.stack(appearances)))
        observations = []
        for (_, row), appearance in zip(frame_rows, appearances):
            identity = str(row["emb_ref"])
            observations.append(
                Observation(
                    gt_id=identity,
                    pos=np.asarray(row["pose_xyz"], dtype=float),
                    quat=np.asarray(row["pose_quat"], dtype=float),
                    appearance=appearance,
                    raw={
                        "vision": np.zeros(4),
                        "geometry": np.zeros(4),
                        "tactile": np.zeros(4),
                    },
                    confidence=float(row["score"]),
                    category=str(row["class_pred"]),
                    frame=frame_idx,
                    gt_category=str(row["class_pred"]),
                )
            )

        if guard is not None:
            result, timed_out, latency_ms = guard.call(
                lambda obs=observations, fi=frame_idx: ingest_fn(obs, fi),
                frame=frame_idx,
            )
        else:
            started = time.perf_counter()
            result = ingest_fn(observations, frame_idx)
            latency_ms = (time.perf_counter() - started) * 1000.0
            timed_out = False
        if timed_out:
            result = {
                "events": {},
                "version": last_version,
                "uncertainty": None,
                "degraded": True,
            }

        assignments: dict[str, str] = {}
        for kind in ("matched", "new", "reid"):
            for track_id, identity in result.get("events", {}).get(kind, []):
                if identity is not None:
                    assignments[str(identity)] = str(track_id)
        for identity, track_id in assignments.items():
            trace_rows.append(
                {"step": step, "t_s": t_s, "gt_id": identity, "pred_id": track_id}
            )

        if not timed_out:
            last_version = int(result.get("version", last_version))
            last_entities = int(len(runtime.entities.graph.tracks))
            last_facts = int(len(runtime.state.facts))
            for track in runtime.entities.graph.tracks.values():
                if int(track.last_seen) == frame_idx:
                    pred_rows.append(
                        {
                            "step": step,
                            "t_s": t_s,
                            "pred_id": track.id,
                            "pos_xyz": [float(v) for v in track.pos[:3]],
                            "category": track.category,
                            "confidence": float(track.confidence),
                        }
                    )
            frame_tracks = [
                track
                for track in runtime.entities.graph.tracks.values()
                if int(track.last_seen) == frame_idx
            ]
            shed_relations = guard is not None and guard.should_shed
            if shed_relations:
                relations_skipped += 1
            if predicates and len(frame_tracks) >= 2 and not shed_relations:
                objects = [
                    RelObject(track.id, track.pos, [0.04, 0.04, 0.04])
                    for track in frame_tracks
                ]
                for subj, pred, obj in compute_relations(
                    objects, predicates, tolerance_m=tolerance
                ):
                    relation_rows.append(
                        {
                            "step": step,
                            "t_s": t_s,
                            "subj": subj,
                            "pred": pred,
                            "obj": obj,
                        }
                    )

        frame_ms = (time.perf_counter() - frame_start) * 1000.0
        pss_row = {
            "t_s": t_s,
            "layer": "entity",
            "latency_ms": float(latency_ms),
            "state_version": int(last_version),
            "entities_n": int(last_entities),
            "facts_n": int(last_facts),
            "expansions": 0,
        }
        if guard is not None:
            pss_row["degraded"] = bool(timed_out)
            pss_row["frame_ms"] = float(frame_ms)
        pss_rows.append(pss_row)

    if guard is not None:
        guard.close()

    trace = pd.DataFrame(trace_rows, columns=["step", "t_s", "gt_id", "pred_id"])
    extra = ["degraded", "frame_ms"] if guard is not None else []
    pss = pd.DataFrame(pss_rows, columns=list(TRACE_SCHEMA.names) + extra)
    preds = pd.DataFrame(pred_rows, columns=list(PRED_STATES_SCHEMA.names))
    relations = pd.DataFrame(relation_rows, columns=list(RELATION_TRACE_SCHEMA.names))
    pss.attrs["relation_rows"] = int(len(relations))
    if guard is not None:
        snapshot = guard.snapshot().to_dict()
        snapshot["relations_skipped"] = int(relations_skipped)
        pss.attrs["deadline_guard"] = snapshot
    if write:
        pq.write_table(
            pa.Table.from_pylist(trace.to_dict(orient="records"), schema=ENTITY_TRACE_SCHEMA),
            root / "entity_trace.parquet",
            compression="zstd",
        )
        pq.write_table(
            pa.Table.from_pylist(preds.to_dict(orient="records"), schema=PRED_STATES_SCHEMA),
            root / "pred_states.parquet",
            compression="zstd",
        )
        pq.write_table(
            pa.Table.from_pylist(relations.to_dict(orient="records"), schema=RELATION_TRACE_SCHEMA),
            root / "pred_relations.parquet",
            compression="zstd",
        )
        pq.write_table(
            pa.Table.from_pylist(
                pss[list(TRACE_SCHEMA.names)].to_dict(orient="records"),
                schema=TRACE_SCHEMA,
            ),
            root / "pss_trace.parquet",
            compression="zstd",
        )
    return trace, pss
