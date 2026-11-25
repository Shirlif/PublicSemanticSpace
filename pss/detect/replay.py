"""Offline PSS replay over a bundle's detection table (T17).

The PSS process consumes only ``obs_detections.parquet``; it never opens
``gt.parquet``.  The replay writes an entity-assignment trace and the per-call
``pss_trace`` table that the metrics and cost layers read.
"""

from __future__ import annotations

import time
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
from pss.runtime.layers import EntityLayer
from pss.runtime.runtime import PSSRuntime


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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(bundle_dir)
    detections = pd.read_parquet(root / "obs_detections.parquet")
    if detections.empty:
        empty = pd.DataFrame(columns=["step", "t_s", "gt_id", "pred_id"])
        return empty, pd.DataFrame(columns=list(TRACE_SCHEMA.names))

    identity_ids = sorted(str(v) for v in detections["emb_ref"].dropna().unique())
    index = {identity: i for i, identity in enumerate(identity_ids)}

    runtime = (
        _real_runtime()
        if "emb" in detections.columns and detections["emb"].notna().any()
        else _oracle_runtime()
    )

    trace_rows: list[dict[str, Any]] = []
    pss_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    relation_rows: list[dict[str, Any]] = []
    predicates, tolerance = _relation_config(root)
    frames = list(detections.groupby("step", sort=True))
    for frame_idx, (step, group) in enumerate(frames):
        step = int(step)
        t_s = float(group.iloc[0]["t_s"])
        observations = []
        for _, row in group.iterrows():
            identity = str(row["emb_ref"])
            embedding = row.get("emb")
            if embedding is not None and len(embedding) > 0:
                appearance = np.asarray(embedding, dtype=float)
            else:
                appearance = _one_hot(index[identity], len(identity_ids))
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

        started = time.perf_counter()
        result = runtime.ingest(observations, frame_idx)
        latency_ms = (time.perf_counter() - started) * 1000.0

        assignments: dict[str, str] = {}
        for kind in ("matched", "new", "reid"):
            for track_id, identity in result.get("events", {}).get(kind, []):
                if identity is not None:
                    assignments[str(identity)] = str(track_id)
        for identity, track_id in assignments.items():
            trace_rows.append(
                {"step": step, "t_s": t_s, "gt_id": identity, "pred_id": track_id}
            )

        pss_rows.append(
            {
                "t_s": t_s,
                "layer": "entity",
                "latency_ms": float(latency_ms),
                "state_version": int(result.get("version", 0)),
                "entities_n": int(len(runtime.entities.graph.tracks)),
                "facts_n": int(len(runtime.state.facts)),
                "expansions": 0,
            }
        )
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
        if predicates and len(frame_tracks) >= 2:
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

    trace = pd.DataFrame(trace_rows, columns=["step", "t_s", "gt_id", "pred_id"])
    pss = pd.DataFrame(pss_rows, columns=list(TRACE_SCHEMA.names))
    preds = pd.DataFrame(pred_rows, columns=list(PRED_STATES_SCHEMA.names))
    relations = pd.DataFrame(relation_rows, columns=list(RELATION_TRACE_SCHEMA.names))
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
            pa.Table.from_pylist(pss.to_dict(orient="records"), schema=TRACE_SCHEMA),
            root / "pss_trace.parquet",
            compression="zstd",
        )
    return trace, pss
