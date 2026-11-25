"""S6 session replay: one PSS runtime across sub-episodes (T35/T36)."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pss.isaac.bundle import TRACE_SCHEMA
from pss.memory.perception import Observation
from pss.runtime.variants import build_runtime

from .replay import _one_hot


def _state_hash(runtime) -> str:
    state = runtime.state
    payload = {
        "entities": {k: v.to_dict() for k, v in sorted(state.entities.items())},
        "facts": {
            f"{k[0]}.{k[1]}": v.to_dict() for k, v in sorted(state.facts.items())
        },
        "relations": [r.to_dict() for r in state.relations],
    }
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def replay_session(
    bundle_dirs: list[str | Path],
    *,
    rollback_test: bool = True,
    variant: str = "full",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    bundles = [Path(b) for b in bundle_dirs]
    identity_ids = set()
    detections = {}
    for bundle in bundles:
        det = pd.read_parquet(bundle / "obs_detections.parquet")
        detections[bundle] = det
        identity_ids.update(str(v) for v in det["emb_ref"].dropna().unique())
    identity_ids = sorted(identity_ids)
    index = {identity: i for i, identity in enumerate(identity_ids)}

    tier = (
        "frozen_real"
        if any(
            "emb" in det.columns and det["emb"].notna().any()
            for det in detections.values()
        )
        else "oracle"
    )
    runtime = build_runtime(variant, tier=tier)
    rows: list[dict[str, Any]] = []
    frame_offset = 0
    pss_rows: list[dict[str, Any]] = []
    for episode, bundle in enumerate(bundles):
        if variant == "no_memory" and episode > 0:
            runtime = build_runtime(variant, tier=tier)
            # make IDs unique across the session so "forgot everything" is visible
            runtime.entities.graph._next = episode * 1000 + 1
        det = detections[bundle]
        if det.empty:
            continue
        gt = pd.read_parquet(bundle / "gt.parquet")
        gt_by_step = {int(step): group for step, group in gt.groupby("physics_step")}
        frames = list(det.groupby("step", sort=True))
        for local_idx, (step, group) in enumerate(frames):
            step = int(step)
            frame_idx = frame_offset + local_idx
            t_s = float(group.iloc[0]["t_s"])
            observations = []
            for _, row in group.iterrows():
                identity = str(row["emb_ref"])
                observations.append(
                    Observation(
                        gt_id=identity,
                        pos=np.asarray(row["pose_xyz"], dtype=float),
                        quat=np.asarray(row["pose_quat"], dtype=float),
                        appearance=_one_hot(index[identity], len(identity_ids)),
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
            latency = (time.perf_counter() - started) * 1000.0
            pss_rows.append(
                {
                    "t_s": t_s,
                    "layer": "entity",
                    "latency_ms": float(latency),
                    "state_version": int(result.get("version", 0)),
                    "entities_n": int(len(runtime.entities.graph.tracks)),
                    "facts_n": int(len(runtime.state.facts)),
                    "expansions": 0,
                }
            )
            assignments: dict[str, str] = {}
            for kind in ("matched", "new", "reid"):
                for track_id, identity in result.get("events", {}).get(kind, []):
                    if identity is not None:
                        assignments[str(identity)] = str(track_id)
            for identity, track_id in assignments.items():
                track = runtime.entities.graph.get(track_id)
                if track is None:
                    continue
                gt_rows = gt_by_step.get(step)
                gt_pos = None
                if gt_rows is not None:
                    match = gt_rows[gt_rows["instance_id"].astype(str) == identity]
                    if not match.empty:
                        gt_pos = [float(v) for v in match.iloc[0]["pose_xyz"]]
                rows.append(
                    {
                        "episode": episode,
                        "frame": frame_idx,
                        "step": step,
                        "gt_id": identity,
                        "pred_id": track_id,
                        "pss_xyz": [float(v) for v in track.pos[:3]],
                        "gt_xyz": gt_pos,
                    }
                )
        frame_offset += len(frames)

    trace = pd.DataFrame(rows)
    report: dict[str, Any] = {
        "episodes": len(bundles),
        "pss_frames": len(pss_rows),
        "tracks": int(len(runtime.entities.graph.tracks)),
        "frames": int(len(trace)),
    }
    if rollback_test and runtime.entities.graph.tracks and runtime.audit_enabled:
        before_hash = _state_hash(runtime)
        before_version = int(runtime.state.version)
        track_id = sorted(runtime.entities.graph.tracks)[0]
        decision = runtime.assert_fact(
            track_id, "memory_test", "edited", source="memory_test"
        )
        after_hash = _state_hash(runtime)
        runtime.rollback(before_version)
        restored_hash = _state_hash(runtime)
        report.update(
            {
                "rollback_version": before_version,
                "rollback_changed_state": before_hash != after_hash,
                "rollback_exactness": 1.0 if restored_hash == before_hash else 0.0,
                "rollback_decision": decision,
            }
        )
    else:
        report["rollback_exactness"] = 0.0

    report["pss_trace"] = pss_rows
    return trace, report
