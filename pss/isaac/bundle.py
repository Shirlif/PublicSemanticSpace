"""Episode bundle writer (T09).

A bundle is the evaluation atom: simulation writes tables, metrics read them
offline.  The W1 bundle already carries the full table set from the design
(``PSS-IsaacSim-评测平台设计.md`` §3.6) -- tables that cannot be produced yet
are written empty, with their schema, so downstream readers never branch on
"W1 vs later".
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

BUNDLE_VERSION = "1.0"

OBS_SCHEMA = pa.schema(
    [
        ("step", pa.int64()),
        ("t_s", pa.float64()),
        ("camera", pa.string()),
        ("image_ref", pa.string()),
        ("depth_ref", pa.string()),
        ("robot_q", pa.list_(pa.float64())),
        ("ee_pose", pa.list_(pa.float64())),
        ("camera_pose", pa.list_(pa.float64())),
    ]
)

GT_SCHEMA = pa.schema(
    [
        ("physics_step", pa.int64()),
        ("t_s", pa.float64()),
        ("instance_id", pa.string()),
        ("class", pa.string()),
        ("twin_group", pa.string()),
        ("pose_xyz", pa.list_(pa.float64())),
        ("pose_quat", pa.list_(pa.float64())),
        ("bbox_min", pa.list_(pa.float64())),
        ("bbox_max", pa.list_(pa.float64())),
        ("bbox_3d", pa.list_(pa.float64())),
        ("visibility_ratio", pa.float64()),
        ("occluded_by", pa.list_(pa.string())),
        ("occluder_of", pa.list_(pa.string())),
        ("contact_points", pa.list_(pa.string())),
        ("net_force_n", pa.float64()),
        ("in_contact_with", pa.list_(pa.string())),
        ("grasped_by", pa.string()),
        ("q", pa.list_(pa.float64())),
        ("qd", pa.list_(pa.float64())),
        ("tau", pa.list_(pa.float64())),
        ("relations", pa.list_(pa.string())),
    ]
)

DETECTIONS_SCHEMA = pa.schema(
    [
        ("step", pa.int64()),
        ("t_s", pa.float64()),
        ("camera", pa.string()),
        ("det_id", pa.string()),
        ("bbox_xyxy", pa.list_(pa.float64())),
        ("score", pa.float64()),
        ("class_pred", pa.string()),
        ("pose_xyz", pa.list_(pa.float64())),
        ("pose_quat", pa.list_(pa.float64())),
        ("emb_ref", pa.string()),
    ]
)

ACTIONS_SCHEMA = pa.schema(
    [
        ("step", pa.int64()),
        ("t_s", pa.float64()),
        ("action_type", pa.string()),
        ("payload_json", pa.string()),
        ("executor_status", pa.string()),
        ("rejected_reason", pa.string()),
    ]
)

EVENTS_SCHEMA = pa.schema(
    [
        ("t_s", pa.float64()),
        ("kind", pa.string()),
        ("source", pa.string()),
        ("detail_json", pa.string()),
    ]
)

TRACE_SCHEMA = pa.schema(
    [
        ("t_s", pa.float64()),
        ("layer", pa.string()),
        ("latency_ms", pa.float64()),
        ("state_version", pa.int64()),
        ("entities_n", pa.int64()),
        ("facts_n", pa.int64()),
        ("expansions", pa.int64()),
    ]
)

ROBOT_STATE_SCHEMA = pa.schema(
    [
        ("physics_step", pa.int64()),
        ("t_s", pa.float64()),
        ("base_pose", pa.list_(pa.float64())),
        ("q", pa.list_(pa.float64())),
        ("qd", pa.list_(pa.float64())),
        ("tau", pa.list_(pa.float64())),
    ]
)

ENTITY_TRACE_SCHEMA = pa.schema(
    [
        ("step", pa.int64()),
        ("t_s", pa.float64()),
        ("gt_id", pa.string()),
        ("pred_id", pa.string()),
    ]
)

TABLES: dict[str, tuple[str, pa.Schema]] = {
    "obs_frames": ("obs_frames.parquet", OBS_SCHEMA),
    "obs_detections": ("obs_detections.parquet", DETECTIONS_SCHEMA),
    "gt": ("gt.parquet", GT_SCHEMA),
    "robot_state": ("robot_state.parquet", ROBOT_STATE_SCHEMA),
    "entity_trace": ("entity_trace.parquet", ENTITY_TRACE_SCHEMA),
    "actions": ("actions.parquet", ACTIONS_SCHEMA),
    "events": ("events.parquet", EVENTS_SCHEMA),
    "pss_trace": ("pss_trace.parquet", TRACE_SCHEMA),
}


def _write_table(path: Path, rows: Iterable[Mapping[str, Any]], schema: pa.Schema) -> int:
    table = pa.Table.from_pylist(list(rows), schema=schema)
    pq.write_table(table, path, compression="zstd")
    return table.num_rows


def write_bundle(
    out_dir: str | Path,
    *,
    episode_id: str,
    spec_id: str,
    spec_hash: str,
    device_class: str,
    seed: int,
    stack: Mapping[str, Any],
    tables: Mapping[str, Iterable[Mapping[str, Any]]] | None = None,
    variant: str = "full",
    detector_tier: str = "oracle",
    outcome: Mapping[str, Any] | None = None,
    events: Iterable[Mapping[str, Any]] | None = None,
    faults_injected: int = 0,
) -> dict[str, Any]:
    """Write manifest + six parquet tables.  Missing tables are written empty."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    tables = dict(tables or {})
    if events is not None:
        tables["events"] = events

    files: dict[str, str] = {}
    counts: dict[str, int] = {}
    for key, (filename, schema) in TABLES.items():
        rows = tables.get(key, [])
        counts[key] = _write_table(root / filename, rows, schema)
        files[key] = filename

    now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest = {
        "bundle_version": BUNDLE_VERSION,
        "episode_id": episode_id,
        "created_utc": now,
        "spec": {"id": spec_id, "hash": spec_hash},
        "stack": dict(stack),
        "variant": variant,
        "detector_tier": detector_tier,
        "device_class": device_class,
        "seed": seed,
        "outcome": dict(outcome or {"task_success": None, "steps": None, "wall_s": None}),
        "counts": {
            "obs_frames": counts["obs_frames"],
            "gt_frames": counts["gt"],
            "robot_frames": counts.get("robot_state", 0),
            "faults_injected": int(faults_injected),
            "replans": 0,
        },
        "files": files,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def read_manifest(bundle_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(bundle_dir) / "manifest.json").read_text(encoding="utf-8"))
