"""Oracle detection tier (T17).

GT boxes are projected into the image and emitted as detections.  This tier is
for calibration only: it checks that the GT, the bundle contract and the
metrics agree, not that PSS can perceive.  The detection stream never reaches
``gt.parquet``; it is a separate table with detector-like rows.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pss.isaac.bundle import DETECTIONS_SCHEMA
from pss.isaac.visibility import quat_inverse, quat_rotate, transform_point
from pss.spec import DEFAULT_REGISTRY, load_spec


def _camera_params(bundle_dir: Path, camera_name: str) -> tuple[int, int, float]:
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    spec_id = manifest.get("spec", {}).get("id")
    spec_hash = manifest.get("spec", {}).get("hash")
    if DEFAULT_REGISTRY.exists():
        for line in DEFAULT_REGISTRY.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("spec_id") != spec_id or entry.get("sha256") != spec_hash:
                continue
            spec_path = Path(entry["path"])
            if not spec_path.is_absolute():
                spec_path = DEFAULT_REGISTRY.parents[1] / spec_path
            if not spec_path.exists():
                continue
            spec = load_spec(spec_path)
            for camera in spec.cameras:
                if camera.get("name") == camera_name:
                    return (
                        int(camera.get("width", 640)),
                        int(camera.get("height", 480)),
                        float(camera.get("fov_deg", 60.0)),
                    )
    return 640, 480, 60.0


def project_point(
    point: list[float],
    camera_pose: list[float],
    width: int,
    height: int,
    fov_deg: float,
) -> tuple[float, float] | None:
    q = (
        (camera_pose[3], camera_pose[4], camera_pose[5], camera_pose[6])
        if len(camera_pose) >= 7
        else (1.0, 0.0, 0.0, 0.0)
    )
    rel = (point[0] - camera_pose[0], point[1] - camera_pose[1], point[2] - camera_pose[2])
    local = quat_rotate(quat_inverse(q), rel)
    z = -local[2]
    if z <= 1e-6:
        return None
    tan_h = math.tan(math.radians(fov_deg) / 2.0)
    tan_v = tan_h * (height / width)
    u = (local[0] / (z * tan_h) * 0.5 + 0.5) * width
    v = (0.5 - local[1] / (z * tan_v) * 0.5) * height
    return u, v


def _project_bbox(
    bbox_3d: list[float],
    camera_pose: list[float],
    width: int,
    height: int,
    fov_deg: float,
) -> list[float]:
    points = [bbox_3d[i : i + 3] for i in range(0, len(bbox_3d), 3)]
    projected = [project_point(list(p), camera_pose, width, height, fov_deg) for p in points]
    projected = [p for p in projected if p is not None]
    if not projected:
        return [0.0, 0.0, 0.0, 0.0]
    us = [max(0.0, min(float(width), p[0])) for p in projected]
    vs = [max(0.0, min(float(height), p[1])) for p in projected]
    return [min(us), min(vs), max(us), max(vs)]


def detect_oracle(
    bundle_dir: str | Path,
    *,
    min_visibility: float = 0.001,
) -> list[dict[str, Any]]:
    root = Path(bundle_dir)
    obs = pd.read_parquet(root / "obs_frames.parquet")
    gt = pd.read_parquet(root / "gt.parquet")
    gt_by_step = {int(step): frame for step, frame in gt.groupby("physics_step")}

    rows: list[dict[str, Any]] = []
    for _, frame in obs.iterrows():
        step = int(frame["step"])
        instance_rows = gt_by_step.get(step)
        if instance_rows is None:
            continue
        width, height, fov = _camera_params(root, str(frame["camera"]))
        camera_pose = [float(v) for v in frame["camera_pose"]]
        for _, inst in instance_rows.iterrows():
            if float(inst["visibility_ratio"]) <= min_visibility:
                continue
            instance_id = str(inst["instance_id"])
            bbox = _project_bbox(
                [float(v) for v in inst["bbox_3d"]],
                camera_pose,
                width,
                height,
                fov,
            )
            rows.append(
                {
                    "step": step,
                    "t_s": float(frame["t_s"]),
                    "camera": str(frame["camera"]),
                    "det_id": f"oracle_{instance_id}",
                    "bbox_xyxy": bbox,
                    "score": 1.0,
                    "class_pred": str(inst["class"]),
                    "pose_xyz": [float(v) for v in inst["pose_xyz"]],
                    "pose_quat": [float(v) for v in inst["pose_quat"]],
                    "emb_ref": instance_id,
                }
            )
    return rows


def write_detections(bundle_dir: str | Path, rows: list[dict[str, Any]]) -> Path:
    path = Path(bundle_dir) / "obs_detections.parquet"
    table = pa.Table.from_pylist(rows, schema=DETECTIONS_SCHEMA)
    pq.write_table(table, path, compression="zstd")
    return path
