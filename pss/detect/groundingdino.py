"""GroundingDINO Swin-T detection tier (T19).

Runs offline over a bundle's RGB frames, writes detector-like rows into
``obs_detections.parquet``.  3D pose comes from the depth frame; appearance is
a DINOv2 CLS embedding of the crop (the tracker's ReID channel).

The PSS process consumes only the resulting detections, never GT.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from pss.isaac.bundle import DETECTIONS_SCHEMA
from pss.isaac.visibility import quat_rotate
from pss.spec import DEFAULT_REGISTRY, load_spec

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = Path(
    os.environ.get("PSS_GDINO_DIR", REPO_ROOT / "_external/models/grounding-dino-tiny")
)
DINOV2_ID = "facebook/dinov2-small"


def _camera_params(bundle_dir: Path, camera_name: str) -> tuple[int, int, float]:
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    spec_id = manifest.get("spec", {}).get("id")
    spec_hash = manifest.get("spec", {}).get("hash")
    if DEFAULT_REGISTRY.exists():
        for line in DEFAULT_REGISTRY.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("spec_id") != spec_id or entry.get("sha256") != spec_hash:
                continue
            path = Path(entry["path"])
            if not path.is_absolute():
                path = DEFAULT_REGISTRY.parents[1] / path
            if path.exists():
                spec = load_spec(path)
                for camera in spec.cameras:
                    if camera.get("name") == camera_name:
                        return (
                            int(camera["width"]),
                            int(camera["height"]),
                            float(camera.get("fov_deg", 60.0)),
                        )
    return 640, 480, 60.0


def _backproject(
    u: float,
    v: float,
    depth_m: float,
    camera_pose: list[float],
    width: int,
    height: int,
    fov_deg: float,
) -> list[float]:
    tan_h = math.tan(math.radians(fov_deg) / 2.0)
    tan_v = tan_h * (height / width)
    fx = (width / 2.0) / tan_h
    fy = (height / 2.0) / tan_v
    local = (
        (u - width / 2.0) * depth_m / fx,
        (height / 2.0 - v) * depth_m / fy,
        -depth_m,
    )
    q = (
        (camera_pose[3], camera_pose[4], camera_pose[5], camera_pose[6])
        if len(camera_pose) >= 7
        else (1.0, 0.0, 0.0, 0.0)
    )
    rotated = quat_rotate(q, local)
    return [
        camera_pose[0] + rotated[0],
        camera_pose[1] + rotated[1],
        camera_pose[2] + rotated[2],
    ]


def _backproject_plane(
    u: float,
    v: float,
    camera_pose: list[float],
    width: int,
    height: int,
    fov_deg: float,
    plane_z: float,
) -> list[float]:
    """Intersect the pixel ray with a horizontal plane (tabletop objects)."""
    tan_h = math.tan(math.radians(fov_deg) / 2.0)
    tan_v = tan_h * (height / width)
    fx = (width / 2.0) / tan_h
    fy = (height / 2.0) / tan_v
    local = ((u - width / 2.0) / fx, (height / 2.0 - v) / fy, -1.0)
    q = (
        (camera_pose[3], camera_pose[4], camera_pose[5], camera_pose[6])
        if len(camera_pose) >= 7
        else (1.0, 0.0, 0.0, 0.0)
    )
    direction = quat_rotate(q, local)
    if abs(direction[2]) < 1e-9:
        return [camera_pose[0], camera_pose[1], plane_z]
    t = (plane_z - camera_pose[2]) / direction[2]
    return [
        camera_pose[0] + t * direction[0],
        camera_pose[1] + t * direction[1],
        plane_z,
    ]


def _depth_at(depth: np.ndarray | None, u: int, v: int, radius: int = 2) -> float:
    if depth is None:
        return 0.5
    h, w = depth.shape[:2]
    v0, v1 = max(0, v - radius), min(h, v + radius + 1)
    u0, u1 = max(0, u - radius), min(w, u + radius + 1)
    patch = depth[v0:v1, u0:u1].astype(np.float32)
    valid = patch[patch > 0]
    return float(np.median(valid)) / 1000.0 if valid.size else 0.5


class GroundingDinoDetector:
    def __init__(self, model_dir: Path = DEFAULT_MODEL_DIR, device: str | None = None):
        import torch
        from transformers import (
            AutoImageProcessor,
            AutoModel,
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
        )

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(str(model_dir))
        self.model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(str(model_dir))
            .to(self.device)
            .eval()
        )
        self.dino_processor = AutoImageProcessor.from_pretrained(DINOV2_ID)
        self.dino = AutoModel.from_pretrained(DINOV2_ID).to(self.device).eval()

    def embed(self, image: Image.Image) -> list[float]:
        inputs = self.dino_processor(images=image, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            output = self.dino(**inputs)
        vector = output.last_hidden_state[:, 0].float().cpu().numpy()[0]
        norm = float(np.linalg.norm(vector))
        return (vector / norm).tolist() if norm > 0 else vector.tolist()

    def detect(
        self,
        image: Image.Image,
        prompt: str,
        *,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
    ) -> list[dict[str, Any]]:
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(
            self.device
        )
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        try:
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[image.size[::-1]],
            )[0]
        except TypeError:
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[image.size[::-1]],
            )[0]

        detections = []
        width, height = image.size
        labels = results.get("text_labels", results.get("labels"))
        for box, score, label in zip(
            results["boxes"].tolist(),
            results["scores"].tolist(),
            labels,
        ):
            u0, v0, u1, v1 = [float(x) for x in box]
            u0, v0 = max(0.0, u0), max(0.0, v0)
            u1, v1 = min(float(width), u1), min(float(height), v1)
            if u1 - u0 < 2 or v1 - v0 < 2:
                continue
            crop = image.crop((int(u0), int(v0), int(u1), int(v1)))
            detections.append(
                {
                    "bbox_xyxy": [u0, v0, u1, v1],
                    "score": float(score),
                    "class_pred": str(label).strip().lower(),
                    "emb": self.embed(crop),
                }
            )
        return detections


def detect_bundle(
    bundle_dir: str | Path,
    *,
    prompt: str | None = None,
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
    max_frames: int | None = None,
    plane_z: float | None = 0.79,
) -> list[dict[str, Any]]:
    root = Path(bundle_dir)
    obs = pd.read_parquet(root / "obs_frames.parquet")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    spec_id = manifest.get("spec", {}).get("id")
    if prompt is None:
        prompt = "cube."
        if DEFAULT_REGISTRY.exists():
            for line in DEFAULT_REGISTRY.read_text(encoding="utf-8").splitlines():
                entry = json.loads(line)
                if entry.get("spec_id") == spec_id:
                    path = Path(entry["path"])
                    if not path.is_absolute():
                        path = DEFAULT_REGISTRY.parents[1] / path
                    if path.exists():
                        classes = sorted(
                            {str(a["class"]) for a in load_spec(path).assets}
                        )
                        prompt = " ".join(f"{c}." for c in classes)
                    break

    detector = GroundingDinoDetector()
    rows: list[dict[str, Any]] = []
    frames = obs[obs["image_ref"].notna()]
    if max_frames is not None:
        frames = frames.head(max_frames)
    for frame_index, (_, frame) in enumerate(frames.iterrows()):
        image_path = root / str(frame["image_ref"])
        depth_path = root / str(frame["depth_ref"]) if pd.notna(frame["depth_ref"]) else None
        image = Image.open(image_path).convert("RGB")
        depth = (
            np.asarray(Image.open(depth_path), dtype=np.uint16)
            if depth_path is not None and depth_path.exists()
            else None
        )
        width, height, fov = _camera_params(root, str(frame["camera"]))
        camera_pose = [float(v) for v in frame["camera_pose"]]
        for det_index, det in enumerate(
            detector.detect(
                image, prompt, box_threshold=box_threshold, text_threshold=text_threshold
            )
        ):
            u0, v0, u1, v1 = det["bbox_xyxy"]
            u = int((u0 + u1) / 2)
            v = int((v0 + v1) / 2)
            if plane_z is not None:
                pos = _backproject_plane(
                    u, v, camera_pose, width, height, fov, plane_z
                )
            else:
                depth_m = _depth_at(depth, u, v)
                pos = _backproject(u, v, depth_m, camera_pose, width, height, fov)
            rows.append(
                {
                    "step": int(frame["step"]),
                    "t_s": float(frame["t_s"]),
                    "camera": str(frame["camera"]),
                    "det_id": f"gdino_{int(frame['step']):06d}_{det_index:03d}",
                    "bbox_xyxy": det["bbox_xyxy"],
                    "score": det["score"],
                    "class_pred": det["class_pred"],
                    "pose_xyz": pos,
                    "pose_quat": [1.0, 0.0, 0.0, 0.0],
                    "emb": det["emb"],
                    "emb_ref": f"gdino_{int(frame['step']):06d}_{det_index:03d}",
                }
            )
        if frame_index % 20 == 0:
            print(f"[gdino] frame {frame_index}/{len(frames)}", flush=True)
    return rows


def write_groundingdino_detections(
    bundle_dir: str | Path, rows: list[dict[str, Any]]
) -> Path:
    path = Path(bundle_dir) / "obs_detections.parquet"
    table = pa.Table.from_pylist(rows, schema=DETECTIONS_SCHEMA)
    pq.write_table(table, path, compression="zstd")
    return path
