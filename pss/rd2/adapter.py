"""RD2 save/load and conversion to the simulator sample schema."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .schema import RDEpisode, RDFrame


def save_episode(path: str | Path, episode: RDEpisode) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    frames = episode.frames
    payload = {
        "timestamps_ns": np.array([f.timestamp_ns for f in frames], dtype=np.int64),
        "sync_offsets_ns": np.array([f.sync_offset_ns for f in frames], dtype=np.int64),
        "pose_sources": np.array([f.pose_source for f in frames]),
        "environment_json": np.array(
            json.dumps([f.environment for f in frames], ensure_ascii=False)
        ),
    }
    if all(f.rgb is not None for f in frames):
        payload["rgb"] = np.stack([f.rgb for f in frames])
    if all(f.depth is not None for f in frames):
        payload["depth"] = np.stack([f.depth for f in frames])
    if all(f.mask is not None for f in frames):
        payload["mask"] = np.stack([f.mask for f in frames])
    if all(f.tactile is not None for f in frames):
        payload["tactile"] = np.stack([f.tactile for f in frames])
    if all(f.sensor_pose is not None for f in frames):
        payload["sensor_pose"] = np.stack([f.sensor_pose for f in frames])
    if all(f.object_pose is not None for f in frames):
        payload["object_pose"] = np.stack([f.object_pose for f in frames])
    np.savez_compressed(path / "frames.npz", **payload)
    meta = {
        "episode_id": episode.episode_id,
        "instance_id": episode.instance_id,
        "category": episode.category,
        "calibration": episode.calibration,
        "consent": episode.consent,
        "privacy_scrubbed": episode.privacy_scrubbed,
        "source": episode.source,
    }
    (path / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_episode(path: str | Path) -> RDEpisode:
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    with np.load(path / "frames.npz", allow_pickle=False) as f:
        environments = json.loads(str(f["environment_json"]))
        frames = []
        n = len(f["timestamps_ns"])
        for i in range(n):
            frames.append(
                RDFrame(
                    timestamp_ns=int(f["timestamps_ns"][i]),
                    sync_offset_ns=int(f["sync_offsets_ns"][i]),
                    pose_source=str(f["pose_sources"][i]),
                    environment=environments[i],
                    rgb=f["rgb"][i] if "rgb" in f.files else None,
                    depth=f["depth"][i] if "depth" in f.files else None,
                    mask=f["mask"][i] if "mask" in f.files else None,
                    tactile=f["tactile"][i] if "tactile" in f.files else None,
                    sensor_pose=f["sensor_pose"][i] if "sensor_pose" in f.files else None,
                    object_pose=f["object_pose"][i] if "object_pose" in f.files else None,
                )
            )
    return RDEpisode(
        episode_id=meta["episode_id"],
        instance_id=meta["instance_id"],
        category=meta["category"],
        frames=frames,
        calibration=meta.get("calibration", {}),
        consent=bool(meta.get("consent", True)),
        privacy_scrubbed=bool(meta.get("privacy_scrubbed", True)),
        source=meta.get("source", "real"),
    )


def episode_to_sample(episode: RDEpisode) -> dict:
    """Convert one RD2 instance episode to the simulator sample schema.

    The mask is placed in segmentation channel 0 as object id 1, with a
    ``geom_to_instance`` mapping so the existing tracker/perception path works.
    """
    frames = episode.frames
    n = len(frames)
    rgb = np.stack([f.rgb for f in frames])
    depth = np.stack([f.depth for f in frames])
    mask = np.stack([f.mask for f in frames]).astype(np.int32)
    seg = np.zeros((n, mask.shape[1], mask.shape[2], 2), dtype=np.int32)
    seg[..., 0] = (mask > 0).astype(np.int32)
    seg[..., 1] = 5
    tactile = np.stack([f.tactile for f in frames])[:, None, :, :]
    poses = np.stack([f.object_pose for f in frames])[:, None, :]
    return {
        "rgb": rgb,
        "depth": depth,
        "segmentation": seg,
        "tactile": tactile,
        "poses": poses,
        "categories": [episode.category],
        "instance_ids": [episode.instance_id],
        "geom_to_instance": {"1": episode.instance_id},
        "relations": [[] for _ in range(n)],
        "camera": episode.calibration,
        "seed": 0,
        "kind": "rd2",
    }
