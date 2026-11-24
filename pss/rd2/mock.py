"""Mock RD2 episodes with domain shift.

This is **not real data**. It exercises the schema, adapter, and validator with
sim-derived episodes whose appearance/tactile/pose statistics are perturbed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pss.encoders.data import load_index
from pss.sim.dataset import load_sample

from .adapter import save_episode
from .schema import RDEpisode, RDFrame


def _shift_rgb(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = rgb.astype(np.float32) * 0.85 + 15.0
    out += rng.normal(scale=5.0, size=out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def make_mock_episode(sample: dict, seed: int) -> RDEpisode:
    rng = np.random.default_rng(seed)
    geom_id = int(sample["geom_id"])
    object_pose = np.asarray(sample["pose"], dtype=float)
    frames = []
    for v in range(sample["rgb"].shape[0]):
        frames.append(
            RDFrame(
                timestamp_ns=v * 33_333_333,
                rgb=_shift_rgb(sample["rgb"][v], rng),
                depth=sample["depth"][v]
                + rng.normal(scale=0.005, size=sample["depth"][v].shape).astype(np.float32),
                mask=sample["segmentation"][v][..., 0] == geom_id,
                tactile=np.clip(
                    sample["tactile"][v] * 0.8
                    + 0.05
                    + rng.normal(scale=0.02, size=sample["tactile"][v].shape),
                    0.0,
                    1.0,
                ).astype(np.float32),
                sensor_pose=np.zeros(7),
                object_pose=object_pose
                + np.array([*rng.normal(scale=0.003, size=3), 0.0, 0.0, 0.0, 0.0]),
                pose_source="mocap",
                sync_offset_ns=int(rng.integers(-1_000_000, 1_000_000)),
                environment={"lighting": "shifted", "device_class": "mock"},
            )
        )
    return RDEpisode(
        episode_id=f"mock_{seed:04d}",
        instance_id=str(sample["instance_id"]),
        category=str(sample["category"]),
        frames=frames,
        calibration={"camera": "mock", "hand_eye": "mock"},
        consent=True,
        privacy_scrubbed=True,
        source="mock_domain_shift",
    )


def make_mock_dataset(
    out_dir: str | Path,
    data_dir: str | Path,
    n_episodes: int = 8,
    seed: int = 0,
) -> dict:
    out = Path(out_dir)
    index = load_index(data_dir)
    entries = [e for e in index["entries"] if e["kind"] == "category"][:n_episodes]
    out_entries = []
    for i, entry in enumerate(entries):
        sample = load_sample(Path(data_dir) / entry["path"])
        episode = make_mock_episode(sample, seed + i)
        rel = f"episodes/ep_{i:04d}"
        save_episode(out / rel, episode)
        out_entries.append(
            {
                "id": episode.episode_id,
                "path": rel,
                "category": episode.category,
                "instance": episode.instance_id,
            }
        )
    (out / "index.json").write_text(
        json.dumps(
            {"version": "rd2_mock", "source": "mock_domain_shift", "episodes": out_entries},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"out_dir": str(out), "episodes": len(out_entries)}
