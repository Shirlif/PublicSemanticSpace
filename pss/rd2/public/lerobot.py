"""LeRobot v3.0 dataset adapter.

Supports the Parquet + videos layout used by WT-MM, OpenGraphLabs, and
Daimon-Infinity. Only meta + a small data shard are needed for a smoke test.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd


def _to_array(value) -> np.ndarray:
    if isinstance(value, np.ndarray) and value.dtype != object:
        return value.astype(np.float32)
    if isinstance(value, np.ndarray) and value.dtype == object:
        return np.asarray(value.tolist(), dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


@dataclass
class LeRobotInfo:
    repo_id: str | None
    root: Path
    info: dict

    @property
    def fps(self) -> float:
        return float(self.info.get("fps", 0.0))

    @property
    def robot_type(self) -> str:
        return str(self.info.get("robot_type", "unknown"))

    @property
    def total_episodes(self) -> int:
        return int(self.info.get("total_episodes", 0))

    @property
    def total_frames(self) -> int:
        return int(self.info.get("total_frames", 0))

    @property
    def features(self) -> dict:
        return self.info.get("features", {})

    def tactile_keys(self) -> list[str]:
        return [k for k in self.features if k.startswith("observation.tactile")]

    def video_keys(self) -> list[str]:
        return [k for k in self.features if k.startswith("observation.images")]


class LeRobotDataset:
    def __init__(self, root: str | Path, repo_id: str | None = None):
        self.root = Path(root)
        self.repo_id = repo_id
        info_path = self.root / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"missing {info_path}")
        self.info = LeRobotInfo(repo_id, self.root, json.loads(info_path.read_text(encoding="utf-8")))

    def data_files(self) -> list[Path]:
        return sorted(self.root.glob("data/chunk-*/file-*.parquet"))

    def read_frames(self, max_rows: int | None = None) -> pd.DataFrame:
        files = self.data_files()
        if not files:
            raise FileNotFoundError(f"no parquet data under {self.root / 'data'}")
        frames = []
        total = 0
        for path in files:
            df = pd.read_parquet(path)
            frames.append(df)
            total += len(df)
            if max_rows is not None and total >= max_rows:
                break
        df = pd.concat(frames, ignore_index=True)
        return df.head(max_rows) if max_rows is not None else df

    def tactile_matrix(self, key: str | None = None, max_rows: int | None = None):
        keys = self.info.tactile_keys()
        if not keys:
            raise KeyError("no observation.tactile.* feature")
        key = key or keys[0]
        df = self.read_frames(max_rows=max_rows)
        if key not in df.columns:
            raise KeyError(f"column {key!r} not in parquet")
        arrays = [_to_array(v) for v in df[key]]
        shapes = {a.shape for a in arrays}
        if len(shapes) == 1:
            tactile = np.stack(arrays)
        else:
            tactile = np.stack([a.reshape(-1) for a in arrays])
        return tactile, df

    def state_matrix(self, max_rows: int | None = None):
        df = self.read_frames(max_rows=max_rows)
        if "observation.state" not in df.columns:
            return None, df
        states = np.stack([_to_array(v) for v in df["observation.state"]])
        return states, df

    def summary(self, max_rows: int | None = 2000) -> dict:
        result = {
            "root": str(self.root),
            "repo_id": self.repo_id,
            "robot_type": self.info.robot_type,
            "fps": self.info.fps,
            "total_episodes": self.info.total_episodes,
            "total_frames": self.info.total_frames,
            "tactile_keys": self.info.tactile_keys(),
            "video_keys": self.info.video_keys(),
            "parquet_files": len(self.data_files()),
        }
        try:
            tactile, df = self.tactile_matrix(max_rows=max_rows)
            flat = tactile.reshape(len(tactile), -1)
            contact = flat.max(axis=1) > 0
            result["loaded_frames"] = int(len(tactile))
            result["tactile_shape"] = list(tactile.shape[1:])
            result["tactile_range"] = [float(tactile.min()), float(tactile.max())]
            result["tactile_mean"] = float(tactile.mean())
            result["contact_frame_fraction"] = float(contact.mean())
            if "episode_index" in df.columns:
                result["loaded_episodes"] = sorted(
                    int(x) for x in df["episode_index"].unique().tolist()
                )
        except Exception as exc:
            result["tactile_error"] = f"{type(exc).__name__}: {exc}"
        try:
            states, _ = self.state_matrix(max_rows=max_rows)
            if states is not None:
                result["state_shape"] = list(states.shape[1:])
        except Exception as exc:
            result["state_error"] = f"{type(exc).__name__}: {exc}"
        return result
