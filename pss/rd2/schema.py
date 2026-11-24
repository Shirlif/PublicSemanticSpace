"""RD2 real synchronized data schema (Phase RD).

This is the contract for real instance-level visual/tactile/pose episodes. The
mock generator only exercises the schema; it is not real data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

POSE_SOURCES = ("mocap", "apriltag", "slam")


@dataclass
class RDFrame:
    timestamp_ns: int
    rgb: np.ndarray | None = None
    depth: np.ndarray | None = None
    mask: np.ndarray | None = None
    tactile: np.ndarray | None = None
    sensor_pose: np.ndarray | None = None
    object_pose: np.ndarray | None = None
    pose_source: str = "mocap"
    sync_offset_ns: int = 0
    environment: dict = field(default_factory=dict)


@dataclass
class RDEpisode:
    episode_id: str
    instance_id: str
    category: str
    frames: list[RDFrame]
    calibration: dict = field(default_factory=dict)
    consent: bool = True
    privacy_scrubbed: bool = True
    source: str = "real"
