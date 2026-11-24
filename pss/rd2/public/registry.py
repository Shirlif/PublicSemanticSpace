"""Registry of public embodied datasets and their RD2 suitability (metadata only)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PublicDatasetSpec:
    key: str
    repo_id: str
    license: str
    commercial_ok: bool
    modalities: tuple[str, ...]
    sync_tier: str  # hardware | ptp | frame_timestamp | pose_paired
    tactile_sensor: str | None
    has_object_pose: bool
    has_instance_ids: bool
    format: str
    notes: str
    default_patterns: tuple[str, ...] = ()


PUBLIC_DATASETS: dict[str, PublicDatasetSpec] = {
    "wt_mm": PublicDatasetSpec(
        key="wt_mm",
        repo_id="WT-MM/lerobot_tactile",
        license="apache-2.0",
        commercial_ok=True,
        modalities=("rgb", "tactile", "joint_state", "action"),
        sync_tier="frame_timestamp",
        tactile_sensor="12x32 tactile array (sensor model unspecified)",
        has_object_pose=False,
        has_instance_ids=False,
        format="lerobot_v3",
        notes="SO-100; 86 episodes; 103,541 frames; one task; small and permissive. Minimal pair: data file-004 + video file-005 (episode 50, 1027 frames).",
        default_patterns=(
            "meta/info.json",
            "meta/stats.json",
            "meta/tasks.parquet",
            "meta/episodes/chunk-000/file-004.parquet",
            "data/chunk-000/file-004.parquet",
            "videos/observation.images.cam/chunk-000/file-005.mp4",
        ),
    ),
    "opengraph_ego_tactile": PublicDatasetSpec(
        key="opengraph_ego_tactile",
        repo_id="OpenGraphLabs-Research/ego-tactile-manipulation",
        license="cc-by-4.0",
        commercial_ok=True,
        modalities=("ego_rgb", "tactile_glove", "imu", "action_labels", "object_labels"),
        sync_tier="frame_timestamp",
        tactile_sensor="OGLO tactile gloves, 160 channels (80/hand)",
        has_object_pose=False,
        has_instance_ids=False,
        format="lerobot_v3",
        notes="Human egocentric; 72 episodes; 1.28 h; commercial use allowed with attribution.",
        default_patterns=(
            "meta/info.json",
            "meta/stats.json",
            "meta/tasks.parquet",
            "meta/opengraph/episode_manifest.json",
            "data/chunk-000/file-000.parquet",
        ),
    ),
    "daimon_infinity": PublicDatasetSpec(
        key="daimon_infinity",
        repo_id="daimonrobotics/Daimon-Infinity",
        license="cc-by-nc-sa-4.0",
        commercial_ok=False,
        modalities=("rgb", "tactile", "ee_pose", "joint_state", "action", "language", "audio"),
        sync_tier="frame_timestamp",
        tactile_sensor="Daimon high-resolution visuotactile",
        has_object_pose=False,
        has_instance_ids=False,
        format="lerobot_v3",
        notes="Gated; non-commercial; strongest high-res tactile source. No object 6D pose/instance IDs.",
        default_patterns=(
            "meta/info.json",
            "data/chunk-000/file-000.parquet",
        ),
    ),
    "visuotactile_gs_oxt": PublicDatasetSpec(
        key="visuotactile_gs_oxt",
        repo_id="jsun/VisuoTactile-GS-OXT",
        license="unknown",
        commercial_ok=False,
        modalities=("rgb", "tactile", "wrist_pose", "hand_joints"),
        sync_tier="pose_paired",
        tactile_sensor="DenseTact",
        has_object_pose=False,
        has_instance_ids=False,
        format="imagefolder",
        notes="RGB and tactile independently swept, paired by pose; not time-synchronized.",
        default_patterns=(),
    ),
}
