"""RD2 validators."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .schema import POSE_SOURCES, RDEpisode


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_episode(
    episode: RDEpisode,
    max_sync_ns: int = 5_000_000,
    tactile_shape: tuple[int, int] = (8, 8),
) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    if not episode.episode_id:
        errors.append("episode_id is empty")
    if not episode.instance_id:
        errors.append("instance_id is empty")
    if not episode.category:
        errors.append("category is empty")
    if not episode.frames:
        errors.append("no frames")
    if not episode.consent:
        errors.append("consent is false")
    if not episode.privacy_scrubbed:
        errors.append("privacy_scrubbed is false")

    last_t = None
    for i, frame in enumerate(episode.frames):
        if last_t is not None and frame.timestamp_ns <= last_t:
            errors.append(f"frame {i}: timestamp not increasing")
        last_t = frame.timestamp_ns
        if abs(frame.sync_offset_ns) > max_sync_ns:
            errors.append(f"frame {i}: sync offset {frame.sync_offset_ns} ns exceeds {max_sync_ns}")
        if frame.pose_source not in POSE_SOURCES:
            errors.append(f"frame {i}: unknown pose_source {frame.pose_source!r}")
        if frame.tactile is not None and tuple(frame.tactile.shape) != tactile_shape:
            errors.append(
                f"frame {i}: tactile shape {frame.tactile.shape} != {tactile_shape}"
            )
        if frame.rgb is not None and frame.mask is not None:
            if frame.rgb.shape[:2] != frame.mask.shape[:2]:
                errors.append(f"frame {i}: rgb/mask shape mismatch")
        if frame.object_pose is not None and np.asarray(frame.object_pose).shape != (7,):
            errors.append(f"frame {i}: object_pose must be (7,)")
        if frame.sensor_pose is not None and np.asarray(frame.sensor_pose).shape != (7,):
            errors.append(f"frame {i}: sensor_pose must be (7,)")

    if episode.source == "real" and not episode.calibration:
        warnings.append("real episode has no calibration block")
    return ValidationResult(ok=not errors, errors=errors, warnings=warnings)


def validate_dataset(path: str | Path, **kwargs) -> dict:
    import json

    from .adapter import load_episode

    root = Path(path)
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    results = []
    for entry in index.get("episodes", []):
        episode = load_episode(root / entry["path"])
        result = validate_episode(episode, **kwargs)
        results.append(
            {
                "id": entry.get("id", episode.episode_id),
                "ok": result.ok,
                "errors": result.errors,
                "warnings": result.warnings,
            }
        )
    return {
        "episodes": len(results),
        "ok": sum(1 for r in results if r["ok"]),
        "failed": [r for r in results if not r["ok"]],
    }
