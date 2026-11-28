"""Load CLEVR scenes into the minimal 3D scene model."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .model import ObjectState, SceneModel

CACHE_FILES = {
    "traina": "traina_8000.pt",
    "vala": "vala_3000.pt",
    "valb": "valb_3000.pt",
}


def load_cache(root: str | Path, split: str) -> dict:
    import torch

    path = Path(root) / "cache" / CACHE_FILES[split]
    return torch.load(path, map_location="cpu", weights_only=False)


def scene_from_dict(payload: dict, scene_index: int) -> SceneModel:
    objects = {}
    for oi, obj in enumerate(payload.get("objects", [])):
        oid = f"obj{oi:02d}"
        objects[oid] = ObjectState.from_clevr(obj, oid)
    return SceneModel(scene_id=f"scene{scene_index:05d}", objects=objects)


def load_scenes(
    root: str | Path,
    split: str,
    *,
    max_scenes: int = 0,
    seed: int = 0,
    min_objects: int = 2,
) -> list[SceneModel]:
    cache = load_cache(root, split)
    scenes = list(cache.get("scenes", []))
    if max_scenes and len(scenes) > max_scenes:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(scenes), size=max_scenes, replace=False)
        scenes = [scenes[int(i)] for i in idx]
    out = []
    for i, payload in enumerate(scenes):
        scene = scene_from_dict(payload, i)
        if len(scene.objects) >= min_objects:
            out.append(scene)
    return out


__all__ = ["CACHE_FILES", "load_cache", "load_scenes", "scene_from_dict"]
