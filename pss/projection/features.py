"""Extract frozen expert features for the projection stage.

The dataset has category-level pairing: each view has a vision crop, a tactile
proxy, and geometry features from the same simulated view. It does **not** have
synchronized real instance-level visual-tactile pairs (RD2).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from pss.encoders.data import crop_object, load_index, normalize_image
from pss.encoders.geometry import geometry_features
from pss.encoders.tactile import TactileExpert
from pss.encoders.vision import VisionExpert
from pss.sim.dataset import load_sample

ATTR_SCALE = np.array([1.0, 1.0, 1.0, 10.0, 10.0], dtype=np.float32)


@dataclass
class PairedFeatures:
    vision: np.ndarray  # (N, vision_dim)
    tactile: np.ndarray  # (N, tactile_dim)
    geometry: np.ndarray  # (N, GEOMETRY_DIM)
    attrs: np.ndarray  # (N, 5) scaled: hardness, roughness, thermal, size_x, size_y
    categories: list[str]
    instances: list[str]
    splits: list[str]
    sample_ids: list[str]
    views: list[int]


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _encode_chunked(expert, X: torch.Tensor, chunk: int = 64) -> np.ndarray:
    """Encode a large batch in chunks so a full dataset fits on an 8 GB GPU."""

    outs = []
    for start in range(0, len(X), int(chunk)):
        outs.append(_to_numpy(expert.encode(X[start : start + int(chunk)])))
    if not outs:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(outs, axis=0)


def load_paired_features(
    data_dir: str | Path, vision: VisionExpert, tactile: TactileExpert
) -> PairedFeatures:
    data_dir = Path(data_dir)
    index = load_index(data_dir)
    preserve_aspect = bool(index.get("config", {}).get("crop_preserve_aspect", False))
    images, tacs, geos, attrs = [], [], [], []
    categories, instances, splits, sample_ids, views = [], [], [], [], []

    for entry in index["entries"]:
        if entry["kind"] != "category":
            continue
        sample = load_sample(data_dir / entry["path"])
        geom_id = int(sample["geom_id"])
        material = sample["material"]
        size = sample["size"]
        size_x = float(size[0])
        size_y = float(size[1]) if len(size) > 1 else float(size[0])
        attr = np.array(
            [material["hardness"], material["roughness"], material["thermal"], size_x, size_y],
            dtype=np.float32,
        )
        for v in range(sample["rgb"].shape[0]):
            crop = crop_object(
                sample["rgb"][v],
                sample["segmentation"][v],
                geom_id,
                size=vision.image_size,
                preserve_aspect=preserve_aspect,
            )
            images.append(normalize_image(crop))
            tacs.append(sample["tactile"][v].reshape(-1).astype(np.float32))
            geos.append(geometry_features(sample["depth"][v], sample["segmentation"][v], geom_id))
            attrs.append(attr)
            categories.append(str(sample["category"]))
            instances.append(str(sample["instance_id"]))
            splits.append(str(sample["split"]))
            sample_ids.append(str(entry["id"]))
            views.append(v)

    Xv = torch.as_tensor(np.stack(images), dtype=torch.float32)
    Xt = torch.as_tensor(np.stack(tacs), dtype=torch.float32)
    V = _encode_chunked(vision, Xv).astype(np.float32)
    T = _encode_chunked(tactile, Xt).astype(np.float32)
    G = np.stack(geos).astype(np.float32)
    A = np.stack(attrs).astype(np.float32) * ATTR_SCALE
    return PairedFeatures(
        vision=V,
        tactile=T,
        geometry=G,
        attrs=A,
        categories=categories,
        instances=instances,
        splits=splits,
        sample_ids=sample_ids,
        views=views,
    )
