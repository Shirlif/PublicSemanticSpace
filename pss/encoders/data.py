"""Dataset loading and object cropping for the sensory experts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from pss.sim.dataset import load_sample

from .geometry import GEOMETRY_DIM, geometry_features

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class CategoryExample:
    image: np.ndarray  # (3, S, S) float32, ImageNet-normalized
    tactile: np.ndarray  # (64,) float32
    category: str
    split: str
    instance: str
    sample_id: str
    view: int


@dataclass
class GeometryExample:
    features: np.ndarray  # (GEOMETRY_DIM,) float32
    category: str
    split: str
    instance: str
    sample_id: str
    view: int


def load_index(data_dir: str | Path) -> dict:
    path = Path(data_dir) / "index.json"
    return json.loads(path.read_text(encoding="utf-8"))


def crop_object(
    rgb: np.ndarray,
    seg: np.ndarray,
    geom_id: int,
    size: int = 64,
    pad: int = 4,
    preserve_aspect: bool = False,
) -> np.ndarray:
    """Crop the object region using the segmentation channel, then resize.

    With ``preserve_aspect`` the bbox is expanded to a square *before* resize,
    so a tall cup and a flat plate are not squeezed into the same silhouette.
    """
    h, w = rgb.shape[:2]
    mask = seg[..., 0] == geom_id
    if mask.any():
        ys, xs = np.where(mask)
        y0 = max(0, int(ys.min()) - pad)
        y1 = min(h - 1, int(ys.max()) + pad)
        x0 = max(0, int(xs.min()) - pad)
        x1 = min(w - 1, int(xs.max()) + pad)
    else:
        y0, y1, x0, x1 = 0, h - 1, 0, w - 1
    if preserve_aspect and mask.any():
        side = max(y1 - y0 + 1, x1 - x0 + 1)
        cy = (y0 + y1) // 2
        cx = (x0 + x1) // 2
        y0 = max(0, cy - side // 2)
        y1 = min(h - 1, y0 + side - 1)
        y0 = max(0, y1 - side + 1)
        x0 = max(0, cx - side // 2)
        x1 = min(w - 1, x0 + side - 1)
        x0 = max(0, x1 - side + 1)
    crop = rgb[y0 : y1 + 1, x0 : x1 + 1]
    return np.asarray(Image.fromarray(crop).resize((size, size), Image.BILINEAR))


def normalize_image(rgb_uint8: np.ndarray) -> np.ndarray:
    x = rgb_uint8.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(x, (2, 0, 1)).astype(np.float32)


def load_category_examples(
    data_dir: str | Path, size: int = 64, preserve_aspect: bool | None = None
) -> list[CategoryExample]:
    data_dir = Path(data_dir)
    index = load_index(data_dir)
    if preserve_aspect is None:
        preserve_aspect = bool(index.get("config", {}).get("crop_preserve_aspect", False))
    examples: list[CategoryExample] = []
    for entry in index["entries"]:
        if entry["kind"] != "category":
            continue
        sample = load_sample(data_dir / entry["path"])
        geom_id = int(sample["geom_id"])
        for v in range(sample["rgb"].shape[0]):
            crop = crop_object(
                sample["rgb"][v],
                sample["segmentation"][v],
                geom_id,
                size,
                preserve_aspect=preserve_aspect,
            )
            examples.append(
                CategoryExample(
                    image=normalize_image(crop),
                    tactile=sample["tactile"][v].reshape(-1).astype(np.float32),
                    category=str(sample["category"]),
                    split=str(sample["split"]),
                    instance=str(sample["instance_id"]),
                    sample_id=str(entry["id"]),
                    view=v,
                )
            )
    return examples


def load_geometry_examples(data_dir: str | Path) -> list[GeometryExample]:
    data_dir = Path(data_dir)
    index = load_index(data_dir)
    examples: list[GeometryExample] = []
    for entry in index["entries"]:
        if entry["kind"] != "category":
            continue
        sample = load_sample(data_dir / entry["path"])
        geom_id = int(sample["geom_id"])
        for v in range(sample["depth"].shape[0]):
            examples.append(
                GeometryExample(
                    features=geometry_features(
                        sample["depth"][v], sample["segmentation"][v], geom_id
                    ),
                    category=str(sample["category"]),
                    split=str(sample["split"]),
                    instance=str(sample["instance_id"]),
                    sample_id=str(entry["id"]),
                    view=v,
                )
            )
    return examples


def build_split(
    examples: list[CategoryExample], split: str, classes: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[CategoryExample]]:
    selected = [e for e in examples if e.split == split and e.category in classes]
    if not selected:
        return (
            np.zeros((0, 3, 0, 0), dtype=np.float32),
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            [],
        )
    images = np.stack([e.image for e in selected]).astype(np.float32)
    tactile = np.stack([e.tactile for e in selected]).astype(np.float32)
    labels = np.array([classes.index(e.category) for e in selected], dtype=np.int64)
    return images, tactile, labels, selected
