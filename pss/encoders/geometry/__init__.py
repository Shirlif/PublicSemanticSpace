"""Geometry expert: handcrafted features from depth + segmentation.

No learning in Phase 3. The goal is an interpretable, stable feature vector
that does not confound "model error" with "geometry error".
"""

from __future__ import annotations

import numpy as np

GEOMETRY_DIM = 16


def geometry_features(depth: np.ndarray, seg: np.ndarray, geom_id: int) -> np.ndarray:
    """Return a 16-d feature vector for the object with ``geom_id``.

    Features: validity, mask area, bbox size/aspect, solidity, depth statistics,
    image centroid, and depth IQR. Depth is normalized by 3 m.
    """
    depth = np.asarray(depth, dtype=np.float64)
    seg = np.asarray(seg)
    mask = seg[..., 0] == geom_id if seg.ndim == 3 else seg == geom_id
    feats = np.zeros(GEOMETRY_DIM, dtype=np.float32)
    if not mask.any():
        return feats

    h, w = depth.shape
    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    bbox_w = (x1 - x0 + 1) / max(w, 1)
    bbox_h = (y1 - y0 + 1) / max(h, 1)
    area = float(mask.mean())
    bbox_area = bbox_w * bbox_h
    solidity = area / bbox_area if bbox_area > 0 else 0.0

    d = np.clip(depth[mask], 0.0, 3.0)
    p10, p25, p75, p90 = np.percentile(d, [10, 25, 75, 90])
    feats[:] = [
        1.0,
        area,
        bbox_w,
        bbox_h,
        (bbox_w / bbox_h) if bbox_h > 0 else 0.0,
        solidity,
        float(d.mean()) / 3.0,
        float(d.std()) / 3.0,
        float(d.min()) / 3.0,
        float(d.max()) / 3.0,
        float(d.max() - d.min()) / 3.0,
        float(p10) / 3.0,
        float(p90) / 3.0,
        float(xs.mean()) / max(w, 1),
        float(ys.mean()) / max(h, 1),
        float(p75 - p25) / 3.0,
    ]
    return np.clip(feats, 0.0, None).astype(np.float32)


def geometry_from_sample(sample: dict, view: int = 0) -> np.ndarray:
    return geometry_features(
        sample["depth"][view],
        sample["segmentation"][view],
        int(sample["geom_id"]),
    )
