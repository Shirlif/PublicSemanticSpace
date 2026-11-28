"""Handcrafted tactile descriptor: zero-parameter contact statistics.

The learned tactile experts (synthetic MLP, WT-MM CNN, PhysiCLeAR CNN) all
consume a contact map and return a learned vector.  This module is the
non-neural control: it describes the same map with interpretable statistics.

For a clip of ``T`` frames it returns one vector per clip:

- per-frame intensity / gradient / LBP / HOG / spatial-grid statistics,
  averaged over frames;
- contact proxies computed from a scale-free threshold (mean + std);
- temporal statistics: consecutive-frame change, temporal std, centroid drift.

Works on both grayscale contact grids (WT-MM ``12x32``) and RGB GelSight
frames (PhysiCLeAR ``64x64x3``).  Nothing is fitted; ``params()`` is zero.
"""

from __future__ import annotations

import numpy as np
import torch

from pss.encoders.vision.handcrafted import hog_features, lbp_features

__all__ = [
    "HandcraftedTactileEncoder",
    "handcrafted_tactile_features",
]


def _as_frames(tactile) -> np.ndarray:
    """Return ``(T, H, W, C)`` float frames in ``[0, 1]``.

    Accepts ``(T, H, W)``, ``(T, H, W, C)``, ``(H, W)``, ``(H, W, C)`` and a
    flattened ``(64,)`` / ``(384,)`` contact map (8x8 / 12x32).
    """
    arr = np.asarray(tactile, dtype=np.float32)
    if arr.ndim == 1:
        if arr.size == 64:
            arr = arr.reshape(1, 8, 8)
        elif arr.size == 384:
            arr = arr.reshape(1, 12, 32)
        else:
            raise ValueError(f"unsupported flat tactile size {arr.size}")
    if arr.ndim == 2:
        arr = arr[None, ..., None]
    elif arr.ndim == 3:
        if arr.shape[-1] in (3, 4):
            arr = arr[None]
        else:
            arr = arr[..., None]
    if arr.ndim != 4:
        raise ValueError(f"unsupported tactile shape {arr.shape}")
    if float(arr.max(initial=0.0)) > 1.5:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _frame_stats(gray: np.ndarray, cell: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame descriptor plus the normalized contact centroid."""
    flat = gray.reshape(-1)
    percentiles = np.percentile(flat, [10, 25, 50, 75, 90])
    gy, gx = np.gradient(gray)
    mag = np.hypot(gx, gy)
    mean = float(gray.mean())
    std = float(gray.std())
    threshold = mean + std

    # Scale-free contact mask: pixels above mean + std.
    weights = np.clip(gray - float(gray.min()), 0.0, None)
    total = float(weights.sum()) + 1e-9
    h, w = gray.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cx = float((weights * xx).sum() / total) / max(w - 1, 1)
    cy = float((weights * yy).sum() / total) / max(h - 1, 1)
    spread = float(
        np.sqrt(
            (weights * ((xx / max(w - 1, 1) - cx) ** 2 + (yy / max(h - 1, 1) - cy) ** 2)).sum()
            / total
        )
    )
    high = float((gray > threshold).mean())

    gray_u8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    grid = 4 if min(h, w) >= 8 else 2
    gh, gw = h // grid, w // grid
    if gh > 0 and gw > 0:
        colour_grid = gray.reshape(grid, gh, grid, gw).mean(axis=(1, 3)).reshape(-1)
    else:
        colour_grid = np.array([mean], dtype=np.float32)

    stats = np.asarray(
        [
            mean,
            std,
            float(flat.min()),
            float(flat.max()),
            *percentiles.tolist(),
            float(mag.mean()),
            float(mag.std()),
            float(np.percentile(mag, 90)),
            high,
            cx,
            cy,
            spread,
        ],
        dtype=np.float32,
    )
    feat = np.concatenate(
        [
            stats,
            hog_features(gray, cell=cell),
            lbp_features(gray_u8),
            colour_grid.astype(np.float32),
        ]
    )
    return feat, np.asarray([cx, cy], dtype=np.float32)


def handcrafted_tactile_features(
    tactile,
    cell: int | None = None,
) -> np.ndarray:
    """Return one fixed-length descriptor for a frame or a clip."""
    frames = _as_frames(tactile)
    t, h, w, _ = frames.shape
    gray = frames[..., :3].mean(axis=-1) if frames.shape[-1] >= 3 else frames[..., 0]
    if cell is None:
        cell = max(4, min(h, w) // 4)

    per_frame = []
    centroids = []
    for index in range(t):
        feat, centroid = _frame_stats(gray[index], cell)
        per_frame.append(feat)
        centroids.append(centroid)

    mean_feat = np.mean(per_frame, axis=0)
    if t > 1:
        diffs = np.abs(np.diff(gray, axis=0)).mean(axis=(1, 2))
        centroids_arr = np.stack(centroids)
        drift = np.linalg.norm(np.diff(centroids_arr, axis=0), axis=1)
        temporal = np.asarray(
            [
                float(diffs.mean()),
                float(diffs.max()),
                float(gray.std(axis=0).mean()),
                float(drift.mean()),
            ],
            dtype=np.float32,
        )
    else:
        temporal = np.zeros(4, dtype=np.float32)

    # Colour means/stds are meaningful for GelSight RGB, harmless for grids.
    if frames.shape[-1] >= 3:
        colour = np.concatenate(
            [
                frames[..., :3].mean(axis=(0, 1, 2)),
                frames[..., :3].std(axis=(0, 1, 2)),
            ]
        ).astype(np.float32)
    else:
        colour = np.zeros(6, dtype=np.float32)

    return np.concatenate([mean_feat, temporal, colour]).astype(np.float32)


class HandcraftedTactileEncoder:
    """Registry-compatible wrapper: ``encode(tactile) -> (B, out_dim)``.

    ``tactile`` may be a batch (``(B, H, W)`` / ``(B, 1, H, W)``) or a single
    flattened contact map.  The output dimension is fixed at construction from
    a zero frame; it is deterministic and parameter-free.
    """

    def __init__(self, out_shape: tuple[int, int] = (8, 8), cell: int | None = None) -> None:
        self.out_shape = tuple(int(v) for v in out_shape)
        self.cell = cell
        self.input_dim = int(self.out_shape[0] * self.out_shape[1])
        probe = np.zeros(self.out_shape, dtype=np.float32)
        self.out_dim = int(handcrafted_tactile_features(probe, cell=cell).size)

    def eval(self) -> "HandcraftedTactileEncoder":
        return self

    def params(self) -> int:
        return 0

    def encode(self, tactile: torch.Tensor) -> torch.Tensor:
        arr = tactile.detach().cpu().numpy()
        if arr.ndim == 1:
            arr = arr[None]
        elif arr.ndim == 2:
            if tuple(arr.shape) in ((8, 8), (12, 32)):
                arr = arr[None]
            else:
                arr = arr.reshape(arr.shape[0], -1)
        elif arr.ndim == 4 and arr.shape[1] == 1:
            arr = arr[:, 0]
        feats = []
        for row in arr:
            feats.append(handcrafted_tactile_features(row.reshape(self.out_shape), cell=self.cell))
        return torch.as_tensor(np.stack(feats), dtype=torch.float32)


def load_handcrafted_tactile(
    out_shape: tuple[int, int] = (8, 8), cell: int | None = None
) -> HandcraftedTactileEncoder:
    return HandcraftedTactileEncoder(out_shape=out_shape, cell=cell)
