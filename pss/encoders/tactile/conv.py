"""Convolutional tactile encoder for 8x8 (and 12x32) contact grids.

The baseline ``TactileMLP`` flattens the contact grid, so it cannot exploit
spatial structure. This encoder keeps the grid and adds two properties the
tactile identity signal needs:

- a local convolution stack over the contact map;
- a residual path from the spatially pooled, contrast-normalized map, so the
  identity pattern is never bottlenecked away by the learned head.

The public interface matches ``TactileMLP``: ``encode(x) -> (B, out_dim)``
with ``x`` either ``(B, 64)``/``(B, 384)`` or ``(B, H, W)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

GRID_SHAPES = {64: (8, 8), 384: (12, 32)}


def _to_grid(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        shape = GRID_SHAPES.get(int(x.shape[1]))
        if shape is None:
            side = int(round(int(x.shape[1]) ** 0.5))
            if side * side != int(x.shape[1]):
                raise ValueError(f"unsupported tactile input dim {int(x.shape[1])}")
            shape = (side, side)
        x = x.reshape(-1, 1, *shape)
    elif x.dim() == 3:
        x = x.unsqueeze(1)
    if tuple(x.shape[-2:]) not in set(GRID_SHAPES.values()):
        x = F.interpolate(x, size=(8, 8), mode="bilinear", align_corners=False)
    return x


class TactileConvEncoder(nn.Module):
    def __init__(self, in_dim: int = 64, out_dim: int = 128, residual: bool = True):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.residual = bool(residual)
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(64, self.out_dim)
        self.skip = nn.Linear(64, self.out_dim, bias=False) if self.residual else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        grid = _to_grid(x)
        # Contrast-normalize so brightness/contact-force scale does not dominate.
        flat = grid.flatten(1)
        mean = flat.mean(dim=1, keepdim=True)
        std = flat.std(dim=1, keepdim=True).clamp_min(1e-6)
        grid = (grid - mean[:, :, None, None]) / std[:, :, None, None]
        h = self.net(grid).flatten(1)
        z = self.proj(h)
        if self.skip is not None:
            pooled = flat - mean
            # Pad/truncate the raw grid into the 64-d skip space deterministically.
            if pooled.shape[1] >= 64:
                pooled = pooled[:, :64]
            else:
                pooled = F.pad(pooled, (0, 64 - pooled.shape[1]))
            z = z + self.skip(pooled)
        return z


@dataclass
class TactileConvExpert:
    encoder: TactileConvEncoder
    classes: list[str]
    input_dim: int
    device: str = "cpu"

    def encode(self, tactile) -> torch.Tensor:
        self.encoder.eval()
        if not isinstance(tactile, torch.Tensor):
            tactile = torch.as_tensor(np.asarray(tactile, dtype=np.float32))
        with torch.no_grad():
            return self.encoder(tactile.to(self.device))


def save_tactile_conv_expert(
    path: str | Path,
    encoder: TactileConvEncoder,
    classes: list[str],
    input_dim: int,
) -> None:
    torch.save(
        {
            "state_dict": encoder.state_dict(),
            "classes": list(classes),
            "input_dim": int(input_dim),
            "out_dim": int(encoder.out_dim),
            "arch": "conv",
        },
        Path(path),
    )


def load_tactile_conv_expert(path: str | Path, device: str = "cpu") -> TactileConvExpert:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    encoder = TactileConvEncoder(
        int(payload["input_dim"]), int(payload["out_dim"])
    ).to(device)
    encoder.load_state_dict(payload["state_dict"])
    encoder.eval()
    return TactileConvExpert(
        encoder=encoder,
        classes=list(payload["classes"]),
        input_dim=int(payload["input_dim"]),
        device=device,
    )
