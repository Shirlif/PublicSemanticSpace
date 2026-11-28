"""Tactile expert: a small MLP on an 8x8 contact grid.

The training data is a **synthetic proxy** (material parameters for category
samples, coarse contact projection for episodes). Treat the result as a
pipeline check, not evidence about real tactile separability.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn


class TactileMLP(nn.Module):
    def __init__(self, in_dim: int = 64, out_dim: int = 64):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class TactileExpert:
    encoder: TactileMLP
    classes: list[str]
    input_dim: int
    device: str = "cpu"

    def encode(self, tactile: torch.Tensor) -> torch.Tensor:
        self.encoder.eval()
        with torch.no_grad():
            return self.encoder(tactile.to(self.device))


def save_tactile_expert(
    path: str | Path, encoder: TactileMLP, classes: list[str], input_dim: int
) -> None:
    torch.save(
        {
            "state_dict": encoder.state_dict(),
            "classes": list(classes),
            "input_dim": int(input_dim),
            "out_dim": int(encoder.out_dim),
        },
        Path(path),
    )


def load_tactile_expert(path: str | Path, device: str = "cpu") -> TactileExpert:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("arch") == "conv":
        from .conv import load_tactile_conv_expert

        return load_tactile_conv_expert(path, device=device)  # type: ignore[return-value]
    encoder = TactileMLP(int(payload["input_dim"]), int(payload["out_dim"])).to(device)
    encoder.load_state_dict(payload["state_dict"])
    encoder.eval()
    return TactileExpert(
        encoder=encoder,
        classes=list(payload["classes"]),
        input_dim=int(payload["input_dim"]),
        device=device,
    )


from .conv import (  # noqa: E402  (kept after the MLP loader to avoid cycles)
    TactileConvEncoder,
    TactileConvExpert,
    load_tactile_conv_expert,
    save_tactile_conv_expert,
)
from .handcrafted import (  # noqa: E402
    HandcraftedTactileEncoder,
    handcrafted_tactile_features,
    load_handcrafted_tactile,
)

__all__ = [
    "HandcraftedTactileEncoder",
    "TactileConvEncoder",
    "TactileConvExpert",
    "TactileExpert",
    "TactileMLP",
    "handcrafted_tactile_features",
    "load_handcrafted_tactile",
    "load_tactile_conv_expert",
    "load_tactile_expert",
    "save_tactile_conv_expert",
    "save_tactile_expert",
]
