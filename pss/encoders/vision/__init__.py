"""Vision expert: a small CNN feature extractor.

Phase 3 smoke test uses a from-scratch CNN because no torchvision/timm is
installed. The production path is a real-image-pretrained backbone (DINOv2 /
ResNet); the interface (``encode(images) -> (B, out_dim)``) is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn


class SmallCNNEncoder(nn.Module):
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.out_dim = int(out_dim)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x).flatten(1)
        return self.proj(h)


@dataclass
class VisionExpert:
    encoder: SmallCNNEncoder
    classes: list[str]
    image_size: int
    device: str = "cpu"

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        self.encoder.eval()
        with torch.no_grad():
            return self.encoder(images.to(self.device))


def save_vision_expert(
    path: str | Path, encoder: SmallCNNEncoder, classes: list[str], image_size: int
) -> None:
    torch.save(
        {
            "state_dict": encoder.state_dict(),
            "classes": list(classes),
            "image_size": int(image_size),
            "out_dim": int(encoder.out_dim),
        },
        Path(path),
    )


def load_vision_expert(path: str | Path, device: str = "cpu") -> VisionExpert:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    encoder = SmallCNNEncoder(int(payload["out_dim"])).to(device)
    encoder.load_state_dict(payload["state_dict"])
    encoder.eval()
    return VisionExpert(
        encoder=encoder,
        classes=list(payload["classes"]),
        image_size=int(payload["image_size"]),
        device=device,
    )


__all__ = ["SmallCNNEncoder", "VisionExpert", "load_vision_expert", "save_vision_expert"]
