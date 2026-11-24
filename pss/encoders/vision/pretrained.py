"""Frozen pretrained backbone adapter for the vision expert (Phase 3 upgrade)."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from . import SmallCNNEncoder


class ResNet18Encoder(nn.Module):
    """ResNet18 backbone (optionally pretrained) + trainable projection head.

    The backbone is frozen and kept in eval mode so BatchNorm statistics do not
    drift while only the head is trained.
    """

    def __init__(self, out_dim: int = 256, pretrained: bool = True):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = nn.Identity()
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.backbone.eval()
        self.out_dim = int(out_dim)
        self.pretrained = bool(pretrained)
        self.proj = nn.Linear(512, self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.backbone(x))

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self


class SqueezeNetEncoder(nn.Module):
    """SqueezeNet1.1 backbone (real ImageNet pretrained) + trainable head."""

    def __init__(self, out_dim: int = 256, pretrained: bool = True):
        super().__init__()
        from torchvision.models import SqueezeNet1_1_Weights, squeezenet1_1

        weights = SqueezeNet1_1_Weights.DEFAULT if pretrained else None
        backbone = squeezenet1_1(weights=weights)
        self.features = backbone.features
        for parameter in self.features.parameters():
            parameter.requires_grad = False
        self.features.eval()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = int(out_dim)
        self.pretrained = bool(pretrained)
        self.proj = nn.Linear(512, self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        h = self.pool(h).flatten(1)
        return self.proj(h)

    def train(self, mode: bool = True):
        super().train(mode)
        self.features.eval()
        return self


def save_pretrained_expert(
    path: str | Path,
    encoder: nn.Module,
    classes: list[str],
    image_size: int,
    backbone: str,
) -> None:
    torch.save(
        {
            "state_dict": encoder.state_dict(),
            "classes": list(classes),
            "image_size": int(image_size),
            "out_dim": int(encoder.out_dim),
            "pretrained": bool(getattr(encoder, "pretrained", False)),
            "backbone": backbone,
        },
        Path(path),
    )


def load_pretrained_expert(path: str | Path, device: str = "cpu") -> nn.Module:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    backbone = payload.get("backbone", "resnet18")
    pretrained = bool(payload.get("pretrained", False))
    if backbone == "resnet18":
        encoder = ResNet18Encoder(int(payload["out_dim"]), pretrained=pretrained)
    elif backbone == "squeezenet1_1":
        encoder = SqueezeNetEncoder(int(payload["out_dim"]), pretrained=pretrained)
    else:
        raise ValueError(f"unknown backbone {backbone!r}")
    encoder = encoder.to(device)
    encoder.load_state_dict(payload["state_dict"])
    encoder.eval()
    return encoder


def save_resnet_expert(
    path: str | Path,
    encoder: ResNet18Encoder,
    classes: list[str],
    image_size: int,
) -> None:
    torch.save(
        {
            "state_dict": encoder.state_dict(),
            "classes": list(classes),
            "image_size": int(image_size),
            "out_dim": int(encoder.out_dim),
            "pretrained": bool(encoder.pretrained),
            "backbone": "resnet18",
        },
        Path(path),
    )


def load_resnet_expert(path: str | Path, device: str = "cpu") -> ResNet18Encoder:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    encoder = ResNet18Encoder(
        int(payload["out_dim"]), pretrained=bool(payload.get("pretrained", False))
    ).to(device)
    encoder.load_state_dict(payload["state_dict"])
    encoder.eval()
    return encoder
