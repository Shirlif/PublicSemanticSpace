"""Local-descriptor aggregation for frozen vision backbones.

Global CLS features on 128x128 primitive renders are coarse: the prior
DINOv2 experiment lost to a task-trained CNN because a single global vector
throws away instance detail. VLAD aggregates patch tokens into a fixed-size
vector and keeps that detail while staying inside the existing PSS contract
(``encode(images) -> (B, out_dim)``).

The codebook is fitted on the projection-training split only; it is data, not
a learned network, and is saved next to the projection head.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class VLADEncoder:
    backbone: object
    codebook: torch.Tensor  # (K, D)
    device: str = "cpu"
    image_size: int = 224
    normalize: bool = True

    @property
    def out_dim(self) -> int:
        return int(self.codebook.shape[0] * self.codebook.shape[1])

    def encode(self, images: torch.Tensor, batch_size: int = 32) -> torch.Tensor:
        """Return one VLAD vector per image, ``(B, K*D)``."""
        outs = []
        for start in range(0, len(images), max(1, batch_size)):
            chunk = images[start : start + max(1, batch_size)]
            patches = self.backbone.encode_patches(chunk.to(self.device))  # (B, P, D)
            outs.append(self._aggregate(patches))
        return torch.cat(outs, dim=0)

    def _aggregate(self, patches: torch.Tensor) -> torch.Tensor:
        centers = self.codebook.to(patches.device, patches.dtype)
        k, d = centers.shape
        assign = (patches @ centers.T).argmax(dim=-1)  # (B, P)
        out = patches.new_zeros((patches.shape[0], k * d))
        for j in range(k):
            mask = (assign == j).to(patches.dtype).unsqueeze(-1)
            residual = (patches - centers[j]).mul(mask).sum(dim=1)
            out[:, j * d : (j + 1) * d] = residual
        if self.normalize:
            out = torch.nn.functional.normalize(out, dim=-1)
        return out


def fit_vlad_codebook(
    backbone,
    images: torch.Tensor,
    k: int = 16,
    seed: int = 0,
    max_patches: int = 80_000,
    batch_size: int = 64,
    device: str | None = None,
) -> VLADEncoder:
    """Fit a k-means codebook over patch tokens from the training images."""
    from sklearn.cluster import MiniBatchKMeans

    device = device or getattr(backbone, "device", "cpu")
    chunks = []
    total = 0
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            patches = backbone.encode_patches(images[start : start + batch_size].to(device))
            chunks.append(patches.reshape(-1, patches.shape[-1]).cpu().numpy())
            total += chunks[-1].shape[0]
            if total >= max_patches:
                break
    flat = np.concatenate(chunks, axis=0)
    rng = np.random.default_rng(seed)
    if len(flat) > max_patches:
        flat = flat[rng.choice(len(flat), size=max_patches, replace=False)]
    km = MiniBatchKMeans(
        n_clusters=int(k), random_state=int(seed), n_init=3, batch_size=4096
    ).fit(flat)
    return VLADEncoder(
        backbone=backbone,
        codebook=torch.as_tensor(km.cluster_centers_, dtype=torch.float32),
        device=device,
        image_size=int(getattr(backbone, "image_size", 224)),
    )


def save_vlad(path: str | Path, encoder: VLADEncoder, meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "codebook": encoder.codebook.cpu(),
            "image_size": int(encoder.image_size),
            "normalize": bool(encoder.normalize),
            "meta": dict(meta or {}),
        },
        path,
    )


def load_vlad(path: str | Path, backbone, device: str = "cpu") -> VLADEncoder:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    return VLADEncoder(
        backbone=backbone,
        codebook=payload["codebook"].float(),
        device=device,
        image_size=int(payload.get("image_size", getattr(backbone, "image_size", 224))),
        normalize=bool(payload.get("normalize", True)),
    )
