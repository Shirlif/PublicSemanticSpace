"""Per-modality projection heads (Phase 4).

Each expert gets its own head that maps native features into the public 128-d
space. Two additions make the head an *alignment/calibration* layer rather
than a lossy MLP:

- ``whiten_dim``: a PCA-whitening front-end fitted on the training split. This
  is what turned raw high-dimensional DINOv2 features from an overfitting
  channel into a usable one in the prior encoder experiment.
- ``residual``: a linear skip from the (whitened) input straight to the public
  space. The head can no longer bottleneck away an identity signal, which
  matters most for the low-dimensional tactile map.

Outputs stay L2-normalized, so cosine similarity is still the distance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pss.contracts.schema import SPACE_VERSION

PUBLIC_DIM = 128


class WhiteningFrontend(nn.Module):
    """PCA whitening fitted on frozen expert features (data, not a network)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.register_buffer("mean", torch.zeros(self.in_dim))
        components = torch.zeros(self.out_dim, self.in_dim)
        side = min(self.out_dim, self.in_dim)
        components[:side, :side] = torch.eye(side)
        self.register_buffer("components", components)
        self.register_buffer("scale", torch.ones(self.out_dim))
        self.fitted = False

    def fit(self, X: np.ndarray) -> "WhiteningFrontend":
        from sklearn.decomposition import PCA

        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.in_dim:
            raise ValueError(f"expected ({self.in_dim},) features, got {X.shape}")
        k = min(self.out_dim, X.shape[1], len(X) - 1)
        if k < 1:
            raise ValueError("not enough samples to fit whitening")
        pca = PCA(n_components=k, whiten=True, random_state=0).fit(X)
        self.mean.copy_(torch.as_tensor(pca.mean_, dtype=torch.float32))
        self.components.zero_()
        self.components[:k].copy_(
            torch.as_tensor(np.asarray(pca.components_), dtype=torch.float32)
        )
        self.scale.fill_(1.0)
        self.scale[:k].copy_(
            torch.as_tensor(np.sqrt(pca.explained_variance_), dtype=torch.float32).clamp_min(1e-6)
        )
        self.fitted = True
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (((x - self.mean) @ self.components.T) / self.scale)


class ProjectionHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int = PUBLIC_DIM,
        hidden: int = 256,
        whiten_dim: int | None = None,
        residual: bool = False,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.hidden = int(hidden)
        self.whiten_dim = None if whiten_dim is None else int(whiten_dim)
        self.residual = bool(residual)
        work_dim = self.whiten_dim or self.in_dim
        self.whiten = WhiteningFrontend(self.in_dim, work_dim) if self.whiten_dim else None
        self.net = nn.Sequential(
            nn.Linear(work_dim, self.hidden),
            nn.LayerNorm(self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.out_dim),
        )
        self.skip = nn.Linear(work_dim, self.out_dim, bias=False) if self.residual else None
        self.space_version = SPACE_VERSION

    def fit_whitening(self, X: np.ndarray) -> "ProjectionHead":
        if self.whiten is None:
            raise ValueError("head was built without whiten_dim")
        self.whiten.fit(X)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.whiten(x) if self.whiten is not None else x
        z = self.net(h)
        if self.skip is not None:
            z = z + self.skip(h)
        return F.normalize(z, dim=-1)


def save_projection(path: str | Path, head: ProjectionHead) -> None:
    torch.save(
        {
            "state_dict": head.state_dict(),
            "in_dim": int(head.in_dim),
            "out_dim": int(head.out_dim),
            "hidden": int(head.hidden),
            "whiten_dim": head.whiten_dim,
            "residual": bool(head.residual),
            "whitening_fitted": bool(getattr(head.whiten, "fitted", False)),
            "space_version": head.space_version,
        },
        Path(path),
    )


def load_projection(path: str | Path, device: str = "cpu") -> ProjectionHead:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    head = ProjectionHead(
        int(payload["in_dim"]),
        int(payload["out_dim"]),
        hidden=int(payload.get("hidden", 256)),
        whiten_dim=payload.get("whiten_dim"),
        residual=bool(payload.get("residual", False)),
    ).to(device)
    head.load_state_dict(payload["state_dict"])
    if head.whiten is not None:
        head.whiten.fitted = bool(payload.get("whitening_fitted", True))
    head.space_version = str(payload.get("space_version", "0.1.1"))
    head.eval()
    return head
