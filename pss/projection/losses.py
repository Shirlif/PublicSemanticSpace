"""Contrastive losses for cross-modal alignment."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def supcon_cross(
    z1: torch.Tensor,
    z2: torch.Tensor,
    labels: torch.Tensor,
    temp: float = 0.07,
) -> torch.Tensor:
    """Supervised contrastive loss between two modality embeddings.

    Positive pairs are samples with the same label (category or instance).
    Multiple positives per anchor are supported, which matters because every
    category has several instances/views in a batch.

    ``z1`` and ``z2`` are row-aligned: row ``i`` of each is the same sample.
    """
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    sim = z1 @ z2.T / temp
    mask = (labels[:, None] == labels[None, :]).float()
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    denom = mask.sum().clamp(min=1.0)
    return -(log_prob * mask).sum() / denom


def material_within_category(
    z: torch.Tensor,
    category_labels: torch.Tensor,
    material_labels: torch.Tensor,
    temp: float = 0.07,
    margin: float = 0.5,
) -> torch.Tensor:
    """Push different material bins apart *inside* the same category.

    Positives: same category and same material bin.
    Negatives: same category, different material bin.
    Anchors without both a positive and a negative are skipped.

    A cosine margin term adds a direct penalty for same-category /
    different-material pairs above ``margin``. This is the direct counterweight
    to category-level SupCon, which otherwise collapses hard/soft samples of
    the same object into one point.
    """
    z = F.normalize(z, dim=-1)
    sim = z @ z.T / temp
    n = z.shape[0]
    self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(self_mask, -1e9)
    same_cat = category_labels[:, None] == category_labels[None, :]
    same_mat = material_labels[:, None] == material_labels[None, :]
    pos = same_cat & same_mat & ~self_mask
    neg = same_cat & ~same_mat & ~self_mask
    valid = pos.any(dim=1) & neg.any(dim=1)
    if not bool(valid.any()):
        return z.sum() * 0.0
    cat_mask = same_cat & ~self_mask
    log_denom = torch.logsumexp(sim.masked_fill(~cat_mask, -1e9), dim=1)
    log_pos = torch.logsumexp(sim.masked_fill(~pos, -1e9), dim=1)
    loss = -(log_pos - log_denom)
    cosine = z @ z.T
    margin_loss = F.relu(cosine - margin)[neg].mean() if neg.any() else z.sum() * 0.0
    return loss[valid].mean() + margin_loss
