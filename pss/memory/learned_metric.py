"""Learned re-ID metric for the PSS scene graph.

The hand-designed re-ID cost in ``TrackerConfig`` is replaced by a learned
Siamese embedding. Each entity token is projected into a metric space; re-ID
uses cosine distance, then the same Hungarian matching as the tracker.

This keeps the explicit scene graph and only learns the association metric.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from pss.sim.scene import CATEGORY_SPECS

CATEGORIES = sorted(CATEGORY_SPECS)
CATEGORY_TO_INDEX = {c: i for i, c in enumerate(CATEGORIES)}
VISION_DIM = 128
GEOMETRY_DIM = 128
TOKEN_DIM = VISION_DIM + GEOMETRY_DIM + 3 + len(CATEGORIES)
TACTILE_DIM = 128
DINO_DIM = 384
TACTILE_TOKEN_DIM = TOKEN_DIM + TACTILE_DIM
DINO_TOKEN_DIM = TOKEN_DIM + DINO_DIM
FULL_TOKEN_DIM = TOKEN_DIM + TACTILE_DIM + DINO_DIM


def category_onehot(category: str | None) -> np.ndarray:
    out = np.zeros(len(CATEGORIES), dtype=np.float32)
    if category in CATEGORY_TO_INDEX:
        out[CATEGORY_TO_INDEX[category]] = 1.0
    return out


def token_dim(include_tactile: bool = False, include_dino: bool = False) -> int:
    return TOKEN_DIM + (TACTILE_DIM if include_tactile else 0) + (DINO_DIM if include_dino else 0)


def token_layout(
    include_tactile: bool = False, include_dino: bool = False
) -> dict[str, tuple[int, int]]:
    """Offsets of each token block, in concatenation order."""
    layout = {
        "vision": (0, VISION_DIM),
        "geometry": (VISION_DIM, VISION_DIM + GEOMETRY_DIM),
        "pose": (VISION_DIM + GEOMETRY_DIM, VISION_DIM + GEOMETRY_DIM + 3),
        "category": (
            VISION_DIM + GEOMETRY_DIM + 3,
            VISION_DIM + GEOMETRY_DIM + 3 + len(CATEGORIES),
        ),
    }
    offset = TOKEN_DIM
    if include_tactile:
        layout["tactile"] = (offset, offset + TACTILE_DIM)
        offset += TACTILE_DIM
    if include_dino:
        layout["dino"] = (offset, offset + DINO_DIM)
    return layout


def token_flags(dim: int) -> tuple[bool, bool]:
    """Infer (include_tactile, include_dino) from a saved token width."""
    if dim == FULL_TOKEN_DIM:
        return True, True
    if dim == DINO_TOKEN_DIM:
        return False, True
    if dim == TACTILE_TOKEN_DIM:
        return True, False
    if dim == TOKEN_DIM:
        return False, False
    raise ValueError(f"unknown token width: {dim}")


def entity_token(
    pos,
    raw: dict,
    category: str | None,
    include_tactile: bool = False,
    include_dino: bool = False,
) -> np.ndarray:
    parts = [
        np.asarray(raw["vision"], dtype=np.float32),
        np.asarray(raw["geometry"], dtype=np.float32),
        np.asarray(pos, dtype=np.float32),
        category_onehot(category),
    ]
    if include_tactile:
        parts.append(np.asarray(raw.get("tactile", np.zeros(TACTILE_DIM)), dtype=np.float32))
    if include_dino:
        parts.append(np.asarray(raw.get("dino", np.zeros(DINO_DIM)), dtype=np.float32))
    return np.concatenate(parts).astype(np.float32)


def pad_observations(
    observations,
    n: int = 3,
    include_tactile: bool = False,
    include_dino: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    dim = token_dim(include_tactile, include_dino)
    items = list(observations[:n])
    mask = np.zeros((n,), dtype=np.float32)

    # Fast path: every entity has both core modalities. Avoids one
    # concatenate + one one-hot allocation per entity, which dominates when
    # the pruned matcher scores hundreds of small components.
    raws = [
        obs.valid_raw() if getattr(obs, "streams", None) else getattr(obs, "raw", {}) or {}
        for obs in items
    ]
    if items and all("vision" in r and "geometry" in r for r in raws):
        count = len(items)
        vision = np.stack([np.asarray(r["vision"], dtype=np.float32) for r in raws])
        geometry = np.stack([np.asarray(r["geometry"], dtype=np.float32) for r in raws])
        pos = np.stack([np.asarray(obs.pos, dtype=np.float32) for obs in items])
        category = np.zeros((count, len(CATEGORIES)), dtype=np.float32)
        for i, obs in enumerate(items):
            c = getattr(obs, "category", None) or getattr(obs, "gt_category", None)
            if c in CATEGORY_TO_INDEX:
                category[i, CATEGORY_TO_INDEX[c]] = 1.0
        parts = [vision, geometry, pos, category]
        if include_tactile:
            parts.append(
                np.stack(
                    [
                        np.asarray(r.get("tactile", np.zeros(TACTILE_DIM)), dtype=np.float32)
                        for r in raws
                    ]
                )
            )
        if include_dino:
            parts.append(
                np.stack(
                    [
                        np.asarray(r.get("dino", np.zeros(DINO_DIM)), dtype=np.float32)
                        for r in raws
                    ]
                )
            )
        tokens = np.concatenate(parts, axis=1).astype(np.float32)
        if count < n:
            tokens = np.concatenate(
                [tokens, np.zeros((n - count, dim), dtype=np.float32)], axis=0
            )
        mask[:count] = 1.0
        return tokens, mask

    tokens = np.zeros((n, dim), dtype=np.float32)
    for i, obs in enumerate(items):
        category = getattr(obs, "category", None) or getattr(obs, "gt_category", None)
        raw = (
            obs.valid_raw()
            if getattr(obs, "streams", None)
            else getattr(obs, "raw", {}) or {}
        )
        if "vision" not in raw or "geometry" not in raw:
            continue
        tokens[i] = entity_token(
            obs.pos,
            raw,
            category,
            include_tactile=include_tactile,
            include_dino=include_dino,
        )
        mask[i] = 1.0
    return tokens, mask


class MetricEmbedding(nn.Module):
    def __init__(self, token_dim: int = TOKEN_DIM, hidden: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(token_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


def _batch(pairs, device):
    pre = torch.as_tensor(np.stack([p.pre for p in pairs]), dtype=torch.float32, device=device)
    post = torch.as_tensor(np.stack([p.post for p in pairs]), dtype=torch.float32, device=device)
    pre_mask = torch.as_tensor(np.stack([p.pre_mask for p in pairs]), dtype=torch.float32, device=device)
    post_mask = torch.as_tensor(np.stack([p.post_mask for p in pairs]), dtype=torch.float32, device=device)
    labels = torch.as_tensor(np.stack([p.labels for p in pairs]), dtype=torch.float32, device=device)
    return pre, post, pre_mask, post_mask, labels


def _assignment_accuracy(scores: np.ndarray, labels: np.ndarray) -> float:
    rows, cols = linear_sum_assignment(-scores)
    correct = sum(float(labels[r, c]) for r, c in zip(rows, cols))
    return correct / max(1, len(labels))


def evaluate_metric(model: MetricEmbedding, pairs, device: str = "cpu") -> dict:
    model.eval()
    rows = []
    with torch.no_grad():
        for pair in pairs:
            pre, post, pm, qm, labels = _batch([pair], device)
            z_pre = model(pre)[0]
            z_post = model(post)[0]
            sim = (z_pre @ z_post.T).cpu().numpy()
            rows.append(
                {
                    "sample": pair.sample_id,
                    "subset": pair.subset,
                    "reid_accuracy": _assignment_accuracy(sim, pair.labels),
                }
            )
    return _summarize(rows)


def train_metric(
    train_pairs,
    val_pairs,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 16,
    temperature: float = 0.1,
    seed: int = 0,
    device: str = "cpu",
    hidden: int = 256,
    out_dim: int = 128,
    assignment_weight: float = 0.0,
) -> tuple[MetricEmbedding, dict]:
    torch.manual_seed(seed)
    model = MetricEmbedding(TOKEN_DIM, hidden, out_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    best_state = None
    best_val = -1.0
    history = []
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(train_pairs))
        losses = []
        for start in range(0, len(order), batch_size):
            batch = [train_pairs[i] for i in order[start : start + batch_size]]
            pre, post, pm, qm, labels = _batch(batch, device)
            z_pre = model(pre)
            z_post = model(post)
            sim = torch.einsum("bid,bjd->bij", z_pre, z_post) / temperature
            mask = (pm[:, :, None] * qm[:, None, :]).bool()
            loss = F.binary_cross_entropy_with_logits(sim[mask], labels[mask])
            if assignment_weight > 0:
                # One-to-one assignment loss: each pre token should pick one
                # post token, and vice versa. This is the set-level signal that
                # pairwise cosine lacks.
                row_log = F.log_softmax(sim, dim=2)
                row_target = labels.argmax(dim=2)
                row_loss = -row_log.gather(2, row_target.unsqueeze(2)).squeeze(2)
                row_valid = (labels.sum(dim=2) > 0) & pm.bool()
                col_log = F.log_softmax(sim, dim=1)
                col_target = labels.argmax(dim=1)
                col_loss = -col_log.gather(1, col_target.unsqueeze(1)).squeeze(1)
                col_valid = (labels.sum(dim=1) > 0) & qm.bool()
                assign = 0.0
                if row_valid.any():
                    assign = assign + row_loss[row_valid].mean()
                if col_valid.any():
                    assign = assign + col_loss[col_valid].mean()
                loss = loss + float(assignment_weight) * assign
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        val = evaluate_metric(model, val_pairs, device)
        history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "val_reid": val["reid_accuracy"],
            }
        )
        if val["reid_accuracy"] > best_val:
            best_val = val["reid_accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_reid": best_val, "history": history}


def save_metric(model: MetricEmbedding, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    first = model.net[0]
    last = model.net[-1]
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hidden": int(first.out_features),
            "out_dim": int(last.out_features),
        },
        path,
    )


def load_metric(path: str | Path, device: str = "cpu") -> MetricEmbedding:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    model = MetricEmbedding(
        TOKEN_DIM,
        int(payload.get("hidden", 256)),
        int(payload.get("out_dim", 128)),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


class LearnedMetricReID:
    """Adapter used by the PSS tracker for learned re-ID."""

    def __init__(self, model: MetricEmbedding, pose_gate: float | None = None, device: str = "cpu"):
        self.model = model
        self.pose_gate = pose_gate
        self.device = device

    def cost_matrix(self, pre_observations, post_observations) -> np.ndarray:
        pre_tokens, pre_mask = pad_observations(pre_observations)
        post_tokens, post_mask = pad_observations(post_observations)
        with torch.no_grad():
            z_pre = self.model(
                torch.as_tensor(pre_tokens, dtype=torch.float32, device=self.device)
            ).cpu().numpy()
            z_post = self.model(
                torch.as_tensor(post_tokens, dtype=torch.float32, device=self.device)
            ).cpu().numpy()
        cost = 1.0 - z_pre @ z_post.T
        cost = cost[: len(pre_observations), : len(post_observations)]
        if self.pose_gate is not None:
            for i, a in enumerate(pre_observations):
                for j, b in enumerate(post_observations):
                    if np.linalg.norm(np.asarray(a.pos) - np.asarray(b.pos)) > self.pose_gate:
                        cost[i, j] = 1e6
        return cost

    def cost(self, pre, post) -> float:
        return float(self.cost_matrix([pre], [post])[0, 0])


def _summarize(rows: list[dict]) -> dict:
    def mean(key, subset=None):
        values = [r[key] for r in rows if subset is None or r.get("subset") == subset]
        return float(np.mean(values)) if values else 0.0

    return {
        "n": len(rows),
        "reid_accuracy": mean("reid_accuracy"),
        "per_subset": {
            subset: {
                "n": sum(1 for r in rows if r.get("subset") == subset),
                "reid_accuracy": mean("reid_accuracy", subset),
            }
            for subset in ("mixed", "samecat", "near_identical")
        },
        "rows": rows,
    }
