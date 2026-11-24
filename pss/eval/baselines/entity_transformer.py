"""Entity-token transformer baseline for PSS P1.

Both PSS and this baseline receive the same per-object tokens:
vision projection, geometry projection, pose, and category one-hot.
PSS scores pairs with the hand-designed re-ID cost; the transformer learns a
cross-attention pair score from the train split.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from pss.encoders.data import load_index
from pss.memory.learned_metric import GEOMETRY_DIM, TOKEN_DIM, VISION_DIM, pad_observations
from pss.memory.perception import PoseNoise, extract_frame, load_perception_models
from pss.memory.tracking import TrackerConfig, _category_distance
from pss.sim.dataset import load_sample


@dataclass
class TokenPair:
    pre: np.ndarray  # (3, TOKEN_DIM)
    post: np.ndarray  # (3, TOKEN_DIM)
    pre_mask: np.ndarray  # (3,)
    post_mask: np.ndarray  # (3,)
    labels: np.ndarray  # (3, 3)
    subset: str
    sample_id: str


def sequence_pair(
    sample: dict,
    models,
    pre_t: int | None = None,
    post_t: int | None = None,
    include_tactile: bool = False,
    include_dino: bool = False,
) -> TokenPair:
    pre_t = int(sample["occlusion_start"]) - 1 if pre_t is None else int(pre_t)
    post_t = int(sample["occlusion_end"]) if post_t is None else int(post_t)
    pre = extract_frame(sample, pre_t, models, PoseNoise(seed=0, enabled=False))
    post = extract_frame(sample, post_t, models, PoseNoise(seed=0, enabled=False))
    n = max(3, len(pre), len(post))
    pre_tokens, pre_mask = pad_observations(
        pre,
        n=n,
        include_tactile=include_tactile,
        include_dino=include_dino,
    )
    post_tokens, post_mask = pad_observations(
        post,
        n=n,
        include_tactile=include_tactile,
        include_dino=include_dino,
    )
    labels = np.zeros((n, n), dtype=np.float32)
    for i, a in enumerate(pre[:n]):
        for j, b in enumerate(post[:n]):
            labels[i, j] = float(a.gt_id == b.gt_id)
    return TokenPair(
        pre=pre_tokens,
        post=post_tokens,
        pre_mask=pre_mask,
        post_mask=post_mask,
        labels=labels,
        subset=str(sample.get("subset", "unknown")),
        sample_id=str(sample.get("seed", "")),
    )


def build_pairs(
    data_dir: str | Path,
    expert_dir: str | Path,
    projection_dir: str | Path,
    split: str,
    device: str = "cpu",
    include_tactile: bool = False,
    include_dino: bool = False,
    vision_expert=None,
    tactile_expert=None,
) -> list[TokenPair]:
    data_dir = Path(data_dir)
    index = load_index(data_dir)
    models = load_perception_models(
        expert_dir,
        projection_dir,
        prototype_dir=None,
        device=device,
        dino=include_dino,
        vision_expert=vision_expert,
        tactile_expert=tactile_expert,
    )
    pairs = []
    for entry in index["entries"]:
        if entry.get("kind") != "occlusion" or entry.get("split") != split:
            continue
        sample = load_sample(data_dir / entry["path"])
        pairs.append(
            sequence_pair(
                sample,
                models,
                include_tactile=include_tactile,
                include_dino=include_dino,
            )
        )
    return pairs


def pss_cost_matrix(pre, post, cfg: TrackerConfig | None = None) -> np.ndarray:
    cfg = cfg or TrackerConfig()
    n_pre = len(pre)
    n_post = len(post)
    cost = np.full((n_pre, n_post), 1e6, dtype=float)
    for i, a in enumerate(pre):
        pa = np.asarray(a.pos, dtype=float)
        aa = np.asarray(a.appearance, dtype=float)
        na = np.linalg.norm(aa)
        ga = np.asarray(a.raw.get("geometry", np.zeros_like(aa)), dtype=float)
        nga = np.linalg.norm(ga)
        for j, b in enumerate(post):
            d_pos = float(np.linalg.norm(pa - np.asarray(b.pos, dtype=float)))
            if d_pos > cfg.reid_pose_gate:
                continue
            bb = np.asarray(b.appearance, dtype=float)
            nb = np.linalg.norm(bb)
            d_vis = 1.0 - float(np.dot(aa, bb) / (na * nb)) if na and nb else 1.0
            gb = np.asarray(b.raw.get("geometry", np.zeros_like(bb)), dtype=float)
            ngb = np.linalg.norm(gb)
            d_geo = 1.0 - float(np.dot(ga, gb) / (nga * ngb)) if nga and ngb else 1.0
            d_app = 0.5 * d_vis + 0.5 * d_geo
            d_cat = _category_distance(a.category, b.category)
            cost[i, j] = (
                cfg.reid_w_pose * (d_pos / cfg.reid_pose_gate)
                + cfg.reid_w_app * d_app
                + 0.1 * d_cat
            )
    return cost


class EntityTransformer(nn.Module):
    def __init__(
        self,
        token_dim: int = TOKEN_DIM,
        d_model: int = 128,
        nhead: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
        sparse_topk: int | None = None,
    ):
        super().__init__()
        self.sparse_topk = None if sparse_topk is None else int(sparse_topk)
        self.nhead = int(nhead)
        self.pre_proj = nn.Linear(token_dim, d_model)
        self.post_proj = nn.Linear(token_dim, d_model)
        self.type_embed = nn.Embedding(2, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.d_model = d_model

    def forward(self, pre, post, pre_mask, post_mask):
        pre_h = self.pre_proj(pre) + self.type_embed(
            torch.zeros(pre.shape[0], dtype=torch.long, device=pre.device)
        ).unsqueeze(1)
        post_h = self.post_proj(post) + self.type_embed(
            torch.ones(post.shape[0], dtype=torch.long, device=post.device)
        ).unsqueeze(1)
        x = torch.cat([pre_h, post_h], dim=1)
        valid = torch.cat([pre_mask, post_mask], dim=1).bool()
        mask = None
        if self.sparse_topk is not None and self.sparse_topk > 0:
            pose_start = VISION_DIM + GEOMETRY_DIM
            poses = torch.cat([pre, post], dim=1)[:, :, pose_start : pose_start + 3]
            dist = torch.cdist(poses, poses)
            k = min(self.sparse_topk + 1, dist.shape[-1])
            keep = dist.topk(k, dim=2, largest=False).indices
            allowed = torch.zeros_like(dist, dtype=torch.bool)
            allowed.scatter_(2, keep, True)
            allowed |= torch.eye(dist.shape[-1], dtype=torch.bool, device=dist.device)
            mask = (~allowed).repeat_interleave(self.nhead, dim=0)
        h = self.encoder(x, mask=mask, src_key_padding_mask=~valid)
        n = pre.shape[1]
        h_pre, h_post = h[:, :n], h[:, n:]
        return torch.einsum("bid,bjd->bij", h_pre, h_post) / math.sqrt(self.d_model)


def _batch(pairs, device):
    pre = torch.as_tensor(np.stack([p.pre for p in pairs]), dtype=torch.float32, device=device)
    post = torch.as_tensor(np.stack([p.post for p in pairs]), dtype=torch.float32, device=device)
    pre_mask = torch.as_tensor(np.stack([p.pre_mask for p in pairs]), dtype=torch.float32, device=device)
    post_mask = torch.as_tensor(np.stack([p.post_mask for p in pairs]), dtype=torch.float32, device=device)
    labels = torch.as_tensor(np.stack([p.labels for p in pairs]), dtype=torch.float32, device=device)
    return pre, post, pre_mask, post_mask, labels


def _assignment_accuracy(scores: np.ndarray, labels: np.ndarray, maximize: bool) -> float:
    matrix = -scores if maximize else scores
    rows, cols = linear_sum_assignment(matrix)
    correct = sum(float(labels[r, c]) for r, c in zip(rows, cols))
    return correct / max(1, len(labels))


def evaluate_pss_tracker(
    data_dir: str | Path,
    expert_dir: str | Path,
    projection_dir: str | Path,
    split: str = "test",
    device: str = "cpu",
    cfg: TrackerConfig | None = None,
) -> dict:
    data_dir = Path(data_dir)
    index = load_index(data_dir)
    models = load_perception_models(
        expert_dir, projection_dir, prototype_dir=None, device=device
    )
    cfg = cfg or TrackerConfig()
    rows = []
    for entry in index["entries"]:
        if entry.get("kind") != "occlusion" or entry.get("split") != split:
            continue
        sample = load_sample(data_dir / entry["path"])
        pre_t = int(sample["occlusion_start"]) - 1
        post_t = int(sample["occlusion_end"])
        pre = extract_frame(sample, pre_t, models, PoseNoise(seed=0, enabled=False))
        post = extract_frame(sample, post_t, models, PoseNoise(seed=0, enabled=False))
        cost = pss_cost_matrix(pre, post, cfg)
        assign_rows, assign_cols = linear_sum_assignment(cost)
        correct = sum(
            1
            for r, c in zip(assign_rows, assign_cols)
            if pre[r].gt_id == post[c].gt_id
        )
        rows.append(
            {
                "sample": entry["id"],
                "subset": entry.get("subset", "unknown"),
                "reid_accuracy": correct / max(1, min(len(pre), len(post))),
                "n_pre": len(pre),
                "n_post": len(post),
            }
        )
    return _summarize(rows)


def train_transformer(
    train_pairs: list[TokenPair],
    val_pairs: list[TokenPair],
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 16,
    seed: int = 0,
    device: str = "cpu",
    token_dim: int = TOKEN_DIM,
    sparse_topk: int | None = None,
) -> tuple[EntityTransformer, dict]:
    torch.manual_seed(seed)
    model = EntityTransformer(token_dim=token_dim, sparse_topk=sparse_topk).to(device)
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
            scores = model(pre, post, pm, qm)
            mask = (pm[:, :, None] * qm[:, None, :]).bool()
            loss = F.binary_cross_entropy_with_logits(scores[mask], labels[mask])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        val = evaluate_transformer(model, val_pairs, device)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_reid": val["reid_accuracy"]})
        if val["reid_accuracy"] > best_val:
            best_val = val["reid_accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_reid": best_val, "history": history}


def evaluate_transformer(
    model: EntityTransformer,
    pairs: list[TokenPair],
    device: str = "cpu",
) -> dict:
    model.eval()
    rows = []
    with torch.no_grad():
        for pair in pairs:
            pre, post, pm, qm, labels = _batch([pair], device)
            scores = model(pre, post, pm, qm).cpu().numpy()[0]
            acc = _assignment_accuracy(scores, pair.labels, maximize=True)
            rows.append(
                {
                    "sample": pair.sample_id,
                    "subset": pair.subset,
                    "reid_accuracy": acc,
                }
            )
    return _summarize(rows)


def _summarize(rows: list[dict]) -> dict:
    def mean(key, subset=None):
        values = [
            r[key] for r in rows if subset is None or r.get("subset") == subset
        ]
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
