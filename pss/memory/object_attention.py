"""Minimal object-centric multi-head attention for PSS.

Instead of attending over tokens, each head attends over persistent object
slots. Attention logits combine content similarity with typed relation biases:

- spatial: relative position / distance;
- semantic: category match;
- tactile: tactile similarity.

The module returns updated object slots and per-head attention weights.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pss.memory.learned_metric import token_layout


class ObjectAttentionBlock(nn.Module):
    def __init__(self, d_model: int = 128, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert d_model % heads == 0
        self.d_model = d_model
        self.heads = heads
        self.dh = d_model // heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
        self.spatial = nn.Linear(4, heads)
        self.semantic = nn.Parameter(torch.zeros(heads))
        self.tactile = nn.Parameter(torch.zeros(heads))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, h, valid, pose, category, tactile):
        b, n, _ = h.shape
        qkv = self.qkv(h).view(b, n, 3, self.heads, self.dh)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dh)

        # Spatial bias: [B, N, N, 4] -> [B, H, N, N]
        pos = pose[:, :, None, :] - pose[:, None, :, :]
        dist = torch.linalg.norm(pos, dim=-1, keepdim=True)
        rel = torch.cat([pos, dist], dim=-1)
        spatial = self.spatial(rel).permute(0, 3, 1, 2)
        logits = logits + spatial

        cat = torch.matmul(category, category.transpose(-1, -2))
        logits = logits + cat.unsqueeze(1) * self.semantic.view(1, -1, 1, 1)

        tac = F.normalize(tactile, dim=-1)
        tac_sim = torch.matmul(tac, tac.transpose(-1, -2))
        logits = logits + tac_sim.unsqueeze(1) * self.tactile.view(1, -1, 1, 1)

        key_valid = valid[:, None, None, :].bool()
        logits = logits.masked_fill(~key_valid, -1e9)
        weights = torch.softmax(logits, dim=-1)
        weights = weights * valid[:, None, :, None].float()
        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).reshape(b, n, self.d_model)
        h = self.norm1(h + self.out(out))
        h = self.norm2(h + self.ffn(h))
        return h, weights


class ObjectAttentionEncoder(nn.Module):
    def __init__(
        self,
        token_dim: int,
        d_model: int = 128,
        heads: int = 4,
        layers: int = 2,
        include_tactile: bool = True,
    ):
        super().__init__()
        self.proj = nn.Linear(token_dim, d_model)
        self.blocks = nn.ModuleList(
            [ObjectAttentionBlock(d_model, heads) for _ in range(layers)]
        )
        self.include_tactile = bool(include_tactile)
        self.layout = token_layout(include_tactile, False)

    def split(self, tokens):
        pose = tokens[:, :, self.layout["pose"][0] : self.layout["pose"][1]]
        category = tokens[:, :, self.layout["category"][0] : self.layout["category"][1]]
        if "tactile" in self.layout:
            s, e = self.layout["tactile"]
            tactile = tokens[:, :, s:e]
        else:
            tactile = torch.zeros_like(pose)
        return pose, category, tactile

    def forward(self, tokens, valid):
        pose, category, tactile = self.split(tokens)
        h = self.proj(tokens)
        weights = None
        for block in self.blocks:
            h, weights = block(h, valid, pose, category, tactile)
        return h, weights


class ObjectAttentionMatcher(nn.Module):
    def __init__(
        self,
        token_dim: int,
        d_model: int = 128,
        heads: int = 4,
        layers: int = 2,
        include_tactile: bool = True,
    ):
        super().__init__()
        self.encoder = ObjectAttentionEncoder(
            token_dim, d_model, heads, layers, include_tactile
        )
        self.pair = nn.Sequential(
            nn.Linear(d_model * 4, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, pre, post, pre_mask, post_mask):
        hp, _ = self.encoder(pre, pre_mask)
        hq, _ = self.encoder(post, post_mask)
        n, m = hp.shape[1], hq.shape[1]
        hp = hp.unsqueeze(2).expand(-1, n, m, -1)
        hq = hq.unsqueeze(1).expand(-1, n, m, -1)
        pair = torch.cat([hp, hq, hp - hq, hp * hq], dim=-1)
        return self.pair(pair).squeeze(-1)


def _batch(pairs, device):
    pre = torch.as_tensor(np.stack([p.pre for p in pairs]), dtype=torch.float32, device=device)
    post = torch.as_tensor(np.stack([p.post for p in pairs]), dtype=torch.float32, device=device)
    pm = torch.as_tensor(np.stack([p.pre_mask for p in pairs]), dtype=torch.float32, device=device)
    qm = torch.as_tensor(np.stack([p.post_mask for p in pairs]), dtype=torch.float32, device=device)
    labels = torch.as_tensor(np.stack([p.labels for p in pairs]), dtype=torch.float32, device=device)
    return pre, post, pm, qm, labels


def _assignment_accuracy(scores, labels):
    from scipy.optimize import linear_sum_assignment

    rows, cols = linear_sum_assignment(-scores)
    correct = sum(float(labels[r, c]) for r, c in zip(rows, cols))
    return correct / max(1, len(labels))


def evaluate_object_attention(model, pairs, device="cpu"):
    model.eval()
    rows = []
    with torch.no_grad():
        for pair in pairs:
            pre, post, pm, qm, labels = _batch([pair], device)
            scores = model(pre, post, pm, qm).cpu().numpy()[0]
            rows.append(
                {
                    "sample": pair.sample_id,
                    "subset": pair.subset,
                    "reid_accuracy": _assignment_accuracy(scores, pair.labels),
                }
            )
    return _summarize(rows)


def train_object_attention(
    train_pairs,
    val_pairs,
    token_dim,
    epochs=200,
    lr=1e-3,
    batch_size=16,
    seed=0,
    device="cpu",
):
    torch.manual_seed(seed)
    model = ObjectAttentionMatcher(token_dim=token_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    best_state, best_val = None, -1.0
    for _epoch in range(epochs):
        model.train()
        order = rng.permutation(len(train_pairs))
        for start in range(0, len(order), batch_size):
            batch = [train_pairs[i] for i in order[start : start + batch_size]]
            pre, post, pm, qm, labels = _batch(batch, device)
            logits = model(pre, post, pm, qm)
            mask = (pm[:, :, None] * qm[:, None, :]).bool()
            loss = F.binary_cross_entropy_with_logits(logits[mask], labels[mask])
            opt.zero_grad()
            loss.backward()
            opt.step()
        val = evaluate_object_attention(model, val_pairs, device)
        if val["reid_accuracy"] > best_val:
            best_val = val["reid_accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {"best_val_reid": best_val}


def _summarize(rows):
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


class ObjectAttentionReID:
    """Tracker adapter: object attention scores -> re-ID cost matrix."""

    def __init__(self, model, device="cpu", max_cost: float = 0.9, include_tactile: bool = True):
        self.model = model
        self.device = device
        self.max_cost = float(max_cost)
        self.include_tactile = bool(include_tactile)

    def cost_matrix(self, tracks, observations) -> np.ndarray:
        from pss.memory.learned_metric import pad_observations

        n = max(3, len(tracks), len(observations))
        pre, pre_mask = pad_observations(
            tracks, n, include_tactile=self.include_tactile
        )
        post, post_mask = pad_observations(
            observations, n, include_tactile=self.include_tactile
        )
        with torch.no_grad():
            scores = self.model(
                torch.as_tensor(pre, dtype=torch.float32, device=self.device)[None],
                torch.as_tensor(post, dtype=torch.float32, device=self.device)[None],
                torch.as_tensor(pre_mask, dtype=torch.float32, device=self.device)[None],
                torch.as_tensor(post_mask, dtype=torch.float32, device=self.device)[None],
            ).sigmoid().cpu().numpy()[0]
        cost = 1.0 - scores[: len(tracks), : len(observations)]
        invalid = (pre_mask[: len(tracks)] <= 0)[:, None] | (
            post_mask[: len(observations)] <= 0
        )[None, :]
        cost[invalid] = 1e6
        return cost

    def cost(self, pre, post) -> float:
        return float(self.cost_matrix([pre], [post])[0, 0])


def save_object_attention(model, path) -> None:
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    first = model.encoder.proj
    torch.save(
        {
            "state_dict": model.state_dict(),
            "token_dim": int(first.in_features),
            "d_model": int(model.encoder.proj.out_features),
            "heads": int(model.encoder.blocks[0].heads),
            "layers": int(len(model.encoder.blocks)),
            "include_tactile": bool(model.encoder.include_tactile),
        },
        path,
    )


def load_object_attention(path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    model = ObjectAttentionMatcher(
        token_dim=int(payload["token_dim"]),
        d_model=int(payload.get("d_model", 128)),
        heads=int(payload.get("heads", 4)),
        layers=int(payload.get("layers", 2)),
        include_tactile=bool(payload.get("include_tactile", True)),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
