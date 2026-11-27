"""Learnable typed relation layer for the PSS scene graph.

The first version of PSS wrote relation edges by hand: relative position and
category match in the graph matcher, geometric ``in/contact/near`` records in
the reasoning layer. This module replaces the hand-written *edge features*
with typed relation heads:

- spatial:  left / right / above / below / near
- semantic: same_category
- tactile:  contact / similar_tactile
- temporal: approaching / receding / stable (contextual; see below)
- task:     goal-satisfied (contextual; used by the P5b planner)

The first three are pairwise and trained end-to-end from simulation labels.
Temporal and task need context (track history / a goal) and are represented by
the same schema but are supplied by the caller; this module ships the schema
and scene-graph write path, and the experiment script evaluates the pairwise
heads and a learned spatial relation head on P5b.

Design rule: relation logits live *next to* the symbolic predicates, not
instead of them. The experiment results show that learned edges are a
robustness/calibration layer, while exact geometric predicates remain stronger
on clean grid geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pss.memory.learned_metric import (
    CATEGORIES,
    TACTILE_TOKEN_DIM,
    entity_token,
    token_layout,
)
from pss.memory.scene_graph import SceneGraph, Track
from pss.sim.scene import CATEGORY_SPECS

PAIRWISE_TYPES = ("spatial", "semantic", "tactile")
RELATION_TYPES = PAIRWISE_TYPES + ("temporal", "task")

PREDICATES = {
    "spatial": ("left", "right", "above", "below", "near"),
    "semantic": ("same_category",),
    "tactile": ("contact", "similar_tactile"),
    "temporal": ("approaching", "receding", "stable"),
    "task": ("goal_satisfied",),
}

NEAR_DIST = 0.12
SIMILAR_TACTILE = 0.80
CONTACT_PAD = 0.02
N_SLOTS = 3


def _cat(o) -> str | None:
    return getattr(o, "category", None) or getattr(o, "gt_category", None)


def _radius(category: str | None) -> float:
    size = CATEGORY_SPECS.get(category or "", {}).get("size", (0.05, 0.05))
    return float(max(size)) / 2.0


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


# ── labels ──────────────────────────────────────────────────────────────────


@dataclass
class FrameTargets:
    spatial: np.ndarray  # (N, N, 5)
    semantic: np.ndarray  # (N, N, 1)
    tactile: np.ndarray  # (N, N, 2)
    valid: np.ndarray  # (N, N)

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "spatial": self.spatial,
            "semantic": self.semantic,
            "tactile": self.tactile,
        }


def frame_targets(observations, n_slots: int = N_SLOTS) -> FrameTargets:
    """Derive frame-local relation labels from observations (training only)."""
    n = min(len(observations), n_slots)
    spatial = np.zeros((n_slots, n_slots, len(PREDICATES["spatial"])), dtype=np.float32)
    semantic = np.zeros((n_slots, n_slots, 1), dtype=np.float32)
    tactile = np.zeros((n_slots, n_slots, 2), dtype=np.float32)
    valid = np.zeros((n_slots, n_slots), dtype=np.float32)
    radii = [_radius(_cat(o)) for o in observations[:n]]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a, b = observations[i], observations[j]
            dx = float(b.pos[0] - a.pos[0])
            dy = float(b.pos[1] - a.pos[1])
            d = float(np.hypot(dx, dy))
            spatial[i, j] = [dx < 0, dx > 0, dy > 0, dy < 0, d < NEAR_DIST]
            semantic[i, j] = float(_cat(a) == _cat(b))
            ta = np.asarray(a.raw.get("tactile", np.zeros(128)), dtype=np.float32)
            tb = np.asarray(b.raw.get("tactile", np.zeros(128)), dtype=np.float32)
            sim = float(np.dot(ta, tb) / (np.linalg.norm(ta) * np.linalg.norm(tb) + 1e-9))
            contact = float(d <= radii[i] + radii[j] + CONTACT_PAD)
            tactile[i, j] = [contact, float(sim > SIMILAR_TACTILE)]
            valid[i, j] = 1.0
    return FrameTargets(spatial, semantic, tactile, valid)


def temporal_targets(dist_pre: float, dist_post: float, eps: float = 0.01) -> np.ndarray:
    """One-hot temporal label from a pair's distance change."""
    delta = float(dist_post - dist_pre)
    out = np.zeros(3, dtype=np.float32)
    if delta < -eps:
        out[0] = 1.0
    elif delta > eps:
        out[1] = 1.0
    else:
        out[2] = 1.0
    return out


# ── data ────────────────────────────────────────────────────────────────────


@dataclass
class RelationPair:
    pre: np.ndarray
    post: np.ndarray
    pre_mask: np.ndarray
    post_mask: np.ndarray
    pre_targets: FrameTargets
    post_targets: FrameTargets
    labels: np.ndarray
    subset: str
    sample_id: str
    gt_pre: list[str] = field(default_factory=list)
    gt_post: list[str] = field(default_factory=list)


def _tokens(observations, n_slots: int = N_SLOTS):
    toks = np.zeros((n_slots, TACTILE_TOKEN_DIM), dtype=np.float32)
    mask = np.zeros(n_slots, dtype=np.float32)
    ids = []
    for i, o in enumerate(observations[:n_slots]):
        toks[i] = entity_token(
            o.pos, o.raw, _cat(o), include_tactile=True
        )
        mask[i] = 1.0
        ids.append(str(o.gt_id))
    return toks, mask, ids


def collect_relation_pairs(data_dir, split: str, device: str = "cpu", **perception_kwargs):
    """Build pre/post relation pairs from the tracking bench (GT used only for labels)."""
    from pss.encoders.data import load_index
    from pss.memory.perception import PoseNoise, extract_frame, load_perception_models
    from pss.sim.dataset import load_sample

    index = load_index(data_dir)
    models = load_perception_models(device=device, **perception_kwargs)
    out: list[RelationPair] = []
    for entry in index["entries"]:
        if entry.get("kind") != "occlusion" or entry.get("split") != split:
            continue
        sample = load_sample(f"{data_dir}/{entry['path']}")
        f0 = int(sample["occlusion_start"]) - 1
        f1 = int(sample["occlusion_end"])
        pre_obs = extract_frame(sample, f0, models, PoseNoise(seed=0, enabled=False))
        post_obs = extract_frame(sample, f1, models, PoseNoise(seed=0, enabled=False))
        pre, pre_m, pre_ids = _tokens(pre_obs)
        post, post_m, post_ids = _tokens(post_obs)
        labels = np.zeros((N_SLOTS, N_SLOTS), dtype=np.float32)
        for i, a in enumerate(pre_ids):
            for j, b in enumerate(post_ids):
                labels[i, j] = float(a == b)
        out.append(
            RelationPair(
                pre=pre,
                post=post,
                pre_mask=pre_m,
                post_mask=post_m,
                pre_targets=frame_targets(pre_obs),
                post_targets=frame_targets(post_obs),
                labels=labels,
                subset=str(entry.get("subset", "unknown")),
                sample_id=str(entry.get("id", "")),
                gt_pre=pre_ids,
                gt_post=post_ids,
            )
        )
    return out


def perturb_tokens(tokens: np.ndarray, kind: str, sigma: float = 0.03, seed: int = 0):
    """Corrupt a token array for the robustness evaluation."""
    rng = np.random.default_rng(seed)
    x = np.asarray(tokens, dtype=np.float32).copy()
    layout = token_layout(True, False)
    if kind == "pose":
        s, e = layout["pose"]
        x[..., s:e] += rng.normal(scale=sigma, size=x[..., s:e].shape)
    elif kind == "tactile":
        s, e = layout["tactile"]
        x[..., s:e] = 0.0
    elif kind == "vision":
        x[..., : layout["vision"][1]] = 0.0
    elif kind == "mixed":
        s, e = layout["pose"]
        x[..., s:e] += rng.normal(scale=sigma, size=x[..., s:e].shape)
        s, e = layout["tactile"]
        x[..., s:e] *= 0.3
    else:
        raise ValueError(f"unknown corruption {kind!r}")
    return x


def _batch(pairs: list[RelationPair], device: str):
    pre = torch.as_tensor(np.stack([p.pre for p in pairs]), device=device)
    post = torch.as_tensor(np.stack([p.post for p in pairs]), device=device)
    pm = torch.as_tensor(np.stack([p.pre_mask for p in pairs]), device=device)
    qm = torch.as_tensor(np.stack([p.post_mask for p in pairs]), device=device)
    labels = torch.as_tensor(np.stack([p.labels for p in pairs]), device=device)
    targets = {
        split: {
            t: torch.as_tensor(
                np.stack([getattr(p, f"{split}_targets").as_dict()[t] for p in pairs]),
                device=device,
            )
            for t in PAIRWISE_TYPES
        }
        for split in ("pre", "post")
    }
    valid = {
        split: torch.as_tensor(
            np.stack([getattr(p, f"{split}_targets").valid for p in pairs]),
            device=device,
        )
        for split in ("pre", "post")
    }
    return pre, post, pm, qm, labels, targets, valid


# ── heads ───────────────────────────────────────────────────────────────────


class TypedRelationHead(nn.Module):
    """Pairwise typed relation head over entity tokens."""

    def __init__(self, token_dim: int = TACTILE_TOKEN_DIM, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.pair = nn.Sequential(
            nn.Linear(token_dim * 4, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.heads = nn.ModuleDict(
            {
                name: nn.Linear(hidden, len(PREDICATES[name]))
                for name in PAIRWISE_TYPES
            }
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> dict[str, torch.Tensor]:
        aa = a.unsqueeze(2)
        bb = b.unsqueeze(1)
        pair = torch.cat(
            [
                aa.expand(-1, -1, b.shape[1], -1),
                bb.expand(-1, a.shape[1], -1, -1),
                aa - bb,
                aa * bb,
            ],
            dim=-1,
        )
        h = self.pair(pair)
        return {name: head(h) for name, head in self.heads.items()}

    def probabilities(self, a: torch.Tensor, b: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: torch.sigmoid(logits) for name, logits in self.forward(a, b).items()}


class SpatialRelationHead(nn.Module):
    """Coordinate pair -> left/right/above/below, for the P5b grid.

    ``feature_mode="absolute"`` uses the original 8-dim pair feature
    ``[ax, ay, bx, by, dx, dy, |dx|, |dy|]``.
    ``feature_mode="relative"`` drops the absolute positions and uses only
    ``[dx, dy, |dx|, |dy|]``; this tests whether the head can rely on the
    relative geometry alone instead of memorising absolute grid positions.
    """

    def __init__(self, hidden: int = 64, feature_mode: str = "absolute"):
        super().__init__()
        if feature_mode not in ("absolute", "relative"):
            raise ValueError(f"unknown feature_mode {feature_mode!r}")
        self.feature_mode = feature_mode
        in_dim = 8 if feature_mode == "absolute" else 4
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 4),
        )

    @staticmethod
    def features(pa, pb, mode: str = "absolute") -> torch.Tensor:
        if mode not in ("absolute", "relative"):
            raise ValueError(f"unknown feature mode {mode!r}")
        pa = torch.as_tensor(pa, dtype=torch.float32)
        pb = torch.as_tensor(pb, dtype=torch.float32)
        dx = pb[..., 0] - pa[..., 0]
        dy = pb[..., 1] - pa[..., 1]
        if mode == "relative":
            return torch.stack([dx, dy, dx.abs(), dy.abs()], dim=-1)
        return torch.stack(
            [
                pa[..., 0], pa[..., 1], pb[..., 0], pb[..., 1],
                dx, dy, dx.abs(), dy.abs(),
            ],
            dim=-1,
        )

    def forward(self, pa, pb) -> torch.Tensor:
        return self.net(self.features(pa, pb, self.feature_mode))

    @staticmethod
    def labels(pa, pb, eps: float = 0.1) -> torch.Tensor:
        """left/right/above/below with ``a`` as the subject (matches ``_rel``).

        ``eps`` is a dead zone for aligned pairs, mirroring the strict
        geometric predicate.
        """
        pa = torch.as_tensor(pa, dtype=torch.float32)
        pb = torch.as_tensor(pb, dtype=torch.float32)
        dx = pb[..., 0] - pa[..., 0]
        dy = pb[..., 1] - pa[..., 1]
        return torch.stack(
            [dx > eps, dx < -eps, dy < -eps, dy > eps], dim=-1
        ).to(torch.float32)


class SpatialRelationGate(nn.Module):
    """Pair-level gate: predict which relation expert is correct.

    The gate consumes a feature vector per object pair (geometry margins,
    learned-head scores, a local noise estimate, contact/near flags, ...) and
    emits one logit per candidate expert.  The logit is trained against
    ``expert_prediction == true_relation``, so ``sigmoid(logit)`` is an
    estimate of that expert's correctness.  Routing is ``argmax`` over the
    experts, with the caller free to override the choice on OOD inputs.

    This is deliberately a *routing* head, not another relation predictor:
    the symbolic geometric predicate stays available as the safe fallback.
    """

    def __init__(self, in_dim: int, n_experts: int, hidden: int = 32, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.GELU()]
        if dropout:
            layers.append(nn.Dropout(dropout))
        layers += [nn.Linear(hidden, hidden), nn.GELU()]
        if dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden, n_experts))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def correctness(self, x: torch.Tensor) -> torch.Tensor:
        """Per-expert probability that the expert's 4-bit prediction is right."""
        return torch.sigmoid(self.forward(x))


# ── relation head training / evaluation ─────────────────────────────────────


def train_relation_head(
    train_pairs,
    val_pairs=None,
    epochs: int = 60,
    lr: float = 1e-3,
    batch_size: int = 32,
    seed: int = 0,
    device: str = "cpu",
    hidden: int = 128,
):
    torch.manual_seed(seed)
    head = TypedRelationHead(hidden=hidden).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    best, best_state = -1.0, None
    for _epoch in range(epochs):
        head.train()
        order = rng.permutation(len(train_pairs))
        for start in range(0, len(order), batch_size):
            batch = [train_pairs[i] for i in order[start : start + batch_size]]
            pre, post, _pm, _qm, _labels, targets, valid = _batch(batch, device)
            loss = 0.0
            terms = 0
            for split, tokens in (("pre", pre), ("post", post)):
                logits = head(tokens, tokens)
                for name in PAIRWISE_TYPES:
                    m = valid[split].bool()
                    loss = loss + F.binary_cross_entropy_with_logits(
                        logits[name][m], targets[split][name][m]
                    )
                    terms += 1
            loss = loss / max(1, terms)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if val_pairs:
            score = evaluate_relation_head(head, val_pairs, device)["overall_f1"]
            if score > best:
                best = score
                best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()
    return head, {"best_val_f1": best}


def _f1(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def evaluate_relation_head(
    head: TypedRelationHead,
    pairs: list[RelationPair],
    device: str = "cpu",
    corrupt: str | None = None,
    batch_size: int = 32,
) -> dict:
    head.eval()
    counts = {name: {"tp": 0, "fp": 0, "fn": 0} for name in PAIRWISE_TYPES}
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start : start + batch_size]
            pre = np.stack([p.pre for p in chunk])
            post = np.stack([p.post for p in chunk])
            if corrupt:
                pre = np.stack(
                    [perturb_tokens(x, corrupt, seed=start + i) for i, x in enumerate(pre)]
                )
                post = np.stack(
                    [perturb_tokens(x, corrupt, seed=start + 1000 + i) for i, x in enumerate(post)]
                )
            pre_t = torch.as_tensor(pre, device=device)
            post_t = torch.as_tensor(post, device=device)
            targets = {
                split: {
                    t: torch.as_tensor(
                        np.stack([getattr(p, f"{split}_targets").as_dict()[t] for p in chunk]),
                        device=device,
                    )
                    for t in PAIRWISE_TYPES
                }
                for split in ("pre", "post")
            }
            valid = {
                split: torch.as_tensor(
                    np.stack([getattr(p, f"{split}_targets").valid for p in chunk]),
                    device=device,
                )
                for split in ("pre", "post")
            }
            for split, tokens in (("pre", pre_t), ("post", post_t)):
                logits = head(tokens, tokens)
                for name in PAIRWISE_TYPES:
                    pred = (torch.sigmoid(logits[name]) > 0.5)
                    truth = targets[split][name] > 0.5
                    m = valid[split].bool().unsqueeze(-1).expand_as(pred)
                    counts[name]["tp"] += int((pred & truth & m).sum())
                    counts[name]["fp"] += int((pred & ~truth & m).sum())
                    counts[name]["fn"] += int((~pred & truth & m).sum())
    per = {name: _f1(**counts[name]) for name in PAIRWISE_TYPES}
    f1s = [v["f1"] for v in per.values()]
    return {"per_type": per, "overall_f1": float(np.mean(f1s)), "corrupt": corrupt}


# ── relation-aware matchers ─────────────────────────────────────────────────


def _rel_feat(head: TypedRelationHead, a, b) -> torch.Tensor:
    probs = head.probabilities(a, b)
    return torch.cat([probs[name] for name in PAIRWISE_TYPES], dim=-1)


class RelationAwareGraphMatcher(nn.Module):
    """Pure learned-edge matcher: no hand-written distance/category edge."""

    def __init__(self, token_dim: int = TACTILE_TOKEN_DIM, rel_head=None, d_model: int = 128, layers: int = 2):
        super().__init__()
        self.rel_head = rel_head or TypedRelationHead(token_dim)
        for p in self.rel_head.parameters():
            p.requires_grad = False
        self.node_proj = nn.Linear(token_dim, d_model)
        self.ctx_proj = nn.Linear(sum(len(PREDICATES[t]) for t in PAIRWISE_TYPES), d_model)
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model * 2, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
                )
                for _ in range(layers)
            ]
        )
        feat = sum(len(PREDICATES[t]) for t in PAIRWISE_TYPES)
        self.pair_head = nn.Sequential(
            nn.Linear(d_model * 2 + feat * 3, 128), nn.ReLU(), nn.Linear(128, 1)
        )

    def encode(self, tokens, mask):
        h = self.node_proj(tokens)
        sig = _rel_feat(self.rel_head, tokens, tokens) * mask[:, None, :, None]
        ctx = self.ctx_proj(sig.sum(dim=2) / mask.sum(dim=1, keepdim=True).clamp(min=1.0)[:, :, None])
        for layer in self.layers:
            h = h + layer(torch.cat([h, ctx], dim=-1))
        return h

    def forward(self, pre, post, pre_mask, post_mask):
        hp = self.encode(pre, pre_mask)
        hq = self.encode(post, post_mask)
        cross = _rel_feat(self.rel_head, pre, post)
        n, m = hp.shape[1], hq.shape[1]
        sp = _rel_feat(self.rel_head, pre, pre) * pre_mask[:, None, :, None]
        sq = _rel_feat(self.rel_head, post, post) * post_mask[:, None, :, None]
        sp = sp.sum(dim=2) / pre_mask.sum(dim=1).clamp(min=1.0)[:, None, None]
        sq = sq.sum(dim=2) / post_mask.sum(dim=1).clamp(min=1.0)[:, None, None]
        pair = torch.cat(
            [
                hp.unsqueeze(2).expand(-1, n, m, -1),
                hq.unsqueeze(1).expand(-1, n, -1, -1),
                cross,
                sp.unsqueeze(2).expand(-1, -1, m, -1),
                sq.unsqueeze(1).expand(-1, n, -1, -1),
            ],
            dim=-1,
        )
        return self.pair_head(pair).squeeze(-1)


class HybridRelationMatcher(nn.Module):
    """Hand-written edges + learned typed relation logits.

    The hand edges stay because they are stronger on clean geometry; the
    learned heads add a calibrated, typed channel that helps when a modality
    is missing.
    """

    def __init__(self, token_dim: int = TACTILE_TOKEN_DIM, rel_head=None, d_model: int = 128, layers: int = 2):
        super().__init__()
        from pss.memory.graph_matcher import EDGE_DIM, _edge_features

        self._edge_features = staticmethod(_edge_features)
        self.edge_dim = EDGE_DIM
        self.rel_head = rel_head or TypedRelationHead(token_dim)
        for p in self.rel_head.parameters():
            p.requires_grad = False
        feat = sum(len(PREDICATES[t]) for t in PAIRWISE_TYPES)
        self.node_proj = nn.Linear(token_dim, d_model)
        self.edge_proj = nn.Linear(EDGE_DIM, d_model)
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model * 2, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
                )
                for _ in range(layers)
            ]
        )
        self.pair_head = nn.Sequential(
            nn.Linear(d_model * 2 + EDGE_DIM + feat, 128), nn.ReLU(), nn.Linear(128, 1)
        )

    def encode(self, tokens):
        h = self.node_proj(tokens)
        edge = self.edge_proj(self._edge_features(tokens))
        for layer in self.layers:
            h = h + layer(torch.cat([h, edge.mean(dim=1)], dim=-1))
        return h

    def forward(self, pre, post, pre_mask, post_mask):
        hp = self.encode(pre)
        hq = self.encode(post)
        hand = self._edge_features(pre, post)
        learned = _rel_feat(self.rel_head, pre, post)
        n, m = hp.shape[1], hq.shape[1]
        pair = torch.cat(
            [
                hp.unsqueeze(2).expand(-1, n, m, -1),
                hq.unsqueeze(1).expand(-1, n, -1, -1),
                hand,
                learned,
            ],
            dim=-1,
        )
        return self.pair_head(pair).squeeze(-1)


def _graph_batch(pairs: list[RelationPair], device: str):
    pre = torch.as_tensor(np.stack([p.pre for p in pairs]), device=device)
    post = torch.as_tensor(np.stack([p.post for p in pairs]), device=device)
    pm = torch.as_tensor(np.stack([p.pre_mask for p in pairs]), device=device)
    qm = torch.as_tensor(np.stack([p.post_mask for p in pairs]), device=device)
    labels = torch.as_tensor(np.stack([p.labels for p in pairs]), device=device)
    return pre, post, pm, qm, labels


def train_relation_matcher(model, train_pairs, epochs: int = 200, seed: int = 0, device: str = "cpu", lr: float = 1e-3):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4
    )
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(train_pairs))
        for start in range(0, len(order), 16):
            batch = [train_pairs[i] for i in order[start : start + 16]]
            pre, post, pm, qm, labels = _graph_batch(batch, device)
            logits = model(pre, post, pm, qm)
            mask = (pm[:, :, None] * qm[:, None, :]).bool()
            loss = F.binary_cross_entropy_with_logits(logits[mask], labels[mask])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def evaluate_relation_matcher(model, pairs, device: str = "cpu", corrupt: str | None = None) -> dict:
    from scipy.optimize import linear_sum_assignment

    model.eval()
    rows = []
    with torch.no_grad():
        for i, p in enumerate(pairs):
            pre = p.pre
            post = p.post
            if corrupt:
                pre = perturb_tokens(pre, corrupt, seed=i)
                post = perturb_tokens(post, corrupt, seed=i + 1000)
            pre_t = torch.as_tensor(pre[None], device=device)
            post_t = torch.as_tensor(post[None], device=device)
            pm = torch.as_tensor(p.pre_mask[None], device=device)
            qm = torch.as_tensor(p.post_mask[None], device=device)
            scores = model(pre_t, post_t, pm, qm)[0].cpu().numpy()
            n = int(p.pre_mask.sum())
            m = int(p.post_mask.sum())
            r, c = linear_sum_assignment(-scores[:n, :m])
            rows.append(
                {
                    "subset": p.subset,
                    "acc": sum(p.labels[a, b] for a, b in zip(r, c)) / max(n, m),
                }
            )
    subsets = ("mixed", "samecat", "near_identical")
    return {
        "overall": float(np.mean([r["acc"] for r in rows])),
        "per_subset": {
            s: float(np.mean([r["acc"] for r in rows if r["subset"] == s]))
            for s in subsets
        },
        "n": len(rows),
    }


# ── scene graph write path ──────────────────────────────────────────────────


@dataclass
class RelationEdge:
    source: str
    target: str
    relation_type: str
    predicate: str
    probability: float

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "type": self.relation_type,
            "predicate": self.predicate,
            "probability": float(self.probability),
        }


def track_token(track: Track) -> np.ndarray:
    """Build the same entity token the tracker/matcher uses, from a Track."""
    return entity_token(
        track.pos, track.raw, track.category, include_tactile=True
    )


def write_relation_edges(
    graph: SceneGraph,
    head: TypedRelationHead,
    tracks: list[Track] | None = None,
    threshold: float = 0.5,
    device: str = "cpu",
) -> list[RelationEdge]:
    """Run the typed head on active tracks and store relation logits in the graph."""
    tracks = tracks if tracks is not None else graph.active()
    if len(tracks) < 2:
        return []
    tokens = torch.as_tensor(
        np.stack([track_token(t) for t in tracks]), dtype=torch.float32, device=device
    )[None]
    head.eval()
    with torch.no_grad():
        probs = {
            name: value[0].cpu().numpy()
            for name, value in head.probabilities(tokens, tokens).items()
        }
    edges: list[RelationEdge] = []
    for i, a in enumerate(tracks):
        for j, b in enumerate(tracks):
            if i == j:
                continue
            for relation_type in PAIRWISE_TYPES:
                values = probs[relation_type][i, j]
                for k, predicate in enumerate(PREDICATES[relation_type]):
                    p = float(values[k])
                    if p < threshold:
                        continue
                    edge = RelationEdge(a.id, b.id, relation_type, predicate, p)
                    graph.set_relation(edge)
                    edges.append(edge)
    return edges


def save_relation_head(path, head: TypedRelationHead, meta: dict | None = None) -> None:
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    hidden = int(head.pair[0].out_features)
    torch.save(
        {
            "state_dict": head.state_dict(),
            "hidden": hidden,
            "token_dim": int(head.pair[0].in_features // 4),
            "meta": dict(meta or {}),
        },
        path,
    )


def load_relation_head(path, device: str = "cpu") -> TypedRelationHead:
    payload = torch.load(path, map_location=device, weights_only=False)
    head = TypedRelationHead(
        token_dim=int(payload.get("token_dim", TACTILE_TOKEN_DIM)),
        hidden=int(payload.get("hidden", 128)),
    ).to(device)
    head.load_state_dict(payload["state_dict"])
    head.eval()
    return head
