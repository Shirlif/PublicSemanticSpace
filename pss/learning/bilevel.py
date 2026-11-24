"""End-to-end / bilevel training for PSS (fifth priority).

Encoders stay frozen; the trainable graph is:

    frozen features -> projections -> binder -> typed relations -> matcher

Training combines:

- task loss: episodic prototype classification (the support mean is a
  differentiable inner memory, the query loss is the outer task);
- re-ID loss: assignment/pair BCE on tracking pairs;
- relation loss: supervised typed relation heads;
- consistency / audit constraints:
    * spatial antisymmetry (left(a,b) == right(b,a));
    * mutual exclusivity (left+right <= 1, above+below <= 1);
    * permutation invariance of the matcher;
    * stability under small input noise.

The module is deliberately small: the point is the training graph and the
loss composition, not another large model. ``mode="sequential"`` trains the
same architecture in stages (recognition first, then freeze projections and
train relations/matching) for a fair joint-vs-sequential comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from pss.memory.typed_relations import PAIRWISE_TYPES, PREDICATES, TypedRelationHead
from pss.projection.heads import ProjectionHead
from pss.projection.losses import supcon_cross

REL_FEAT = sum(len(PREDICATES[t]) for t in PAIRWISE_TYPES)  # spatial+semantic+tactile


# ── data containers ─────────────────────────────────────────────────────────


@dataclass
class RawBatch:
    vision: torch.Tensor
    tactile: torch.Tensor
    geometry: torch.Tensor
    pose: torch.Tensor
    category: torch.Tensor
    mask: torch.Tensor | None = None


@dataclass
class RawPairs:
    pre: RawBatch
    post: RawBatch
    labels: torch.Tensor
    pre_targets: dict[str, torch.Tensor] = field(default_factory=dict)
    post_targets: dict[str, torch.Tensor] = field(default_factory=dict)
    pre_valid: torch.Tensor | None = None
    post_valid: torch.Tensor | None = None
    subsets: list[str] = field(default_factory=list)


# ── model ───────────────────────────────────────────────────────────────────


class EndToEndPSS(nn.Module):
    def __init__(
        self,
        vision_dim: int = 256,
        tactile_dim: int = 64,
        geometry_dim: int = 16,
        n_categories: int = 16,
        category_dim: int = 20,
        token_dim: int = 128,
        hidden: int = 128,
        binder: str = "mlp",
    ):
        super().__init__()
        self.token_dim = int(token_dim)
        self.config = dict(
            vision_dim=int(vision_dim),
            tactile_dim=int(tactile_dim),
            geometry_dim=int(geometry_dim),
            n_categories=int(n_categories),
            category_dim=int(category_dim),
            token_dim=int(token_dim),
            hidden=int(hidden),
            binder=str(binder),
        )
        self.proj_vision = ProjectionHead(vision_dim, token_dim, hidden=hidden)
        self.proj_tactile = ProjectionHead(tactile_dim, token_dim, hidden=hidden)
        self.proj_geometry = ProjectionHead(geometry_dim, token_dim, hidden=hidden)
        concat = token_dim * 3 + 3 + category_dim
        if binder == "mlp":
            self.binder = nn.Sequential(
                nn.Linear(concat, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Linear(hidden, token_dim),
            )
        elif binder == "linear":
            self.binder = nn.Linear(concat, token_dim)
        else:
            raise ValueError(f"unknown binder {binder!r}")
        self.binder_mode = binder
        self.relation = TypedRelationHead(token_dim=token_dim, hidden=hidden // 2)
        self.pair = nn.Sequential(
            nn.Linear(token_dim * 4 + REL_FEAT, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.classifier = nn.Linear(token_dim, n_categories)

    def bind(self, batch: RawBatch, include_category: bool = True) -> torch.Tensor:
        zv = self.proj_vision(batch.vision)
        zt = self.proj_tactile(batch.tactile)
        zg = self.proj_geometry(batch.geometry)
        category = batch.category if include_category else torch.zeros_like(batch.category)
        h = torch.cat([zv, zt, zg, batch.pose, category], dim=-1)
        return F.normalize(self.binder(h), dim=-1)

    def bind_rec(self, batch: RawBatch) -> torch.Tensor:
        """Recognition view: category is masked to avoid label leakage."""
        return self.bind(batch, include_category=False)

    def relation_probs(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.relation.probabilities(tokens, tokens)

    def pair_scores(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        rel = self.relation(pre, post)
        rel = torch.cat([torch.sigmoid(rel[name]) for name in ("spatial", "semantic", "tactile")], dim=-1)
        n, m = pre.shape[1], post.shape[1]
        a = pre.unsqueeze(2).expand(-1, n, m, -1)
        b = post.unsqueeze(1).expand(-1, n, -1, -1)
        pair = torch.cat([a, b, a - b, a * b, rel], dim=-1)
        return self.pair(pair).squeeze(-1)

    def forward(self, batch: RawBatch) -> torch.Tensor:
        return self.bind(batch)


def to_device(batch: RawBatch, device: str) -> RawBatch:
    return RawBatch(
        vision=batch.vision.to(device),
        tactile=batch.tactile.to(device),
        geometry=batch.geometry.to(device),
        pose=batch.pose.to(device),
        category=batch.category.to(device),
        mask=None if batch.mask is None else batch.mask.to(device),
    )


# ── losses ──────────────────────────────────────────────────────────────────


def prototype_logits(
    query: torch.Tensor,
    support: torch.Tensor,
    support_labels: torch.Tensor,
    n_classes: int,
    temperature: float = 0.1,
) -> torch.Tensor:
    protos = []
    for c in range(n_classes):
        rows = support[support_labels == c]
        if len(rows) == 0:
            protos.append(torch.zeros_like(query[0]))
        else:
            protos.append(F.normalize(rows.mean(dim=0), dim=-1))
    prototypes = torch.stack(protos)
    return query @ prototypes.T / temperature, prototypes


def episodic_prototype_loss(
    tokens: torch.Tensor,
    labels: torch.Tensor,
    n_way: int = 8,
    n_support: int = 2,
    n_query: int = 2,
    temperature: float = 0.1,
) -> torch.Tensor:
    support_idx, query_idx = [], []
    for c in labels.unique():
        idx = torch.nonzero(labels == c, as_tuple=False).flatten()
        if len(idx) < n_support + n_query:
            continue
        perm = idx[torch.randperm(len(idx), device=labels.device)]
        support_idx.append(perm[:n_support])
        query_idx.append(perm[n_support : n_support + n_query])
    if len(support_idx) < 2:
        return tokens.sum() * 0.0
    s = torch.cat(support_idx)
    q = torch.cat(query_idx)
    logits, _ = prototype_logits(
        tokens[q], tokens[s], labels[s], int(labels.max()) + 1, temperature
    )
    return F.cross_entropy(logits, labels[q])


def recognition_linear_loss(
    model: EndToEndPSS, batch: RawBatch, labels: torch.Tensor
) -> torch.Tensor:
    tokens = model.bind_rec(batch)
    return F.cross_entropy(model.classifier(tokens), labels)


def masked_bce(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.bool()
    if valid.sum() == 0:
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits[valid], labels[valid])


def reid_loss(
    model: EndToEndPSS, pairs: RawPairs, assignment_weight: float = 0.5
) -> torch.Tensor:
    pre = model.bind(pairs.pre)
    post = model.bind(pairs.post)
    scores = model.pair_scores(pre, post)
    mask = (pairs.pre.mask[:, :, None] * pairs.post.mask[:, None, :]).bool()
    loss = masked_bce(scores, pairs.labels, mask)
    if assignment_weight > 0:
        row_log = F.log_softmax(scores, dim=2)
        row_t = pairs.labels.argmax(dim=2)
        row_loss = -row_log.gather(2, row_t.unsqueeze(2)).squeeze(2)
        row_valid = (pairs.labels.sum(dim=2) > 0) & pairs.pre.mask.bool()
        col_log = F.log_softmax(scores, dim=1)
        col_t = pairs.labels.argmax(dim=1)
        col_loss = -col_log.gather(1, col_t.unsqueeze(1)).squeeze(1)
        col_valid = (pairs.labels.sum(dim=1) > 0) & pairs.post.mask.bool()
        assign = 0.0
        if row_valid.any():
            assign = assign + row_loss[row_valid].mean()
        if col_valid.any():
            assign = assign + col_loss[col_valid].mean()
        loss = loss + assignment_weight * assign
    return loss


def relation_loss(model: EndToEndPSS, pairs: RawPairs) -> torch.Tensor:
    pre = model.bind(pairs.pre)
    post = model.bind(pairs.post)
    loss = 0.0
    terms = 0
    for split, tokens, targets, valid in (
        ("pre", pre, pairs.pre_targets, pairs.pre_valid),
        ("post", post, pairs.post_targets, pairs.post_valid),
    ):
        logits = model.relation(tokens, tokens)
        for name in ("spatial", "semantic", "tactile"):
            if name not in targets or valid is None:
                continue
            m = valid.bool()
            if m.sum() == 0:
                continue
            positives = targets[name][m].mean().clamp(min=1e-3, max=0.999)
            pos_weight = ((1.0 - positives) / positives).clamp(1.0, 20.0)
            loss = loss + F.binary_cross_entropy_with_logits(
                logits[name][m], targets[name][m], pos_weight=pos_weight
            )
            terms += 1
    return loss / max(1, terms)


def antisymmetry_loss(model: EndToEndPSS, pre: RawBatch) -> torch.Tensor:
    tokens = model.bind(pre)
    probs = model.relation_probs(tokens)["spatial"]
    left, right, above, below = probs[..., 0], probs[..., 1], probs[..., 2], probs[..., 3]
    return (left - right.transpose(1, 2)).abs().mean() + (
        above - below.transpose(1, 2)
    ).abs().mean()


def exclusivity_loss(model: EndToEndPSS, pre: RawBatch) -> torch.Tensor:
    tokens = model.bind(pre)
    probs = model.relation_probs(tokens)["spatial"]
    left, right, above, below = probs[..., 0], probs[..., 1], probs[..., 2], probs[..., 3]
    return F.relu(left + right - 1).mean() + F.relu(above + below - 1).mean()


def permutation_loss(model: EndToEndPSS, pre: RawBatch, post: RawBatch) -> torch.Tensor:
    tokens_pre = model.bind(pre)
    tokens_post = model.bind(post)
    n = tokens_pre.shape[1]
    if n < 2:
        return tokens_pre.sum() * 0.0
    perm = torch.randperm(n, device=tokens_pre.device)
    scores = model.pair_scores(tokens_pre, tokens_post)
    shuffled = model.pair_scores(tokens_pre[:, perm], tokens_post)
    return F.mse_loss(shuffled, scores[:, perm, :])


def stability_loss(model: EndToEndPSS, pre: RawBatch, post: RawBatch, sigma: float = 0.01) -> torch.Tensor:
    tokens_pre = model.bind(pre)
    tokens_post = model.bind(post)
    noise = sigma * torch.randn_like(tokens_pre)
    scores = model.pair_scores(tokens_pre, tokens_post)
    noisy = model.pair_scores(tokens_pre + noise, tokens_post)
    return F.mse_loss(noisy, scores)


# ── training ────────────────────────────────────────────────────────────────


def _sample_category_batch(data: dict, batch_size: int = 64):
    n = len(data["labels"])
    idx = torch.randint(0, n, (batch_size,))
    batch = RawBatch(
        vision=data["vision"][idx],
        tactile=data["tactile"][idx],
        geometry=data["geometry"][idx],
        pose=data["pose"][idx],
        category=data["category"][idx],
    )
    instances = data.get("instances")
    return batch, data["labels"][idx], (None if instances is None else instances[idx])


def _sample_pair_batch(pairs: RawPairs, batch_size: int = 8) -> RawPairs:
    n = pairs.labels.shape[0]
    idx = torch.randint(0, n, (batch_size,))
    return RawPairs(
        pre=RawBatch(
            vision=pairs.pre.vision[idx],
            tactile=pairs.pre.tactile[idx],
            geometry=pairs.pre.geometry[idx],
            pose=pairs.pre.pose[idx],
            category=pairs.pre.category[idx],
            mask=pairs.pre.mask[idx],
        ),
        post=RawBatch(
            vision=pairs.post.vision[idx],
            tactile=pairs.post.tactile[idx],
            geometry=pairs.post.geometry[idx],
            pose=pairs.post.pose[idx],
            category=pairs.post.category[idx],
            mask=pairs.post.mask[idx],
        ),
        labels=pairs.labels[idx],
        pre_targets={k: v[idx] for k, v in pairs.pre_targets.items()},
        post_targets={k: v[idx] for k, v in pairs.post_targets.items()},
        pre_valid=None if pairs.pre_valid is None else pairs.pre_valid[idx],
        post_valid=None if pairs.post_valid is None else pairs.post_valid[idx],
    )


@dataclass
class TrainConfig:
    mode: str = "joint"  # joint | sequential | no_consistency | no_relation | linear_head | concat
    task: str = "category"  # category | instance
    epochs: int = 60
    lr: float = 1e-3
    seed: int = 0
    n_categories: int = 16
    token_dim: int = 128
    hidden: int = 128
    lambda_task: float = 1.0
    lambda_reid: float = 0.5
    lambda_relation: float = 0.3
    lambda_consistency: float = 0.1
    consistency_terms: tuple[str, ...] = ("antisymmetry", "exclusivity", "permutation", "stability")


def train_bilevel(
    category_train: dict,
    pair_train: RawPairs,
    config: TrainConfig | None = None,
    device: str = "cpu",
) -> tuple[EndToEndPSS, dict]:
    config = config or TrainConfig()
    torch.manual_seed(config.seed)
    binder = "linear" if config.mode == "concat" else "mlp"
    model = EndToEndPSS(
        vision_dim=int(category_train["vision"].shape[1]),
        tactile_dim=int(category_train["tactile"].shape[1]),
        geometry_dim=int(category_train["geometry"].shape[1]),
        n_categories=config.n_categories,
        category_dim=int(category_train["category"].shape[1]),
        token_dim=config.token_dim,
        hidden=config.hidden,
        binder=binder,
    ).to(device)
    data = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in category_train.items()}
    pairs = RawPairs(
        pre=to_device(pair_train.pre, device),
        post=to_device(pair_train.post, device),
        labels=pair_train.labels.to(device),
        pre_targets={k: v.to(device) for k, v in pair_train.pre_targets.items()},
        post_targets={k: v.to(device) for k, v in pair_train.post_targets.items()},
        pre_valid=None if pair_train.pre_valid is None else pair_train.pre_valid.to(device),
        post_valid=None if pair_train.post_valid is None else pair_train.post_valid.to(device),
    )
    history = []

    def task_loss() -> torch.Tensor:
        batch, labels, instances = _sample_category_batch(data)
        batch = to_device(batch, device)
        labels = labels.to(device)
        if config.mode == "linear_head":
            return config.lambda_task * recognition_linear_loss(model, batch, labels)
        tokens = model.bind_rec(batch)
        if config.task == "instance" and instances is not None:
            return config.lambda_task * supcon_cross(
                tokens, tokens, instances.to(device)
            )
        return config.lambda_task * episodic_prototype_loss(tokens, labels)

    def aux_losses(pair_batch: RawPairs) -> tuple[torch.Tensor, dict]:
        terms = {}
        terms["reid"] = reid_loss(model, pair_batch)
        if config.mode != "no_relation":
            terms["relation"] = relation_loss(model, pair_batch)
        if config.mode not in ("no_consistency",):
            cons = 0.0
            for name in config.consistency_terms:
                if name == "antisymmetry":
                    cons = cons + antisymmetry_loss(model, pair_batch.pre)
                elif name == "exclusivity":
                    cons = cons + exclusivity_loss(model, pair_batch.pre)
                elif name == "permutation":
                    cons = cons + permutation_loss(model, pair_batch.pre, pair_batch.post)
                elif name == "stability":
                    cons = cons + stability_loss(model, pair_batch.pre, pair_batch.post)
            terms["consistency"] = cons
        total = config.lambda_reid * terms["reid"]
        total = total + config.lambda_relation * terms.get(
            "relation", torch.zeros((), device=device)
        )
        total = total + config.lambda_consistency * terms.get(
            "consistency", torch.zeros((), device=device)
        )
        return total, terms

    # Sequential: first task-only, then freeze projections and train the rest.
    if config.mode == "sequential":
        opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
        for _ in range(max(1, config.epochs // 2)):
            model.train()
            loss = task_loss()
            opt.zero_grad()
            loss.backward()
            opt.step()
        for name in ("proj_vision", "proj_tactile", "proj_geometry"):
            for parameter in getattr(model, name).parameters():
                parameter.requires_grad = False
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.lr,
            weight_decay=1e-4,
        )
        for _ in range(max(1, config.epochs // 2)):
            model.train()
            pair_batch = _sample_pair_batch(pairs)
            aux, terms = aux_losses(pair_batch)
            loss = task_loss() + aux
            opt.zero_grad()
            loss.backward()
            opt.step()
            history.append({k: float(v.detach()) for k, v in terms.items()})
        model.eval()
        return model, {"mode": config.mode, "history": history}

    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    for _ in range(config.epochs):
        model.train()
        pair_batch = _sample_pair_batch(pairs)
        aux, terms = aux_losses(pair_batch)
        loss = task_loss() + aux
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append({k: float(v.detach()) for k, v in terms.items()})
    model.eval()
    return model, {"mode": config.mode, "history": history}


# ── evaluation ──────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate_recognition(model: EndToEndPSS, data: dict, device: str = "cpu") -> float:
    model.eval()
    data = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in data.items()}
    train_tokens = model.bind_rec(
        RawBatch(
            vision=data["train_vision"],
            tactile=data["train_tactile"],
            geometry=data["train_geometry"],
            pose=data["train_pose"],
            category=data["train_category"],
        )
    )
    test_tokens = model.bind_rec(
        RawBatch(
            vision=data["test_vision"],
            tactile=data["test_tactile"],
            geometry=data["test_geometry"],
            pose=data["test_pose"],
            category=data["test_category"],
        )
    )
    protos = []
    for c in range(int(data["n_classes"])):
        rows = train_tokens[data["train_labels"] == c]
        protos.append(F.normalize(rows.mean(dim=0), dim=-1))
    prototypes = torch.stack(protos)
    pred = (test_tokens @ prototypes.T).argmax(dim=1)
    return float((pred == data["test_labels"]).float().mean())


@torch.no_grad()
def evaluate_reid(model: EndToEndPSS, pairs: RawPairs, device: str = "cpu") -> dict:
    model.eval()
    pre = model.bind(to_device(pairs.pre, device))
    post = model.bind(to_device(pairs.post, device))
    scores = model.pair_scores(pre, post).cpu().numpy()
    rows = []
    for i in range(len(scores)):
        n = int(pairs.pre.mask[i].sum())
        m = int(pairs.post.mask[i].sum())
        if n == 0 or m == 0:
            continue
        r, c = linear_sum_assignment(-scores[i, :n, :m])
        labels = pairs.labels[i].cpu().numpy()
        rows.append(sum(labels[a, b] for a, b in zip(r, c)) / max(n, m))
    subsets = ("mixed", "samecat", "near_identical")
    per = {}
    for s in subsets:
        vals = [v for v, sub in zip(rows, pairs.subsets) if sub == s]
        per[s] = float(np.mean(vals)) if vals else 0.0
    return {"overall": float(np.mean(rows)) if rows else 0.0, "per_subset": per, "n": len(rows)}


@torch.no_grad()
def evaluate_relations(model: EndToEndPSS, pairs: RawPairs, device: str = "cpu") -> dict:
    model.eval()
    pre = model.bind(to_device(pairs.pre, device))
    post = model.bind(to_device(pairs.post, device))
    counts = {name: {"tp": 0, "fp": 0, "fn": 0} for name in ("spatial", "semantic", "tactile")}
    for tokens, targets, valid in (
        (pre, pairs.pre_targets, pairs.pre_valid),
        (post, pairs.post_targets, pairs.post_valid),
    ):
        logits = model.relation(tokens, tokens)
        for name in counts:
            if name not in targets or valid is None:
                continue
            pred = (torch.sigmoid(logits[name]) > 0.5).cpu()
            truth = (targets[name] > 0.5).cpu()
            m = valid.bool().cpu().unsqueeze(-1)
            counts[name]["tp"] += int((pred & truth & m).sum())
            counts[name]["fp"] += int((pred & ~truth & m).sum())
            counts[name]["fn"] += int((~pred & truth & m).sum())
    out = {}
    for name, c in counts.items():
        p = c["tp"] / max(1, c["tp"] + c["fp"])
        r = c["tp"] / max(1, c["tp"] + c["fn"])
        out[name] = {"precision": p, "recall": r, "f1": 2 * p * r / max(1e-9, p + r)}
    return out


@torch.no_grad()
def consistency_metrics(model: EndToEndPSS, pairs: RawPairs, device: str = "cpu") -> dict:
    model.eval()
    pre = to_device(pairs.pre, device)
    post = to_device(pairs.post, device)
    tokens = model.bind(pre)
    probs = model.relation_probs(tokens)["spatial"]
    left, right, above, below = probs[..., 0], probs[..., 1], probs[..., 2], probs[..., 3]
    antisym = float((left - right.transpose(1, 2)).abs().mean())
    exclusivity = float(
        F.relu(left + right - 1).mean() + F.relu(above + below - 1).mean()
    )
    perm = float(permutation_loss(model, pre, post))
    stab = float(stability_loss(model, pre, post))
    return {
        "antisymmetry_error": antisym,
        "exclusivity_violation": exclusivity,
        "permutation_mse": perm,
        "stability_mse": stab,
    }


def save_joint(model: EndToEndPSS, path) -> None:
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "config": model.config, "token_dim": model.token_dim},
        path,
    )


def load_joint(path, device: str = "cpu") -> EndToEndPSS:
    payload = torch.load(path, map_location=device, weights_only=False)
    config = payload.get("config") or {"token_dim": int(payload.get("token_dim", 128))}
    model = EndToEndPSS(**config).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
