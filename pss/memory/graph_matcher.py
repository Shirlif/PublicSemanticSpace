"""Set-level graph matcher for PSS re-ID.

Unlike the pairwise metric, this model encodes within-set relations (relative
pose and category match) before scoring pre/post pairs. It keeps the PSS scene
graph and adds a set-level matching head.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from pss.memory.learned_metric import (
    CATEGORIES,
    CATEGORY_TO_INDEX,
    DINO_DIM,
    GEOMETRY_DIM,
    TACTILE_DIM,
    TOKEN_DIM,
    VISION_DIM,
)

POSE_START = VISION_DIM + GEOMETRY_DIM
CAT_START = POSE_START + 3
EDGE_DIM = 5  # distance + category match + relative position (3)


def _edge_features(pre, post=None):
    if post is None:
        post = pre
    pos_a = pre[..., POSE_START : POSE_START + 3]
    pos_b = post[..., POSE_START : POSE_START + 3]
    rel = pos_b.unsqueeze(2) - pos_a.unsqueeze(1)
    dist = torch.linalg.norm(rel, dim=-1, keepdim=True)
    cat_a = pre[..., CAT_START : CAT_START + len(CATEGORIES)]
    cat_b = post[..., CAT_START : CAT_START + len(CATEGORIES)]
    match = torch.einsum("bid,bjd->bij", cat_a, cat_b).unsqueeze(-1)
    return torch.cat([dist, match, rel], dim=-1)


class GraphMatcher(nn.Module):
    def __init__(
        self,
        token_dim: int = TOKEN_DIM,
        d_model: int = 128,
        layers: int = 2,
        use_edges: bool = True,
    ):
        super().__init__()
        self.use_edges = bool(use_edges)
        self.node_proj = nn.Linear(token_dim, d_model)
        self.edge_proj = nn.Linear(EDGE_DIM, d_model)
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model * 2, d_model),
                    nn.ReLU(inplace=True),
                    nn.Linear(d_model, d_model),
                )
                for _ in range(layers)
            ]
        )
        self.pair_head = nn.Sequential(
            nn.Linear(d_model * 2 + EDGE_DIM, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )
        self.d_model = d_model

    def encode(self, tokens):
        h = self.node_proj(tokens)
        if self.use_edges:
            edge = _edge_features(tokens)
            for layer in self.layers:
                msg = self.edge_proj(edge) * h.unsqueeze(1)
                agg = msg.mean(dim=1)
                h = h + layer(torch.cat([h, agg], dim=-1))
        return h

    def forward(self, pre, post, pre_mask, post_mask):
        hp = self.encode(pre)
        hq = self.encode(post)
        pair = _edge_features(pre, post)
        n, m = hp.shape[1], hq.shape[1]
        hp = hp.unsqueeze(2).expand(-1, n, m, -1)
        hq = hq.unsqueeze(1).expand(-1, n, m, -1)
        return self.pair_head(torch.cat([hp, hq, pair], dim=-1)).squeeze(-1)


def _batch(pairs, device):
    pre = torch.as_tensor(np.stack([p.pre for p in pairs]), dtype=torch.float32, device=device)
    post = torch.as_tensor(np.stack([p.post for p in pairs]), dtype=torch.float32, device=device)
    pm = torch.as_tensor(np.stack([p.pre_mask for p in pairs]), dtype=torch.float32, device=device)
    qm = torch.as_tensor(np.stack([p.post_mask for p in pairs]), dtype=torch.float32, device=device)
    labels = torch.as_tensor(np.stack([p.labels for p in pairs]), dtype=torch.float32, device=device)
    return pre, post, pm, qm, labels


def _assignment_accuracy(scores, labels):
    rows, cols = linear_sum_assignment(-scores)
    correct = sum(float(labels[r, c]) for r, c in zip(rows, cols))
    return correct / max(1, len(labels))


def _greedy_accuracy(scores, labels):
    scores = np.array(scores, dtype=float)
    used_rows, used_cols = set(), set()
    correct = 0.0
    for _ in range(min(scores.shape)):
        masked = scores.copy()
        for r in used_rows:
            masked[r, :] = -np.inf
        for c in used_cols:
            masked[:, c] = -np.inf
        r, c = np.unravel_index(int(np.argmax(masked)), masked.shape)
        if not np.isfinite(masked[r, c]):
            break
        correct += float(labels[r, c])
        used_rows.add(int(r))
        used_cols.add(int(c))
    return correct / max(1, len(labels))


def evaluate_graph(model, pairs, device="cpu", matching="hungarian"):
    model.eval()
    rows = []
    with torch.no_grad():
        for pair in pairs:
            pre, post, pm, qm, labels = _batch([pair], device)
            scores = model(pre, post, pm, qm).cpu().numpy()[0]
            if matching == "greedy":
                acc = _greedy_accuracy(scores, pair.labels)
            else:
                acc = _assignment_accuracy(scores, pair.labels)
            rows.append(
                {
                    "sample": pair.sample_id,
                    "subset": pair.subset,
                    "reid_accuracy": acc,
                }
            )
    return _summarize(rows)


def train_graph(
    train_pairs,
    val_pairs,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 16,
    seed: int = 0,
    device: str = "cpu",
    assignment_weight: float = 1.0,
    token_dim: int = TOKEN_DIM,
    use_edges: bool = True,
):
    torch.manual_seed(seed)
    model = GraphMatcher(token_dim=token_dim, use_edges=use_edges).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    best_state, best_val = None, -1.0
    history = []
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(train_pairs))
        losses = []
        for start in range(0, len(order), batch_size):
            batch = [train_pairs[i] for i in order[start : start + batch_size]]
            pre, post, pm, qm, labels = _batch(batch, device)
            logits = model(pre, post, pm, qm)
            mask = (pm[:, :, None] * qm[:, None, :]).bool()
            loss = F.binary_cross_entropy_with_logits(logits[mask], labels[mask])
            if assignment_weight > 0:
                row_log = F.log_softmax(logits, dim=2)
                row_t = labels.argmax(dim=2)
                row_loss = -row_log.gather(2, row_t.unsqueeze(2)).squeeze(2)
                row_valid = (labels.sum(dim=2) > 0) & pm.bool()
                col_log = F.log_softmax(logits, dim=1)
                col_t = labels.argmax(dim=1)
                col_loss = -col_log.gather(1, col_t.unsqueeze(1)).squeeze(1)
                col_valid = (labels.sum(dim=1) > 0) & qm.bool()
                assign = 0.0
                if row_valid.any():
                    assign = assign + row_loss[row_valid].mean()
                if col_valid.any():
                    assign = assign + col_loss[col_valid].mean()
                loss = loss + assignment_weight * assign
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        val = evaluate_graph(model, val_pairs, device)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_reid": val["reid_accuracy"]})
        if val["reid_accuracy"] > best_val:
            best_val = val["reid_accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_reid": best_val, "history": history}


def save_graph(model: GraphMatcher, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, path)


def load_graph(path: str | Path, device="cpu", token_dim: int | None = None):
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    state = payload["state_dict"]
    if token_dim is None:
        token_dim = int(state["node_proj.weight"].shape[1])
    model = GraphMatcher(token_dim=token_dim).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


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


class GraphMatcherReID:
    """Set-level re-ID adapter for the PSS tracker."""

    def __init__(
        self,
        model: GraphMatcher,
        device: str = "cpu",
        max_cost: float = 0.9,
        include_tactile: bool = False,
        include_dino: bool = False,
    ):
        self.model = model
        self.device = device
        self.max_cost = float(max_cost)
        self.include_tactile = bool(include_tactile)
        self.include_dino = bool(include_dino)

    def cost_matrix(self, tracks, observations) -> np.ndarray:
        from pss.memory.learned_metric import pad_observations

        n = max(3, len(tracks), len(observations))
        pre_tokens, pre_mask = pad_observations(
            tracks,
            n,
            include_tactile=self.include_tactile,
            include_dino=self.include_dino,
        )
        post_tokens, post_mask = pad_observations(
            observations,
            n,
            include_tactile=self.include_tactile,
            include_dino=self.include_dino,
        )
        with torch.no_grad():
            logits = self.model(
                torch.as_tensor(pre_tokens, dtype=torch.float32, device=self.device)[None],
                torch.as_tensor(post_tokens, dtype=torch.float32, device=self.device)[None],
                torch.as_tensor(pre_mask, dtype=torch.float32, device=self.device)[None],
                torch.as_tensor(post_mask, dtype=torch.float32, device=self.device)[None],
            ).sigmoid().cpu().numpy()[0]
        cost = 1.0 - logits[: len(tracks), : len(observations)]
        invalid = (pre_mask[: len(tracks)] <= 0)[:, None] | (
            post_mask[: len(observations)] <= 0
        )[None, :]
        cost[invalid] = 1e6
        return cost



class PrunedGraphMatcherReID(GraphMatcherReID):
    """Candidate-pruned, component-wise graph re-ID.

    The dense matcher scores every track/observation pair and runs one global
    Hungarian assignment. This variant first builds a sparse candidate graph
    (pose gate + category gate + appearance top-k), then runs the learned graph
    matcher only inside each connected component. It preserves the dense
    matcher's within-set context locally while avoiding the full N x N pair
    tensor.
    """

    def __init__(
        self,
        model: GraphMatcher,
        device: str = "cpu",
        max_cost: float = 0.9,
        include_tactile: bool = False,
        include_dino: bool = False,
        pose_gate: float = 0.15,
        top_k: int = 16,
        category_gate: bool = True,
    ):
        super().__init__(
            model,
            device=device,
            max_cost=max_cost,
            include_tactile=include_tactile,
            include_dino=include_dino,
        )
        self.pose_gate = float(pose_gate)
        self.top_k = int(top_k)
        self.category_gate = bool(category_gate)

    @staticmethod
    def _cat_ok(a, b) -> bool:
        return a is None or b is None or a == b

    def _candidate_edges(self, tracks, observations) -> list[tuple[int, int]]:
        if not tracks or not observations:
            return []
        track_pos = np.stack([np.asarray(t.pos, dtype=float)[:3] for t in tracks])
        obs_pos = np.stack([np.asarray(o.pos, dtype=float)[:3] for o in observations])
        tree = cKDTree(track_pos)
        neighbors = tree.query_ball_point(obs_pos, r=self.pose_gate)

        track_app = np.stack([np.asarray(t.appearance, dtype=float) for t in tracks])
        obs_app = np.stack([np.asarray(o.appearance, dtype=float) for o in observations])
        track_app = track_app / (np.linalg.norm(track_app, axis=1, keepdims=True) + 1e-9)
        obs_app = obs_app / (np.linalg.norm(obs_app, axis=1, keepdims=True) + 1e-9)
        track_cat = np.asarray([self._cat_index(t.category) for t in tracks], dtype=int)
        obs_cat = np.asarray([self._cat_index(o.category) for o in observations], dtype=int)

        edges: set[tuple[int, int]] = set()
        for j, near in enumerate(neighbors):
            if not near:
                continue
            idx = np.asarray(near, dtype=int)
            # Sparse scenes usually have one or two candidates; skip the
            # appearance matmul and sorting in that common case.
            if len(idx) <= self.top_k:
                for i in idx:
                    if not self.category_gate or obs_cat[j] < 0 or track_cat[i] == obs_cat[j]:
                        edges.add((int(i), j))
                continue
            sims = track_app[idx] @ obs_app[j]
            if self.category_gate and obs_cat[j] >= 0:
                sims[track_cat[idx] != obs_cat[j]] = -np.inf
            keep = np.argsort(-sims)[: self.top_k]
            for pos in keep:
                if np.isfinite(sims[pos]):
                    edges.add((int(idx[pos]), j))
        return sorted(edges)

    @staticmethod
    def _cat_index(category) -> int:
        try:
            return CATEGORY_TO_INDEX.get(category, -1)
        except TypeError:
            return -1

    def _component_costs(self, tracks, observations):
        edges = self._candidate_edges(tracks, observations)
        if not edges:
            return []
        n_tracks, n_obs = len(tracks), len(observations)
        parent = list(range(n_tracks + n_obs))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i, j in edges:
            union(i, n_tracks + j)

        comps: dict[int, tuple[list[int], list[int]]] = {}
        for i, j in edges:
            root = find(i)
            ti, oi = comps.setdefault(root, ([], []))
            if i not in ti:
                ti.append(i)
            if j not in oi:
                oi.append(j)

        return self._score_components(list(comps.values()), tracks, observations)

    def _score_components(self, comps, tracks, observations):
        """Batch components into padded buckets and run one graph forward each."""
        from pss.memory.learned_metric import pad_observations

        if not comps:
            return []
        pre_all, _ = pad_observations(
            tracks,
            n=len(tracks),
            include_tactile=self.include_tactile,
            include_dino=self.include_dino,
        )
        post_all, _ = pad_observations(
            observations,
            n=len(observations),
            include_tactile=self.include_tactile,
            include_dino=self.include_dino,
        )
        dim = pre_all.shape[1]
        bucket_sizes = (3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128)

        def bucket_for(k):
            for size in bucket_sizes:
                if size >= k:
                    return size
            return k

        out = []
        # Most sparse-scene components are one track / one observation. Batch
        # them as B x 1 x 1 instead of paying one padded 3-node forward each.
        singles = [(ti[0], oi[0]) for ti, oi in comps if len(ti) == 1 and len(oi) == 1]
        if singles:
            pre = pre_all[np.asarray([i for i, _j in singles], dtype=int)][:, None, :]
            post = post_all[np.asarray([j for _i, j in singles], dtype=int)][:, None, :]
            pm = np.ones((len(singles), 1), dtype=np.float32)
            qm = np.ones((len(singles), 1), dtype=np.float32)
            with torch.no_grad():
                logits = self.model(
                    torch.as_tensor(pre, device=self.device),
                    torch.as_tensor(post, device=self.device),
                    torch.as_tensor(pm, device=self.device),
                    torch.as_tensor(qm, device=self.device),
                ).sigmoid().cpu().numpy()
            for idx, (i, j) in enumerate(singles):
                out.append(([i], [j], 1.0 - logits[idx, :1, :1]))

        buckets: dict[int, list[tuple[list[int], list[int]]]] = {}
        for ti, oi in comps:
            if len(ti) == 1 and len(oi) == 1:
                continue
            k = bucket_for(max(3, len(ti), len(oi)))
            buckets.setdefault(k, []).append((ti, oi))

        for k, items in buckets.items():
            b = len(items)
            pre = np.zeros((b, k, dim), dtype=np.float32)
            post = np.zeros((b, k, dim), dtype=np.float32)
            pm = np.zeros((b, k), dtype=np.float32)
            qm = np.zeros((b, k), dtype=np.float32)
            for idx, (ti, oi) in enumerate(items):
                pre[idx, : len(ti)] = pre_all[ti]
                post[idx, : len(oi)] = post_all[oi]
                pm[idx, : len(ti)] = 1.0
                qm[idx, : len(oi)] = 1.0

            with torch.no_grad():
                logits = self.model(
                    torch.as_tensor(pre, device=self.device),
                    torch.as_tensor(post, device=self.device),
                    torch.as_tensor(pm, device=self.device),
                    torch.as_tensor(qm, device=self.device),
                ).sigmoid().cpu().numpy()

            for idx, (ti, oi) in enumerate(items):
                block = 1.0 - logits[idx, : len(ti), : len(oi)]
                out.append((ti, oi, block))
        return out

    def cost_matrix(self, tracks, observations) -> np.ndarray:
        cost = np.full((len(tracks), len(observations)), 1e6, dtype=float)
        for ti, oi, block in self._component_costs(tracks, observations):
            cost[np.ix_(ti, oi)] = block
        return cost

    def match(self, tracks, observations) -> list[tuple[int, int, float]]:
        """Return accepted (track_index, observation_index, cost) matches."""
        out = []
        for ti, oi, block in self._component_costs(tracks, observations):
            rows, cols = linear_sum_assignment(block)
            for r, c in zip(rows, cols):
                value = float(block[r, c])
                if value <= self.max_cost:
                    out.append((ti[int(r)], oi[int(c)], value))
        return out
