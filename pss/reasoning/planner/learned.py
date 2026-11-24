"""Learned cost-to-go heuristic for A*.

Row 5 of the gap table: keep the search, learn the heuristic. The heuristic is
trained from optimal A* costs on random states/goals; A* remains the planner,
so success/validity semantics do not change. The hand heuristic remains the
fallback.

The feature encoding is deliberately small: per entity, one-hot current
region + one-hot goal region + a missing-goal flag. This is enough for the
tabletop region domain and keeps the model auditable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pss.world_model.actions import Operator, OperatorLibrary

from . import AStarPlanner

DEFAULT_ENTITIES = ("e0", "e1", "e2")
DEFAULT_REGIONS = ("A", "B", "C", "D", "E")


def _region_loc(region: str, regions) -> tuple[int, int]:
    # A small ring/grid layout; only relative distances matter.
    coords = {
        "A": (-1.0, 0.0),
        "B": (1.0, 0.0),
        "C": (0.0, 1.0),
        "D": (0.0, -1.0),
        "E": (0.0, 0.0),
    }
    return coords.get(region, (0.0, 0.0))


def build_library(entities=DEFAULT_ENTITIES, regions=DEFAULT_REGIONS, distance_cost=True):
    ops = []
    for entity in entities:
        for src in regions:
            for dst in regions:
                if src == dst:
                    continue
                if distance_cost:
                    ax, ay = _region_loc(src, regions)
                    bx, by = _region_loc(dst, regions)
                    cost = float(np.hypot(ax - bx, ay - by))
                else:
                    cost = 1.0
                ops.append(
                    Operator(
                        f"move_{entity}_{src}_{dst}",
                        frozenset({(entity, "location", src)}),
                        frozenset({(entity, "location", dst)}),
                        frozenset({(entity, "location", src)}),
                        cost=cost,
                    )
                )
    return OperatorLibrary(ops)


def encode_state(state, goal, entities=DEFAULT_ENTITIES, regions=DEFAULT_REGIONS):
    current = {e: None for e in entities}
    targets = {e: None for e in entities}
    for entity, attribute, value in state:
        if attribute == "location" and entity in current:
            current[entity] = value
    for entity, attribute, value in goal:
        if attribute == "location" and entity in targets:
            targets[entity] = value
    parts = []
    missing = 0.0
    for entity in entities:
        cur = np.zeros(len(regions), dtype=np.float32)
        tgt = np.zeros(len(regions), dtype=np.float32)
        if current[entity] in regions:
            cur[regions.index(current[entity])] = 1.0
        if targets[entity] in regions:
            tgt[regions.index(targets[entity])] = 1.0
        else:
            missing += 1.0
        parts.extend([cur, tgt, np.array([float(targets[entity] is None)], dtype=np.float32)])
    parts.append(np.array([missing / max(1, len(entities))], dtype=np.float32))
    return np.concatenate(parts)


def make_dataset(
    n: int = 4000,
    entities=DEFAULT_ENTITIES,
    regions=DEFAULT_REGIONS,
    seed: int = 0,
    distance_cost: bool = True,
):
    rng = np.random.default_rng(seed)
    library = build_library(entities, regions, distance_cost)
    planner = AStarPlanner(library, heuristic=lambda s, g: 0.0)
    states, goals, costs = [], [], []
    for _ in range(n):
        assignment = rng.choice(len(regions), size=len(entities), replace=False)
        state = frozenset(
            (entity, "location", regions[int(region)])
            for entity, region in zip(entities, assignment)
        )
        n_goal = int(rng.integers(1, len(entities) + 1))
        goal_entities = rng.choice(len(entities), size=n_goal, replace=False)
        goal = frozenset(
            (entities[int(i)], "location", regions[int(rng.integers(0, len(regions)))])
            for i in goal_entities
        )
        result = planner.plan(state, goal)
        if not result.success:
            continue
        states.append(state)
        goals.append(goal)
        costs.append(float(result.cost))
    return states, goals, costs, library


class LearnedHeuristic(nn.Module):
    def __init__(self, feature_dim: int, hidden: int = 128):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features):
        return self.net(features).squeeze(-1)

    def predict(self, state, goal, entities=DEFAULT_ENTITIES, regions=DEFAULT_REGIONS) -> float:
        features = encode_state(state, goal, entities, regions)
        with torch.no_grad():
            value = float(self.net(torch.as_tensor(features[None], dtype=torch.float32)))
        return max(0.0, value)


def train_heuristic(
    states,
    goals,
    costs,
    entities=DEFAULT_ENTITIES,
    regions=DEFAULT_REGIONS,
    epochs: int = 200,
    lr: float = 1e-3,
    seed: int = 0,
):
    torch.manual_seed(seed)
    X = torch.as_tensor(
        np.stack([encode_state(s, g, entities, regions) for s, g in zip(states, goals)]),
        dtype=torch.float32,
    )
    y = torch.as_tensor(costs, dtype=torch.float32)
    model = LearnedHeuristic(int(X.shape[1]))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(X))
        for start in range(0, len(perm), 128):
            idx = perm[start : start + 128]
            loss = F.mse_loss(model(X[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


def save_heuristic(path, model: LearnedHeuristic, meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_dim": model.feature_dim,
            "meta": dict(meta or {}),
        },
        path,
    )


def load_heuristic(path, device: str = "cpu") -> LearnedHeuristic:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = LearnedHeuristic(int(payload["feature_dim"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
