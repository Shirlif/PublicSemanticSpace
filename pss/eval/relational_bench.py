"""P1-5 / P5b: compositional relational planning.

Three objects sit in five regions. The goal is object 0 at G, object 1 in
relation R1 to object 0, and object 2 in relation R2 to object 1. Training
uses four relation pairs; test uses the four held-out pairs, so every relation
is seen individually but the pair is novel.

PSS: explicit geometric goal check + BFS over placements.
Transformer: behavior-cloned token policy over object/region/relation tokens.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pss._console import ensure_utf8

REGIONS = {
    "A": (-1.0, 0.0),
    "B": (1.0, 0.0),
    "C": (0.0, 1.0),
    "D": (0.0, -1.0),
    "G": (0.0, 0.0),
}
NAMES = list(REGIONS)
INDEX = {n: i for i, n in enumerate(NAMES)}
GOAL = INDEX["G"]
OBJECTS = 3
ACTIONS = [(obj, region) for obj in range(OBJECTS) for region in range(len(NAMES))]

TRAIN_PAIRS = (
    ("left", "above"),
    ("right", "below"),
    ("above", "left"),
    ("below", "right"),
)
TEST_PAIRS = (
    ("left", "below"),
    ("right", "above"),
    ("above", "right"),
    ("below", "left"),
)
RELATIONS = ("left", "right", "above", "below")


def _rel(a: str, b: str, rel: str) -> bool:
    ax, ay = REGIONS[a]
    bx, by = REGIONS[b]
    if rel == "left":
        return ax < bx
    if rel == "right":
        return ax > bx
    if rel == "above":
        return ay > by
    if rel == "below":
        return ay < by
    raise ValueError(rel)


def goal_ok(state, rel1, rel2, rel_fn=None):
    rel_fn = rel_fn or _rel
    p0, p1, p2 = (NAMES[i] for i in state)
    return p0 == "G" and rel_fn(p1, p0, rel1) and rel_fn(p2, p1, rel2)


def plan(start, rel1, rel2, max_nodes=20000, rel_fn=None):
    rel_fn = rel_fn or _rel
    start = tuple(start)
    if goal_ok(start, rel1, rel2, rel_fn):
        return []
    frontier = deque([start])
    parent = {start: None}
    while frontier and len(parent) < max_nodes:
        state = frontier.popleft()
        for obj in range(OBJECTS):
            for region in range(len(NAMES)):
                if state[obj] == region or region in state:
                    continue
                nxt = list(state)
                nxt[obj] = region
                nxt = tuple(nxt)
                if nxt in parent:
                    continue
                parent[nxt] = (state, (obj, region))
                if goal_ok(nxt, rel1, rel2, rel_fn):
                    actions = []
                    cur = nxt
                    while parent[cur] is not None:
                        prev, act = parent[cur]
                        actions.append(act)
                        cur = prev
                    return list(reversed(actions))
                frontier.append(nxt)
    return None


def make_episode(rng, rel1, rel2):
    while True:
        state = tuple(rng.choice(len(NAMES), size=OBJECTS, replace=False))
        solution = plan(state, rel1, rel2)
        if solution is not None:
            return state, solution


def _rows(rel_pairs, n_per_pair, rng):
    rows = []
    for rel1, rel2 in rel_pairs:
        for _ in range(n_per_pair):
            state, solution = make_episode(rng, rel1, rel2)
            cur = state
            for obj, region in solution:
                rows.append(
                    {
                        "state": cur,
                        "rel1": RELATIONS.index(rel1),
                        "rel2": RELATIONS.index(rel2),
                        "action": obj * len(NAMES) + region,
                    }
                )
                nxt = list(cur)
                nxt[obj] = region
                cur = tuple(nxt)
    return rows


class RelationalPolicy(nn.Module):
    def __init__(self, d_model=64, layers=2, heads=4):
        super().__init__()
        self.region = nn.Embedding(len(NAMES), d_model)
        self.obj = nn.Embedding(OBJECTS, d_model)
        self.rel = nn.Embedding(len(RELATIONS), d_model)
        self.pair = nn.Linear(2 * d_model, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 2,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Linear(d_model, len(ACTIONS))

    def forward(self, states, rel1, rel2):
        b = states.shape[0]
        obj_ids = torch.arange(OBJECTS, device=states.device)[None].expand(b, -1)
        tokens = self.region(states) + self.obj(obj_ids)
        pair = self.pair(
            torch.cat([self.rel(rel1), self.rel(rel2)], dim=1)
        ).unsqueeze(1)
        h = self.encoder(torch.cat([tokens, pair], dim=1))
        return self.head(h[:, :OBJECTS].mean(dim=1))


def _tensors(rows, device):
    states = torch.as_tensor(
        np.stack([r["state"] for r in rows]), dtype=torch.long, device=device
    )
    rel1 = torch.as_tensor([r["rel1"] for r in rows], dtype=torch.long, device=device)
    rel2 = torch.as_tensor([r["rel2"] for r in rows], dtype=torch.long, device=device)
    labels = torch.as_tensor([r["action"] for r in rows], dtype=torch.long, device=device)
    return states, rel1, rel2, labels


def train_policy(rows, seed, epochs=300, device="cpu"):
    torch.manual_seed(seed)
    model = RelationalPolicy().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    data = _tensors(rows, device)
    best_state, best_loss = None, float("inf")
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(rows), device=device)
        total = 0.0
        for start in range(0, len(perm), 64):
            idx = perm[start : start + 64]
            logits = model(data[0][idx], data[1][idx], data[2][idx])
            loss = F.cross_entropy(logits, data[3][idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        if total / len(rows) < best_loss:
            best_loss = total / len(rows)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model


def rollout(model, state, rel1, rel2, device, max_steps=8):
    cur = tuple(state)
    steps = 0
    invalid = 0
    for _ in range(max_steps):
        if goal_ok(cur, rel1, rel2):
            return True, steps, invalid
        s = torch.as_tensor([cur], dtype=torch.long, device=device)
        r1 = torch.as_tensor([RELATIONS.index(rel1)], dtype=torch.long, device=device)
        r2 = torch.as_tensor([RELATIONS.index(rel2)], dtype=torch.long, device=device)
        with torch.no_grad():
            act = int(model(s, r1, r2).argmax(dim=1).cpu())
        obj, region = ACTIONS[act]
        if cur[obj] == region or region in cur:
            invalid += 1
            return False, steps, invalid
        nxt = list(cur)
        nxt[obj] = region
        cur = tuple(nxt)
        steps += 1
    return goal_ok(cur, rel1, rel2), steps, invalid


def evaluate(model, rel_pairs, n_per_pair, rng, device):
    ok, steps, invalid = [], [], []
    for rel1, rel2 in rel_pairs:
        for _ in range(n_per_pair):
            state, solution = make_episode(rng, rel1, rel2)
            good, s, inv = rollout(model, state, rel1, rel2, device)
            ok.append(float(good))
            steps.append(s)
            invalid.append(float(inv > 0))
    return {
        "success": float(np.mean(ok)),
        "steps": float(np.mean([s for s, g in zip(steps, ok) if g]))
        if any(ok)
        else 0.0,
        "invalid_rate": float(np.mean(invalid)),
        "n": len(ok),
    }


def run(seeds=5, n_train_per_pair=80, n_test_per_pair=40, epochs=300, device="cpu"):
    rng = np.random.default_rng(0)
    train_rows = _rows(TRAIN_PAIRS, n_train_per_pair, rng)
    test_rng = np.random.default_rng(1234)
    runs = []
    for seed in range(seeds):
        model = train_policy(train_rows, seed, epochs, device)
        id_rng = np.random.default_rng(100 + seed)
        ood_rng = np.random.default_rng(200 + seed)
        runs.append(
            {
                "seed": seed,
                "id": evaluate(model, TRAIN_PAIRS, n_test_per_pair, id_rng, device),
                "ood": evaluate(model, TEST_PAIRS, n_test_per_pair, ood_rng, device),
            }
        )

    # PSS success is complete by construction; report the oracle step count.
    oracle_steps = []
    for rel1, rel2 in TEST_PAIRS:
        r = np.random.default_rng(7)
        for _ in range(n_test_per_pair):
            _state, solution = make_episode(r, rel1, rel2)
            oracle_steps.append(len(solution))

    def agg(key, sub):
        vals = [r[sub][key] for r in runs]
        return float(np.mean(vals)), float(np.std(vals))

    metrics = {
        "train_pairs": TRAIN_PAIRS,
        "test_pairs": TEST_PAIRS,
        "n_train": len(train_rows),
        "pss": {
            "success": 1.0,
            "steps": float(np.mean(oracle_steps)),
        },
        "transformer": {
            "id_success": agg("success", "id"),
            "id_steps": agg("steps", "id"),
            "ood_success": agg("success", "ood"),
            "ood_steps": agg("steps", "ood"),
            "ood_invalid": agg("invalid_rate", "ood"),
            "runs": runs,
        },
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    return metrics


def _report(path, metrics):
    t = metrics["transformer"]
    lines = [
        "# P1-5 / P5b: compositional relational planning",
        "",
        f"- train pairs: {metrics['train_pairs']}",
        f"- held-out pairs: {metrics['test_pairs']}",
        f"- train demonstrations: {metrics['n_train']}",
        "",
        "| method | ID success | ID steps | OOD success | OOD steps | OOD invalid |",
        "|---|---:|---:|---:|---:|---:|",
        f"| PSS explicit planner | **1.000** | {metrics['pss']['steps']:.2f} | "
        f"**1.000** | {metrics['pss']['steps']:.2f} | 0.000 |",
        f"| Transformer BC | {t['id_success'][0]:.3f} ± {t['id_success'][1]:.3f} | "
        f"{t['id_steps'][0]:.2f} | "
        f"{t['ood_success'][0]:.3f} ± {t['ood_success'][1]:.3f} | "
        f"{t['ood_steps'][0]:.2f} | {t['ood_invalid'][0]:.3f} |",
        "",
        "Goal: object 0 at G, object 1 in relation R1 to object 0, object 2 in R2 to object 1.",
        "Each relation appears in training; the relation pair is held out.",
        "PSS goal check is explicit geometry; transformer is behavior-cloned on seen pairs.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/results/pss_vs_transformer")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    metrics = run(seeds=args.seeds, epochs=args.epochs, device=args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "p15_relational.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _report(out / "p15_relational.md", metrics)
    t = metrics["transformer"]
    print(
        f"ID: PSS 1.000 vs TR {t['id_success'][0]:.3f}; "
        f"OOD: PSS 1.000 vs TR {t['ood_success'][0]:.3f}; "
        f"OOD invalid {t['ood_invalid'][0]:.3f}"
    )
    print(f"output: {args.out}")


if __name__ == "__main__":
    main()
