"""P1.3: explicit relation planner vs a transformer policy.

The state is the same discrete tabletop relation graph used by the PSS planner:
five waypoints and a set of clear edges. The PSS side runs A* over explicit
move operators. The transformer side is a behavior-cloned region policy over
the same five region tokens and edge mask, trained on A* demonstrations.

Train layouts contain one obstacle. In-distribution test layouts also contain
one obstacle; OOD layouts contain two, so the policy has to compose edge
constraints it never saw together. Both methods see the same mask.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pss._console import ensure_utf8
from pss.reasoning.grounding import REGIONS, segment_clear
from pss.reasoning.planner import AStarPlanner
from pss.world_model.actions import Operator, OperatorLibrary

NAMES = list(REGIONS)
GOAL = "G"
START_NAMES = [n for n in NAMES if n != GOAL]


def clear_mask(obstacles) -> np.ndarray:
    n = len(NAMES)
    mask = np.zeros((n, n), dtype=np.float32)
    for i, src in enumerate(NAMES):
        for j, dst in enumerate(NAMES):
            if i == j:
                continue
            mask[i, j] = float(
                segment_clear(REGIONS[src], REGIONS[dst], obstacles, clearance=0.09)
            )
    return mask


def library_from_mask(mask: np.ndarray) -> OperatorLibrary:
    ops = []
    for i, src in enumerate(NAMES):
        for j, dst in enumerate(NAMES):
            if i == j or not mask[i, j]:
                continue
            ops.append(
                Operator(
                    f"move_{src}_{dst}",
                    frozenset({("target", "at", src)}),
                    frozenset({("target", "at", dst)}),
                    frozenset({("target", "at", src)}),
                    cost=float(np.linalg.norm(REGIONS[src] - REGIONS[dst])),
                )
            )
    return OperatorLibrary(ops)


def reachable(mask: np.ndarray, start: str) -> bool:
    seen, stack = {start}, [start]
    while stack:
        cur = stack.pop()
        i = NAMES.index(cur)
        for j, dst in enumerate(NAMES):
            if mask[i, j] and dst not in seen:
                seen.add(dst)
                stack.append(dst)
    return GOAL in seen


def plan_from(mask: np.ndarray, start: str) -> list[str] | None:
    if not reachable(mask, start):
        return None
    result = AStarPlanner(library_from_mask(mask), max_expansions=2000).plan(
        frozenset({("target", "at", start)}),
        frozenset({("target", "at", GOAL)}),
    )
    if not result.success:
        return None
    return [name.split("_")[2] for name in result.actions]


def sample_layout(rng, n_obstacles: int, tries: int = 500):
    for _ in range(tries):
        obstacles = []
        while len(obstacles) < n_obstacles:
            p = rng.uniform([-0.26, -0.20], [0.26, 0.20])
            if min(np.linalg.norm(p - c) for c in REGIONS.values()) < 0.10:
                continue
            if any(np.linalg.norm(p - q) < 0.08 for q in obstacles):
                continue
            obstacles.append(p)
        mask = clear_mask(obstacles)
        if any(reachable(mask, s) for s in START_NAMES):
            return np.asarray(obstacles), mask
    raise RuntimeError("could not sample a solvable layout")


def episodes(
    rng,
    n_layouts: int,
    n_obstacles: int,
    starts_per_layout: int = 2,
    min_plan_len: int = 2,
):
    out = []
    while len(out) < n_layouts:
        obstacles, mask = sample_layout(rng, n_obstacles)
        viable = []
        for s in START_NAMES:
            plan = plan_from(mask, s)
            if plan is not None and len(plan) >= min_plan_len:
                viable.append(s)
        rng.shuffle(viable)
        for start in viable[:starts_per_layout]:
            out.append({"obstacles": obstacles, "mask": mask, "start": start})
            if len(out) >= n_layouts:
                break
    return out


def demonstrations(episodes_):
    rows = []
    for ep in episodes_:
        plan = plan_from(ep["mask"], ep["start"])
        if plan is None:
            continue
        cur = ep["start"]
        for dst in plan:
            rows.append(
                {
                    "mask": ep["mask"],
                    "current": NAMES.index(cur),
                    "goal": NAMES.index(GOAL),
                    "label": NAMES.index(dst),
                }
            )
            cur = dst
    return rows


class RegionPolicy(nn.Module):
    """Transformer over five region tokens with an edge-mask feature."""

    def __init__(self, d_model: int = 64, layers: int = 2, heads: int = 4):
        super().__init__()
        self.region = nn.Embedding(len(NAMES), d_model)
        self.kind = nn.Embedding(3, d_model)  # 0 normal, 1 current, 2 goal
        self.edge = nn.Linear(len(NAMES), d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 2,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Linear(d_model, len(NAMES))

    def forward(self, mask, current, goal):
        b = mask.shape[0]
        ids = torch.arange(len(NAMES), device=mask.device)[None].expand(b, -1)
        tokens = self.region(ids) + self.edge(mask)
        kind = torch.zeros_like(ids)
        kind[torch.arange(b), current] = 1
        kind[torch.arange(b), goal] = 2
        tokens = tokens + self.kind(kind)
        h = self.encoder(tokens)
        return self.head(h[torch.arange(b), current])


def _tensors(rows, device):
    mask = torch.as_tensor(
        np.stack([r["mask"] for r in rows]), dtype=torch.float32, device=device
    )
    cur = torch.as_tensor([r["current"] for r in rows], dtype=torch.long, device=device)
    goal = torch.as_tensor([r["goal"] for r in rows], dtype=torch.long, device=device)
    label = torch.as_tensor([r["label"] for r in rows], dtype=torch.long, device=device)
    return mask, cur, goal, label


def train_policy(train_rows, val_rows, seed, epochs=300, device="cpu"):
    torch.manual_seed(seed)
    model = RegionPolicy().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    tr = _tensors(train_rows, device)
    va = _tensors(val_rows, device)
    best_state, best_val = None, -1.0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(train_rows), device=device)
        for s in range(0, len(perm), 64):
            idx = perm[s : s + 64]
            logits = model(tr[0][idx], tr[1][idx], tr[2][idx])
            loss = F.cross_entropy(logits, tr[3][idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(va[0], va[1], va[2]).argmax(dim=1)
            acc = float((pred == va[3]).float().mean().cpu())
        if acc > best_val:
            best_val = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model, best_val


def run_policy(model, mask, start, max_steps=8, device="cpu"):
    cur = start
    invalid = 0
    steps = 0
    for _ in range(max_steps):
        m = torch.as_tensor(mask[None], dtype=torch.float32, device=device)
        c = torch.as_tensor([NAMES.index(cur)], dtype=torch.long, device=device)
        g = torch.as_tensor([NAMES.index(GOAL)], dtype=torch.long, device=device)
        with torch.no_grad():
            dst_idx = int(model(m, c, g).argmax(dim=1).cpu())
        dst = NAMES[dst_idx]
        if dst == cur or not mask[NAMES.index(cur), dst_idx]:
            invalid += 1
            return False, steps, invalid
        cur = dst
        steps += 1
        if cur == GOAL:
            return True, steps, invalid
    return False, steps, invalid


def _pss_rollout(mask, start, max_steps=8):
    cur = start
    steps = 0
    plan = plan_from(mask, cur)
    if plan is None:
        return False, steps
    for dst in plan[:max_steps]:
        if not mask[NAMES.index(cur), NAMES.index(dst)]:
            return False, steps
        cur = dst
        steps += 1
        if cur == GOAL:
            return True, steps
    return cur == GOAL, steps


def _dynamic_rollout(ep, model, device, policy: bool, max_steps=10):
    """Move once under the original layout, then switch to a new obstacle set."""
    cur = ep["start"]
    steps = 0
    mask = ep["mask"]
    plan = None if policy else plan_from(mask, cur)
    if not policy and plan is None:
        return False, 0, 0
    invalid = 0
    changed = False
    for _ in range(max_steps):
        if not changed and steps == 1:
            mask = ep["new_mask"]
            changed = True
            if not reachable(mask, cur):
                return False, steps, invalid
            if not policy:
                plan = plan_from(mask, cur)
                if plan is None:
                    return False, steps, invalid
        if policy:
            m = torch.as_tensor(mask[None], dtype=torch.float32, device=device)
            c = torch.as_tensor([NAMES.index(cur)], dtype=torch.long, device=device)
            g = torch.as_tensor([NAMES.index(GOAL)], dtype=torch.long, device=device)
            with torch.no_grad():
                dst_idx = int(model(m, c, g).argmax(dim=1).cpu())
            dst = NAMES[dst_idx]
            if dst == cur or not mask[NAMES.index(cur), dst_idx]:
                invalid += 1
                return False, steps, invalid
        else:
            if plan is None:
                plan = plan_from(mask, cur)
                if plan is None:
                    return False, steps, invalid
            dst = plan.pop(0)
            if not mask[NAMES.index(cur), NAMES.index(dst)]:
                return False, steps, 0
        cur = dst
        steps += 1
        if cur == GOAL:
            return True, steps, invalid
    return cur == GOAL, steps, invalid


def evaluate_episodes(model, eps, device, dynamic=False):
    pss_ok, pss_steps, tr_ok, tr_steps, tr_invalid = [], [], [], [], []
    for ep in eps:
        if dynamic:
            ok, s = _dynamic_rollout(ep, model, device, policy=False)[:2]
        else:
            ok, s = _pss_rollout(ep["mask"], ep["start"])
        pss_ok.append(float(ok))
        pss_steps.append(s)
        if dynamic:
            ok, s, inv = _dynamic_rollout(ep, model, device, policy=True)
        else:
            ok, s, inv = run_policy(model, ep["mask"], ep["start"], device=device)
        tr_ok.append(float(ok))
        tr_steps.append(s)
        tr_invalid.append(inv)
    return {
        "pss": {
            "success": float(np.mean(pss_ok)),
            "steps": float(np.mean([s for s, ok in zip(pss_steps, pss_ok) if ok]))
            if any(pss_ok)
            else 0.0,
        },
        "transformer": {
            "success": float(np.mean(tr_ok)),
            "steps": float(np.mean([s for s, ok in zip(tr_steps, tr_ok) if ok]))
            if any(tr_ok)
            else 0.0,
            "invalid_rate": float(np.mean([float(i > 0) for i in tr_invalid])),
        },
        "n": len(eps),
    }


def _make_dynamic(eps, rng, n_obstacles):
    out = []
    for ep in eps:
        plan = plan_from(ep["mask"], ep["start"])
        after = plan[0] if plan else ep["start"]
        for _ in range(100):
            obstacles, mask = sample_layout(rng, n_obstacles)
            if reachable(mask, after):
                out.append({**ep, "new_mask": mask, "new_obstacles": obstacles})
                break
    return out


def run(seed=0, n_train=200, n_val=50, n_test=100, epochs=300, device="cpu", seeds=5):
    rng = np.random.default_rng(seed)
    train_eps = episodes(rng, n_train, n_obstacles=1, min_plan_len=2)
    val_eps = episodes(rng, n_val, n_obstacles=1, min_plan_len=2)
    id_eps = episodes(rng, n_test, n_obstacles=1, min_plan_len=2)
    ood_eps = episodes(rng, n_test, n_obstacles=2, min_plan_len=2)
    dyn_id = _make_dynamic(id_eps, rng, n_obstacles=1)
    dyn_ood = _make_dynamic(ood_eps, rng, n_obstacles=2)

    train_rows = demonstrations(train_eps)
    val_rows = demonstrations(val_eps)
    runs = []
    for s in range(seeds):
        model, best_val = train_policy(train_rows, val_rows, s, epochs, device)
        runs.append(
            {
                "seed": s,
                "val_action_acc": best_val,
                "n_params": sum(p.numel() for p in model.parameters()),
                "id": evaluate_episodes(model, id_eps, device),
                "ood": evaluate_episodes(model, ood_eps, device),
                "dynamic_id": evaluate_episodes(model, dyn_id, device, dynamic=True),
                "dynamic_ood": evaluate_episodes(model, dyn_ood, device, dynamic=True),
            }
        )
    return {"runs": runs, "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}


def aggregate(metrics):
    def mean(split, key, sub="transformer"):
        vals = [r[split][sub][key] for r in metrics["runs"]]
        return float(np.mean(vals)), float(np.std(vals))

    def pss(split, key):
        vals = [r[split]["pss"][key] for r in metrics["runs"]]
        return float(np.mean(vals)), float(np.std(vals))

    out = {}
    for split in ("id", "ood", "dynamic_id", "dynamic_ood"):
        out[split] = {
            "pss_success": pss(split, "success"),
            "pss_steps": pss(split, "steps"),
            "tr_success": mean(split, "success"),
            "tr_steps": mean(split, "steps"),
            "tr_invalid": mean(split, "invalid_rate")
            if split.startswith("dynamic")
            else (0.0, 0.0),
        }
    out["params"] = metrics["runs"][0]["n_params"]
    out["val_action_acc"] = (
        float(np.mean([r["val_action_acc"] for r in metrics["runs"]])),
        float(np.std([r["val_action_acc"] for r in metrics["runs"]])),
    )
    return out


def _report(path, agg):
    lines = [
        "# P1.3: explicit relation planner vs transformer policy",
        "",
        f"- transformer params: {agg['params']}",
        f"- val next-action accuracy: {agg['val_action_acc'][0]:.3f} ± {agg['val_action_acc'][1]:.3f}",
        "",
        "| split | PSS success | PSS steps | Transformer success | Transformer steps | invalid action rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "id": "ID (1 obstacle)",
        "ood": "OOD (2 obstacles)",
        "dynamic_id": "dynamic ID",
        "dynamic_ood": "dynamic OOD",
    }
    for split, label in labels.items():
        a = agg[split]
        lines.append(
            f"| {label} | {a['pss_success'][0]:.3f} | {a['pss_steps'][0]:.2f} | "
            f"{a['tr_success'][0]:.3f} ± {a['tr_success'][1]:.3f} | "
            f"{a['tr_steps'][0]:.2f} | {a['tr_invalid'][0]:.3f} |"
        )
    lines += [
        "",
        "PSS runs A* over explicit move operators rebuilt from the current edge mask.",
        "Transformer is behavior-cloned on A* plans with one obstacle.",
        "Dynamic splits relocate the obstacle after the first move; PSS replans, transformer reacts.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/results/pss_vs_transformer_v2")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    metrics = run(epochs=args.epochs, device=args.device, seeds=args.seeds)
    agg = aggregate(metrics)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "p13_planning.json").write_text(
        json.dumps({**metrics, "aggregate": agg}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _report(out / "p13_planning.md", agg)
    for split in ("id", "ood", "dynamic_id", "dynamic_ood"):
        a = agg[split]
        print(
            f"{split}: PSS {a['pss_success'][0]:.3f} vs TR {a['tr_success'][0]:.3f} "
            f"(invalid {a['tr_invalid'][0]:.3f})"
        )
    print(f"output: {out}")


if __name__ == "__main__":
    main()
