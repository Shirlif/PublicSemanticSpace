"""Version B: true (a)/(b)/(c) attribution on the P5b relation-composition task.

Task: three objects in five regions; goal = object 0 at G, object 1 in relation
R1 to object 0, object 2 in relation R2 to object 1. Train on four relation
pairs, test on four held-out pairs (relation labels seen, pair composition
novel).

Arms (same train rows, same seeds, same episodes):

  (a) flat probe      linear / small MLP policy on one-hot state + relation
                      pair; no explicit goal check, no search. This is the
                      "readout lower bound" arm.
  (b) PSS composition learned spatial-relation head + explicit goal check +
                      BFS over placements. ``pss_hand`` is the hand-written
                      relation upper version of the same planner.
  (c) transformer     end-to-end token policy trained by behavior cloning.

Outputs success / invalid-action rate on ID (seen pairs) and OOD (held-out
pairs), plus relation-probe accuracy for the (a) knowledge probe.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pss._console import ensure_utf8
from pss.eval.relational_bench import (
    NAMES,
    RELATIONS,
    TEST_PAIRS,
    TRAIN_PAIRS,
    _rows,
    goal_ok,
    make_episode,
    plan,
    rollout,
    train_policy,
)
from pss.eval.scripts.run_typed_relations import (
    _apply,
    _learned_rel,
    _train_spatial_head,
)

N_ACTIONS = 3 * len(NAMES)
STATE_DIM = 3 * len(NAMES)
REL_DIM = len(RELATIONS)


class FlatPolicy(nn.Module):
    """One-hot state + relation pair -> action. No goal check, no search."""

    def __init__(self, hidden: int | None = None):
        super().__init__()
        in_dim = STATE_DIM + 2 * REL_DIM
        if hidden is None:
            self.net = nn.Linear(in_dim, N_ACTIONS)
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, N_ACTIONS),
            )

    def forward(self, states, rel1, rel2):
        s = F.one_hot(states, num_classes=len(NAMES)).float().flatten(1)
        r = torch.cat(
            [F.one_hot(rel1, num_classes=REL_DIM).float(),
             F.one_hot(rel2, num_classes=REL_DIM).float()], dim=1
        )
        return self.net(torch.cat([s, r], dim=1))


def _tensors(rows, device):
    states = torch.as_tensor(np.stack([r["state"] for r in rows]), dtype=torch.long, device=device)
    rel1 = torch.as_tensor([r["rel1"] for r in rows], dtype=torch.long, device=device)
    rel2 = torch.as_tensor([r["rel2"] for r in rows], dtype=torch.long, device=device)
    labels = torch.as_tensor([r["action"] for r in rows], dtype=torch.long, device=device)
    return states, rel1, rel2, labels


def train_flat(rows, seed, epochs=300, device="cuda", hidden=128):
    torch.manual_seed(seed)
    model = FlatPolicy(hidden=hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    data = _tensors(rows, device)
    best_state, best_loss = None, float("inf")
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(rows), device=device)
        total = 0.0
        for start in range(0, len(perm), 64):
            idx = perm[start:start + 64]
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


def rollout_flat(model, state, rel1, rel2, device, max_steps=8):
    cur = tuple(state)
    r1 = torch.as_tensor([RELATIONS.index(rel1)], dtype=torch.long, device=device)
    r2 = torch.as_tensor([RELATIONS.index(rel2)], dtype=torch.long, device=device)
    steps, invalid = 0, 0
    for _ in range(max_steps):
        if goal_ok(cur, rel1, rel2):
            return True, steps, invalid
        s = torch.as_tensor([cur], dtype=torch.long, device=device)
        with torch.no_grad():
            act = int(model(s, r1, r2).argmax(dim=1).cpu())
        obj, region = divmod(act, len(NAMES))
        if cur[obj] == region or region in cur:
            invalid += 1
            return False, steps, invalid
        nxt = list(cur)
        nxt[obj] = region
        cur = tuple(nxt)
        steps += 1
    return goal_ok(cur, rel1, rel2), steps, invalid


def evaluate_flat(model, pairs, n, rng, device):
    ok, inv = [], []
    for rel1, rel2 in pairs:
        for _ in range(n):
            state, _ = make_episode(rng, rel1, rel2)
            good, _steps, invalid = rollout_flat(model, state, rel1, rel2, device)
            ok.append(float(good))
            inv.append(float(invalid > 0))
    return {"success": float(np.mean(ok)), "invalid": float(np.mean(inv)), "n": len(ok)}


def evaluate_pss(learned_rel, pairs, n, rng):
    ok = []
    for rel1, rel2 in pairs:
        for _ in range(n):
            state, _ = make_episode(rng, rel1, rel2)
            sol = plan(state, rel1, rel2, rel_fn=learned_rel)
            ok.append(float(sol is not None and goal_ok(_apply(state, sol), rel1, rel2)))
    return {"success": float(np.mean(ok)), "n": len(ok)}


def evaluate_transformer(policy, pairs, n, rng, device):
    ok, inv = [], []
    for rel1, rel2 in pairs:
        for _ in range(n):
            state, _ = make_episode(rng, rel1, rel2)
            good, _steps, invalid = rollout(policy, state, rel1, rel2, device)
            ok.append(float(good))
            inv.append(float(invalid > 0))
    return {"success": float(np.mean(ok)), "invalid": float(np.mean(inv)), "n": len(ok)}


def run(device="cuda", seeds=3, epochs=300, n_test=50, out_dir="eval/results/version_b"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = _rows(TRAIN_PAIRS, 80, np.random.default_rng(0))

    # relation probe (the (a) knowledge probe) and PSS relation head
    head_indist, _, exact_indist, per_indist = _train_spatial_head(device, train_regions=None)
    head_held, held_pairs, exact_held, per_held = _train_spatial_head(
        device, train_regions=("A", "B", "C")
    )
    learned_indist = _learned_rel(head_indist, device)
    learned_held = _learned_rel(head_held, device)

    runs = []
    for seed in range(seeds):
        linear = train_flat(rows, seed, epochs, device, hidden=None)
        mlp = train_flat(rows, seed, epochs, device, hidden=128)
        transformer = train_policy(rows, seed, epochs, device)
        id_rng = np.random.default_rng(100 + seed)
        ood_rng = np.random.default_rng(200 + seed)
        runs.append({
            "seed": seed,
            "id": {
                "probe_linear": evaluate_flat(linear, TRAIN_PAIRS, n_test, id_rng, device),
                "probe_mlp": evaluate_flat(mlp, TRAIN_PAIRS, n_test, id_rng, device),
                "pss_learned": evaluate_pss(learned_indist, TRAIN_PAIRS, n_test, id_rng),
                "transformer": evaluate_transformer(transformer, TRAIN_PAIRS, n_test, id_rng, device),
            },
            "ood": {
                "probe_linear": evaluate_flat(linear, TEST_PAIRS, n_test, ood_rng, device),
                "probe_mlp": evaluate_flat(mlp, TEST_PAIRS, n_test, ood_rng, device),
                "pss_learned": evaluate_pss(learned_indist, TEST_PAIRS, n_test, ood_rng),
                "pss_learned_heldout": evaluate_pss(learned_held, TEST_PAIRS, n_test, ood_rng),
                "transformer": evaluate_transformer(transformer, TEST_PAIRS, n_test, ood_rng, device),
            },
        })
        print(f"seed {seed}: " + ", ".join(
            f"{k} ID/OOD {runs[-1]['id'][k]['success']:.3f}/{runs[-1]['ood'][k]['success']:.3f}"
            for k in ("probe_linear", "probe_mlp", "pss_learned", "transformer")
        ), flush=True)

    def agg(split, method, key="success"):
        vals = [
            r[split][method][key]
            for r in runs
            if method in r[split] and key in r[split][method]
        ]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))} if vals else None

    payload = {
        "task": "P5b relation-composition OOD",
        "train_pairs": TRAIN_PAIRS,
        "test_pairs": TEST_PAIRS,
        "seeds": seeds,
        "epochs": epochs,
        "n_test_per_pair": n_test,
        "relation_probe": {
            "indist": {"exact4": exact_indist, "per_label": per_indist},
            "heldout_regions": {"exact4": exact_held, "per_label": per_held, "pairs": held_pairs},
        },
        "summary": {
            split: {method: {"success": agg(split, method), "invalid": agg(split, method, "invalid")}
                    for method in ("probe_linear", "probe_mlp", "pss_learned", "pss_learned_heldout", "transformer")
                    if any(method in r[split] for r in runs)}
            for split in ("id", "ood")
        },
        "runs": runs,
    }
    (out / "version_b.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "VERSION_B_REPORT.md").write_text(render(payload), encoding="utf-8")
    return payload


def _fmt(x, d=3):
    return "n/a" if x is None else f"{x:.{d}f}"


def render(p: dict) -> str:
    lines = [
        "# Version B：关系/组合 OOD 的三臂归因",
        "",
        f"- task: {p['task']}",
        f"- train pairs: {p['train_pairs']}",
        f"- test pairs: {p['test_pairs']}",
        f"- seeds: {p['seeds']} · epochs: {p['epochs']} · n/pair: {p['n_test_per_pair']}",
        "",
        "## 1. 主表（planning success）",
        "",
        "| arm | ID success | OOD success | OOD invalid |",
        "|---|---:|---:|---:|",
    ]
    names = {
        "probe_linear": "(a) 线性平坦策略",
        "probe_mlp": "(a) MLP 平坦策略",
        "pss_learned": "(b) PSS 学习关系 + BFS",
        "pss_learned_heldout": "(b) PSS held-out 区域关系 + BFS",
        "transformer": "(c) transformer 端到端策略",
    }
    for key, label in names.items():
        idv = p["summary"]["id"].get(key, {}).get("success")
        ood = p["summary"]["ood"].get(key, {}).get("success")
        inv = p["summary"]["ood"].get(key, {}).get("invalid")
        lines.append(
            f"| {label} | {_fmt(idv['mean'] if idv else None)} ± {_fmt(idv['std'] if idv else None)} "
            f"| {_fmt(ood['mean'] if ood else None)} ± {_fmt(ood['std'] if ood else None)} "
            f"| {_fmt((inv['mean'] if inv else None))} |"
        )
    rel = p["relation_probe"]
    lines += [
        "",
        "## 2. 关系探针（(a) 的知识读出）",
        "",
        "| 训练区域 | exact-4 | per-label |",
        "|---|---:|---:|",
        f"| 全部区域（in-dist） | {_fmt(rel['indist']['exact4'])} | {_fmt(rel['indist']['per_label'])} |",
        f"| A/B/C（held-out 区域） | {_fmt(rel['heldout_regions']['exact4'])} | {_fmt(rel['heldout_regions']['per_label'])} |",
        "",
        "## 3. 判读",
        "",
        "- **(b) > (a)**：显式关系 + goal check + 搜索从同一符号输入里多榨出了组合泛化 → 结构先验有用。",
        "- **(b) ≈ (a)**：平坦探针也能从训练对里学到组合；结构先验的增量小。",
        "- **(b) < (a)**：关系头/搜索引入误差，反而不如端到端平坦策略。",
        "- **(c) ≫ (b)**：端到端高容量策略在 ID 上更强，但看 OOD；若 OOD 掉得快，说明它在记训练对而不是组合。",
        "- 注意：P5b 是符号任务，`(a)` 的“冻结特征”是状态/区域 one-hot，不是感知编码器；它测的是**组合读出**，不是感知信息。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--n-test", type=int, default=50)
    ap.add_argument("--out", default="eval/results/version_b")
    args = ap.parse_args()
    run(args.device, args.seeds, args.epochs, args.n_test, args.out)


if __name__ == "__main__":
    main()
