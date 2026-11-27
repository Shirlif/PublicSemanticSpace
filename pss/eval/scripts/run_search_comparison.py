"""B3: BFS vs placement-space A* on paired P5b episodes.

The plan keeps BFS as the default and only adopts A* if it is not worse on
success, validity, steps **and latency** (the plan's metric list includes
latency).  The operational gate is therefore, per arm and per split:

    switchable := success >= BFS success
                  AND step regret vs BFS <= 0
                  AND p50 latency <= BFS p50 latency

p50 is the robust latency statistic; p95 is reported next to it so tail
behaviour is visible even when p50 passes.  All planners run on the same
paired episode set, so the comparison is not two independent draws.

Arms:
  bfs          reference planner from ``pss.eval.relational_bench.plan``,
               re-implemented here with a node counter
  astar_w1     placement A*, heuristic weight 1.0 (unmet goal constraints)
  astar_w15    weighted A*, weight 1.5
  astar_w2     weighted A*, weight 2.0

Metrics: success, steps, expanded nodes, wall-clock latency, step regret vs
BFS.  A plan is only counted successful when the final state satisfies the
goal and keeps all objects in distinct regions.

Example
-------
python -m pss.eval.scripts.run_search_comparison --n 60 --out eval/results/search_comparison
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np

from pss._console import ensure_utf8
from pss.eval.relational_bench import (
    NAMES,
    TEST_PAIRS,
    TRAIN_PAIRS,
    _rel,
    goal_ok,
    make_episode,
)
from pss.reasoning.planner.astar import apply_actions, plan_placements

WEIGHTS = (1.0, 1.5, 2.0)


def bfs_plan(state, rel1, rel2, rel_fn=_rel, max_nodes: int = 20000):
    """BFS over placements with the same successors as ``relational_bench.plan``."""

    start = tuple(int(x) for x in state)
    if goal_ok(start, rel1, rel2, rel_fn):
        return [], 0
    frontier = deque([start])
    parent = {start: None}
    nodes = 0
    while frontier and nodes < max_nodes:
        current = frontier.popleft()
        nodes += 1
        for obj in range(len(current)):
            for region in range(len(NAMES)):
                if current[obj] == region or region in current:
                    continue
                nxt = list(current)
                nxt[obj] = region
                nxt = tuple(nxt)
                if nxt in parent:
                    continue
                parent[nxt] = (current, (obj, region))
                if goal_ok(nxt, rel1, rel2, rel_fn):
                    actions = []
                    cur = nxt
                    while parent[cur] is not None:
                        prev, action = parent[cur]
                        actions.append(action)
                        cur = prev
                    actions.reverse()
                    return actions, nodes
                frontier.append(nxt)
    return None, nodes


def _valid(start, actions, rel1, rel2, rel_fn=_rel) -> bool:
    if actions is None:
        return False
    final = apply_actions(start, actions)
    if len(set(final)) != len(final):
        return False
    return goal_ok(final, rel1, rel2, rel_fn)


def _one_episode(state, rel1, rel2):
    out = {}
    t0 = time.perf_counter()
    bfs_actions, bfs_nodes = bfs_plan(state, rel1, rel2)
    bfs_ms = (time.perf_counter() - t0) * 1000.0
    out["bfs"] = {
        "actions": bfs_actions,
        "steps": 0 if bfs_actions is None else len(bfs_actions),
        "nodes": bfs_nodes,
        "ms": bfs_ms,
        "valid": _valid(state, bfs_actions, rel1, rel2),
    }
    for weight in WEIGHTS:
        t0 = time.perf_counter()
        plan = plan_placements(state, rel1, rel2, NAMES, _rel, weight=weight)
        ms = (time.perf_counter() - t0) * 1000.0
        actions = list(plan.actions) if plan.success else None
        out[f"astar_w{weight:g}"] = {
            "actions": actions,
            "steps": 0 if actions is None else len(actions),
            "nodes": plan.nodes,
            "ms": ms,
            "valid": _valid(state, actions, rel1, rel2),
        }
    return out


def _aggregate(rows: list[dict], arm: str) -> dict:
    vals = [r[arm] for r in rows]
    return {
        "n": len(vals),
        "success": float(np.mean([float(v["valid"]) for v in vals])),
        "steps": float(np.mean([v["steps"] for v in vals])),
        "nodes": float(np.mean([v["nodes"] for v in vals])),
        "latency_ms": float(np.mean([v["ms"] for v in vals])),
        "latency_p50_ms": float(np.median([v["ms"] for v in vals])),
        "latency_p95_ms": float(np.percentile([v["ms"] for v in vals], 95)),
        "regret_vs_bfs": float(
            np.mean([v["steps"] - r["bfs"]["steps"] for v, r in zip(vals, rows)])
        ),
    }


def run(
    n_per_pair: int = 60,
    seed: int = 4321,
    out_dir: str = "eval/results/search_comparison",
) -> dict:
    rng = np.random.default_rng(seed)
    splits = {}
    for label, pairs in (("id", TRAIN_PAIRS), ("ood", TEST_PAIRS)):
        rows = []
        for rel1, rel2 in pairs:
            for _ in range(n_per_pair):
                state, _solution = make_episode(rng, rel1, rel2)
                rows.append(_one_episode(state, rel1, rel2))
        splits[label] = {
            "pairs": [list(p) for p in pairs],
            "rows": rows,
            "summary": {
                arm: _aggregate(rows, arm)
                for arm in ("bfs",) + tuple(f"astar_w{w:g}" for w in WEIGHTS)
            },
        }

    switches = {}
    for label, split in splits.items():
        baseline = split["summary"]["bfs"]
        switches[label] = {}
        for arm, summary in split["summary"].items():
            if not arm.startswith("astar"):
                continue
            gates = {
                "success": summary["success"] >= baseline["success"],
                "regret": summary["regret_vs_bfs"] <= 0,
                "latency_p50": summary["latency_p50_ms"] <= baseline["latency_p50_ms"],
            }
            switches[label][arm] = {**gates, "switchable": all(gates.values())}

    payload = {
        "task": "P5b placement search: BFS vs A*",
        "regions": list(NAMES),
        "n_per_pair": int(n_per_pair),
        "episode_seed": int(seed),
        "switch_condition": (
            "success >= BFS success AND step regret vs BFS <= 0 AND "
            "p50 latency <= BFS p50 latency"
        ),
        "splits": splits,
        "switches": switches,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "search_comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "SEARCH_REPORT.md").write_text(_render(payload), encoding="utf-8")
    return payload


def _gate_label(gate) -> str:
    """Compact gate cell for the report table."""

    if gate is None:
        return "baseline"
    if gate["switchable"]:
        return "通过"
    failed = [
        label
        for label, key in (
            ("success", "success"),
            ("regret", "regret"),
            ("p50", "latency_p50"),
        )
        if not gate[key]
    ]
    return "未过: " + ",".join(failed)


def _render(p: dict) -> str:
    lines = [
        "# B3：BFS vs placement A*（配对 episode）",
        "",
        f"- regions: {', '.join(p['regions'])} · episodes per pair: {p['n_per_pair']}",
        f"- episode seed: {p['episode_seed']}",
        "",
        "## 1. 汇总",
        "",
        "| split | planner | success | steps | nodes | p50 (ms) | p95 (ms) | "
        "regret | gate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for split in ("id", "ood"):
        for arm, s in p["splits"][split]["summary"].items():
            gate = p["switches"][split].get(arm)
            lines.append(
                f"| {split} | {arm} | {s['success']:.3f} | {s['steps']:.2f} | "
                f"{s['nodes']:.1f} | {s['latency_p50_ms']:.3f} | "
                f"{s['latency_p95_ms']:.3f} | {s['regret_vs_bfs']:+.3f} | "
                f"{_gate_label(gate)} |"
            )

    lines += [
        "",
        "## 2. 判读",
        "",
        f"- 预定义切换条件：`{p['switch_condition']}`。全部满足才可切换，否则 BFS 保持默认。",
        "- nodes 是正确性之外的效率指标：A* 的节点数应显著低于 BFS，否则只是换了遍历顺序。",
        "- 加权 A*（w>1）用步数换扩展数；只有 regret 仍为 0 时才值得保留。",
        "",
    ]
    for split in ("id", "ood"):
        summary = p["splits"][split]["summary"]
        baseline = summary["bfs"]
        lines.append(
            f"- **{split} 基线 BFS**：success {baseline['success']:.3f}、"
            f"steps {baseline['steps']:.2f}、p50 {baseline['latency_p50_ms']:.3f} ms、"
            f"p95 {baseline['latency_p95_ms']:.3f} ms。"
        )
        for arm, gate in p["switches"][split].items():
            s = summary[arm]
            failed = [
                label
                for label, key in (
                    ("success", "success"),
                    ("regret", "regret"),
                    ("p50", "latency_p50"),
                )
                if not gate[key]
            ]
            verdict = "满足" if gate["switchable"] else "不满足"
            suffix = f"（未过：{', '.join(failed)}）" if failed else ""
            p95_note = (
                f"；p95 比 BFS 差（{s['latency_p95_ms']:.3f} vs "
                f"{baseline['latency_p95_ms']:.3f} ms）"
                if s["latency_p95_ms"] > baseline["latency_p95_ms"]
                else ""
            )
            lines.append(
                f"  - `{arm}`：success {s['success']:.3f}、regret "
                f"{s['regret_vs_bfs']:+.3f}、p50 {s['latency_p50_ms']:.3f} ms → "
                f"**{verdict}**{suffix}{p95_note}"
            )

    lines += [
        "",
        "## 3. 边界",
        "",
        "- 3 个物体、5 个区域的符号任务；状态空间只有 60 个，延迟差异在毫秒级。",
        "- 每个 episode 只跑一次，没有重复计时的方差估计；p50 比 mean 更抗抖动。",
        "- 成功以最终状态满足显式 `goal_ok` 为准，不是用 BFS 的解做对照答案。",
        "- 若计时与其它训练任务并行，绝对延迟会受 CPU 竞争影响；只作相对比较。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=4321)
    ap.add_argument("--out", default="eval/results/search_comparison")
    args = ap.parse_args()
    payload = run(n_per_pair=args.n, seed=args.seed, out_dir=args.out)
    for split in ("id", "ood"):
        summary = payload["splits"][split]["summary"]
        print(
            f"{split}: BFS success={summary['bfs']['success']:.3f} "
            f"nodes={summary['bfs']['nodes']:.1f} · "
            f"A* w1 success={summary['astar_w1']['success']:.3f} "
            f"nodes={summary['astar_w1']['nodes']:.1f} "
            f"regret={summary['astar_w1']['regret_vs_bfs']:+.3f}"
        )
    print(f"output: {Path(args.out) / 'SEARCH_REPORT.md'}")


if __name__ == "__main__":
    main()
