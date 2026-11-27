"""Version B, paired episodes: every arm is scored on exactly the same samples.

``run_version_b_composition`` draws episodes per arm from a shared RNG, so
each arm is evaluated on a *different* sample.  The report already flags this
("同一分布但不是同一批 episode，seed 级未配对").  This script fixes the episode
set first and evaluates every arm on it, then reports paired bootstrap CIs on
the arm differences.

Task, training rows, relation-head construction and seeds are unchanged from
the original script.  Only the evaluation protocol changes, so the two reports
are directly comparable: the means should be close, the CIs become meaningful.

Example
-------
python -m pss.eval.scripts.run_version_b_paired \
  --device cuda --seeds 3 --epochs 300 --n-test 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pss._console import ensure_utf8
from pss.eval.relational_bench import (
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
from pss.eval.scripts.run_version_b_composition import rollout_flat, train_flat

ARMS = ("probe_linear", "probe_mlp", "pss_learned", "pss_learned_heldout", "transformer")

# Fixed episode seeds: identical for every arm and every training seed.
ID_EPISODE_SEED = 1234
OOD_EPISODE_SEED = 4321

COMPARISONS = (
    ("pss_learned_minus_probe_mlp", "pss_learned", "probe_mlp"),
    ("pss_learned_minus_probe_linear", "pss_learned", "probe_linear"),
    ("pss_learned_minus_transformer", "pss_learned", "transformer"),
    ("pss_learned_minus_heldout", "pss_learned", "pss_learned_heldout"),
    ("heldout_minus_probe_mlp", "pss_learned_heldout", "probe_mlp"),
    ("transformer_minus_probe_mlp", "transformer", "probe_mlp"),
)


# --------------------------------------------------------------------------- #
# episodes
# --------------------------------------------------------------------------- #
def build_episodes(rel_pairs, n_per_pair: int, seed: int) -> list[dict]:
    """Draw a fixed episode set: same seed -> same states, in the same order."""

    rng = np.random.default_rng(seed)
    episodes = []
    for rel1, rel2 in rel_pairs:
        for _ in range(n_per_pair):
            state, _solution = make_episode(rng, rel1, rel2)
            episodes.append(
                {
                    "state": tuple(int(v) for v in state),
                    "rel1": rel1,
                    "rel2": rel2,
                }
            )
    return episodes


# --------------------------------------------------------------------------- #
# per-arm evaluation on an explicit episode list
# --------------------------------------------------------------------------- #
def evaluate_flat(model, episodes, device) -> tuple[np.ndarray, np.ndarray]:
    ok, invalid = [], []
    for ep in episodes:
        good, _steps, inv = rollout_flat(
            model, ep["state"], ep["rel1"], ep["rel2"], device
        )
        ok.append(float(good))
        invalid.append(float(inv > 0))
    return np.asarray(ok), np.asarray(invalid)


def evaluate_transformer(policy, episodes, device) -> tuple[np.ndarray, np.ndarray]:
    ok, invalid = [], []
    for ep in episodes:
        good, _steps, inv = rollout(
            policy, ep["state"], ep["rel1"], ep["rel2"], device
        )
        ok.append(float(good))
        invalid.append(float(inv > 0))
    return np.asarray(ok), np.asarray(invalid)


def evaluate_pss(rel_fn, episodes) -> np.ndarray:
    ok = []
    for ep in episodes:
        solution = plan(ep["state"], ep["rel1"], ep["rel2"], rel_fn=rel_fn)
        ok.append(
            float(
                solution is not None
                and goal_ok(_apply(ep["state"], solution), ep["rel1"], ep["rel2"])
            )
        )
    return np.asarray(ok)


def _eval_split(linear, mlp, transformer, learned_all, learned_held, episodes, device):
    ok_linear, inv_linear = evaluate_flat(linear, episodes, device)
    ok_mlp, inv_mlp = evaluate_flat(mlp, episodes, device)
    ok_tf, inv_tf = evaluate_transformer(transformer, episodes, device)
    return {
        "probe_linear": {"success": ok_linear, "invalid": inv_linear},
        "probe_mlp": {"success": ok_mlp, "invalid": inv_mlp},
        "transformer": {"success": ok_tf, "invalid": inv_tf},
        "pss_learned": {"success": evaluate_pss(learned_all, episodes), "invalid": None},
        "pss_learned_heldout": {"success": evaluate_pss(learned_held, episodes), "invalid": None},
    }


# --------------------------------------------------------------------------- #
# paired statistics
# --------------------------------------------------------------------------- #
def paired_bootstrap(delta: np.ndarray, n_boot: int = 10000, seed: int = 0) -> dict:
    """Bootstrap the mean of a per-episode delta vector.

    The input is already paired: ``delta[i]`` is the difference on episode i.
    """

    delta = np.asarray(delta, dtype=np.float64)
    if delta.size == 0:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, delta.size, size=(n_boot, delta.size))
    boot = delta[idx].mean(axis=1)
    return {
        "mean": float(delta.mean()),
        "lo": float(np.percentile(boot, 2.5)),
        "hi": float(np.percentile(boot, 97.5)),
        "n": int(delta.size),
    }


def _stack(runs, split: str, arm: str, key: str) -> np.ndarray | None:
    if runs[0][split][arm].get(key) is None:
        return None
    return np.stack([np.asarray(r[split][arm][key], dtype=np.float64) for r in runs])


def _agg(runs, split: str, arm: str, key: str) -> dict | None:
    stacked = _stack(runs, split, arm, key)
    if stacked is None:
        return None
    per_seed = [float(v.mean()) for v in stacked]
    return {"mean": float(np.mean(per_seed)), "std": float(np.std(per_seed))}


def _paired(runs, split: str, arm_a: str, arm_b: str, n_boot: int) -> dict:
    a = _stack(runs, split, arm_a, "success")
    b = _stack(runs, split, arm_b, "success")
    # Same episodes across seeds, so average over seeds first, then pair by
    # episode and bootstrap over episodes.
    delta = (a - b).mean(axis=0)
    return paired_bootstrap(delta, n_boot=n_boot)


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def run(device="cuda", seeds=3, epochs=300, n_test=50, n_boot=10000,
        out_dir="eval/results/version_b_paired") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_rows = _rows(TRAIN_PAIRS, 80, np.random.default_rng(0))
    id_episodes = build_episodes(TRAIN_PAIRS, n_test, seed=ID_EPISODE_SEED)
    ood_episodes = build_episodes(TEST_PAIRS, n_test, seed=OOD_EPISODE_SEED)

    head_all, _, exact_all, per_all = _train_spatial_head(device, train_regions=None)
    head_held, _, exact_held, per_held = _train_spatial_head(
        device, train_regions=("A", "B", "C")
    )
    learned_all = _learned_rel(head_all, device)
    learned_held = _learned_rel(head_held, device)

    runs = []
    for seed in range(seeds):
        linear = train_flat(train_rows, seed, epochs, device, hidden=None)
        mlp = train_flat(train_rows, seed, epochs, device, hidden=128)
        transformer = train_policy(train_rows, seed, epochs, device)
        runs.append(
            {
                "seed": seed,
                "id": _eval_split(linear, mlp, transformer, learned_all, learned_held,
                                  id_episodes, device),
                "ood": _eval_split(linear, mlp, transformer, learned_all, learned_held,
                                   ood_episodes, device),
            }
        )
        print(
            f"seed {seed}: "
            + ", ".join(
                f"{arm} ID/OOD {runs[-1]['id'][arm]['success'].mean():.3f}/"
                f"{runs[-1]['ood'][arm]['success'].mean():.3f}"
                for arm in ARMS
            ),
            flush=True,
        )

    summary = {
        split: {
            arm: {
                "success": _agg(runs, split, arm, "success"),
                "invalid": _agg(runs, split, arm, "invalid"),
            }
            for arm in ARMS
        }
        for split in ("id", "ood")
    }
    paired = {
        split: {
            name: _paired(runs, split, arm_a, arm_b, n_boot)
            for name, arm_a, arm_b in COMPARISONS
        }
        for split in ("id", "ood")
    }

    payload = {
        "task": "P5b relation-composition OOD (paired episodes)",
        "protocol": "identical episodes for every arm; paired bootstrap over episodes",
        "episode_seeds": {"id": ID_EPISODE_SEED, "ood": OOD_EPISODE_SEED},
        "train_pairs": [list(p) for p in TRAIN_PAIRS],
        "test_pairs": [list(p) for p in TEST_PAIRS],
        "seeds": seeds,
        "epochs": epochs,
        "n_test_per_pair": n_test,
        "relation_probe": {
            "indist": {"exact4": exact_all, "per_label": per_all},
            "heldout_regions": {"exact4": exact_held, "per_label": per_held},
        },
        "summary": summary,
        "paired_deltas": paired,
        "runs": [
            {
                "seed": r["seed"],
                **{
                    split: {
                        arm: {
                            "success": float(r[split][arm]["success"].mean()),
                            "invalid": (
                                None
                                if r[split][arm]["invalid"] is None
                                else float(r[split][arm]["invalid"].mean())
                            ),
                        }
                        for arm in ARMS
                    }
                    for split in ("id", "ood")
                },
            }
            for r in runs
        ],
    }

    (out / "version_b_paired.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "REPORT.md").write_text(_render(payload), encoding="utf-8")
    return payload


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def _fmt(value, digits=3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _render(payload: dict) -> str:
    lines = [
        "# Version B（配对版）：同一批 episode 上的三臂对照",
        "",
        f"- task: {payload['task']}",
        f"- protocol: {payload['protocol']}",
        f"- episode seeds: id={payload['episode_seeds']['id']}, "
        f"ood={payload['episode_seeds']['ood']}",
        f"- seeds: {payload['seeds']} · epochs: {payload['epochs']} · "
        f"n/pair: {payload['n_test_per_pair']}",
        "- 脚本：`pss/eval/scripts/run_version_b_paired.py`",
        "",
        "---",
        "",
        "## 1. 主表（配对 episode）",
        "",
        "| arm | ID success | OOD success | OOD invalid |",
        "|---|---:|---:|---:|",
    ]
    for arm in ARMS:
        id_s = payload["summary"]["id"][arm]["success"]
        ood_s = payload["summary"]["ood"][arm]["success"]
        ood_i = payload["summary"]["ood"][arm]["invalid"]
        lines.append(
            f"| {arm} | {_fmt(id_s['mean'])} ± {_fmt(id_s['std'])} | "
            f"{_fmt(ood_s['mean'])} ± {_fmt(ood_s['std'])} | "
            f"{'n/a' if ood_i is None else _fmt(ood_i['mean'])} |"
        )

    lines += [
        "",
        "## 2. 配对差值与 bootstrap 95% CI",
        "",
        "| 对照 | split | Δ | 95% CI | 排除 0 |",
        "|---|---|---:|---|---|",
    ]
    for split in ("id", "ood"):
        for name, _a, _b in COMPARISONS:
            ci = payload["paired_deltas"][split][name]
            excludes = ci["lo"] > 0.0 or ci["hi"] < 0.0
            lines.append(
                f"| {name} | {split} | {_fmt(ci['mean'])} | "
                f"[{_fmt(ci['lo'])}, {_fmt(ci['hi'])}] | "
                f"{'是' if excludes else '否'} |"
            )

    probe = payload["relation_probe"]
    lines += [
        "",
        "## 3. 关系探针（(a) 的知识读出）",
        "",
        "| 训练区域 | exact-4 | per-label |",
        "|---|---:|---:|",
        f"| 全部区域（in-dist） | {_fmt(probe['indist']['exact4'])} | "
        f"{_fmt(probe['indist']['per_label'])} |",
        f"| A/B/C（held-out 区域） | {_fmt(probe['heldout_regions']['exact4'])} | "
        f"{_fmt(probe['heldout_regions']['per_label'])} |",
        "",
        "## 4. 判读",
        "",
        "1. 本表与 `VERSION_B_REPORT.md` 使用同一任务、同一训练行、同一 seed，"
        "差别只在**评测协议**：这里所有臂在同一批 episode 上打分。",
        "2. 均值应与原报告接近；主要变化是 CI 现在按 episode 配对，"
        "因此可用于后续 B1/B2/B3 的增量比较。",
        "3. `pss_learned − probe_mlp` 的 OOD 置信区间若完全排除 0，"
        "则 §5.7 的 +42.7 点从效应量升级为可报告的统计结论。",
        "",
        "---",
        "",
        "边界：P5b 是符号、小规模任务；关系头覆盖仍是上限（见 §5.7）。",
        "",
    ]
    return "\n".join(lines)


def _cuda() -> bool:
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency
        return False
    return bool(torch.cuda.is_available())


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda" if _cuda() else "cpu")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--n-test", type=int, default=50)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", default="eval/results/version_b_paired")
    args = ap.parse_args()
    run(
        device=args.device,
        seeds=args.seeds,
        epochs=args.epochs,
        n_test=args.n_test,
        n_boot=args.n_boot,
        out_dir=args.out,
    )


if __name__ == "__main__":
    main()
