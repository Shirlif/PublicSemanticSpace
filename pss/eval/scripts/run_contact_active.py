"""C4 benchmark: passive vs active contact-event perception on live MuJoCo.

Three observation conditions are compared:

- ``fine``:  passive sees one frame every 10 steps (20 ms); full 0.8 s horizon.
- ``coarse``: passive sees one frame every 30 steps (60 ms); full 0.8 s horizon.
- ``short``: passive stops at a short nominal horizon (drop 110 steps,
  collide 30 steps); the active arm may wait beyond it.

The active policy is observation-only (re-observe / wait); no physical push or
tap is applied.  Ground truth is the MuJoCo contact force over the full
simulated trajectory, shared by both arms.

Example
-------
python -u -m pss.eval.scripts.run_contact_active \
  --conditions fine coarse short --seeds 0 1 2 \
  --out eval/results/contact_active
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pss._console import ensure_utf8
from pss.events.active import ActiveConfig, run_live_scenario

CONDITIONS = {
    "fine": {
        "base_every": 10,
        "nominal": {"drop": 400, "collide": 400, "near_miss": 400},
    },
    "coarse": {
        "base_every": 30,
        "nominal": {"drop": 400, "collide": 400, "near_miss": 400},
    },
    "short": {
        "base_every": 10,
        "nominal": {"drop": 110, "collide": 30, "near_miss": 400},
    },
}

ARMS = ("passive", "active")


def _agg(rows: list[dict]) -> dict:
    keys = (
        "precision",
        "recall",
        "f1",
        "onset_latency",
        "intensity_mae",
        "extra_frames",
        "extra_steps",
        "probes",
        "passive_frames",
        "active_frames",
    )
    out: dict = {}
    for key in keys:
        vals = [row[key] for row in rows if row.get(key) is not None]
        out[key] = {
            "mean": float(np.mean(vals)) if vals else None,
            "std": float(np.std(vals)) if vals else None,
            "n": len(vals),
        }
    tp = int(sum(row["tp"] for row in rows))
    fp = int(sum(row["fp"] for row in rows))
    fn = int(sum(row["fn"] for row in rows))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    out["tp"] = tp
    out["fp"] = fp
    out["fn"] = fn
    out["n_gt"] = int(sum(row["n_gt"] for row in rows))
    out["micro"] = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
    return out


def _flatten(result: dict, condition: str, seed: int) -> list[dict]:
    rows = []
    for arm in ARMS:
        metrics = result[arm]
        rows.append(
            {
                "condition": condition,
                "seed": seed,
                "kind": result["kind"],
                "scenario": result["scenario"],
                "arm": arm,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "onset_latency": metrics["onset_latency"],
                "intensity_mae": metrics["intensity_mae"],
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "n_gt": metrics["n_gt"],
                "extra_frames": result["extra_frames"] if arm == "active" else 0,
                "extra_steps": result["extra_steps"] if arm == "active" else 0,
                "probes": len(result["probes"]) if arm == "active" else 0,
                "probe_kinds": (
                    [p["kind"] for p in result["probes"]] if arm == "active" else []
                ),
                "passive_frames": result["passive_frames"],
                "active_frames": result["active_frames"],
                "probes_detail": result["probes"] if arm == "active" else [],
            }
        )
    return rows


def run(args) -> dict:
    seeds = list(args.seeds)
    conditions = list(args.conditions)
    rows: list[dict] = []
    runs: list[dict] = []
    for condition in conditions:
        preset = CONDITIONS[condition]
        for seed in seeds:
            for kind in args.scenarios:
                cfg = ActiveConfig(
                    kind=kind,
                    seed=seed,
                    nominal_steps=preset["nominal"][kind],
                    base_every=preset["base_every"],
                    dense_every=args.dense_every,
                    reobserve_steps=args.reobserve_steps,
                    max_wait_steps=args.max_wait_steps,
                )
                result = run_live_scenario(kind, cfg)
                result["condition"] = condition
                rows.extend(_flatten(result, condition, seed))
                runs.append(result)
                print(
                    f"[{condition} seed {seed} {kind}] "
                    f"passive F1 {result['passive']['f1']:.3f} "
                    f"active F1 {result['active']['f1']:.3f} "
                    f"probes {len(result['probes'])}",
                    flush=True,
                )

    groups: dict[str, dict[str, dict]] = {}
    event_kinds = [kind for kind in args.scenarios if kind != "near_miss"]
    for condition in conditions:
        groups[condition] = {}
        for arm in ARMS:
            arm_rows = [r for r in rows if r["condition"] == condition and r["arm"] == arm]
            event_rows = [r for r in arm_rows if r["kind"] in event_kinds]
            groups[condition][arm] = {
                "overall": _agg(arm_rows),
                "overall_events": _agg(event_rows) if event_rows else _agg(arm_rows),
                "by_kind": {
                    kind: _agg([r for r in arm_rows if r["kind"] == kind])
                    for kind in args.scenarios
                },
            }

    payload = {
        "stage": "contact_active",
        "config": {
            "conditions": {
                name: CONDITIONS[name] for name in conditions
            },
            "seeds": seeds,
            "scenarios": list(args.scenarios),
            "dense_every": args.dense_every,
            "reobserve_steps": args.reobserve_steps,
            "max_wait_steps": args.max_wait_steps,
            "note": (
                "Active arm is observation-only (re-observe/wait); no physical "
                "push/tap, so both arms share the same physics trajectory and GT. "
                "Uncertainty signal = force/geometry agreement (C2 fusion absent)."
            ),
        },
        "groups": groups,
        "runs": runs,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "contact_active.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "CONTACT_ACTIVE_REPORT.md").write_text(_render(payload), encoding="utf-8")
    return payload


def _fmt(row: dict, key: str, digits: int = 3) -> str:
    value = row.get(key)
    if not value or value.get("mean") is None:
        return "—"
    return f"{value['mean']:.{digits}f} ± {value['std']:.{digits}f}"


def _fmt_ms(row: dict, key: str) -> str:
    value = row.get(key)
    if not value or value.get("mean") is None:
        return "—"
    return f"{1000.0 * value['mean']:.1f} ± {1000.0 * value['std']:.1f}"


def _render(p: dict) -> str:
    lines = [
        "# PSS ContactEvent C4：主动感知（passive vs active）",
        "",
        f"- conditions: {list(p['config']['conditions'])}",
        f"- scenarios: {p['config']['scenarios']} · seeds {p['config']['seeds']}",
        f"- dense_every {p['config']['dense_every']} · "
        f"reobserve_steps {p['config']['reobserve_steps']} · "
        f"max_wait_steps {p['config']['max_wait_steps']}",
        "",
        "## 1. 事件场景总表（drop+collide；每格为跨 run mean ± std）",
        "",
        "| condition | arm | precision | recall | F1 | micro F1 | onset latency (ms) | extra frames | extra steps | probes |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, arms in p["groups"].items():
        for arm, group in arms.items():
            row = group["overall_events"]
            lines.append(
                f"| `{condition}` | `{arm}` | {_fmt(row, 'precision')} | "
                f"{_fmt(row, 'recall')} | {_fmt(row, 'f1')} | "
                f"{row['micro']['f1']:.3f} | {_fmt_ms(row, 'onset_latency')} | "
                f"{_fmt(row, 'extra_frames', 1)} | "
                f"{_fmt(row, 'extra_steps', 1)} | {_fmt(row, 'probes', 1)} |"
            )
    lines += [
        "",
        "## 2. 分场景（micro F1 = tp/fp/fn 汇总）",
        "",
        "| condition | arm | kind | P | R | F1 | micro F1 | onset latency (ms) | extra frames | extra steps | probes |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, arms in p["groups"].items():
        for arm, group in arms.items():
            for kind, row in group["by_kind"].items():
                lines.append(
                    f"| `{condition}` | `{arm}` | `{kind}` | "
                    f"{_fmt(row, 'precision')} | {_fmt(row, 'recall')} | "
                    f"{_fmt(row, 'f1')} | {row['micro']['f1']:.3f} | "
                    f"{_fmt_ms(row, 'onset_latency')} | "
                    f"{_fmt(row, 'extra_frames', 1)} | "
                    f"{_fmt(row, 'extra_steps', 1)} | "
                    f"{_fmt(row, 'probes', 1)} |"
                )
    lines += [
        "",
        "## 3. 近失对照（near_miss，GT 为空）",
        "",
        "| condition | arm | false alarms (fp) | F1 |",
        "|---|---|---:|---:|",
    ]
    for condition, arms in p["groups"].items():
        for arm, group in arms.items():
            row = group["by_kind"].get("near_miss")
            if row is None:
                continue
            lines.append(
                f"| `{condition}` | `{arm}` | {row['fp']} | {_fmt(row, 'f1')} |"
            )
    lines += [
        "",
        "## 4. 主动动作明细",
        "",
        "| condition | seed | scenario | action | step | t (s) | reason | pair | extra frames | extra steps |",
        "|---|---:|---|---|---:|---:|---|---|---:|---:|",
    ]
    for run in p["runs"]:
        for probe in run["probes"]:
            pair = "/".join(probe["pair"]) if probe["pair"] else "—"
            lines.append(
                f"| `{run['condition']}` | {run['seed']} | `{run['scenario']}` | "
                f"`{probe['kind']}` | {probe['step']} | {probe['t']:.3f} | "
                f"{probe['reason']} | {pair} | {probe['extra_frames']} | "
                f"{probe['extra_steps']} |"
            )
    lines += [
        "",
        "## 5. 口径与边界",
        "",
        "- 两臂共享同一条物理轨迹；主动臂只改变观测（re-observe/wait），不下发物理 push/tap。",
        "- GT 来自完整轨迹的 MuJoCo contact force（threshold 1 N，min_duration 20 ms，max_gap 50 ms）。",
        "- 事件估计：force 通道（score threshold 0.5 对应 1 N），事件匹配 IoU ≥0.3 或 onset 差 ≤50 ms。",
        "- C2 fusion 尚未实现，不确定性信号是 force/geometry 简单合议，不是校准后验。",
        "- MuJoCo 图元是 proxy；force/tactile 近 oracle；不声称真实机器人或 sim2real。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--conditions", nargs="+", default=["fine", "coarse", "short"],
        choices=sorted(CONDITIONS),
    )
    ap.add_argument(
        "--scenarios", nargs="+", default=["drop", "collide", "near_miss"],
    )
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--dense-every", type=int, default=2)
    ap.add_argument("--reobserve-steps", type=int, default=100)
    ap.add_argument("--max-wait-steps", type=int, default=150)
    ap.add_argument("--out", default="eval/results/contact_active")
    args = ap.parse_args()
    payload = run(args)
    for condition, arms in payload["groups"].items():
        for arm, group in arms.items():
            row = group["overall"]
            print(
                f"{condition:<7} {arm:<7} F1 {row['f1']['mean']:.3f} "
                f"P {row['precision']['mean']:.3f} R {row['recall']['mean']:.3f} "
                f"microF1 {row['micro']['f1']:.3f} "
                f"lat {row['onset_latency']['mean']} "
                f"extra {row['extra_frames']['mean']:.1f} frames "
                f"{row['extra_steps']['mean']:.1f} steps "
                f"probes {row['probes']['mean']:.1f}"
            )


if __name__ == "__main__":
    main()
