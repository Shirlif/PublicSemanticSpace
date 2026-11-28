"""Contact-event benchmark: scenarios -> channel evidence -> events -> metrics.

Example
-------
python -u -m pss.eval.scripts.run_contact_event_bench \
  --scenarios drop collide near_miss --seeds 0 1 2 \
  --out eval/results/contact_event
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pss._console import ensure_utf8
from pss.events import ARMS, ContactConfig, build_scenario, run_bench


def _aggregate(seed_rows: list[dict]) -> dict:
    if not seed_rows:
        return {}
    arms = set()
    for row in seed_rows:
        arms.update(row)
    out = {}
    for arm in sorted(arms):
        metrics = ("precision", "recall", "f1", "onset_latency", "offset_latency", "intensity_mae")
        agg = {}
        for key in metrics:
            values = [
                row[arm][key]["mean"]
                for row in seed_rows
                if arm in row and row[arm].get(key) and row[arm][key]["mean"] is not None
            ]
            agg[key] = {
                "mean": float(np.mean(values)) if values else None,
                "std": float(np.std(values)) if values else None,
                "n": len(values),
            }
        agg["tp"] = int(sum(row[arm]["tp"] for row in seed_rows if arm in row))
        agg["fp"] = int(sum(row[arm]["fp"] for row in seed_rows if arm in row))
        agg["fn"] = int(sum(row[arm]["fn"] for row in seed_rows if arm in row))
        agg["n_gt"] = int(sum(row[arm]["n_gt"] for row in seed_rows if arm in row))
        out[arm] = agg
    return out


def _fmt(row: dict, key: str) -> str:
    value = row.get(key)
    if not value or value.get("mean") is None:
        return "—"
    return f"{value['mean']:.3f} ± {value['std']:.3f}"


def run(args) -> dict:
    seeds = list(args.seeds)
    arms = list(args.arms) if args.arms else ["force", "tactile", "geometry", "vision", "audio"] + sorted(ARMS)
    seed_rows = []
    scenario_meta = []
    for seed in seeds:
        config = ContactConfig(
            width=args.width,
            height=args.height,
            max_steps=args.max_steps,
            capture_every=args.capture_every,
            render=args.render,
            threshold=args.threshold,
            gt_threshold=args.gt_threshold,
            hysteresis=args.hysteresis,
            min_duration=args.min_duration,
            max_gap=args.max_gap,
            strict=args.strict,
            iou_threshold=args.iou_threshold,
            onset_tolerance=args.onset_tolerance,
            offset_tolerance=args.offset_tolerance,
            seed=seed,
        )
        scenarios = [build_scenario(name, config) for name in args.scenarios]
        for s in scenarios:
            scenario_meta.append(
                {
                    "seed": seed,
                    "scenario": s.scenario_id,
                    "kind": s.metadata.get("kind"),
                    "frames": len(s.frames),
                    "gt_events": len(s.gt_events),
                    "gt_pairs": sorted({e.pair for e in s.gt_events}),
                }
            )
        seed_rows.append(run_bench(arms, scenarios, config))
        print(f"[seed {seed}] done; arms {arms}", flush=True)

    payload = {
        "stage": "contact_event",
        "config": {
            "width": args.width,
            "height": args.height,
            "max_steps": args.max_steps,
            "capture_every": args.capture_every,
            "render": args.render,
            "threshold": args.threshold,
            "gt_threshold": args.gt_threshold,
            "hysteresis": args.hysteresis,
            "min_duration": args.min_duration,
            "max_gap": args.max_gap,
            "strict": args.strict,
            "iou_threshold": args.iou_threshold,
            "onset_tolerance": args.onset_tolerance,
            "offset_tolerance": args.offset_tolerance,
            "scenarios": list(args.scenarios),
            "seeds": seeds,
        },
        "scenarios": scenario_meta,
        "arms": _aggregate(seed_rows),
        "note": (
            "GT = MuJoCo contact forces above gt_threshold, merged with "
            "max_gap/min_duration. Channels are evidence, not calibrated "
            "probabilities. vision/audio are placeholders/proxies."
        ),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "contact_event.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "CONTACT_EVENT_REPORT.md").write_text(
        _render(payload), encoding="utf-8"
    )
    return payload


def _render(p: dict) -> str:
    lines = [
        "# PSS ContactEvent 基准（C0/C1）",
        "",
        f"- scenarios: {p['config']['scenarios']} · seeds {p['config']['seeds']}",
        f"- frames: {p['config']['max_steps']} steps × {p['config']['capture_every']} · "
        f"render {p['config']['render']}",
        f"- threshold {p['config']['threshold']} · GT force {p['config']['gt_threshold']} · "
        f"min duration {p['config']['min_duration']} · max gap {p['config']['max_gap']}",
        f"- strict {p['config']['strict']} · IoU {p['config']['iou_threshold']} · "
        f"onset tol {p['config']['onset_tolerance']} · offset tol {p['config']['offset_tolerance']}",
        "",
        "## 0. 场景",
        "",
        "| seed | scenario | kind | frames | GT events | GT pairs |",
        "|---:|---|---|---:|---:|---|",
    ]
    for s in p["scenarios"]:
        lines.append(
            f"| {s['seed']} | {s['scenario']} | {s['kind']} | {s['frames']} | "
            f"{s['gt_events']} | {s['gt_pairs']} |"
        )
    lines += [
        "",
        "## 1. 主表（按 seed 聚合）",
        "",
        "| arm | precision | recall | F1 | onset latency | offset latency | intensity MAE | tp/fp/fn |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for arm, row in p["arms"].items():
        lines.append(
            f"| `{arm}` | {_fmt(row, 'precision')} | {_fmt(row, 'recall')} | "
            f"{_fmt(row, 'f1')} | {_fmt(row, 'onset_latency')} | "
            f"{_fmt(row, 'offset_latency')} | {_fmt(row, 'intensity_mae')} | "
            f"{row['tp']}/{row['fp']}/{row['fn']} |"
        )
    lines += [
        "",
        "## 2. 口径",
        "",
        "- GT 来自 MuJoCo contact force，不是人工标签；table/floor 接触被过滤。",
        f"- 事件匹配：strict={p['config']['strict']}；同 pair + IoU ≥"
        f"{p['config']['iou_threshold']} + onset ≤{p['config']['onset_tolerance']} s"
        f" + offset ≤{p['config']['offset_tolerance']} s（loose 口径仅备查）。",
        "- 通道输出是**证据**，不是校准概率；fusion 由 C2 做。",
        "- `vision` 目前是占位；`audio` 是由 force 合成的代理，不是真实音频。",
        "",
        "## 3. 边界",
        "",
        "- MuJoCo 图元是 proxy；force/tactile 近 oracle；audio 是合成。",
        "- 场景只有 drop/collide/near-miss；遮挡/冲突由 C3 补。",
        "- 不声称真实机器人/真实传感器性能。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenarios", nargs="+", default=["drop", "collide", "near_miss"])
    ap.add_argument("--arms", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--height", type=int, default=72)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--capture-every", type=int, default=2)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--gt-threshold", type=float, default=1.0)
    ap.add_argument("--hysteresis", type=float, default=0.0)
    ap.add_argument("--min-duration", type=float, default=0.02)
    ap.add_argument("--max-gap", type=float, default=0.05)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--iou-threshold", type=float, default=0.3)
    ap.add_argument("--onset-tolerance", type=float, default=0.05)
    ap.add_argument("--offset-tolerance", type=float, default=0.1)
    ap.add_argument("--out", default="eval/results/contact_event")
    args = ap.parse_args()
    payload = run(args)
    for arm, row in payload["arms"].items():
        print(
            f"{arm:<10} F1 {row['f1']['mean']} "
            f"P {row['precision']['mean']} R {row['recall']['mean']} "
            f"tp/fp/fn {row['tp']}/{row['fp']}/{row['fn']}"
        )


if __name__ == "__main__":
    main()
