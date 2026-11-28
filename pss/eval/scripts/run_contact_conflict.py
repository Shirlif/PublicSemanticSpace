"""C3 conflict benchmark: fusion vs single channels, loose and strict matching.

Scenarios (see ``pss.events.scenarios_conflict``):
``occlusion``, ``conflict_force_only``, ``conflict_geometry_only``,
``near_miss_hard``, ``near_miss_clear``.

Arms: ``force`` (clean oracle-like control), ``force_noisy`` (C2 degraded
sensor), ``geometry`` (the conflict-prone channel), ``fusion_noisy`` (C2
calibrated fusion over noisy force + tactile + geometry).

Two matching rules are reported side by side:

- **loose** (preregistered event rule): same pair and temporal IoU >= 0.3 *or*
  onset difference <= 50 ms.  This rule rewards over-long intervals.
- **strict** (post-hoc diagnostic): same pair and temporal IoU >= 0.5 with the
  onset bypass disabled.

``clean force`` and ``geometry`` near 1.0 under the loose rule are not
evidence of fusion success: force shares the GT signal, and the loose rule
accepts intervals with large offset error.  Read the strict table and the
conflict/false-alarm columns together.

Example
-------
python -u -m pss.eval.scripts.run_contact_conflict \
  --scenarios occlusion conflict_force_only conflict_geometry_only \
              near_miss_hard near_miss_clear \
  --seeds 0 1 2 --arms force force_noisy geometry fusion_noisy \
  --out eval/results/contact_event/c3_conflict
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pss._console import ensure_utf8
from pss.events import (
    ARMS,
    ContactConfig,
    build_scenario,
    evaluate_events,
    run_channel,
)
from pss.events.segmenter import Interval, temporal_iou

DEFAULT_ARMS = ("force", "force_noisy", "geometry", "fusion_noisy")
DEFAULT_SCENARIOS = (
    "occlusion",
    "conflict_force_only",
    "conflict_geometry_only",
    "near_miss_hard",
    "near_miss_clear",
)
NO_CONTACT_SCENARIOS = ("near_miss_hard", "near_miss_clear")
# Scenarios whose GT defines a force/geometry disagreement.  `occlusion` is a
# visibility test, not a channel conflict, so it is excluded from resolution.
CONFLICT_SCENARIOS = (
    "conflict_force_only",
    "conflict_geometry_only",
    "near_miss_hard",
    "near_miss_clear",
)

LOOSE = {"iou_threshold": 0.3, "onset_tolerance": 0.05}
STRICT = {"iou_threshold": 0.5, "onset_tolerance": -1.0}  # no onset bypass


def _run_arm(name: str, scenario, config: ContactConfig):
    if name in ARMS:
        return ARMS[name](scenario, config)
    return run_channel(scenario, config, channel=name)


def _latency_errors(predicted, ground_truth, *, iou_threshold, onset_tolerance):
    """Onset/offset errors for matches under the same rule as ``evaluate_events``."""
    by_pair_gt: dict[tuple[str, str], list] = {}
    by_pair_pred: dict[tuple[str, str], list] = {}
    for e in ground_truth:
        by_pair_gt.setdefault(e.pair, []).append(e)
    for e in predicted:
        by_pair_pred.setdefault(e.pair, []).append(e)

    onsets: list[float] = []
    offsets: list[float] = []
    for pair in set(by_pair_gt) | set(by_pair_pred):
        gts = list(by_pair_gt.get(pair, []))
        preds = list(by_pair_pred.get(pair, []))
        used = [False] * len(preds)
        for gt in gts:
            gi = Interval(gt.onset, gt.offset if gt.offset is not None else gt.onset)
            best = None
            best_iou = 0.0
            for i, pred in enumerate(preds):
                if used[i]:
                    continue
                pi = Interval(
                    pred.onset, pred.offset if pred.offset is not None else pred.onset
                )
                iou = temporal_iou(gi, pi)
                onset_ok = abs(pred.onset - gt.onset) <= onset_tolerance
                if (iou >= iou_threshold or onset_ok) and iou >= best_iou:
                    best, best_iou = i, iou
            if best is None:
                continue
            used[best] = True
            pred = preds[best]
            gt_offset = gt.offset if gt.offset is not None else gt.onset
            pred_offset = pred.offset if pred.offset is not None else pred.onset
            onsets.append(abs(float(pred.onset) - float(gt.onset)))
            offsets.append(abs(float(pred_offset) - float(gt_offset)))
    return {
        "onset_error": float(np.mean(onsets)) if onsets else None,
        "offset_error": float(np.mean(offsets)) if offsets else None,
        "n_matched": int(len(onsets)),
    }


def _micro(rows: list[dict]) -> dict:
    tp = int(sum(r["tp"] for r in rows))
    fp = int(sum(r["fp"] for r in rows))
    fn = int(sum(r["fn"] for r in rows))
    n_gt = int(sum(r["n_gt"] for r in rows))
    n_pred = int(sum(r["n_pred"] for r in rows))
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    if precision is not None and recall is not None:
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    elif fp > 0:
        f1 = 0.0
    else:
        f1 = None
    return {
        "precision": None if precision is None else float(precision),
        "recall": None if recall is None else float(recall),
        "f1": None if f1 is None else float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_gt": n_gt,
        "n_pred": n_pred,
    }


def _decide(row: dict) -> bool:
    """Contact-vs-no-contact decision: was the GT state chosen correctly?"""
    if row["n_gt"] > 0:
        return bool(row["tp"] > 0)
    return bool(row["n_pred"] == 0)


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def run(args) -> dict:
    seeds = list(args.seeds)
    arms = list(args.arms)
    scenario_names = list(args.scenarios)
    rows: list[dict] = []
    scenario_meta: list[dict] = []

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
            seed=seed,
        )
        for name in scenario_names:
            scenario = build_scenario(name, config)
            scenario_meta.append(
                {
                    "seed": seed,
                    "scenario": scenario.scenario_id,
                    "kind": scenario.metadata.get("kind"),
                    "frames": len(scenario.frames),
                    "gt_events": len(scenario.gt_events),
                    "n_pairs": len(scenario.frames[0].poses) if scenario.frames else 0,
                }
            )
            for arm in arms:
                predicted = _run_arm(arm, scenario, config)
                loose = evaluate_events(predicted, scenario.gt_events, **LOOSE)
                strict = evaluate_events(predicted, scenario.gt_events, **STRICT)
                row = {
                    "seed": seed,
                    "scenario": name,
                    "arm": arm,
                    "n_gt": int(loose["n_gt"]),
                    "n_pred": int(loose["n_pred"]),
                    "loose": loose,
                    "strict": strict,
                    "loose_latency": _latency_errors(
                        predicted, scenario.gt_events, **LOOSE
                    ),
                    "strict_latency": _latency_errors(
                        predicted, scenario.gt_events, **STRICT
                    ),
                    "resolved": _decide(loose),
                }
                rows.append(row)
        print(f"[seed {seed}] done", flush=True)

    payload = {
        "stage": "contact_event_c3_conflict",
        "config": {
            "width": args.width,
            "height": args.height,
            "max_steps": args.max_steps,
            "capture_every": args.capture_every,
            "render": bool(args.render),
            "threshold": args.threshold,
            "gt_threshold": args.gt_threshold,
            "min_duration": args.min_duration,
            "max_gap": args.max_gap,
            "seeds": seeds,
            "arms": arms,
            "scenarios": scenario_names,
            "loose_rule": LOOSE,
            "strict_rule": STRICT,
        },
        "scenarios": scenario_meta,
        "arms": {},
        "note": (
            "GT = MuJoCo contact force; clean force shares the GT signal. "
            "Loose = IoU>=0.3 or onset<=50 ms (preregistered event rule). "
            "Strict = pure IoU>=0.5 with no onset bypass (post-hoc diagnostic). "
            "near_miss_hard has surface gap < eps; near_miss_clear has gap > eps."
        ),
    }

    arm_out: dict[str, dict] = {}
    for arm in arms:
        arm_rows = [r for r in rows if r["arm"] == arm]
        by_scenario: dict[str, dict] = {}
        for name in scenario_names:
            sel = [r for r in arm_rows if r["scenario"] == name]
            by_scenario[name] = {
                "loose": _micro([r["loose"] for r in sel]),
                "strict": _micro([r["strict"] for r in sel]),
                "loose_onset_error": _mean_or_none(
                    [r["loose_latency"]["onset_error"] for r in sel]
                ),
                "loose_offset_error": _mean_or_none(
                    [r["loose_latency"]["offset_error"] for r in sel]
                ),
                "strict_onset_error": _mean_or_none(
                    [r["strict_latency"]["onset_error"] for r in sel]
                ),
                "strict_offset_error": _mean_or_none(
                    [r["strict_latency"]["offset_error"] for r in sel]
                ),
                "seeds": [
                    {
                        "seed": r["seed"],
                        "n_pred": r["n_pred"],
                        "loose_tp/fp/fn": [r["loose"]["tp"], r["loose"]["fp"], r["loose"]["fn"]],
                        "strict_tp/fp/fn": [
                            r["strict"]["tp"], r["strict"]["fp"], r["strict"]["fn"],
                        ],
                        "resolved": r["resolved"],
                    }
                    for r in sel
                ],
            }
        overall = {
            "loose": _micro([r["loose"] for r in arm_rows]),
            "strict": _micro([r["strict"] for r in arm_rows]),
            "loose_onset_error": _mean_or_none(
                [r["loose_latency"]["onset_error"] for r in arm_rows]
            ),
            "loose_offset_error": _mean_or_none(
                [r["loose_latency"]["offset_error"] for r in arm_rows]
            ),
            "strict_onset_error": _mean_or_none(
                [r["strict_latency"]["onset_error"] for r in arm_rows]
            ),
            "strict_offset_error": _mean_or_none(
                [r["strict_latency"]["offset_error"] for r in arm_rows]
            ),
        }
        conflict_sel = [r for r in arm_rows if r["scenario"] in CONFLICT_SCENARIOS]
        false_alarm = {}
        for name in NO_CONTACT_SCENARIOS:
            sel = [r for r in arm_rows if r["scenario"] == name]
            runs_with_fp = sum(1 for r in sel if r["loose"]["fp"] > 0)
            false_alarm[name] = {
                "fp_events": int(sum(r["loose"]["fp"] for r in sel)),
                "runs": len(sel),
                "runs_with_fp": int(runs_with_fp),
                "run_fa_rate": float(runs_with_fp / len(sel)) if sel else None,
            }
        arm_out[arm] = {
            "overall": overall,
            "by_scenario": by_scenario,
            "conflict_resolution": {
                "resolved": int(sum(1 for r in conflict_sel if r["resolved"])),
                "total": int(len(conflict_sel)),
                "by_scenario": {
                    name: {
                        "resolved": int(
                            sum(
                                1
                                for r in conflict_sel
                                if r["scenario"] == name and r["resolved"]
                            )
                        ),
                        "total": int(
                            sum(1 for r in conflict_sel if r["scenario"] == name)
                        ),
                    }
                    for name in CONFLICT_SCENARIOS
                },
            },
            "false_alarm": false_alarm,
        }
    payload["arms"] = arm_out

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "contact_event.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "CONTACT_CONFLICT_REPORT.md").write_text(_render(payload), encoding="utf-8")
    return payload


def _mean_or_none(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def _render(p: dict) -> str:
    arms = p["config"]["arms"]
    scenarios = p["config"]["scenarios"]
    lines = [
        "# PSS ContactEvent C3：冲突基准（loose vs strict）",
        "",
        f"- scenarios: {scenarios}",
        f"- seeds: {p['config']['seeds']} · arms: {arms}",
        f"- loose: IoU≥{p['config']['loose_rule']['iou_threshold']} 或 onset≤"
        f"{p['config']['loose_rule']['onset_tolerance']} s",
        f"- strict: IoU≥{p['config']['strict_rule']['iou_threshold']}，无 onset 容差",
        "",
        "## 总体（micro over scenario×seed）",
        "",
        "| arm | loose P | loose R | loose F1 | strict P | strict R | strict F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in arms:
        o = p["arms"][arm]["overall"]
        lines.append(
            f"| `{arm}` | {_fmt(o['loose']['precision'])} | {_fmt(o['loose']['recall'])} | "
            f"{_fmt(o['loose']['f1'])} | {_fmt(o['strict']['precision'])} | "
            f"{_fmt(o['strict']['recall'])} | {_fmt(o['strict']['f1'])} |"
        )
    lines += [
        "",
        "## 分场景 F1",
        "",
        "| scenario | arm | loose F1 | loose tp/fp/fn | strict F1 | strict tp/fp/fn |",
        "|---|---|---:|---|---:|---|",
    ]
    for name in scenarios:
        for arm in arms:
            row = p["arms"][arm]["by_scenario"][name]
            lo, st = row["loose"], row["strict"]
            lines.append(
                f"| {name} | `{arm}` | {_fmt(lo['f1'])} | "
                f"{lo['tp']}/{lo['fp']}/{lo['fn']} | {_fmt(st['f1'])} | "
                f"{st['tp']}/{st['fp']}/{st['fn']} |"
            )
    lines += [
        "",
        "## 冲突处理（conflict_force_only/geometry_only/near_miss_hard/clear）",
        "",
        "| arm | resolved | force_only | geometry_only | near_miss_hard | near_miss_clear |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in arms:
        cr = p["arms"][arm]["conflict_resolution"]
        b = cr["by_scenario"]
        lines.append(
            f"| `{arm}` | {cr['resolved']}/{cr['total']} | "
            f"{b['conflict_force_only']['resolved']}/{b['conflict_force_only']['total']} | "
            f"{b['conflict_geometry_only']['resolved']}/{b['conflict_geometry_only']['total']} | "
            f"{b['near_miss_hard']['resolved']}/{b['near_miss_hard']['total']} | "
            f"{b['near_miss_clear']['resolved']}/{b['near_miss_clear']['total']} |"
        )
    lines += [
        "",
        "## False alarm（no-contact 场景）",
        "",
        "| arm | near_miss_clear FP/runs | near_miss_clear FA | near_miss_hard FP/runs | near_miss_hard FA |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in arms:
        f = p["arms"][arm]["false_alarm"]
        c, h = f["near_miss_clear"], f["near_miss_hard"]
        lines.append(
            f"| `{arm}` | {c['fp_events']}/{c['runs']} | {_fmt(c['run_fa_rate'])} | "
            f"{h['fp_events']}/{h['runs']} | {_fmt(h['run_fa_rate'])} |"
        )
    lines += [
        "",
        "## Onset / offset 误差（匹配事件均值）",
        "",
        "| arm | loose onset | loose offset | strict onset | strict offset |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in arms:
        o = p["arms"][arm]["overall"]
        lines.append(
            f"| `{arm}` | {_fmt(o['loose_onset_error'], 4)} | "
            f"{_fmt(o['loose_offset_error'], 4)} | "
            f"{_fmt(o['strict_onset_error'], 4)} | {_fmt(o['strict_offset_error'], 4)} |"
        )
    lines += [
        "",
        "## 口径",
        "",
        "- GT 来自 MuJoCo contact force；clean `force` 与 GT 同源，其 1.0 不是融合证据。",
        "- loose 规则允许 onset 容差，会把过长区间记为 TP；strict 为事后诊断，不是预注册 bar。",
        "- `near_miss_hard` 表面间隙 < eps，`near_miss_clear` 间隙 > eps。",
        "- 所有结果为 MuJoCo proxy；noisy force 是模拟退化，不是真实传感器。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenarios", nargs="+", default=list(DEFAULT_SCENARIOS))
    ap.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS))
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
    ap.add_argument("--out", default="eval/results/contact_event/c3_conflict")
    args = ap.parse_args()
    payload = run(args)
    for arm in args.arms:
        o = payload["arms"][arm]["overall"]
        cr = payload["arms"][arm]["conflict_resolution"]
        print(
            f"{arm:<12} loose F1 {_fmt(o['loose']['f1'])} "
            f"(P {_fmt(o['loose']['precision'])} R {_fmt(o['loose']['recall'])})  "
            f"strict F1 {_fmt(o['strict']['f1'])} "
            f"(P {_fmt(o['strict']['precision'])} R {_fmt(o['strict']['recall'])})  "
            f"resolved {cr['resolved']}/{cr['total']}"
        )


if __name__ == "__main__":
    main()
