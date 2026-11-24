"""Phase 10 E3 rerun on the fully-occluded occlusion_v3 benchmark.

Compares persistent tracking against the one-frame text-state proxy and
ablations. The K3 criterion is ``main - text_state_proxy >= 0.10``.

Example:
    python -m pss.eval.scripts.run_e3_occlusion \
        --data data/occlusion_v3 --split test \
        --experts checkpoints/category_pretrain_experts \
        --projections checkpoints/phase45_projection \
        --prototypes checkpoints/phase45_prototypes \
        --out eval/results/occlusion_v3
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pss._console import ensure_utf8
from pss.encoders.data import load_index
from pss.eval.baselines import evaluate_tracker, tracker_variants
from pss.memory import Tracker, TrackerConfig
from pss.memory.perception import PoseNoise, load_perception_models
from pss.sim.dataset import load_sample

METRICS = ("id_retention", "id_switches", "reappearance_success", "wake_latency_frames")


def _variants() -> dict:
    variants = dict(tracker_variants())
    # Isolate the effect of the prototype category term.
    variants["main_no_category"] = lambda: Tracker(
        TrackerConfig(w_category=0.0), None
    )
    return variants


def run_e3(
    data_dir: str | Path,
    expert_dir: str | Path,
    projection_dir: str | Path,
    out_dir: str | Path,
    prototype_dir: str | Path | None = None,
    split: str = "test",
    max_samples: int = 100,
    seed: int = 0,
    weights_mode: str = "reliability",
) -> dict:
    data_dir = Path(data_dir)
    index = load_index(data_dir)
    entries = [
        e
        for e in index["entries"]
        if e["kind"] == "occlusion" and e.get("split") == split
    ][:max_samples]
    if not entries:
        raise ValueError(f"no occlusion entries for split={split!r} in {data_dir}")

    samples = [load_sample(data_dir / e["path"]) for e in entries]
    models = load_perception_models(
        expert_dir,
        projection_dir,
        prototype_dir,
        device="cpu",
        weights_mode=weights_mode,
    )

    protocol_ok = all(bool(e.get("fully_occluded")) for e in entries)
    data_category = (index.get("config") or {}).get("category")
    data_near_identical = bool((index.get("config") or {}).get("near_identical"))
    forced = {
        e["id"]: e.get("forced_mask_frames", []) for e in entries
    }

    results: dict[str, dict] = {}
    per_sample: dict[str, list[dict]] = {}
    for name, factory in _variants().items():
        rows = []
        for i, sample in enumerate(samples):
            rows.append(
                evaluate_tracker(
                    sample,
                    models,
                    PoseNoise(seed=seed + i),
                    factory(),
                )
            )
        per_sample[name] = [
            {"sample": e["id"], **{k: row[k] for k in METRICS}}
            for e, row in zip(entries, rows)
        ]
        results[name] = {
            key: float(np.mean([row[key] for row in rows])) for key in METRICS
        }
        results[name]["samples"] = len(rows)

    delta = results["main"]["id_retention"] - results["text_state_proxy"]["id_retention"]
    metrics = {
        "data_dir": str(data_dir),
        "split": split,
        "samples": [e["id"] for e in entries],
        "experts": str(expert_dir),
        "projections": str(projection_dir),
        "prototypes": str(prototype_dir) if prototype_dir is not None else None,
        "recognition_weights_mode": weights_mode,
        "same_category": data_category,
        "near_identical": data_near_identical,
        "protocol": {
            "fully_occluded_all": protocol_ok,
            "forced_mask_frames": forced,
            "occlusion_windows": {
                e["id"]: [e.get("occlusion_start"), e.get("occlusion_end")]
                for e in entries
            },
        },
        "variants": results,
        "per_sample": per_sample,
        "k3": {
            "metric": "ID retention",
            "main_minus_text_state_proxy": delta,
            "threshold": 0.10,
            "pass": bool(protocol_ok and delta >= 0.10),
        },
        "note": (
            "occlusion_v3 hides every object for 14 frames and moves objects "
            "behind the panel. pose-only coasting is not enough to preserve "
            "the one-frame text baseline."
        ),
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_report(out / "REPORT.md", metrics)
    return metrics


def _write_report(path: Path, metrics: dict) -> None:
    rows = metrics["variants"]
    lines = [
        "# Phase 10 E3: occlusion_v3 (fully hidden objects)",
        "",
        f"- data: `{metrics['data_dir']}`",
        f"- split: {metrics['split']}",
        f"- samples: {', '.join(metrics['samples'])}",
        f"- prototypes: `{metrics['prototypes'] or 'none (GT category)'}`",
        f"- same-category: `{metrics.get('same_category') or 'no (mixed)'}`",
        f"- near-identical: **{metrics.get('near_identical', False)}**",
        f"- protocol fully occluded: **{metrics['protocol']['fully_occluded_all']}**",
        "",
        "| variant | ID retention | switches | reappearance | wake latency |",
        "|---|---:|---:|---:|---:|",
    ]
    order = [
        "main",
        "main_no_category",
        "pose_only",
        "appearance_only",
        "text_state_proxy",
        "no_scene_graph",
    ]
    for name in order:
        r = rows.get(name)
        if r is None:
            continue
        lines.append(
            f"| {name} | {r['id_retention']:.3f} | {r['id_switches']:.2f} | "
            f"{r['reappearance_success']:.3f} | {r['wake_latency_frames']:.1f} |"
        )
    k3 = metrics["k3"]
    lines += [
        "",
        f"**K3**: main - text_state_proxy = "
        f"{k3['main_minus_text_state_proxy']:.3f} "
        f"(threshold {k3['threshold']:.2f}) -> "
        f"**{'PASS' if k3['pass'] else 'FAIL'}**",
        "",
        "## Protocol",
        "",
        "- Every object is fully hidden for 14 consecutive frames.",
        "- Objects move behind the panel; last-known pose is stale on reappearance.",
        "- The one-frame text-state proxy has no memory to re-identify them.",
        "",
        "## Caveat",
        "",
    ]
    if metrics.get("near_identical"):
        lines += [
            "- Near-identical variant: all instances share category, size,",
            "  color, and material. The only difference is instance id and",
            "  position, so appearance has no designed identity signal.",
            "",
        ]
    elif metrics.get("same_category"):
        lines += [
            f"- Same-category variant (`{metrics['same_category']}`): every",
            "  object has the same label, so the category term is uninformative.",
            "  Only appearance can preserve identity across the occlusion.",
            "",
        ]
    else:
        lines += [
            "- This benchmark tests persistence, not instance-level appearance.",
            "  The scenes use different categories, so the category term can help",
            "  when it is available. Use the same-category variant to isolate",
            "  appearance-only re-ID.",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/occlusion_v3")
    ap.add_argument("--experts", default="checkpoints/category_pretrain_experts")
    ap.add_argument("--projections", default="checkpoints/phase45_projection")
    ap.add_argument("--prototypes", default=None)
    ap.add_argument("--out", default="eval/results/occlusion_v3")
    ap.add_argument("--split", choices=("train", "val", "test"), default="test")
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--recognition-weights", choices=("reliability", "tuned"), default="reliability")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    metrics = run_e3(
        args.data,
        args.experts,
        args.projections,
        args.out,
        prototype_dir=args.prototypes,
        split=args.split,
        max_samples=args.samples,
        seed=args.seed,
        weights_mode=args.recognition_weights,
    )
    for name, row in metrics["variants"].items():
        print(
            f"{name:22s} retention={row['id_retention']:.3f} "
            f"switches={row['id_switches']:.2f} "
            f"reappear={row['reappearance_success']:.3f}"
        )
    k3 = metrics["k3"]
    print(
        f"K3: main - text_state_proxy = {k3['main_minus_text_state_proxy']:.3f} "
        f"-> {'PASS' if k3['pass'] else 'FAIL'}"
    )
    print(f"output: {args.out}")


if __name__ == "__main__":
    main()
