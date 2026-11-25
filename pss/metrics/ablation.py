"""Ablation harness: same bundles, different PSS runtime variants."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from pss.detect.replay import replay_bundle
from pss.detect.session_replay import replay_session
from pss.metrics.memory import compute_memory_metrics
from pss.metrics.scorecard import discover_bundles, score_bundle
from pss.runtime.variants import VARIANTS


def _aggregate(episodes: list[dict[str, Any]], names: list[str]) -> dict[str, Any]:
    out = {}
    for name in names:
        values = [float(ep[name]) for ep in episodes if name in ep]
        if not values:
            continue
        out[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return out


def ablate_entity_suite(
    bundles_root: str | Path,
    suite: str,
    *,
    variants: tuple[str, ...] = VARIANTS,
    device: str | None = None,
    out: str | Path = "scorecards",
) -> dict[str, Any]:
    bundles = discover_bundles(bundles_root, suite, device)
    if not bundles:
        raise FileNotFoundError(f"no bundles for {suite}")
    names = ["idf1", "id_retention", "id_switches", "duplicate_rate", "reappearance_rate", "slot_purity"]
    results = {}
    for variant in variants:
        episodes = []
        for bundle in bundles:
            replay_bundle(bundle, write=True, variant=variant)
            episodes.append(score_bundle(bundle))
        results[variant] = _aggregate(episodes, names)
    # leave the bundle artifacts in the full state
    for bundle in bundles:
        replay_bundle(bundle, write=True, variant="full")

    report = {
        "suite": suite,
        "device_class": device or "dev",
        "n_episodes": len(bundles),
        "variants": list(variants),
        "metrics": results,
        "note": "preliminary dev ablation; oracle/noisy/real tier as stored in each bundle",
    }
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{suite}_ablation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        f"# {suite} ablation (dev, preliminary)",
        "",
        f"- episodes: {len(bundles)}  variants: {len(variants)}",
        "",
        "| variant | IDF1 | retention | switches | duplicate | reappearance |",
        "|---|---|---|---|---|---|",
    ]
    for variant in variants:
        m = results.get(variant, {})
        def value(name):
            return m.get(name, {}).get("mean", float("nan"))
        lines.append(
            f"| {variant} | {value('idf1'):.4f} | {value('id_retention'):.4f} | "
            f"{value('id_switches'):.3f} | {value('duplicate_rate'):.4f} | "
            f"{value('reappearance_rate'):.3f} |"
        )
    (out_dir / f"{suite}_ablation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def ablate_memory_session(
    session_dir: str | Path,
    *,
    variants: tuple[str, ...] = VARIANTS,
    out: str | Path = "scorecards",
) -> dict[str, Any]:
    root = Path(session_dir)
    bundles = sorted(p for p in root.glob("ep*") if p.is_dir())
    if not bundles:
        raise FileNotFoundError(f"no episode bundles under {root}")
    results = {}
    for variant in variants:
        trace, report = replay_session(bundles, variant=variant)
        metrics = compute_memory_metrics(
            trace, rollback_exactness=report.get("rollback_exactness", 0.0)
        )
        metrics["rollback_changed_state"] = report.get("rollback_changed_state")
        results[variant] = metrics
    # leave full-state traces
    replay_session(bundles, variant="full")
    payload = {
        "session": str(root),
        "variants": list(variants),
        "metrics": results,
        "note": "preliminary dev S6 ablation; oracle detections",
    }
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "S6_ablation.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# S6 memory ablation (dev, preliminary)",
        "",
        "| variant | edited | old_forgetting | forgetting_rate | retention | rollback |",
        "|---|---|---|---|---|---|",
    ]
    for variant in variants:
        m = results.get(variant, {})
        lines.append(
            f"| {variant} | {m.get('edited_accuracy', 0):.4f} | "
            f"{m.get('old_forgetting', 0):.4f} | {m.get('forgetting_rate', 1):.4f} | "
            f"{m.get('id_retention', 0):.4f} | {m.get('rollback_exactness', 0):.4f} |"
        )
    (out_dir / "S6_ablation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload
