"""Compare PSS full vs B1 text-state and B2 SORT-lite on S1 bundles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pss.baselines import sort_lite_predictions, text_state_predictions
from pss.detect.replay import replay_bundle
from pss.metrics.scorecard import (
    discover_bundles,
    score_bundle,
    score_pred_states,
)

NAMES = ["idf1", "id_retention", "id_switches", "duplicate_rate", "reappearance_rate"]


def _steps(bundle: Path) -> list[int]:
    obs = pd.read_parquet(bundle / "obs_frames.parquet")
    return sorted(int(s) for s in obs["step"].unique())


def _detections(bundle: Path) -> pd.DataFrame:
    return pd.read_parquet(bundle / "obs_detections.parquet")


def _mean(rows: list[dict], name: str) -> float:
    return float(np.mean([float(row[name]) for row in rows])) if rows else float("nan")


def _bootstrap_delta(full: list[float], baseline: list[float], rounds: int = 2000) -> tuple[float, float, float]:
    deltas = np.asarray(full) - np.asarray(baseline)
    if len(deltas) == 0:
        return (0.0, 0.0, 0.0)
    rng = np.random.default_rng(0)
    means = [float(rng.choice(deltas, size=len(deltas), replace=True).mean()) for _ in range(rounds)]
    return float(deltas.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="s1_occlusion_20")
    parser.add_argument("--bundles-root", default="bundles")
    parser.add_argument("--device", default="dev")
    parser.add_argument("--out", default="scorecards")
    args = parser.parse_args()

    bundles = discover_bundles(args.bundles_root, args.suite, args.device)
    if not bundles:
        raise FileNotFoundError(f"no bundles for {args.suite}")
    results: dict[str, list[dict]] = {"full": [], "no_entity": [], "B1_text": [], "B2_sort": []}
    for bundle in bundles:
        replay_bundle(bundle, write=True, variant="full")
        results["full"].append(score_bundle(bundle))
        replay_bundle(bundle, write=True, variant="no_entity")
        results["no_entity"].append(score_bundle(bundle))

        det = _detections(bundle)
        steps = _steps(bundle)
        has_emb = "emb" in det.columns and det["emb"].notna().any()
        b1 = text_state_predictions(det, steps=steps)
        results["B1_text"].append(score_pred_states(bundle, b1))
        b2 = sort_lite_predictions(
            det, steps=steps, appearance_weight=0.3 if has_emb else 0.0
        )
        results["B2_sort"].append(score_pred_states(bundle, b2))

    # leave the bundle on the PSS full trace
    for bundle in bundles:
        replay_bundle(bundle, write=True, variant="full")

    aggregate = {
        name: {metric: _mean(rows, metric) for metric in NAMES}
        for name, rows in results.items()
    }
    deltas = {}
    for baseline in ("no_entity", "B1_text", "B2_sort"):
        deltas[baseline] = {}
        for metric in ("idf1", "id_retention", "reappearance_rate"):
            full_values = [float(row[metric]) for row in results["full"]]
            base_values = [float(row[metric]) for row in results[baseline]]
            mean, low, high = _bootstrap_delta(full_values, base_values)
            deltas[baseline][metric] = {"mean": mean, "ci95": [low, high]}

    manifest = json.loads((bundles[0] / "manifest.json").read_text(encoding="utf-8"))
    report = {
        "suite": args.suite,
        "device_class": args.device,
        "detector_tier": manifest.get("detector_tier"),
        "n_episodes": len(bundles),
        "metrics": aggregate,
        "full_minus_baseline": deltas,
        "note": "B1 = memoryless text-state proxy; B2 = constant-velocity SORT-lite",
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.suite}_vs_baselines.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        f"# {args.suite} vs baselines",
        "",
        f"- episodes: {len(bundles)}  detector_tier: {report['detector_tier']}",
        "",
        "| method | IDF1 | retention | switches | duplicate | reappearance |",
        "|---|---|---|---|---|---|",
    ]
    for name in results:
        m = aggregate[name]
        lines.append(
            f"| {name} | {m['idf1']:.4f} | {m['id_retention']:.4f} | "
            f"{m['id_switches']:.2f} | {m['duplicate_rate']:.4f} | {m['reappearance_rate']:.4f} |"
        )
    lines += ["", "## full − baseline (paired bootstrap, 95% CI)", ""]
    for baseline, metrics in deltas.items():
        for metric, stats in metrics.items():
            lines.append(
                f"- {baseline} / {metric}: {stats['mean']:+.4f} "
                f"[{stats['ci95'][0]:+.4f}, {stats['ci95'][1]:+.4f}]"
            )
    (out / f"{args.suite}_vs_baselines.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
