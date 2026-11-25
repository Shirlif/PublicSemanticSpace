"""Scorecard generation from bundle artifacts."""

from __future__ import annotations

import datetime as _dt
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from .entity import compute_entity_metrics
from .twin import twin_ambiguity


def discover_bundles(
    bundles_root: str | Path,
    suite: str,
    device: str | None = None,
) -> list[Path]:
    root = Path(bundles_root)
    if device is not None:
        root = root / device
    found = sorted(
        path.parent
        for path in root.glob(f"**/{suite}/**/seed*/manifest.json")
    )
    return found


def match_states(
    gt: pd.DataFrame,
    preds: pd.DataFrame,
    steps: list[int],
    *,
    gate: float = 0.05,
) -> pd.DataFrame:
    """Evaluation-side pred-track -> GT matching by 3D position."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    step_to_frame = {step: i for i, step in enumerate(steps)}
    gt = gt[(gt["physics_step"].isin(steps)) & (gt["visibility_ratio"] > 0.001)]
    gt_by_step = {int(step): group for step, group in gt.groupby("physics_step")}
    pred_by_step = {int(step): group for step, group in preds.groupby("step")}
    rows = []
    for step in steps:
        g = gt_by_step.get(step)
        p = pred_by_step.get(step)
        if g is None or p is None or g.empty or p.empty:
            continue
        gt_ids = g["instance_id"].astype(str).tolist()
        pred_ids = p["pred_id"].astype(str).tolist()
        gpos = np.stack([np.asarray(v, dtype=float) for v in g["pose_xyz"]])
        ppos = np.stack([np.asarray(v, dtype=float) for v in p["pos_xyz"]])
        cost = np.linalg.norm(gpos[:, None, :] - ppos[None, :, :], axis=2)
        rows_i, cols_j = linear_sum_assignment(cost)
        for i, j in zip(rows_i, cols_j):
            if cost[i, j] <= gate:
                rows.append(
                    {
                        "frame": step_to_frame[step],
                        "gt_id": gt_ids[int(i)],
                        "pred_id": pred_ids[int(j)],
                    }
                )
    return pd.DataFrame(rows, columns=["frame", "gt_id", "pred_id"])


def score_pred_states(bundle: str | Path, preds: pd.DataFrame) -> dict[str, Any]:
    """Score any tracker's per-frame states with the shared matching/metrics."""
    root = Path(bundle)
    obs = pd.read_parquet(root / "obs_frames.parquet")
    gt = pd.read_parquet(root / "gt.parquet")
    steps = sorted(int(s) for s in obs["step"].unique())
    trace = match_states(gt, preds, steps)
    gt_frames = gt[
        (gt["physics_step"].isin(steps)) & (gt["visibility_ratio"] > 0.001)
    ][["physics_step", "instance_id"]].rename(columns={"physics_step": "frame"})
    step_to_frame = {step: i for i, step in enumerate(steps)}
    gt_frames["frame"] = gt_frames["frame"].map(step_to_frame)
    return compute_entity_metrics(gt_frames, trace, pred_total=len(preds))


def _match_states(bundle: Path, gate: float = 0.05) -> pd.DataFrame:
    """Evaluation-side pred/track -> GT matching by 3D position.

    The PSS process never sees GT; this matching happens after replay and is
    what turns track states into an identity trace.
    """
    obs = pd.read_parquet(bundle / "obs_frames.parquet")
    gt = pd.read_parquet(bundle / "gt.parquet")
    preds = pd.read_parquet(bundle / "pred_states.parquet")
    steps = sorted(int(s) for s in obs["step"].unique())
    return match_states(gt, preds, steps, gate=gate)


def _gt_frames(bundle: Path) -> tuple[pd.DataFrame, dict[int, int]]:
    obs = pd.read_parquet(bundle / "obs_frames.parquet")
    gt = pd.read_parquet(bundle / "gt.parquet")
    steps = sorted(int(s) for s in obs["step"].unique())
    step_to_frame = {step: i for i, step in enumerate(steps)}
    visible = gt[(gt["physics_step"].isin(steps)) & (gt["visibility_ratio"] > 0.001)]
    frames = visible[["physics_step", "instance_id"]].rename(columns={"physics_step": "frame"})
    frames["frame"] = frames["frame"].map(step_to_frame)
    return frames, step_to_frame


def score_bundle(bundle: str | Path) -> dict[str, Any]:
    root = Path(bundle)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    gt_frames, step_to_frame = _gt_frames(root)
    if (root / "pred_states.parquet").exists() and not pd.read_parquet(
        root / "pred_states.parquet"
    ).empty:
        trace = _match_states(root)
    else:
        trace = pd.read_parquet(root / "entity_trace.parquet")
        trace = trace.copy()
        trace["frame"] = trace["step"].map(step_to_frame)
        trace = trace.dropna(subset=["frame"])
    if trace.empty:
        raise ValueError(f"{root}: no identity trace; run `pss replay` first")
    pred_states_path = root / "pred_states.parquet"
    pred_total = (
        int(len(pd.read_parquet(pred_states_path)))
        if pred_states_path.exists()
        else None
    )
    metrics = compute_entity_metrics(
        gt_frames, trace[["frame", "gt_id", "pred_id"]], pred_total=pred_total
    )
    det_path = root / "obs_detections.parquet"
    if det_path.exists():
        detections = pd.read_parquet(det_path)
        if "emb" in detections.columns and detections["emb"].notna().any():
            twin = twin_ambiguity(detections, pd.read_parquet(root / "gt.parquet"))
            if twin["frames"] > 0:
                metrics.update(
                    {
                        "twin_ambiguity_rate": twin["twin_ambiguity_rate"],
                        "twin_mean_max_similarity": twin["mean_max_similarity"],
                    }
                )
    metrics.update(
        {
            "episode_id": manifest.get("episode_id"),
            "seed": manifest.get("seed"),
            "bundle": str(root),
            "spec_hash": manifest.get("spec", {}).get("hash"),
            "device_class": manifest.get("device_class"),
            "detector_tier": manifest.get("detector_tier"),
        }
    )
    return metrics


def _ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return 1.96 * math.sqrt(variance / len(values))


def score_suite(
    bundles_root: str | Path,
    suite: str,
    *,
    device: str | None = None,
) -> dict[str, Any]:
    bundles = discover_bundles(bundles_root, suite, device)
    if not bundles:
        raise FileNotFoundError(f"no bundles for suite {suite!r} under {bundles_root}")
    episodes = [score_bundle(bundle) for bundle in bundles]
    devices = {ep["device_class"] for ep in episodes}
    if len(devices) != 1:
        raise ValueError(f"mixed device_class in scorecard: {sorted(devices)}")
    tiers = {ep["detector_tier"] for ep in episodes}
    specs = {ep["spec_hash"] for ep in episodes}

    metric_names = [
        "idf1",
        "id_retention",
        "id_switches",
        "duplicate_rate",
        "reappearance_rate",
        "slot_purity",
    ]
    metric_names += [
        name
        for name in ("twin_ambiguity_rate", "twin_mean_max_similarity")
        if all(name in ep for ep in episodes)
    ]
    aggregate: dict[str, Any] = {}
    for name in metric_names:
        values = [float(ep[name]) for ep in episodes]
        aggregate[name] = {
            "mean": sum(values) / len(values),
            "std": (
                math.sqrt(sum((v - sum(values) / len(values)) ** 2 for v in values) / (len(values) - 1))
                if len(values) > 1
                else 0.0
            ),
            "ci95": _ci95(values),
            "min": min(values),
            "max": max(values),
        }
    return {
        "suite": suite,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "device_class": episodes[0]["device_class"],
        "detector_tier": sorted(tiers)[0] if len(tiers) == 1 else sorted(tiers),
        "caveat": (
            "oracle tier is a calibration upper bound (perfect GT identity/pose); "
            "it is not capability evidence"
            if "oracle" in tiers
            else ""
        ),
        "spec_hashes": sorted(specs),
        "n_episodes": len(episodes),
        "seeds": sorted(int(ep["seed"]) for ep in episodes),
        "metrics": aggregate,
        "episodes": episodes,
    }


def write_scorecard(report: dict[str, Any], out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"{report['suite']}.json"
    md_path = out / f"{report['suite']}.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        f"# {report['suite']} scorecard",
        "",
        f"- device_class: `{report['device_class']}`",
        f"- detector_tier: `{report['detector_tier']}`",
        f"- episodes: {report['n_episodes']}  seeds: {report['seeds']}",
        f"- spec hashes: {', '.join('`' + h + '`' for h in report['spec_hashes'])}",
    ]
    if report.get("caveat"):
        lines.append(f"- caveat: {report['caveat']}")
    lines.extend(
        [
            "",
            "| metric | mean | std | 95% CI | min | max |",
            "|---|---|---|---|---|---|",
        ]
    )
    for name, stats in report["metrics"].items():
        lines.append(
            f"| {name} | {stats['mean']:.4f} | {stats['std']:.4f} | "
            f"±{stats['ci95']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} |"
        )
    lines.append("")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path
