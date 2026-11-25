"""Relation scorecard: per-predicate F1 + consistency violations (T25/T27)."""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import pandas as pd

from .relation import consistency_violations, per_predicate_prf
from .scorecard import discover_bundles, _match_states


def _parse_relation(value) -> list[tuple[str, str]]:
    out = []
    if value is None:
        return out
    for item in value:
        text = str(item)
        if "(" in text and text.endswith(")"):
            pred, obj = text[:-1].split("(", 1)
            out.append((pred, obj))
    return out


def score_relations_bundle(bundle: str | Path) -> dict:
    root = Path(bundle)
    obs = pd.read_parquet(root / "obs_frames.parquet")
    gt = pd.read_parquet(root / "gt.parquet")
    pred_rel = pd.read_parquet(root / "pred_relations.parquet")
    steps = sorted(int(s) for s in obs["step"].unique())
    step_to_frame = {step: i for i, step in enumerate(steps)}

    matches = _match_states(root)
    pred_to_gt: dict[int, dict[str, str]] = {}
    for _, row in matches.iterrows():
        pred_to_gt.setdefault(int(row["frame"]), {})[str(row["pred_id"])] = str(row["gt_id"])

    gt_by_step = {int(step): group for step, group in gt.groupby("physics_step")}
    truth: set[tuple] = set()
    computed: set[tuple] = set()
    gt_triples: list[tuple] = []
    pred_triples: list[tuple] = []
    for step in steps:
        frame = step_to_frame[step]
        group = gt_by_step.get(step)
        if group is not None:
            for _, row in group.iterrows():
                for pred, obj in _parse_relation(row["relations"]):
                    triple = (frame, str(row["instance_id"]), pred, obj)
                    truth.add(triple)
                    gt_triples.append((str(row["instance_id"]), pred, obj))
        mapping = pred_to_gt.get(frame, {})
        if pred_rel.empty:
            continue
        for _, row in pred_rel[pred_rel["step"] == step].iterrows():
            subj = mapping.get(str(row["subj"]))
            obj = mapping.get(str(row["obj"]))
            if subj is None or obj is None:
                continue
            triple = (frame, subj, str(row["pred"]), obj)
            computed.add(triple)
            pred_triples.append((subj, str(row["pred"]), obj))

    metrics = per_predicate_prf(computed, truth)
    gt_consistency = consistency_violations(gt_triples)
    pred_consistency = consistency_violations(pred_triples)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    metrics.update(
        {
            "gt_relation_triples": len(gt_triples),
            "pred_relation_triples": len(pred_triples),
            "gt_consistency_violations": gt_consistency["violation_count"],
            "pred_consistency_violations": pred_consistency["violation_count"],
            "pred_consistency_rate": pred_consistency["rate"],
            "bundle": str(root),
            "seed": manifest.get("seed"),
            "device_class": manifest.get("device_class"),
            "spec_hash": manifest.get("spec", {}).get("hash"),
        }
    )
    return metrics


def score_relations_suite(
    bundles_root: str | Path,
    suite: str,
    *,
    device: str | None = None,
) -> dict:
    bundles = discover_bundles(bundles_root, suite, device)
    if not bundles:
        raise FileNotFoundError(f"no bundles for relation suite {suite!r}")
    episodes = [score_relations_bundle(bundle) for bundle in bundles]
    predicates = sorted(
        {p for episode in episodes for p in episode["per_predicate"]}
    )
    per = {}
    for pred in predicates:
        values = [ep["per_predicate"].get(pred, {}).get("f1", 0.0) for ep in episodes]
        per[pred] = {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }
    return {
        "suite": suite,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "device_class": episodes[0].get("device_class", "dev"),
        "n_episodes": len(episodes),
        "macro_f1": sum(ep["macro_f1"] for ep in episodes) / len(episodes),
        "micro_f1": sum(ep["micro"]["f1"] for ep in episodes) / len(episodes),
        "per_predicate": per,
        "pred_consistency_violations": sum(
            ep["pred_consistency_violations"] for ep in episodes
        ),
        "gt_consistency_violations": sum(
            ep["gt_consistency_violations"] for ep in episodes
        ),
        "episodes": episodes,
    }


def write_relation_scorecard(report: dict, out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"{report['suite']}_relations.json"
    md_path = out / f"{report['suite']}_relations.md"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        f"# {report['suite']} relation scorecard",
        "",
        f"- episodes: {report['n_episodes']}",
        f"- macro F1: {report['macro_f1']:.4f}",
        f"- micro F1: {report['micro_f1']:.4f}",
        f"- PSS consistency violations: {report['pred_consistency_violations']}",
        f"- GT consistency violations: {report['gt_consistency_violations']}",
        "",
        "| predicate | mean F1 | min | max |",
        "|---|---|---|---|",
    ]
    for pred, stats in report["per_predicate"].items():
        lines.append(
            f"| {pred} | {stats['mean']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path
