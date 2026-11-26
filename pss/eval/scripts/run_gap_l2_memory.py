"""Claim-gap E12: L2 bundle replay under cognitive-layer ablations.

The Isaac S4-PSS demo has oracle poses and no occlusion, so it cannot test
memory/binding.  The existing bundles do contain occlusion and noisy detection
streams.  This experiment replays those bundles offline with the real runtime
and switches only the architecture variant:

- ``full``: persistent entity layer;
- ``no_memory``: a track is dropped as soon as it misses a frame;
- ``no_entity``: every frame gets fresh ids.

Metrics are identity retention/switches/reappearance from the replayed
``entity_trace``.  Relation activity is reported as mean ``facts_n`` from
``pss_trace``.  This is L2 external validity without re-running Isaac.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from pss._console import ensure_utf8
from pss.detect.replay import replay_bundle
from pss.metrics.entity import compute_entity_metrics

VARIANTS = ("full", "no_memory", "no_entity", "no_relations")


def _has_embedding(detections: Path) -> bool:
    """True when a detection table carries at least one non-empty ``emb``."""
    try:
        frame = pd.read_parquet(detections, columns=["emb"])
    except Exception:
        return False
    if "emb" not in frame.columns or frame.empty:
        return False
    return bool(frame["emb"].map(lambda value: value is not None and len(value) > 0).any())


def _bundles(root: Path, pattern: str, *, require_emb: bool = False) -> list[Path]:
    out = []
    for det in sorted(root.glob(pattern)):
        bundle = det.parent
        if (bundle / "gt.parquet").exists() and (bundle / "obs_detections.parquet").exists():
            if require_emb and not _has_embedding(det):
                continue
            out.append(bundle)
    return out


def _relative_bundle(bundle: Path, root: Path) -> Path:
    """Path relative to the bundle root, skipping a pack's top-level directory."""
    rel = bundle.relative_to(root)
    if len(rel.parts) > 1 and (root / rel.parts[0] / "PACK_MANIFEST.json").exists():
        rel = Path(*rel.parts[1:])
    return rel


def _identity(trace: pd.DataFrame, gt: pd.DataFrame) -> dict:
    if trace is None or trace.empty:
        return {
            "id_retention": 0.0,
            "id_switches": 0.0,
            "reappearance_rate": 0.0,
            "slot_purity": 0.0,
        }
    tr = trace.rename(columns={"step": "frame"})
    gt_frames = gt.rename(
        columns={"physics_step": "frame", "instance_id": "instance_id"}
    )[["frame", "instance_id"]]
    m = compute_entity_metrics(gt_frames, tr)
    return {
        "id_retention": float(m["id_retention"]),
        "id_switches": float(m["id_switches"]),
        "reappearance_rate": float(m["reappearance_rate"]),
        "slot_purity": float(m["slot_purity"]),
    }


def run(
    root: str = "bundles/dev",
    pattern: str = "**/obs_detections.parquet",
    *,
    require_emb: bool = False,
) -> dict:
    root = Path(root)
    bundles = _bundles(root, pattern, require_emb=require_emb)
    if not bundles:
        raise ValueError(f"no bundles under {root}/{pattern}")
    rows = []
    for bundle in bundles:
        rel = _relative_bundle(bundle, root)
        gt = pd.read_parquet(bundle / "gt.parquet")
        for variant in VARIANTS:
            trace, pss = replay_bundle(bundle, write=False, variant=variant)
            identity = _identity(trace, gt)
            facts_mean = (
                float(pss["facts_n"].mean())
                if (pss is not None and not pss.empty and "facts_n" in pss)
                else 0.0
            )
            latency_p95 = (
                float(np.percentile(pss["latency_ms"], 95))
                if (pss is not None and not pss.empty and "latency_ms" in pss)
                else 0.0
            )
            rows.append(
                {
                    "bundle": str(rel),
                    "group": rel.parts[0],
                    "variant": variant,
                    **identity,
                    "facts_mean": facts_mean,
                    "latency_p95_ms": latency_p95,
                }
            )
        print(f"[{len(rows) // len(VARIANTS)}/{len(bundles)}] {rel}")

    summary = {}
    for variant in VARIANTS:
        per = [r for r in rows if r["variant"] == variant]
        summary[variant] = {
            "n": len(per),
            "id_retention": float(np.mean([r["id_retention"] for r in per])),
            "id_switches": float(np.mean([r["id_switches"] for r in per])),
            "reappearance_rate": float(np.mean([r["reappearance_rate"] for r in per])),
            "facts_mean": float(np.mean([r["facts_mean"] for r in per])),
            "latency_p95_ms": float(np.mean([r["latency_p95_ms"] for r in per])),
        }
    groups = {}
    for group in sorted({r["group"] for r in rows}):
        groups[group] = {}
        for variant in VARIANTS:
            per = [r for r in rows if r["group"] == group and r["variant"] == variant]
            groups[group][variant] = {
                "n": len(per),
                "id_retention": float(np.mean([r["id_retention"] for r in per])),
                "id_switches": float(np.mean([r["id_switches"] for r in per])),
                "reappearance_rate": float(np.mean([r["reappearance_rate"] for r in per])),
                "facts_mean": float(np.mean([r["facts_mean"] for r in per])),
            }
    return {
        "root": str(root),
        "pattern": pattern,
        "n_bundles": len(bundles),
        "variants": list(VARIANTS),
        "rows": rows,
        "summary": summary,
        "groups": groups,
    }


def _valid_bundles(m: dict) -> set[str]:
    return {
        r["bundle"]
        for r in m["rows"]
        if r["variant"] == "full" and r["id_retention"] > 0.5
    }


def _valid_summary(m: dict) -> dict:
    valid_bundles = _valid_bundles(m)
    valid_rows = [r for r in m["rows"] if r["bundle"] in valid_bundles]
    summary = {}
    for variant in m["variants"]:
        per = [r for r in valid_rows if r["variant"] == variant]
        if per:
            summary[variant] = {
                "n": len(per),
                "id_retention": float(np.mean([r["id_retention"] for r in per])),
                "id_switches": float(np.mean([r["id_switches"] for r in per])),
            }
    return summary


def _report(
    m: dict,
    title: str = "Claim-gap E12: L2 bundle replay under cognitive ablations",
) -> str:
    valid_bundles = _valid_bundles(m)
    valid_summary = _valid_summary(m)
    lines = [
        f"# {title}",
        "",
        f"> bundles: {m['n_bundles']} under `{m['root']}`",
        "",
        "## Overall",
        "",
        "| variant | n | ID retention | switches | reappearance | facts mean | p95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, r in m["summary"].items():
        lines.append(
            f"| {variant} | {r['n']} | {r['id_retention']:.3f} | {r['id_switches']:.2f} | "
            f"{r['reappearance_rate']:.3f} | {r['facts_mean']:.2f} | {r['latency_p95_ms']:.3f} |"
        )
    if valid_summary:
        lines += [
            "",
            "## Valid bundles only",
            "",
            "Bundles where the full runtime has a non-empty identity trace",
            f"({len(valid_bundles)} of {m['n_bundles']}). Zero-retention bundles are excluded",
            "because they carry no identity signal to ablate.",
            "",
            "| variant | n | ID retention | switches |",
            "|---|---:|---:|---:|",
        ]
        for variant, r in valid_summary.items():
            lines.append(
                f"| {variant} | {r['n']} | {r['id_retention']:.3f} | {r['id_switches']:.2f} |"
            )
    lines += [
        "",
        "## By bundle group",
        "",
        "| group | variant | ID retention | switches | reappearance | facts |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for group, variants in m["groups"].items():
        for variant, r in variants.items():
            lines.append(
                f"| {group} | {variant} | {r['id_retention']:.3f} | "
                f"{r['id_switches']:.2f} | {r['reappearance_rate']:.3f} | {r['facts_mean']:.2f} |"
            )
    lines += [
        "",
        "## How to read this",
        "",
        "The replay consumes only `obs_detections.parquet`; it never reads GT for",
        "association. GT is used after the fact for scoring. `no_memory` and",
        "`no_entity` are architecture ablations on the same detection stream, so the",
        "difference is the state layer, not perception. `no_relations` should keep",
        "identity intact while `facts_n` drops; that is the relation-layer control.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="bundles/dev")
    ap.add_argument("--pattern", default="**/obs_detections.parquet")
    ap.add_argument("--out", default="reports/claim_gaps/l2_memory")
    ap.add_argument(
        "--require-emb",
        action="store_true",
        help="only bundles whose detection table carries non-empty embeddings",
    )
    ap.add_argument(
        "--title",
        default="Claim-gap E12: L2 bundle replay under cognitive ablations",
    )
    args = ap.parse_args()
    metrics = run(root=args.root, pattern=args.pattern, require_emb=args.require_emb)
    metrics["generated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "REPORT.md").write_text(_report(metrics, title=args.title), encoding="utf-8")
    for variant, r in metrics["summary"].items():
        print(
            f"{variant:14s} retention={r['id_retention']:.3f} switches={r['id_switches']:.2f} "
            f"facts={r['facts_mean']:.2f}"
        )
    print(f"output: {out}")


if __name__ == "__main__":
    main()
