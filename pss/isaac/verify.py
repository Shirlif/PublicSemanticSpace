"""Read-back checks for a W1 bundle (D5).

The third assertion is the important one: visibility must actually vary.  If it
is 1.0 for every frame, the occlusion path is not connected and every later S1
number would be fiction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def verify_bundle(
    bundle_dir: str | Path,
    *,
    expect_instances: int = 4,
    expect_obs_frames: int = 100,
    expect_physics_steps: int | None = 1200,
) -> dict[str, Any]:
    root = Path(bundle_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    gt = pd.read_parquet(root / "gt.parquet")
    frames = pd.read_parquet(root / "obs_frames.parquet")

    results: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: Any) -> None:
        results.append({"check": name, "ok": bool(ok), "detail": detail})

    n_instances = int(gt.instance_id.nunique())
    check("instance_count", n_instances == expect_instances, n_instances)
    check("obs_frame_count", len(frames) == expect_obs_frames, len(frames))
    check(
        "visibility_varies",
        float(gt.visibility_ratio.max()) > 0.9 and float(gt.visibility_ratio.min()) < 1.0,
        {
            "min": float(gt.visibility_ratio.min()),
            "max": float(gt.visibility_ratio.max()),
        },
    )
    if expect_physics_steps is not None:
        steps = int(gt.physics_step.nunique())
        check("physics_step_count", steps == expect_physics_steps, steps)
    check(
        "manifest_links_bundle",
        manifest.get("spec", {}).get("id") is not None
        and (root / manifest.get("files", {}).get("gt", "gt.parquet")).exists(),
        manifest.get("spec"),
    )

    per_instance = (
        gt.groupby("instance_id")["visibility_ratio"]
        .agg(["min", "max", "mean", "count"])
        .reset_index()
        .to_dict(orient="records")
    )
    return {
        "bundle": str(root),
        "ok": all(r["ok"] for r in results),
        "checks": results,
        "per_instance": per_instance,
        "manifest": manifest,
    }


def print_report(report: dict[str, Any]) -> None:
    for row in report["checks"]:
        mark = "PASS" if row["ok"] else "FAIL"
        print(f"[{mark}] {row['check']}: {row['detail']}")
    print("visibility per instance:")
    for row in report["per_instance"]:
        print(
            f"  {row['instance_id']:<16} min={row['min']:.3f} "
            f"max={row['max']:.3f} mean={row['mean']:.3f}"
        )
    print("OVERALL:", "PASS" if report["ok"] else "FAIL")
