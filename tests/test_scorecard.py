from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pss.metrics.scorecard import score_suite


def _bundle(root: Path, seed: int) -> None:
    bundle = root / "dev" / "s1_demo" / "abc12345" / f"seed{seed}"
    bundle.mkdir(parents=True)
    manifest = {
        "episode_id": f"s1__seed{seed}__oracle__full",
        "seed": seed,
        "spec": {"id": "s1_demo_v1", "hash": "sha256:" + "a" * 64},
        "device_class": "dev",
        "detector_tier": "oracle",
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    pd.DataFrame(
        [{"step": 0}, {"step": 12}], columns=["step"]
    ).to_parquet(bundle / "obs_frames.parquet", index=False)
    gt_rows = [
        {"physics_step": step, "instance_id": item, "visibility_ratio": 1.0}
        for step in (0, 12)
        for item in ("a", "b")
    ]
    pd.DataFrame(gt_rows).to_parquet(bundle / "gt.parquet", index=False)
    trace_rows = [
        {"step": step, "t_s": step / 120.0, "gt_id": item, "pred_id": f"t_{item}"}
        for step in (0, 12)
        for item in ("a", "b")
    ]
    pd.DataFrame(trace_rows).to_parquet(bundle / "entity_trace.parquet", index=False)


def test_score_suite_aggregates_episodes(tmp_path):
    _bundle(tmp_path, 0)
    _bundle(tmp_path, 1)
    report = score_suite(tmp_path, "s1_demo", device="dev")
    assert report["n_episodes"] == 2
    assert report["seeds"] == [0, 1]
    assert report["device_class"] == "dev"
    assert report["metrics"]["idf1"]["mean"] == 1.0
    assert report["metrics"]["id_switches"]["mean"] == 0.0
