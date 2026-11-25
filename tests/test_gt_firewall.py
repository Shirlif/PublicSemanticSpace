from __future__ import annotations

import json
import shutil
from pathlib import Path

from pss.detect.replay import replay_bundle


def test_replay_does_not_need_gt(tmp_path):
    source = Path("bundles/dev/s1_camera_real/b4bb5f0e/seed0")
    if not source.exists():
        return
    target = tmp_path / "bundle"
    target.mkdir()
    for name in ("manifest.json", "obs_frames.parquet", "obs_detections.parquet"):
        shutil.copyfile(source / name, target / name)
    assert not (target / "gt.parquet").exists()
    trace, pss = replay_bundle(target, write=False)
    assert len(pss) > 0
    assert not trace.empty
