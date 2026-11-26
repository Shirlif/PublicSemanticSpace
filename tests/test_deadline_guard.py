from __future__ import annotations

import json
import time

import pandas as pd

from pss.detect.replay import replay_bundle
from pss.isaac.bundle import TRACE_SCHEMA
from pss.runtime.deadline import DeadlineGuard, LatencyIsolation


def _tiny_bundle(root):
    rows = []
    for step in range(5):
        for j, (identity, pos) in enumerate((("a", [0.0, 0.0, 0.0]), ("b", [1.0, 0.0, 0.0]))):
            rows.append(
                {
                    "step": step,
                    "t_s": step / 10.0,
                    "camera": "overhead",
                    "det_id": f"{step}_{j}",
                    "bbox_xyxy": [0.0, 0.0, 1.0, 1.0],
                    "score": 0.9,
                    "class_pred": "cube",
                    "pose_xyz": pos,
                    "pose_quat": [1.0, 0.0, 0.0, 0.0],
                    "emb": [1.0, 0.0] if j == 0 else [0.0, 1.0],
                    "emb_ref": identity,
                }
            )
    pd.DataFrame(rows).to_parquet(root / "obs_detections.parquet", index=False)
    gt = pd.DataFrame(
        [
            {"physics_step": step, "instance_id": identity}
            for step in range(5)
            for identity in ("a", "b")
        ]
    )
    gt.to_parquet(root / "gt.parquet", index=False)
    (root / "manifest.json").write_text(
        json.dumps({"spec": {"id": "test", "hash": "sha256:test"}}),
        encoding="utf-8",
    )
    return root


def test_fast_call_returns_value():
    guard = DeadlineGuard(budget_ms=100.0, reserve_ms=1.0)
    value, timed_out, elapsed = guard.call(lambda: 42)
    assert value == 42
    assert timed_out is False
    assert elapsed < 100.0
    snapshot = guard.snapshot()
    assert snapshot.frames == 1
    assert snapshot.timeouts == 0
    guard.close()


def test_slow_call_times_out_and_joins():
    guard = DeadlineGuard(budget_ms=20.0, reserve_ms=2.0, switch_interval=0.0005)

    def slow():
        time.sleep(0.06)
        return "late"

    value, timed_out, elapsed = guard.call(slow)
    assert value is None
    assert timed_out is True
    assert elapsed < 30.0
    # The worker is joined before the next call; the late work is recorded.
    snapshot = guard.snapshot()
    assert snapshot.timeouts == 1
    assert snapshot.worker_late_ms["max"] >= 50.0
    assert snapshot.errors == 0
    guard.close()


def test_slow_call_enters_work_shedding_cooldown():
    guard = DeadlineGuard(
        budget_ms=20.0,
        reserve_ms=2.0,
        degraded_cooldown_frames=2,
        switch_interval=0.0005,
    )
    guard.call(lambda: time.sleep(0.06))
    assert guard.should_shed
    guard.call(lambda: 1)
    assert guard.should_shed
    guard.call(lambda: 2)
    assert not guard.should_shed
    guard.close()


def test_latency_isolation_is_reentrant():
    with LatencyIsolation():
        with LatencyIsolation():
            pass


def test_replay_guard_caps_latency_and_adds_columns(tmp_path):
    bundle = _tiny_bundle(tmp_path)
    trace, pss = replay_bundle(
        bundle,
        write=False,
        deadline_ms=10.0,
        slow_stage_ms=50.0,
        slow_every=2,
    )
    assert not trace.empty
    assert {"degraded", "frame_ms"}.issubset(pss.columns)
    assert int(pss["degraded"].sum()) >= 1
    assert float(pss["latency_ms"].max()) < 20.0
    snapshot = pss.attrs["deadline_guard"]
    assert snapshot["timeouts"] >= 1
    assert snapshot["frames"] == 5
    assert "relation_rows" in pss.attrs


def test_replay_guard_keeps_parquet_schema(tmp_path):
    bundle = _tiny_bundle(tmp_path)
    trace, pss = replay_bundle(
        bundle,
        write=True,
        deadline_ms=10.0,
        slow_stage_ms=50.0,
        slow_every=2,
    )
    assert "degraded" in pss.columns
    written = pd.read_parquet(tmp_path / "pss_trace.parquet")
    assert list(written.columns) == list(TRACE_SCHEMA.names)


def test_replay_without_guard_has_no_extra_columns(tmp_path):
    bundle = _tiny_bundle(tmp_path)
    _, pss = replay_bundle(bundle, write=False)
    assert "degraded" not in pss.columns
    assert "frame_ms" not in pss.columns

