from __future__ import annotations

import pandas as pd

from pss.metrics.memory import compute_memory_metrics


def test_memory_metrics_after_relocation():
    trace = pd.DataFrame(
        [
            {"episode": 0, "gt_id": "a", "pred_id": "p1",
             "pss_xyz": [0.0, 0.0, 0.0], "gt_xyz": [0.0, 0.0, 0.0]},
            {"episode": 1, "gt_id": "a", "pred_id": "p1",
             "pss_xyz": [0.2, 0.0, 0.0], "gt_xyz": [0.2, 0.0, 0.0]},
        ]
    )
    out = compute_memory_metrics(trace, rollback_exactness=1.0)
    assert out["edited_accuracy"] == 1.0
    assert out["old_forgetting"] == 1.0
    assert out["forgetting_rate"] == 0.0
    assert out["id_retention"] == 1.0
    assert out["rollback_exactness"] == 1.0


def test_memory_metrics_detect_lost_identity():
    trace = pd.DataFrame(
        [
            {"episode": 0, "gt_id": "a", "pred_id": "p1",
             "pss_xyz": [0.0, 0.0, 0.0], "gt_xyz": [0.0, 0.0, 0.0]},
            {"episode": 1, "gt_id": "a", "pred_id": "p2",
             "pss_xyz": [0.0, 0.0, 0.0], "gt_xyz": [0.2, 0.0, 0.0]},
        ]
    )
    out = compute_memory_metrics(trace)
    assert out["edited_accuracy"] == 0.0
    assert out["old_forgetting"] == 0.0
    assert out["id_retention"] < 1.0
