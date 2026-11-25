from __future__ import annotations

import numpy as np
import pandas as pd

from pss.metrics.twin import twin_ambiguity


def test_twin_ambiguity_detects_identical_embeddings():
    gt = pd.DataFrame(
        [
            {"physics_step": 0, "instance_id": "a", "pose_xyz": [0, 0, 0.79],
             "visibility_ratio": 1.0, "twin_group": "tg"},
            {"physics_step": 0, "instance_id": "b", "pose_xyz": [0.1, 0, 0.79],
             "visibility_ratio": 1.0, "twin_group": "tg"},
        ]
    )
    det = pd.DataFrame(
        [
            {"step": 0, "pose_xyz": [0, 0, 0.79], "emb": np.array([1.0, 0.0])},
            {"step": 0, "pose_xyz": [0.1, 0, 0.79], "emb": np.array([1.0, 0.0])},
        ]
    )
    out = twin_ambiguity(det, gt, sim_threshold=0.98)
    assert out["twin_ambiguity_rate"] == 1.0


def test_distinct_embeddings_are_not_ambiguous():
    gt = pd.DataFrame(
        [
            {"physics_step": 0, "instance_id": "a", "pose_xyz": [0, 0, 0.79],
             "visibility_ratio": 1.0, "twin_group": "tg"},
            {"physics_step": 0, "instance_id": "b", "pose_xyz": [0.1, 0, 0.79],
             "visibility_ratio": 1.0, "twin_group": "tg"},
        ]
    )
    det = pd.DataFrame(
        [
            {"step": 0, "pose_xyz": [0, 0, 0.79], "emb": np.array([1.0, 0.0])},
            {"step": 0, "pose_xyz": [0.1, 0, 0.79], "emb": np.array([0.0, 1.0])},
        ]
    )
    out = twin_ambiguity(det, gt, sim_threshold=0.98)
    assert out["twin_ambiguity_rate"] == 0.0
