from __future__ import annotations

import pandas as pd

from pss.baselines import sort_lite_predictions, text_state_predictions


def _det(rows):
    return pd.DataFrame(
        rows,
        columns=["step", "t_s", "pose_xyz", "class_pred", "score", "emb"],
    )


def test_text_state_persists_across_consecutive_frames():
    det = _det(
        [
            [0, 0.0, [0.0, 0.0, 0.8], "cube", 1.0, None],
            [1, 0.1, [0.0, 0.0, 0.8], "cube", 1.0, None],
        ]
    )
    out = text_state_predictions(det, steps=[0, 1])
    ids = out[out["step"] == 1]["pred_id"].tolist()
    assert ids == ["text_0001"]


def test_text_state_forgets_after_gap():
    det = _det(
        [
            [0, 0.0, [0.0, 0.0, 0.8], "cube", 1.0, None],
            [2, 0.2, [0.0, 0.0, 0.8], "cube", 1.0, None],
        ]
    )
    out = text_state_predictions(det, steps=[0, 1, 2])
    assert out[out["step"] == 2]["pred_id"].tolist() == ["text_0002"]


def test_sort_lite_keeps_id_through_short_gap():
    det = _det(
        [
            [0, 0.0, [0.0, 0.0, 0.8], "cube", 1.0, None],
            [2, 0.2, [0.0, 0.0, 0.8], "cube", 1.0, None],
        ]
    )
    out = sort_lite_predictions(det, steps=[0, 1, 2], max_age=5)
    assert out[out["step"] == 2]["pred_id"].tolist() == ["sort_0001"]
