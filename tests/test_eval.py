import numpy as np
import pytest

from pss.eval.metrics import id_retention, pose_error, relation_prf


def test_id_retention_perfect_and_switch():
    perfect = id_retention({"a": [(0, "t1"), (1, "t1"), (2, "t1")]})
    assert perfect["id_retention"] == 1.0
    assert perfect["id_switches"] == 0.0

    switched = id_retention({"a": [(0, "t1"), (1, "t2")]})
    assert switched["id_retention"] == 0.5
    assert switched["id_switches"] == 1.0


def test_relation_prf():
    result = relation_prf(
        {("a", "near", "b")},
        {("a", "near", "b"), ("c", "in", "bin0")},
    )
    assert result["tp"] == 1
    assert result["fn"] == 1
    assert result["f1"] == pytest.approx(2 / 3)


def test_pose_error():
    pred = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    truth = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert pose_error(pred, truth) == 0.0


def _obs(pos, category="cup"):
    pytest.importorskip("torch")
    from pss.memory.perception import Observation

    return Observation(
        gt_id="a",
        pos=np.asarray(pos, dtype=float),
        quat=np.array([1.0, 0.0, 0.0, 0.0]),
        appearance=np.array([1.0, 0.0, 0.0, 0.0]),
        raw={"vision": np.array([1.0, 0.0, 0.0, 0.0])},
        confidence=1.0,
        category=category,
    )


def test_text_state_tracker_drops_occluded_identity():
    pytest.importorskip("torch")
    from pss.eval.baselines import TextStateTracker

    tracker = TextStateTracker(gate=0.2)
    assert tracker.step([_obs([0.0, 0.0, 0.05])], 0)["new"]
    assert tracker.step([_obs([0.01, 0.0, 0.05])], 1)["matched"]
    tracker.step([], 2)
    assert tracker.step([_obs([0.01, 0.0, 0.05])], 3)["new"]


def test_no_scene_graph_tracker_assigns_new_ids():
    pytest.importorskip("torch")
    from pss.eval.baselines import NoSceneGraphTracker

    tracker = NoSceneGraphTracker()
    first = tracker.step([_obs([0.0, 0.0, 0.05])], 0)["new"][0][0]
    second = tracker.step([_obs([0.01, 0.0, 0.05])], 1)["new"][0][0]
    assert first != second
