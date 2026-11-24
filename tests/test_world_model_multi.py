from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pss.world_model.multi import (
    MAX_ENTITIES,
    MultiDeltaMLP,
    feature_dim,
    multi_transition_feature,
)
from pss.world_model.model import EntityState
from pss.world_model.set_model import (
    FEATURE_DIM,
    SetDeltaMLP,
    entity_features,
    train_set_world_model,
)

DATA = Path("data/v1")


def _states():
    return {
        "a": EntityState(np.array([0.0, 0.0, 0.0]), np.zeros(3)),
        "b": EntityState(np.array([0.1, 0.0, 0.0]), np.zeros(3)),
        "c": EntityState(np.array([0.2, 0.0, 0.0]), np.zeros(3)),
    }


def test_multi_entity_feature_shape_and_target_flag():
    action = {"kind": "push", "target": 1, "params": {"force": 2.0, "direction": [1.0, 0.0]}}
    feat = multi_transition_feature(_states(), action, ["a", "b", "c"], MAX_ENTITIES)
    assert feat.shape == (feature_dim(MAX_ENTITIES),)
    assert np.isfinite(feat).all()


def test_set_entity_feature_includes_target_relation():
    poses = np.array(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.08, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ]
    )
    velocities = np.zeros((3, 3))
    action = {"kind": "push", "target": 1, "params": {"force": 2.0, "direction": [1.0, 0.0]}}
    feats = entity_features(poses, velocities, ["a", "b", "c"], action, MAX_ENTITIES)
    assert feats.shape == (MAX_ENTITIES, FEATURE_DIM)
    # Entity 0 sees the target 0.08 m away and inside the contact band.
    assert feats[0, -2] == pytest.approx(0.08, abs=1e-6)
    assert feats[0, -1] == 1.0
    assert feats[1, -1] == 0.0


def test_models_forward_shapes():
    multi = MultiDeltaMLP(feature_dim(MAX_ENTITIES), MAX_ENTITIES, hidden=16)
    out = multi(torch.zeros(2, feature_dim(MAX_ENTITIES)))
    assert out.shape == (2, MAX_ENTITIES, 4)

    set_model = SetDeltaMLP(FEATURE_DIM, hidden=16)
    out = set_model(torch.zeros(6, FEATURE_DIM))
    assert out.shape == (6, 4)


@pytest.mark.skipif(not DATA.exists(), reason="run dataset generation first")
def test_set_world_model_training_smoke(tmp_path):
    metrics = train_set_world_model(DATA, tmp_path, epochs=2, hidden=16, seed=0)
    assert metrics["n_train"] > 0
    assert metrics["n_val"] > 0
    assert "learned" in metrics["one_step_pos_error"]
    assert (tmp_path / "world_model.pt").exists()
