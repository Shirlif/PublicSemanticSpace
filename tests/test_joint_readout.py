import numpy as np
import pytest

pytest.importorskip("sklearn")

from pss.prototypes.joint_readout import JointReadout, recognize_joint


def _synthetic(n_per_class=40, dim=8, n_classes=4, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.normal(scale=3.0, size=(n_classes, dim))
    Z = {m: [] for m in ("vision", "tactile", "geometry")}
    y = []
    for cls in range(n_classes):
        for modality in Z:
            offset = rng.normal(scale=2.0, size=dim)
            Z[modality].append(
                means[cls] + offset + 0.1 * rng.normal(size=(n_per_class, dim))
            )
        y.extend([cls] * n_per_class)
    return (
        {m: np.concatenate(v, axis=0) for m, v in Z.items()},
        np.asarray(y, dtype=np.int64),
    )


def test_joint_readout_fits_all_masks():
    Z, y = _synthetic()
    classes = [f"c{i}" for i in range(4)]
    readout = JointReadout.fit(Z, y, classes)
    assert set(readout.models) == {
        ("vision",),
        ("tactile",),
        ("geometry",),
        ("vision", "tactile"),
        ("vision", "geometry"),
        ("tactile", "geometry"),
        ("vision", "tactile", "geometry"),
    }
    scores = readout.scores_batch(Z)
    assert scores.shape == (len(y), 4)
    assert float((scores.argmax(axis=1) == y).mean()) > 0.95


def test_joint_readout_missing_modalities_use_mask_model():
    Z, y = _synthetic()
    readout = JointReadout.fit(Z, y, [f"c{i}" for i in range(4)])
    query = {m: Z[m][0] for m in ("vision",)}
    scores = readout.scores(query)
    assert scores.shape == (4,)
    assert readout.predict(query) == "c0"


def test_joint_readout_add_and_update_class():
    Z, y = _synthetic()
    readout = JointReadout.fit(Z, y, [f"c{i}" for i in range(4)])
    new_query = {m: Z[m][0] + 20.0 for m in Z}
    readout.add_class("new", new_query)
    assert readout.predict(new_query) == "new"

    moved = {m: Z[m][1] + 20.0 for m in Z}
    readout.update_class("new", moved)
    assert readout.predict(moved) == "new"


def test_joint_readout_save_load_roundtrip(tmp_path):
    Z, y = _synthetic()
    readout = JointReadout.fit(Z, y, [f"c{i}" for i in range(4)])
    readout.threshold = -10.0
    path = tmp_path / "joint.npz"
    readout.save(path)
    loaded = JointReadout.load(path)
    assert loaded.classes == readout.classes
    assert loaded.threshold == pytest.approx(-10.0)
    np.testing.assert_allclose(loaded.scores_batch(Z), readout.scores_batch(Z))


def test_recognize_joint_returns_probabilities_and_conflict():
    Z, y = _synthetic()
    readout = JointReadout.fit(Z, y, [f"c{i}" for i in range(4)])
    query = {m: Z[m][0] for m in Z}
    result = recognize_joint(query, readout, top_k=3)
    assert result.ranked
    assert 0.0 <= result.ranked[0][1] <= 1.0
    assert not result.unknown
    assert result.candidates == readout.classes
