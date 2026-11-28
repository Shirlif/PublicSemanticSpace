import numpy as np
import pytest

from pss.prototypes.joint_readout import JointReadout, _masks


def _data(n_classes: int = 4, n_per_class: int = 20, dim: int = 4, modalities=("a", "b", "c")):
    rng = np.random.default_rng(0)
    Z = {m: [] for m in modalities}
    y = []
    centers = {m: rng.normal(size=(n_classes, dim)) for m in modalities}
    for cls in range(n_classes):
        for _ in range(n_per_class):
            y.append(cls)
            for m in modalities:
                Z[m].append(centers[m][cls] + rng.normal(scale=0.2, size=dim))
    return {m: np.asarray(rows) for m, rows in Z.items()}, np.asarray(y), [f"c{i}" for i in range(n_classes)]


def test_mask_strategies_enumerate_expected_counts():
    modalities = ("a", "b", "c")
    assert len(list(_masks(modalities, strategy="all"))) == 7
    assert len(list(_masks(modalities, strategy="single_plus_all"))) == 4
    assert len(list(_masks(modalities, strategy="max_size", max_size=2))) == 6
    observed = {("a",), ("a", "b")}
    assert list(_masks(modalities, strategy="observed", observed=observed)) == [
        ("a",),
        ("a", "b"),
    ]
    with pytest.raises(ValueError):
        list(_masks(modalities, strategy="observed"))


def test_four_modalities_avoid_full_combinatorics():
    modalities = ("a", "b", "c", "d")
    Z, y, classes = _data(modalities=modalities)
    full = JointReadout.fit(Z, y, classes, modalities=modalities, mask_strategy="all")
    lean = JointReadout.fit(
        Z, y, classes, modalities=modalities, mask_strategy="single_plus_all"
    )
    assert len(full.models) == 15
    assert len(lean.models) == 5
    assert lean.metadata["mask_strategy"] == "single_plus_all"


def test_fallback_picks_available_mask():
    modalities = ("a", "b", "c")
    Z, y, classes = _data(modalities=modalities)
    model = JointReadout.fit(
        Z, y, classes, modalities=modalities, mask_strategy="single_plus_all"
    )
    query = {"a": Z["a"][0], "b": Z["b"][0], "c": Z["c"][0]}
    # (a, b, c) is trained; (a, c) is not, so it must fall back to a pair or a
    # single modality that was trained.
    mask = model._model_mask(("a", "c"))
    assert mask in model.models
    assert set(mask) <= {"a", "c"}
    assert model.predict(query) in classes


def test_fit_class_centroids_marks_approximate():
    modalities = ("a", "b")
    Z, y, classes = _data(modalities=modalities)
    centroids = {
        m: np.stack([Z[m][y == cls].mean(0) for cls in range(len(classes))])
        for m in modalities
    }
    model = JointReadout.fit_class_centroids(
        centroids, classes, modalities=modalities, mask_strategy="single_plus_all"
    )
    assert model.metadata["approximate"] is True
    assert model.predict({"a": centroids["a"][0], "b": centroids["b"][0]}) == classes[0]


def test_partial_query_hot_update_does_not_crash():
    modalities = ("a", "b", "c")
    Z, y, classes = _data(modalities=modalities)
    model = JointReadout.fit(Z, y, classes, modalities=modalities)
    partial = {"a": Z["a"][0], "b": Z["b"][0]}
    model.add_class("new", partial)
    model.update_class("new", partial)
    assert "new" in model.classes


def test_save_load_preserves_mask_strategy(tmp_path):
    modalities = ("a", "b", "c")
    Z, y, classes = _data(modalities=modalities)
    model = JointReadout.fit(
        Z, y, classes, modalities=modalities, mask_strategy="single_plus_all"
    )
    path = tmp_path / "readout.npz"
    model.save(path)
    loaded = JointReadout.load(path)
    assert loaded.metadata["mask_strategy"] == "single_plus_all"
    assert set(loaded.models) == set(model.models)
