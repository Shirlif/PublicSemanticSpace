import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("scipy")

from pss.prototypes import (
    PrototypeBank,
    build_tree,
    load_tree,
    pre_review,
    recognize,
    save_tree,
)


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def _bank() -> PrototypeBank:
    rng = np.random.default_rng(0)
    categories = ["a", "b", "c", "d"]
    protos = {
        "vision": np.stack([_unit(rng.normal(size=8)) for _ in categories]),
        "geometry": np.stack([_unit(rng.normal(size=8)) for _ in categories]),
    }
    stats = {
        c: {m: {"sample_count": 10, "variance": 0.1} for m in protos}
        for c in categories
    }
    return PrototypeBank(categories, protos, stats)


def test_recognize_top1():
    bank = _bank()
    query = {
        "vision": bank.protos["vision"][0],
        "geometry": bank.protos["geometry"][0],
    }
    res = recognize(
        query, bank, use_tree=False, weights={"vision": 1.0, "geometry": 1.0}
    )
    assert res.ranked[0][0] == "a"
    assert not res.unknown


def test_zero_norm_modality_is_ignored():
    bank = _bank()
    query = {
        "vision": bank.protos["vision"][0],
        "geometry": np.zeros(8),
    }
    res = recognize(
        query, bank, use_tree=False, weights={"vision": 1.0, "geometry": 1.0}
    )
    assert res.ranked[0][0] == "a"
    assert "geometry" not in res.per_modality["a"]


def test_conflict_flag():
    bank = _bank()
    query = {
        "vision": bank.protos["vision"][0],
        "geometry": bank.protos["geometry"][1],
    }
    res = recognize(
        query, bank, use_tree=False, weights={"vision": 1.0, "geometry": 1.0}
    )
    assert res.conflict


def test_open_set_unknown():
    bank = _bank()
    query = {
        "vision": -_unit(bank.protos["vision"].sum(0)),
        "geometry": -_unit(bank.protos["geometry"].sum(0)),
    }
    res = recognize(
        query,
        bank,
        use_tree=False,
        weights={"vision": 1.0, "geometry": 1.0},
        threshold=0.0,
    )
    assert res.unknown


def test_tree_structure_and_save(tmp_path):
    bank = _bank()
    tree = build_tree(bank)
    assert len(tree["nodes"]) == 2 * len(bank.categories) - 1
    leaves = [n for n in tree["nodes"].values() if n["category"] is not None]
    assert len(leaves) == len(bank.categories)

    path = tmp_path / "tree.json"
    save_tree(path, tree)
    loaded = load_tree(path)
    assert loaded["root"] == tree["root"]
    assert len(loaded["nodes"]) == len(tree["nodes"])


def test_admission_rejects_near_duplicate():
    bank = _bank()
    proto = bank.protos["vision"][0]
    rng = np.random.default_rng(0)
    samples = np.stack([_unit(proto + 0.001 * rng.normal(size=8)) for _ in range(30)])
    result = pre_review({"vision": samples}, bank)
    assert result.decision in ("reject", "review")
    assert any("inter_distance_low" in r for r in result.reasons)


def test_admission_accepts_far_candidate():
    bank = _bank()
    far = -_unit(bank.protos["vision"].sum(0))
    rng = np.random.default_rng(1)
    samples = np.stack([_unit(far + 0.01 * rng.normal(size=8)) for _ in range(30)])
    result = pre_review({"vision": samples}, bank)
    assert result.decision == "accept"
