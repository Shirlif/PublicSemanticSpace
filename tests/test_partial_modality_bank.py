import json
from types import SimpleNamespace

import numpy as np
import pytest

from pss.prototypes.bank import PrototypeBank, build_bank
from pss.prototypes.recognize import recognize

CLASSES = ["a", "b", "c"]
DIM = 8


def _fixture(seed: int = 0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(len(CLASSES), DIM))
    Z = {m: [] for m in ("vision", "tactile", "geometry")}
    categories, sample_ids, instances = [], [], []
    for ci, category in enumerate(CLASSES):
        for v in range(6):
            categories.append(category)
            sample_ids.append(f"{category}_v{v}")
            instances.append(f"{category}_i{v}")
            for modality in Z:
                Z[modality].append(centers[ci] + rng.normal(scale=0.2, size=DIM))
    Z = {m: np.asarray(rows, dtype=np.float32) for m, rows in Z.items()}
    feats = SimpleNamespace(
        categories=categories, sample_ids=sample_ids, instances=instances
    )
    # 4 train / 2 test views per class; tactile lacks class c, geometry lacks b.
    train_vision = [i for i, c in enumerate(categories) if (i % 6) < 4]
    train_tactile = [
        i for i, c in enumerate(categories) if (i % 6) < 4 and c in {"a", "b"}
    ]
    train_geometry = [
        i for i, c in enumerate(categories) if (i % 6) < 4 and c in {"a", "c"}
    ]
    train_idx = {
        "vision": train_vision,
        "tactile": train_tactile,
        "geometry": train_geometry,
    }
    return feats, Z, train_idx


def test_partial_coverage_is_recorded():
    feats, Z, train_idx = _fixture()
    bank = build_bank(feats, Z, train_idx, CLASSES, version="partial_v1")
    assert bank.coverage["tactile"]["c"] == 0
    assert bank.coverage["geometry"]["b"] == 0
    assert bank.missing["tactile"] == ["c"]
    assert bank.missing["geometry"] == ["b"]
    assert not bank.has_prototype("tactile", "c")
    assert bank.has_prototype("tactile", "a")
    assert bank.has_prototype("geometry", "c")
    assert not bank.has_prototype("geometry", "b")


def test_recognition_skips_missing_modality_pairs():
    feats, Z, train_idx = _fixture()
    bank = build_bank(feats, Z, train_idx, CLASSES, version="partial_v1")
    test_c = [i for i, c in enumerate(feats.categories) if c == "c" and (i % 6) >= 4]
    query = {m: Z[m][test_c[0]] for m in ("vision", "tactile")}
    result = recognize(query, bank, weights={"vision": 1.0, "tactile": 0.5}, use_tree=False)
    assert result.ranked
    assert result.ranked[0][0] == "c"

    tactile_only = {"tactile": Z["tactile"][test_c[0]]}
    result = recognize(tactile_only, bank, weights={"tactile": 1.0}, use_tree=False)
    assert "c" not in [category for category, _ in result.ranked]


def test_raise_policy_preserves_strict_behaviour():
    feats, Z, train_idx = _fixture()
    with pytest.raises(ValueError):
        build_bank(
            feats,
            Z,
            train_idx,
            CLASSES,
            missing_policy={"tactile": "raise", "geometry": "skip"},
        )


def test_impute_policy_fills_and_marks():
    feats, Z, train_idx = _fixture()
    bank = build_bank(
        feats,
        Z,
        train_idx,
        CLASSES,
        missing_policy={"tactile": "impute", "geometry": "skip"},
    )
    assert bank.imputed["tactile"] == ["c"]
    assert bank.has_prototype("tactile", "c")
    assert np.linalg.norm(bank.protos["tactile"][CLASSES.index("c")]) == pytest.approx(1.0)


def test_save_load_roundtrip(tmp_path):
    feats, Z, train_idx = _fixture()
    bank = build_bank(feats, Z, train_idx, CLASSES, version="partial_v1")
    path = tmp_path / "bank.npz"
    bank.save(path)
    loaded = PrototypeBank.load(path)
    assert loaded.missing == bank.missing
    assert loaded.coverage == bank.coverage
    assert loaded.imputed == bank.imputed
    np.testing.assert_allclose(loaded.protos["tactile"], bank.protos["tactile"])


def test_legacy_bank_without_coverage_loads(tmp_path):
    path = tmp_path / "legacy.npz"
    protos = {m: np.eye(len(CLASSES), DIM, dtype=np.float32) for m in ("vision", "tactile")}
    payload = {
        "categories_json": np.array(json.dumps(CLASSES)),
        "stats_json": np.array(json.dumps({c: {} for c in CLASSES})),
        "version": np.array("legacy"),
    }
    for modality, arr in protos.items():
        payload[f"proto_{modality}"] = arr
    np.savez_compressed(path, **payload)
    bank = PrototypeBank.load(path)
    assert bank.coverage == {}
    assert bank.missing == {}
    assert bank.has_prototype("vision", "a")
