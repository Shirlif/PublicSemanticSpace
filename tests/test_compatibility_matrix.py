"""W7 compatibility matrix: one test per row of the plan's §6 table."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from pss.contracts import PUBLIC_DIM, Entity
from pss.contracts.versioning import IncompatibleArtifactError, assert_compatible
from pss.memory.perception import ModalityObservation, Observation
from pss.prototypes.bank import PrototypeBank, build_bank
from pss.prototypes.joint_readout import JointReadout
from pss.prototypes.recognize import recognize

CLASSES = ["a", "b", "c"]
DIM = PUBLIC_DIM
MODS = ("vision", "tactile", "geometry")


def _data(seed: int = 0, modalities=MODS):
    rng = np.random.default_rng(seed)
    centers = {m: rng.normal(size=(len(CLASSES), DIM)) for m in modalities}
    Z = {m: [] for m in modalities}
    categories = []
    for ci, category in enumerate(CLASSES):
        for _ in range(12):
            categories.append(category)
            for m in modalities:
                Z[m].append(centers[m][ci] + rng.normal(scale=0.1, size=DIM))
    Z = {m: np.asarray(rows) for m, rows in Z.items()}
    feats = SimpleNamespace(categories=categories, sample_ids=list(range(len(categories))), instances=list(range(len(categories))))
    train = [i for i, c in enumerate(categories) if (i % 12) < 8]
    return feats, Z, train


def _bank(feats, Z, train, train_idx=None, **kwargs):
    return build_bank(feats, Z, train_idx or train, CLASSES, **kwargs)


def _obs(**kwargs):
    defaults = dict(
        gt_id="obj0",
        pos=np.zeros(3),
        quat=np.array([1.0, 0.0, 0.0, 0.0]),
        appearance=np.zeros(8),
        raw={},
        confidence=0.9,
        category="a",
    )
    defaults.update(kwargs)
    return Observation(**defaults)


def test_01_missing_single_modality_still_recognizes():
    feats, Z, train = _data()
    bank = _bank(feats, Z, train)
    query = {"vision": Z["vision"][-1]}
    result = recognize(query, bank, weights={"vision": 1.0}, use_tree=False)
    assert result.ranked
    assert result.modality_top and set(result.modality_top) <= {"vision"}


def test_02_missing_multiple_modalities_skips_conflict():
    feats, Z, train = _data()
    bank = _bank(feats, Z, train)
    result = recognize({"tactile": Z["tactile"][-1]}, bank, use_tree=False)
    assert result.ranked
    assert result.conflict is False


def test_03_async_within_budget():
    obs = _obs()
    obs.add_stream(
        ModalityObservation("tactile", np.ones(4), t_capture=9.99, t_arrival=10.0),
        t_ref=10.0,
    )
    obs.refresh_staleness({"tactile": 20.0})
    assert obs.stale_modalities == []


def test_04_async_over_budget_marks_stale():
    obs = _obs()
    obs.add_stream(
        ModalityObservation("tactile", np.ones(4), t_capture=9.0, t_arrival=10.0),
        t_ref=10.0,
    )
    assert obs.stale_modalities == ["tactile"]
    assert "tactile" not in obs.valid_raw()


def test_05_clock_reversal_is_a_contract_error():
    with pytest.raises(ValueError):
        ModalityObservation("vision", np.ones(4), t_capture=10.0, t_arrival=9.0).validate()


def test_06_scale_drift_stays_finite():
    feats, Z, train = _data()
    bank = _bank(feats, Z, train)
    result = recognize({m: Z[m][-1] * 2.0 for m in MODS}, bank, use_tree=False)
    assert result.ranked
    assert np.isfinite(result.ranked[0][1])


def test_07_saturation_stays_finite():
    feats, Z, train = _data()
    bank = _bank(feats, Z, train)
    saturated = {m: np.ones(DIM, dtype=np.float32) for m in MODS}
    result = recognize(saturated, bank, use_tree=False)
    assert result.ranked
    assert all(np.isfinite(score) for _, score in result.ranked)


def test_08_nan_is_rejected_by_the_contract():
    entity = Entity("obj0", raw={"vision": np.full(DIM, np.nan)})
    with pytest.raises(ValueError):
        entity.validate()


def test_09_wrong_dim_is_rejected():
    entity = Entity("obj0", raw={"vision": np.ones(DIM - 1)})
    with pytest.raises(ValueError):
        entity.validate()


def test_10_zero_vector_is_treated_as_missing():
    feats, Z, train = _data()
    bank = _bank(feats, Z, train)
    result = recognize({"vision": np.zeros(DIM)}, bank, use_tree=False)
    assert result.ranked == []
    assert result.unknown is True


def test_11_new_dummy_modality_needs_no_core_edit():
    modalities = MODS + ("audio",)
    feats, Z, train = _data(modalities=modalities)
    bank = _bank(feats, Z, train)
    model = JointReadout.fit(
        {m: Z[m][train] for m in modalities},
        np.array([CLASSES.index(feats.categories[i]) for i in train]),
        CLASSES,
        modalities=modalities,
        mask_strategy="single_plus_all",
    )
    assert len(model.models) == 5
    assert bank.protos["audio"].shape == (len(CLASSES), DIM)


def test_12_old_bank_artifact_loads(tmp_path):
    path = tmp_path / "legacy.npz"
    np.savez_compressed(
        path,
        categories_json=np.array(json.dumps(CLASSES)),
        stats_json=np.array(json.dumps({c: {} for c in CLASSES})),
        version=np.array("legacy"),
        proto_vision=np.eye(len(CLASSES), DIM, dtype=np.float32),
    )
    bank = PrototypeBank.load(path)
    assert bank.space_version == "0.1.1"
    assert bank.has_prototype("vision", "a")


def test_13_incompatible_version_is_rejected():
    with pytest.raises(IncompatibleArtifactError):
        assert_compatible("1.0.0", "0.2.0", artifact_name="bank")
