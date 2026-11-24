import numpy as np

from pss.prototypes import PrototypeBank, conflict_aware_recognize


def _bank():
    return PrototypeBank(
        ["a", "b"],
        {
            "vision": np.stack([np.array([1.0, 0.0]), np.array([0.0, 1.0])]),
            "geometry": np.stack([np.array([1.0, 0.0]), np.array([0.0, 1.0])]),
        },
        {
            "a": {"vision": {"sample_count": 1}, "geometry": {"sample_count": 1}},
            "b": {"vision": {"sample_count": 1}, "geometry": {"sample_count": 1}},
        },
    )


def test_conflict_aware_agreeing_modalities():
    decision = conflict_aware_recognize(
        {"vision": np.array([1.0, 0.0]), "geometry": np.array([1.0, 0.0])},
        _bank(),
        {"vision": 1.0, "geometry": 1.0},
    )
    assert decision.category == "a"
    assert not decision.conflict


def test_conflict_aware_disagreement_abstains():
    decision = conflict_aware_recognize(
        {"vision": np.array([1.0, 0.0]), "geometry": np.array([0.0, 1.0])},
        _bank(),
        {"vision": 1.0, "geometry": 1.0},
    )
    assert decision.category is None
    assert decision.conflict


def test_conflict_aware_zero_modality_is_ignored():
    decision = conflict_aware_recognize(
        {"vision": np.array([1.0, 0.0]), "geometry": np.zeros(2)},
        _bank(),
        {"vision": 1.0, "geometry": 1.0},
    )
    assert decision.category == "a"
    assert not decision.conflict
