import numpy as np
import pytest

pytest.importorskip("torch")

from pss.learning.review import (
    aggregate_proposals,
    calibrated_pre_review_thresholds,
    load_human_decision,
    review_proposal,
    write_human_decision,
)
from pss.learning.signing import generate_keypair
from pss.prototypes import PrototypeBank


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def test_aggregate_proposals_weighted_mean():
    p1 = {
        "category": "x",
        "sample_count": 10,
        "vectors": {"vision": [1.0, 0.0, 0.0, 0.0]},
        "variance": {"vision": 0.1},
    }
    p2 = {
        "category": "x",
        "sample_count": 30,
        "vectors": {"vision": [0.0, 1.0, 0.0, 0.0]},
        "variance": {"vision": 0.2},
    }
    out = aggregate_proposals([p1, p2])[0]
    assert out["sample_count"] == 40
    assert out["proposals"] == 2
    assert np.isclose(np.linalg.norm(out["vectors"]["vision"]), 1.0)


def test_human_decision_signing_and_wrong_key(tmp_path):
    private_key, public_key = generate_keypair()
    path = tmp_path / "human.json"
    write_human_decision(
        path, "c", "reviewer", "accept", private_key=private_key
    )
    loaded = load_human_decision(path, trusted_public_key=public_key)
    assert loaded["decision"] == "accept"

    _, wrong_public = generate_keypair()
    with pytest.raises(Exception):
        load_human_decision(path, trusted_public_key=wrong_public)

    payload = loaded
    payload["decision"] = "reject"
    import json

    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(Exception):
        load_human_decision(path, trusted_public_key=public_key)


def test_review_accepts_separable_candidate():
    rng = np.random.default_rng(0)
    cats = ["a", "b"]
    protos = {"vision": np.stack([_unit(rng.normal(size=4)) for _ in cats])}
    bank = PrototypeBank(
        cats,
        protos,
        {c: {"vision": {"sample_count": 10, "variance": 0.1}} for c in cats},
    )
    target = _unit([0.0, 0.0, 0.0, 1.0])
    samples = np.stack([_unit(target + 0.01 * rng.normal(size=4)) for _ in range(30)])
    holdout = np.stack([_unit(target + 0.01 * rng.normal(size=4)) for _ in range(12)])
    candidate = {"vision": samples.mean(0) / np.linalg.norm(samples.mean(0))}
    result = review_proposal(
        {"vision": samples},
        candidate,
        {"vision": holdout},
        bank,
        "c",
        sample_count=30,
    )
    assert result.accepted
    assert result.regression["new_acc"] > 0.8


def test_review_rejects_insufficient_samples():
    rng = np.random.default_rng(1)
    bank = PrototypeBank(
        ["a"],
        {"vision": np.stack([_unit(rng.normal(size=4))])},
        {"a": {"vision": {"sample_count": 10, "variance": 0.1}}},
    )
    samples = np.stack([_unit([0, 0, 0, 1]) for _ in range(5)])
    result = review_proposal(
        {"vision": samples},
        {"vision": samples.mean(0)},
        {"vision": samples},
        bank,
        "b",
        sample_count=5,
    )
    assert not result.accepted
    assert "insufficient_samples" in result.reasons


def test_calibrated_pre_review_thresholds_are_below_min_nn():
    bank = PrototypeBank(
        ["a", "b"],
        {"vision": np.stack([_unit([1.0, 0.0]), _unit([0.8, 0.2])])},
        {
            "a": {"vision": {"sample_count": 10, "variance": 0.1}},
            "b": {"vision": {"sample_count": 10, "variance": 0.1}},
        },
    )
    thresholds = calibrated_pre_review_thresholds(bank)
    # Both prototypes are within one nearest-neighbor distance of each other.
    min_nn = 1.0 - float(np.dot(_unit([1.0, 0.0]), _unit([0.8, 0.2])))
    assert thresholds["vision"] == pytest.approx(max(0.01, 0.15 * min_nn))


def test_human_gate_accepts_and_vetoes(tmp_path):
    rng = np.random.default_rng(2)
    bank = PrototypeBank(
        ["a", "b"],
        {"vision": np.stack([_unit(rng.normal(size=4)) for _ in range(2)])},
        {c: {"vision": {"sample_count": 10, "variance": 0.1}} for c in ("a", "b")},
    )
    target = _unit([0.0, 0.0, 0.0, 1.0])
    samples = np.stack([_unit(target + 0.01 * rng.normal(size=4)) for _ in range(30)])
    holdout = np.stack([_unit(target + 0.01 * rng.normal(size=4)) for _ in range(12)])
    candidate = {"vision": samples.mean(0) / np.linalg.norm(samples.mean(0))}

    accept_path = tmp_path / "human_accept.json"
    write_human_decision(accept_path, "c", "reviewer", "accept")
    accept = review_proposal(
        {"vision": samples},
        candidate,
        {"vision": holdout},
        bank,
        "c",
        sample_count=30,
        human_decision=load_human_decision(accept_path),
        require_human=True,
    )
    assert accept.accepted
    assert accept.human_decision == "accept"

    reject_path = tmp_path / "human_reject.json"
    write_human_decision(reject_path, "c", "reviewer", "reject")
    reject = review_proposal(
        {"vision": samples},
        candidate,
        {"vision": holdout},
        bank,
        "c",
        sample_count=30,
        human_decision=load_human_decision(reject_path),
        require_human=True,
    )
    assert not reject.accepted
    assert "human_rejected" in reject.reasons

    pending = review_proposal(
        {"vision": samples},
        candidate,
        {"vision": holdout},
        bank,
        "c",
        sample_count=30,
        require_human=True,
    )
    assert not pending.accepted
    assert "human_review_pending" in pending.reasons
