"""Adapter tests: E5 flat runtime and transformer-state baseline."""

from __future__ import annotations

import numpy as np
import pytest

from pss_eval.adapters.flat import FlatNonPSSAdapter
from pss_eval.adapters.transformer import TransformerStateAdapter
from pss_eval.fixtures import make_gaussian_stream
from pss_eval.properties import ContractProperty
from pss_eval.protocol import (
    Observation,
    capabilities_of,
    prediction_digest,
    state_digest,
)


def _drive(system, stream, *, seed: int):
    system.reset(seed=seed)
    predictions = []
    for sample in stream:
        system.ingest(sample.observation)
        predictions.append(system.step())
    return predictions


@pytest.fixture()
def stream():
    return make_gaussian_stream(n=12, dim=32, seed=0)


# ── E5 flat adapter ────────────────────────────────────────────────────────


def test_flat_adapter_runs_and_is_deterministic(stream):
    system = FlatNonPSSAdapter(dim=32)
    predictions = _drive(system, stream, seed=1)

    assert len(predictions) == len(stream)
    assert all(
        prediction.observation_id == sample.observation.observation_id
        for prediction, sample in zip(predictions, stream)
    )
    assert all(prediction.label is not None for prediction in predictions)

    other = FlatNonPSSAdapter(dim=32)
    repeated = _drive(other, stream, seed=1)
    assert [prediction_digest(p) for p in predictions] == [
        prediction_digest(p) for p in repeated
    ]
    assert state_digest(system.state()) == state_digest(other.state())


def test_flat_adapter_is_contract_free(stream):
    system = FlatNonPSSAdapter(dim=32)
    assert capabilities_of(system) == frozenset({"snapshot", "swap", "audit"})

    # E5 accepts every module, including an invalid one.
    assert system.swap({"modality": "vision", "kind": "invalid_dim", "valid": False})
    # No rollback, no audit.
    assert system.rollback("anything") is False
    assert system.audit() == []

    result = ContractProperty().evaluate(
        lambda: FlatNonPSSAdapter(dim=32), stream, seed=0
    )
    assert result.status == "fail"
    assert result.metrics == {
        "swap_accepted": 1.0,
        "invalid_rejected": 0.0,
        "invalid_keeps_state": 0.0,
        "rollback_exact": 0.0,
        "audit_present": 0.0,
    }


def test_flat_adapter_rejects_wrong_feature_dim():
    system = FlatNonPSSAdapter(dim=32)
    with pytest.raises(ValueError):
        system.ingest(Observation("s0", "vision", np.zeros(31)))


def test_e5_reference_scorecard_is_still_contract_free():
    """Parity guard: the wrapped E5 reference keeps its negative metrics."""

    from pss.property.nonpss import run_nonpss_baseline

    metrics = run_nonpss_baseline(seed=0, dim=32)["metrics"]
    assert metrics["SRR"] == 0.0  # invalid modules accepted
    assert metrics["RBE"] == 0.0  # no rollback
    assert metrics["AUD"] == 0.0  # no audit


# ── transformer-state adapter ──────────────────────────────────────────────


def test_transformer_adapter_runs_and_is_deterministic(stream):
    kwargs = {"dim": 32, "model_dim": 16, "n_heads": 2, "context": 8}
    system = TransformerStateAdapter(**kwargs)
    predictions = _drive(system, stream, seed=2)

    assert len(predictions) == len(stream)
    assert all(prediction.label is None for prediction in predictions)
    assert all(prediction.embedding.shape == (16,) for prediction in predictions)

    state = system.state()
    assert state["context_len"] == 8  # stream longer than the context window
    assert np.all(np.isfinite(state["hidden"]))

    other = TransformerStateAdapter(**kwargs)
    repeated = _drive(other, stream, seed=2)
    assert [prediction_digest(p) for p in predictions] == [
        prediction_digest(p) for p in repeated
    ]
    assert state_digest(state) == state_digest(other.state())


def test_transformer_adapter_has_no_contract_capabilities(stream):
    system = TransformerStateAdapter(dim=32)
    assert capabilities_of(system) == frozenset()

    result = ContractProperty().evaluate(
        lambda: TransformerStateAdapter(dim=32), stream, seed=0
    )
    assert result.status == "skip"
    assert "snapshot" in result.notes
    assert "swap" in result.notes
    assert "audit" in result.notes


def test_transformer_adapter_rejects_wrong_feature_dim():
    system = TransformerStateAdapter(dim=32)
    with pytest.raises(ValueError):
        system.ingest(Observation("s0", "vision", np.zeros(16)))
