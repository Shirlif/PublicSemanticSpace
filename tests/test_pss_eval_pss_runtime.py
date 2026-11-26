"""PSS runtime adapter and extended-property tests (synthetic stream)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pss_eval.adapters.pss_runtime import PSSRuntimeAdapter
from pss_eval.fixtures import make_gaussian_stream
from pss_eval.properties import (
    BudgetProperty,
    ContractProperty,
    DeterminismProperty,
    IdentityRetentionProperty,
    LatencyProperty,
    SwapRollbackProperty,
)
from pss_eval.protocol import capabilities_of, state_digest, state_size_bytes

REPO_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = REPO_ROOT / "pss_eval" / "adapters" / "pss_runtime.py"


def _factory(dim: int = 16, budget_ms: float | None = 50.0):
    return lambda: PSSRuntimeAdapter(
        input_dim=dim, public_dim=dim, budget_ms=budget_ms
    )


def _drive(system, stream, *, seed: int):
    system.reset(seed=seed)
    predictions = []
    for sample in stream:
        system.ingest(sample.observation)
        predictions.append(system.step())
    return predictions


def test_adapter_exposes_the_full_contract():
    system = PSSRuntimeAdapter(input_dim=16, public_dim=16)
    assert capabilities_of(system) == frozenset(
        {"snapshot", "swap", "audit", "identity", "budget"}
    )
    # A fresh adapter has an audit log only after it commits something.
    system.reset(seed=0)
    stream = make_gaussian_stream(n=4, dim=16, seed=0)
    _drive(system, stream, seed=0)
    assert len(system.audit()) > 0
    assert all(isinstance(entry, dict) for entry in system.audit())


def test_adapter_is_deterministic_and_resets_exactly():
    stream = make_gaussian_stream(n=8, dim=16, seed=0)
    left = PSSRuntimeAdapter(input_dim=16, public_dim=16)
    right = PSSRuntimeAdapter(input_dim=16, public_dim=16)
    preds_left = _drive(left, stream, seed=1)
    preds_right = _drive(right, stream, seed=1)
    assert [p.observation_id for p in preds_left] == [
        sample.observation.observation_id for sample in stream
    ]
    assert all(
        np.array_equal(a.embedding, b.embedding) for a, b in zip(preds_left, preds_right)
    )
    assert state_digest(left.state()) == state_digest(right.state())

    left.reset(seed=1)
    fresh = PSSRuntimeAdapter(input_dim=16, public_dim=16)
    fresh.reset(seed=1)
    assert state_digest(left.state()) == state_digest(fresh.state())


def test_state_is_small_and_finite():
    stream = make_gaussian_stream(n=12, dim=16, seed=0)
    system = PSSRuntimeAdapter(input_dim=16, public_dim=16)
    _drive(system, stream, seed=0)
    state = system.state()
    assert isinstance(state, dict)
    assert state_size_bytes(state) <= 256 * 1024


def test_contract_property_passes():
    stream = make_gaussian_stream(n=8, dim=16, seed=0)
    result = ContractProperty().evaluate(_factory(), stream, seed=0)
    assert result.status == "pass"
    assert result.metrics == {
        "swap_accepted": 1.0,
        "invalid_rejected": 1.0,
        "invalid_keeps_state": 1.0,
        "rollback_exact": 1.0,
        "audit_present": 1.0,
    }


def test_swap_rollback_property_passes():
    stream = make_gaussian_stream(n=8, dim=16, seed=0)
    result = SwapRollbackProperty().evaluate(_factory(), stream, seed=0)
    assert result.status == "pass"
    assert result.metrics["rollback_exact"] == 1.0
    assert result.metrics["post_rollback_swap"] == 1.0


def test_budget_property_passes_and_records_latency():
    stream = make_gaussian_stream(n=12, dim=16, seed=0)
    result = BudgetProperty().evaluate(_factory(budget_ms=50.0), stream, seed=0)
    assert result.status == "pass"
    assert result.metrics["frames"] == len(stream)
    assert result.metrics["elapsed_p99_ms"] <= 52.0


def test_budget_property_skips_when_disabled():
    stream = make_gaussian_stream(n=4, dim=16, seed=0)
    result = BudgetProperty().evaluate(_factory(budget_ms=None), stream, seed=0)
    assert result.status == "skip"


def test_latency_property_reports_percentiles():
    stream = make_gaussian_stream(n=8, dim=16, seed=0)
    result = LatencyProperty().evaluate(_factory(), stream, seed=0)
    assert result.status == "pass"
    assert {"p50_ms", "p99_ms", "max_ms"} <= set(result.metrics)


def test_identity_retention_skips_without_harness_labels():
    stream = make_gaussian_stream(n=8, dim=16, seed=0)
    result = IdentityRetentionProperty().evaluate(_factory(), stream, seed=0)
    assert result.status == "skip"
    assert "identity labels" in result.notes


def test_determinism_property_passes():
    stream = make_gaussian_stream(n=6, dim=16, seed=0)
    result = DeterminismProperty().evaluate(_factory(), stream, seed=0)
    assert result.status == "pass"


def test_adapter_never_reads_parquet():
    """The adapter gets detections from ``Observation.metadata``, not files."""

    import ast

    tree = ast.parse(ADAPTER_PATH.read_text(encoding="utf-8"))
    readers = [
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"read_parquet", "read_csv", "read_json"}
    ]
    assert readers == []
