import numpy as np
import pytest

from pss.binding import (
    DriftMonitor,
    RebindPolicy,
    build_entity,
    make_slots,
    relative_error,
    unbind_modality,
)


def test_relative_error():
    assert relative_error(np.array([1.0, 0.0]), np.array([1.1, 0.0])) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        relative_error(np.zeros(2), np.ones(2))


def test_policy_trigger():
    policy = RebindPolicy(error_budget=0.10)
    assert not policy.should_rebind(0.10)
    assert policy.should_rebind(0.11)


def test_drift_monitor():
    monitor = DriftMonitor(error_budget=0.10)
    assert not monitor.update("e0", 0.05)
    assert monitor.update("e0", 0.20)
    assert monitor.needs_rebind("e0")
    monitor.clear("e0")
    assert not monitor.needs_rebind("e0")


def test_rebind_restores_from_raw():
    rng = np.random.default_rng(0)
    slot = make_slots("hadamard", 128, 1)[0]
    raw = rng.normal(size=128)
    raw /= np.linalg.norm(raw)
    entity = build_entity("obj0", {"vision": raw}, slot, slot_index=0)
    entity.bound["vision"] = entity.bound["vision"] + 0.5 * rng.normal(size=128)

    policy = RebindPolicy()
    repaired = policy.rebind(entity, slot, slot_index=0)
    assert np.allclose(unbind_modality(repaired, "vision", slot), raw, atol=1e-9)
