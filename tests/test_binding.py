import numpy as np
import pytest

from pss.binding import bind, cosine, hadamard, make_slots, slot_orthogonality, unbind
from pss.binding.ops import _sign_preserving


def test_hadamard_matrix_orthogonal():
    h = hadamard(128)
    assert np.allclose(h @ h.T, 128 * np.eye(128))


def test_hadamard_slots_orthonormal():
    slots = make_slots("hadamard", 128, 4)
    assert np.allclose(slots @ slots.T, np.eye(4), atol=1e-12)
    assert np.allclose(np.linalg.norm(slots, axis=1), 1.0)
    assert slot_orthogonality(slots) < 1e-12


@pytest.mark.parametrize("scheme", ["hadamard", "rademacher", "gaussian"])
def test_roundtrip_exact_without_noise(scheme):
    rng = np.random.default_rng(0)
    slot = make_slots(scheme, 128, 1, seed=1)[0]
    x = rng.normal(size=128)
    x /= np.linalg.norm(x)
    y = unbind(bind(x, slot), slot)
    assert np.allclose(y, x, atol=1e-10)


def test_sign_preserving_guard_does_not_flip_negative():
    slot = np.array([-1e-9, 1e-9, 0.0, 0.5])
    safe = _sign_preserving(slot, eps=1e-6)
    assert safe[0] < 0
    assert safe[1] > 0
    assert safe[2] > 0  # zero maps to +eps by convention


def test_near_zero_component_roundtrip_with_tiny_eps():
    slot = np.array([-1e-9, 1e-9, 0.5])
    x = np.array([3.0, -2.0, 1.0])
    y = unbind(bind(x, slot), slot, eps=1e-12)
    assert np.allclose(y, x, atol=1e-9)


def test_sum_is_not_separable():
    rng = np.random.default_rng(1)
    slot = make_slots("hadamard", 128, 1)[0]
    a = rng.normal(size=128)
    b = rng.normal(size=128)
    recovered = unbind(bind(a, slot) + bind(b, slot), slot)
    assert np.allclose(recovered, a + b, atol=1e-9)
    assert not np.allclose(recovered, a, atol=1e-6)
    assert not np.allclose(recovered, b, atol=1e-6)


def test_hadamard_rebind_no_drift():
    rng = np.random.default_rng(2)
    slot = make_slots("hadamard", 128, 1)[0]
    x0 = rng.normal(size=128)
    x0 /= np.linalg.norm(x0)
    x = x0.copy()
    for _ in range(100):
        x = unbind(bind(x, slot), slot)
    assert np.allclose(x, x0, atol=1e-10)


def test_cosine_safe_for_zero():
    assert cosine(np.zeros(4), np.ones(4)) == 0.0
