import numpy as np

from pss.binding.hrr import (
    hrr_bind,
    hrr_unbind,
    make_roles,
    naive_correlation,
    naive_convolution,
)


def test_fft_matches_naive_convolution():
    rng = np.random.default_rng(0)
    for d in (8, 16):
        a = rng.normal(size=d)
        b = rng.normal(size=d)
        assert np.allclose(hrr_bind(a, b), naive_convolution(a, b), atol=1e-10)
        c = naive_convolution(a, b)
        assert np.allclose(hrr_unbind(c, b), naive_correlation(c, b), atol=1e-10)


def test_delta_identity():
    d = 32
    rng = np.random.default_rng(1)
    b = rng.normal(size=d)
    delta = np.zeros(d)
    delta[0] = 1.0
    assert np.allclose(hrr_bind(delta, b), b)


def test_roundtrip_is_approximate():
    d = 128
    rng = np.random.default_rng(0)
    a = rng.normal(size=d)
    a /= np.linalg.norm(a)
    b = rng.normal(size=d)
    b /= np.linalg.norm(b)
    a2 = hrr_unbind(hrr_bind(a, b), b)
    cos = float(np.dot(a2, a) / np.linalg.norm(a2))
    err = float(np.linalg.norm(a2 - a) / np.linalg.norm(a))
    # HRR is not exact: cosine is well below 1 and the relative error is O(1).
    assert 0.5 < cos < 0.95
    assert err < 1.5


def test_roles_are_unit_norm():
    roles = make_roles(128, 3, seed=0)
    assert np.allclose(np.linalg.norm(roles, axis=1), 1.0)
