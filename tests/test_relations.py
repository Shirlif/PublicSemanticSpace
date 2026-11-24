import numpy as np

from pss.binding import ROLE_NAMES, RelationEncoder, relative_error


def _unit(rng, d=128):
    v = rng.normal(size=d)
    return v / np.linalg.norm(v)


def test_hadamard_parts_decode_exactly():
    rng = np.random.default_rng(0)
    enc = RelationEncoder(128, seed=1)
    truth = (_unit(rng), _unit(rng), _unit(rng))
    rel = enc.encode(*truth)
    rel.validate()
    for role, true in zip(ROLE_NAMES, truth):
        assert np.allclose(enc.decode(rel.parts, role), true, atol=1e-9)


def test_fact_is_not_separable():
    rng = np.random.default_rng(1)
    enc = RelationEncoder(128, seed=1)
    truth = (_unit(rng), _unit(rng), _unit(rng))
    rel = enc.encode(*truth)
    for role, true in zip(ROLE_NAMES, truth):
        err = relative_error(true, enc.decode_fact(rel.fact, role))
        # Cross-talk from the other two roles is O(1); this is why the summed
        # fact is a lookup shortcut, never a storage format.
        assert err > 0.5


def test_crosstalk_small_for_hadamard():
    rng = np.random.default_rng(2)
    enc = RelationEncoder(128, seed=1)
    rel = enc.encode(_unit(rng), _unit(rng), _unit(rng))
    assert enc.crosstalk(rel.parts) < 0.5


def test_hrr_relation_parts_are_approximate():
    rng = np.random.default_rng(3)
    enc = RelationEncoder(128, seed=1, scheme="hrr")
    truth = (_unit(rng), _unit(rng), _unit(rng))
    rel = enc.encode(*truth)
    rel.validate()
    errs = [relative_error(true, enc.decode(rel.parts, role)) for role, true in zip(ROLE_NAMES, truth)]
    assert np.mean(errs) > 0.2
