"""Tests for ordered, non-commutative relation operators."""

from __future__ import annotations

import numpy as np
import pytest

from pss.binding.operators import (
    CompositeOperator,
    LowRankOperator,
    RelationOperatorBank,
    SignedPermutationOperator,
)
from pss.binding.protocol import Operator


def _signed(perm, signs, name: str = "signed_perm") -> SignedPermutationOperator:
    return SignedPermutationOperator(
        perm=np.array(perm, dtype=np.int64),
        signs=np.array(signs, dtype=np.float64),
        name=name,
    )


class _PlainOperator(Operator):
    """A semantic operator with no matrix(), used for diagnostics coverage."""

    name = "plain"
    kind = "semantic"

    def apply(self, x):
        return np.asarray(x, dtype=np.float64)


def test_composition_is_non_commutative():
    # (0 1) and (0 2) do not commute.
    p = _signed([1, 0, 2, 3], [1, 1, 1, 1], name="p")
    q = _signed([2, 1, 0, 3], [1, 1, 1, 1], name="q")

    pq = p.compose(q)
    qp = q.compose(p)
    assert pq is not None and qp is not None
    assert pq != qp
    assert pq.signature() != qp.signature()

    x = np.array([0.3, -0.7, 1.1, 2.0])
    assert not np.allclose(pq.apply(x), qp.apply(x))


def test_composition_is_closed():
    p = SignedPermutationOperator.random(12, seed=1)
    q = SignedPermutationOperator.random(12, seed=2)
    composed = p.compose(q)
    assert isinstance(composed, SignedPermutationOperator)
    # Non-signed operators are not accepted by this family.
    low = LowRankOperator(U=np.ones((12, 1)), V=np.ones((12, 1)))
    assert p.compose(low) is None


def test_inverse_roundtrip_exact_and_norm_preserved():
    op = SignedPermutationOperator.random(16, seed=3)
    x = np.random.default_rng(0).normal(size=16)
    y = op.apply(x)
    recovered = op.inverse().apply(y)

    assert np.allclose(recovered, x, atol=1e-12)
    assert np.isclose(np.linalg.norm(y), np.linalg.norm(x), atol=1e-12)


def test_matrix_is_orthogonal():
    op = SignedPermutationOperator.random(10, seed=4)
    mat = op.matrix()
    assert np.allclose(mat @ mat.T, np.eye(10), atol=1e-12)
    assert np.allclose(mat.T @ mat, np.eye(10), atol=1e-12)


def test_symmetric_and_involution_residuals():
    # Transposition of slots 0 and 1 with equal signs on the pair is symmetric.
    symmetric = _signed([1, 0, 2, 3], [-1, -1, 1, 1], name="symmetric")
    assert symmetric.is_symmetric()
    assert symmetric.is_involution()
    mat = symmetric.matrix()
    assert np.allclose(mat, mat.T, atol=1e-12)
    assert np.allclose(mat @ mat, np.eye(4), atol=1e-12)

    bank = RelationOperatorBank({"symmetric": symmetric})
    residuals = bank.axiom_residuals()["symmetric"]
    assert residuals["symmetry"] < 1e-12
    assert residuals["involution"] < 1e-12

    # A 3-cycle is neither symmetric nor an involution.
    cyclic = _signed([1, 2, 0, 3], [1, 1, 1, 1], name="cyclic")
    assert not cyclic.is_symmetric()
    assert not cyclic.is_involution()


def test_low_rank_inverse_roundtrip():
    rng = np.random.default_rng(5)
    U = rng.normal(scale=0.2, size=(8, 2))
    V = rng.normal(scale=0.2, size=(8, 2))
    op = LowRankOperator(U=U, V=V)
    x = rng.normal(size=8)

    inverse = op.inverse()
    assert inverse is not None
    assert np.allclose(inverse.apply(op.apply(x)), x, atol=1e-6)
    assert np.allclose(op.matrix() @ inverse.matrix(), np.eye(8), atol=1e-6)


def test_low_rank_inverse_returns_none_when_singular():
    # I + V.T @ U == 0, so the Woodbury core is singular.
    U = np.array([[1.0], [0.0], [0.0]])
    V = np.array([[-1.0], [0.0], [0.0]])
    op = LowRankOperator(U=U, V=V)
    assert op.inverse() is None


def test_low_rank_compose_returns_composite():
    op = LowRankOperator(U=np.ones((6, 1)) * 0.1, V=np.ones((6, 1)) * 0.1)
    other = SignedPermutationOperator.random(6, seed=6)
    composed = op.compose(other)
    assert isinstance(composed, CompositeOperator)
    x = np.arange(6, dtype=np.float64)
    assert np.allclose(composed.apply(x), op.apply(other.apply(x)))


def test_composite_apply_matches_manual_sequence_and_inverse_reverses_order():
    rng = np.random.default_rng(7)
    a = SignedPermutationOperator.random(8, seed=7)
    b = SignedPermutationOperator.random(8, seed=8)
    low = LowRankOperator(
        U=rng.normal(scale=0.1, size=(8, 2)),
        V=rng.normal(scale=0.1, size=(8, 2)),
    )
    composite = CompositeOperator((a, low, b))
    x = rng.normal(size=8)

    manual = b.apply(low.apply(a.apply(x)))
    assert np.allclose(composite.apply(x), manual, atol=1e-12)

    inverse = composite.inverse()
    assert isinstance(inverse, CompositeOperator)
    assert inverse.parts[0].signature() == b.inverse().signature()
    assert np.allclose(inverse.parts[1].matrix(), low.inverse().matrix(), atol=1e-12)
    assert inverse.parts[2].signature() == a.inverse().signature()
    assert np.allclose(inverse.apply(composite.apply(x)), x, atol=1e-9)

    # self.compose(other) applies other first.
    other = SignedPermutationOperator.random(8, seed=9)
    extended = composite.compose(other)
    assert extended.parts == (other, *composite.parts)
    assert np.allclose(extended.apply(x), composite.apply(other.apply(x)), atol=1e-12)


def test_composite_inverse_is_none_when_a_part_is_not_invertible():
    class _NotInvertible(Operator):
        name = "not_invertible"
        kind = "semantic"

        def apply(self, x):
            return np.asarray(x, dtype=np.float64)

    composite = CompositeOperator((SignedPermutationOperator.random(4, seed=0), _NotInvertible()))
    assert composite.inverse() is None


def test_bank_compose_follows_path_order_and_compose_path():
    r1 = SignedPermutationOperator.random(8, seed=10)
    r2 = SignedPermutationOperator.random(8, seed=11)
    bank = RelationOperatorBank({"r1": r1, "r2": r2})
    x = np.random.default_rng(10).normal(size=8)

    # a -r1-> b -r2-> c applies r1 first.
    assert np.allclose(bank.compose("r1", "r2", x), r2.apply(r1.apply(x)), atol=1e-12)
    assert np.allclose(bank.compose_path(["r1", "r2"], x), r2.apply(r1.apply(x)), atol=1e-12)
    assert bank.compose("r1", "missing", x) is None
    assert bank.compose("missing", "r2", x) is None


def test_bank_nearest_returns_self_at_zero_distance():
    a = SignedPermutationOperator.random(8, seed=12)
    b = SignedPermutationOperator.random(8, seed=13)
    low = LowRankOperator(U=np.ones((8, 1)) * 0.1, V=np.ones((8, 1)) * 0.1)
    bank = RelationOperatorBank({"a": a, "b": b, "low": low})

    result = bank.nearest(a)
    assert result is not None
    name, distance = result
    assert name == "a"
    assert distance < 1e-12

    result = bank.nearest(b.matrix(), names=["a", "b"])
    assert result is not None
    assert result[0] == "b"
    assert result[1] < 1e-12


def test_axiom_residual_keys_and_ranges():
    symmetric = _signed([1, 0, 2, 3], [-1, -1, 1, 1], name="symmetric")
    cyclic = _signed([1, 2, 0, 3], [1, 1, 1, 1], name="cyclic")
    low = LowRankOperator(
        U=np.ones((4, 1)) * 0.1,
        V=np.ones((4, 1)) * 0.1,
        name="low",
    )
    bank = RelationOperatorBank(
        {"symmetric": symmetric, "cyclic": cyclic, "low": low, "plain": _PlainOperator()}
    )
    residuals = bank.axiom_residuals()

    assert set(residuals) == {"symmetric", "cyclic", "low", "plain"}
    assert residuals["plain"] == {}
    for name in ("symmetric", "cyclic", "low"):
        entry = residuals[name]
        assert {"symmetry", "involution", "idempotence"} <= set(entry)
        assert "inverse_error" in entry
        for value in entry.values():
            assert np.isfinite(value)
            assert value >= 0.0

    # Signed permutations stay in the normalised unit interval.
    for name in ("symmetric", "cyclic"):
        for value in residuals[name].values():
            assert value <= 1.0 + 1e-12

    assert residuals["symmetric"]["symmetry"] < 1e-12
    assert residuals["symmetric"]["involution"] < 1e-12
    assert residuals["symmetric"]["inverse_error"] < 1e-12
    assert residuals["cyclic"]["symmetry"] > 0.0
    assert residuals["cyclic"]["involution"] > 0.0


def test_save_load_roundtrip(tmp_path):
    signed = SignedPermutationOperator.random(8, seed=14, name="signed")
    low = LowRankOperator(
        U=np.ones((8, 1)) * 0.1,
        V=np.ones((8, 1)) * 0.3,
        name="low",
    )
    bank = RelationOperatorBank({"signed": signed, "low": low})
    path = tmp_path / "operators.npz"
    bank.save(path)

    loaded = RelationOperatorBank.load(path)
    assert set(loaded.names()) == {"signed", "low"}
    assert loaded.get("signed").signature() == signed.signature()

    x = np.random.default_rng(14).normal(size=8)
    assert np.allclose(loaded.apply("signed", x), signed.apply(x), atol=0.0)
    assert np.allclose(loaded.apply("low", x), low.apply(x), atol=1e-15)


def test_save_rejects_composite(tmp_path):
    composite = CompositeOperator(
        (SignedPermutationOperator.random(4, seed=15), SignedPermutationOperator.random(4, seed=16))
    )
    bank = RelationOperatorBank({"composite": composite})
    with pytest.raises(ValueError, match="unsupported"):
        bank.save(tmp_path / "composite.npz")


def test_validation_rejects_bad_permutations_and_shapes():
    with pytest.raises(ValueError):
        SignedPermutationOperator(perm=[0, 0], signs=[1, 1])
    with pytest.raises(ValueError):
        SignedPermutationOperator(perm=[0, 1], signs=[1, 2])
    with pytest.raises(ValueError):
        SignedPermutationOperator(perm=[0, 1], signs=[1])
    with pytest.raises(ValueError):
        SignedPermutationOperator(perm=[0.5, 1], signs=[1, 1])
    with pytest.raises(ValueError):
        LowRankOperator(U=np.zeros((4, 2)), V=np.zeros((5, 2)))

    op = SignedPermutationOperator.random(4, seed=0)
    with pytest.raises(ValueError):
        op.apply(np.zeros(3))
