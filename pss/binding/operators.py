"""Ordered relation operators that can express non-commutative composition.

Hadamard binding (``pss.binding.relations``) and complex diagonal binding
(RotatE) are commutative: ``A ∘ B == B ∘ A``.  That makes them unable to
represent an ordered path such as ``a -r1-> b -r2-> c``.

This module adds three operator families:

``SignedPermutationOperator``
    A signed permutation matrix ``M = diag(signs) @ P``.  It is orthogonal
    (norm-preserving), exactly invertible with ``M.T``, closed under
    composition, and generally non-commutative.  Two operators of this family
    compose into a third operator of the same family.

``LowRankOperator``
    An identity plus a rank-``k`` update, ``M = I + U @ V.T``.  It is cheap to
    apply and invertible via the Woodbury identity when ``I + V.T @ U`` is
    non-singular.

``CompositeOperator``
    A heterogeneous chain, stored in *application* order.

Composition order follows :mod:`pss.binding.protocol`: ``self.compose(other)``
is ``self ∘ other`` and applies ``other`` first, matching
``self.matrix() @ other.matrix()``.

Dataclass and array equality
----------------------------
The operator classes are frozen dataclasses, but dataclass-generated equality
would compare NumPy arrays in a boolean context and raise ``ValueError``.
They are therefore declared with ``eq=False``.  ``SignedPermutationOperator``
defines its own equality/hash on top of :meth:`signature`; the other families
keep identity equality and expose ``matrix()`` for comparisons.  Use
``signature()`` or ``matrix()`` rather than dataclass equality.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ops import cosine
from .protocol import Operator

__all__ = [
    "CompositeOperator",
    "LowRankOperator",
    "RelationOperatorBank",
    "SignedPermutationOperator",
]


def _as_vector(x, d: int, label: str = "x") -> np.ndarray:
    """Return ``x`` as a float64 vector of shape ``(d,)``."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.shape != (d,):
        raise ValueError(f"{label} must have shape ({d},), got {arr.shape}")
    return arr


def _normalised_frobenius(a: np.ndarray) -> float:
    """Frobenius norm of a square matrix normalised by ``2 * d``."""
    d = a.shape[0]
    if d == 0:
        return 0.0
    return float(np.linalg.norm(a, ord="fro") / (2 * d))


def _matrix_of(operator: object) -> np.ndarray | None:
    """Return an operator's matrix, or ``None`` when it has no usable one."""
    matrix = getattr(operator, "matrix", None)
    if not callable(matrix):
        return None
    try:
        arr = np.asarray(matrix(), dtype=np.float64)
    except (NotImplementedError, TypeError, ValueError):
        return None
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        return None
    return arr


@dataclass(frozen=True, eq=False)
class SignedPermutationOperator(Operator):
    """A signed permutation: ``apply(x) = x[perm] * signs``.

    ``perm`` is a permutation of ``range(d)`` and ``signs`` contains only
    ``+1``/``-1``.  The corresponding matrix has exactly one non-zero entry per
    row and column, so it is orthogonal and exactly invertible.
    """

    perm: np.ndarray
    signs: np.ndarray
    name: str = "signed_perm"
    kind: str = "semantic"

    def __post_init__(self) -> None:
        raw_perm = np.asarray(self.perm)
        if raw_perm.ndim != 1:
            raise ValueError(f"perm must be one-dimensional, got shape {raw_perm.shape}")
        if raw_perm.size == 0:
            raise ValueError("perm must not be empty")
        if not np.issubdtype(raw_perm.dtype, np.integer):
            if not np.all(np.isfinite(raw_perm)) or not np.all(raw_perm == np.floor(raw_perm)):
                raise ValueError("perm must contain integers")
        perm = raw_perm.astype(np.int64)

        raw_signs = np.asarray(self.signs)
        if raw_signs.ndim != 1:
            raise ValueError(f"signs must be one-dimensional, got shape {raw_signs.shape}")
        signs = raw_signs.astype(np.float64)
        if perm.shape != signs.shape:
            raise ValueError(f"perm shape {perm.shape} does not match signs shape {signs.shape}")
        if not np.array_equal(np.sort(perm), np.arange(perm.size)):
            raise ValueError("perm must be a permutation of range(d)")
        if not np.all((signs == 1.0) | (signs == -1.0)):
            raise ValueError("signs must be +1 or -1")

        object.__setattr__(self, "perm", perm)
        object.__setattr__(self, "signs", signs)

    # -- identity / comparison -------------------------------------------------

    def signature(self) -> tuple[tuple[int, ...], tuple[float, ...]]:
        """Canonical transform identity: ``(perm, signs)`` as Python tuples."""
        return (
            tuple(int(p) for p in self.perm),
            tuple(float(s) for s in self.signs),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SignedPermutationOperator):
            return NotImplemented
        return self.signature() == other.signature()

    def __hash__(self) -> int:
        return hash((type(self).__name__, self.signature()))

    # -- construction ----------------------------------------------------------

    @classmethod
    def random(
        cls,
        d: int,
        name: str = "signed_perm",
        seed: int = 0,
    ) -> "SignedPermutationOperator":
        """Return a uniformly random signed permutation of dimension ``d``."""
        if d < 1:
            raise ValueError("d must be positive")
        rng = np.random.default_rng(seed)
        return cls(
            perm=rng.permutation(d).astype(np.int64),
            signs=rng.choice(np.array([-1.0, 1.0]), size=d),
            name=name,
        )

    # -- Operator interface ----------------------------------------------------

    def apply(self, x) -> np.ndarray:
        """Gather and sign-flip: ``x[perm] * signs``."""
        arr = _as_vector(x, self.perm.size)
        return arr[self.perm] * self.signs

    def inverse(self) -> "SignedPermutationOperator":
        """Inverse operator with the inverse permutation.

        For ``y_i = signs_i * x[perm_i]`` the inverse is
        ``x = y[inv_perm] * signs[inv_perm]``.  Note that the signs are
        therefore re-indexed, not simply copied: copying them is only correct
        when every cycle of ``perm`` has a constant sign.
        """
        inv_perm = np.argsort(self.perm)
        inv_signs = self.signs[inv_perm]
        return SignedPermutationOperator(
            perm=inv_perm,
            signs=inv_signs,
            name=self.name,
            kind=self.kind,
        )

    def compose(self, other: "Operator") -> "SignedPermutationOperator | None":
        """Return ``self ∘ other`` (``other`` applied first), or ``None``.

        For ``other`` mapping ``y = b * x[o]`` and ``self`` mapping
        ``z = a * y[p]`` the composition is ``z = (a * b[p]) * x[o[p]]``.
        """
        if not isinstance(other, SignedPermutationOperator):
            return None
        return SignedPermutationOperator(
            perm=other.perm[self.perm],
            signs=self.signs * other.signs[self.perm],
            name=f"{self.name}∘{other.name}",
            kind=self.kind,
        )

    def residual(self, x, y) -> float:
        """Consistency distance ``1 - cosine(x, y)``."""
        return 1.0 - cosine(x, y)

    # -- linear algebra --------------------------------------------------------

    def matrix(self) -> np.ndarray:
        """Return the ``(d, d)`` signed permutation matrix."""
        d = self.perm.size
        mat = np.zeros((d, d), dtype=np.float64)
        mat[np.arange(d), self.perm] = self.signs
        return mat

    def is_symmetric(self, tol: float = 1e-9) -> bool:
        """Whether the matrix equals its transpose."""
        mat = self.matrix()
        return bool(np.allclose(mat, mat.T, atol=tol))

    def is_involution(self, tol: float = 1e-9) -> bool:
        """Whether applying the operator twice is the identity."""
        mat = self.matrix()
        return bool(np.allclose(mat @ mat, np.eye(mat.shape[0]), atol=tol))


@dataclass(frozen=True, eq=False)
class LowRankOperator(Operator):
    """Identity plus a rank-``k`` update: ``apply(x) = x + U @ (V.T @ x)``.

    The represented matrix is ``M = I + U @ V.T``.  It is invertible through
    the Woodbury identity when ``I + V.T @ U`` is non-singular.
    """

    U: np.ndarray
    V: np.ndarray
    name: str = "low_rank"
    kind: str = "semantic"

    def __post_init__(self) -> None:
        U = np.asarray(self.U, dtype=np.float64)
        V = np.asarray(self.V, dtype=np.float64)
        if U.ndim != 2 or V.ndim != 2:
            raise ValueError(
                f"U and V must be two-dimensional, got {U.shape} and {V.shape}"
            )
        if U.shape != V.shape:
            raise ValueError(f"U shape {U.shape} does not match V shape {V.shape}")
        if U.shape[0] < 1 or U.shape[1] < 1:
            raise ValueError("U and V must have shape (d, k) with d >= 1 and k >= 1")
        if not np.all(np.isfinite(U)) or not np.all(np.isfinite(V)):
            raise ValueError("U and V must contain finite values")
        object.__setattr__(self, "U", U)
        object.__setattr__(self, "V", V)

    # -- Operator interface ----------------------------------------------------

    def apply(self, x) -> np.ndarray:
        """Return ``x + U @ (V.T @ x)``."""
        arr = _as_vector(x, self.U.shape[0])
        return arr + self.U @ (self.V.T @ arr)

    def inverse(self) -> "LowRankOperator | None":
        """Woodbury inverse, or ``None`` when ``I + V.T @ U`` is singular.

        ``(I + U V.T)^{-1} = I - U (I + V.T U)^{-1} V.T``, so the recovered
        right factor is ``V @ (I + V.T U)^{-T}``.
        """
        k = self.V.shape[1]
        core = np.eye(k) + self.V.T @ self.U
        scale = np.linalg.norm(core, ord=2)
        if not np.isfinite(scale) or scale == 0.0:
            return None
        tol = 1e-12 * max(core.shape) * scale
        if np.linalg.matrix_rank(core, tol=tol) < k:
            return None
        try:
            V_recovered = np.linalg.solve(core, self.V.T).T
        except np.linalg.LinAlgError:
            return None
        return LowRankOperator(
            U=-self.U,
            V=V_recovered,
            name=f"{self.name}^-1",
            kind=self.kind,
        )

    def compose(self, other: "Operator") -> "CompositeOperator | None":
        """Wrap ``self ∘ other`` in a :class:`CompositeOperator`."""
        if not isinstance(other, Operator) or not self.composable(other):
            return None
        return CompositeOperator(parts=(other, self), kind=self.kind)

    def residual(self, x, y) -> float:
        """Consistency distance ``1 - cosine(x, y)``."""
        return 1.0 - cosine(x, y)

    # -- linear algebra --------------------------------------------------------

    def matrix(self) -> np.ndarray:
        """Return ``I + U @ V.T``."""
        d = self.U.shape[0]
        return np.eye(d) + self.U @ self.V.T


@dataclass(frozen=True, eq=False)
class CompositeOperator(Operator):
    """A chain of operators stored in **application** order.

    ``parts[0]`` is applied first, so for ``parts=(a, b)``::

        apply(x) == b.apply(a.apply(x))
        matrix() == b.matrix() @ a.matrix()

    ``compose`` prepends the other operator because ``self ∘ other`` applies
    ``other`` first.
    """

    parts: tuple[Operator, ...]
    name: str = "composite"
    kind: str = "semantic"

    def __post_init__(self) -> None:
        parts = tuple(self.parts)
        for part in parts:
            if not isinstance(part, Operator):
                raise TypeError(f"composite parts must be Operators, got {type(part).__name__}")
        object.__setattr__(self, "parts", parts)

    def apply(self, x):
        """Apply every part left to right."""
        for part in self.parts:
            x = part.apply(x)
        return x

    def inverse(self) -> "CompositeOperator | None":
        """Reverse the order and invert every part."""
        inverses = [part.inverse() for part in self.parts]
        if any(inv is None for inv in inverses):
            return None
        return CompositeOperator(
            parts=tuple(reversed(inverses)),
            name=f"{self.name}^-1",
            kind=self.kind,
        )

    def compose(self, other: "Operator") -> "CompositeOperator | None":
        """Return ``self ∘ other`` with application order ``(other, *self.parts)``."""
        if not isinstance(other, Operator) or not self.composable(other):
            return None
        if isinstance(other, CompositeOperator):
            return CompositeOperator(parts=other.parts + self.parts, kind=self.kind)
        return CompositeOperator(parts=(other, *self.parts), kind=self.kind)

    def residual(self, x, y) -> float:
        """Consistency distance between the chain output and ``y``."""
        return 1.0 - cosine(self.apply(x), y)

    def matrix(self) -> np.ndarray:
        """Matrix product in application order: ``M_last @ ... @ M_first``."""
        if not self.parts:
            raise NotImplementedError("an empty composite has no fixed dimension")
        matrices = []
        for part in self.parts:
            mat = _matrix_of(part)
            if mat is None:
                raise NotImplementedError(
                    f"part {part.name!r} does not expose a square matrix()"
                )
            matrices.append(mat)
        result = matrices[0]
        for mat in matrices[1:]:
            result = mat @ result
        return result


class RelationOperatorBank:
    """Registry of named relation operators with diagnostics and persistence.

    ``closure`` is optional and duck-typed: it must expose
    ``compose(r1, r2) -> iterable[(predicate, prior)]``.  It is stored for
    callers that combine vector operators with symbolic closure tables; this
    class deliberately does not import the closure module.
    """

    def __init__(self, operators: dict[str, Operator] | None = None, closure=None):
        self._operators: dict[str, Operator] = {}
        self.closure = closure
        for name, operator in (operators or {}).items():
            self.register(name, operator)

    # -- registry --------------------------------------------------------------

    def register(self, name: str, op: Operator) -> None:
        """Register ``op`` under ``name``, replacing any previous entry."""
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty string")
        if not isinstance(op, Operator):
            raise TypeError(f"op must be an Operator, got {type(op).__name__}")
        self._operators[name] = op

    def get(self, name: str) -> Operator | None:
        """Return the operator registered under ``name``, or ``None``."""
        return self._operators.get(name)

    def names(self) -> list[str]:
        """Return registered names in insertion order."""
        return list(self._operators)

    # -- application -----------------------------------------------------------

    def apply(self, name: str, x) -> np.ndarray:
        """Apply the named operator to ``x``."""
        try:
            op = self._operators[name]
        except KeyError as exc:
            raise KeyError(f"no operator registered as {name!r}") from exc
        return op.apply(x)

    def inverse(self, name: str) -> Operator | None:
        """Return the named operator's inverse, or ``None``."""
        op = self._operators.get(name)
        return None if op is None else op.inverse()

    def compose(self, r1: str, r2: str, x):
        """Apply the path ``a -r1-> b -r2-> c`` to ``x``.

        Following the closure convention, this is ``T_r2 ∘ T_r1``: ``r1`` is
        applied first.  Returns ``None`` when either name is missing or the two
        operators are not composable.  If the family does not implement
        ``compose``, the two operators are applied sequentially.
        """
        op1 = self._operators.get(r1)
        op2 = self._operators.get(r2)
        if op1 is None or op2 is None:
            return None
        if not op2.composable(op1):
            return None
        composed = op2.compose(op1)
        if composed is not None:
            return composed.apply(x)
        return op2.apply(op1.apply(x))

    def compose_path(self, names, x):
        """Apply operators named left to right."""
        for name in names:
            try:
                op = self._operators[name]
            except KeyError as exc:
                raise KeyError(f"no operator registered as {name!r}") from exc
            x = op.apply(x)
        return x

    # -- diagnostics -----------------------------------------------------------

    def axiom_residuals(self) -> dict[str, dict[str, float]]:
        """Report group-like axiom violations for every matrix-backed operator.

        Each residual is a normalised Frobenius distance (divided by ``2 * d``):
        ``symmetry`` for ``M - M.T``, ``involution`` for ``M @ M - I``,
        ``idempotence`` for ``M @ M - M``, and ``inverse_error`` for
        ``M @ M_inv - I`` when an inverse exists.  Operators without a square
        ``matrix()`` get an empty dictionary.
        """
        report: dict[str, dict[str, float]] = {}
        for name, op in self._operators.items():
            mat = _matrix_of(op)
            if mat is None:
                report[name] = {}
                continue
            d = mat.shape[0]
            identity = np.eye(d)
            report[name] = {
                "symmetry": _normalised_frobenius(mat - mat.T),
                "involution": _normalised_frobenius(mat @ mat - identity),
                "idempotence": _normalised_frobenius(mat @ mat - mat),
            }
            inverse_op = op.inverse()
            if inverse_op is not None:
                inv_mat = _matrix_of(inverse_op)
                if inv_mat is None:
                    try:
                        inv_mat = np.linalg.inv(mat)
                    except np.linalg.LinAlgError:
                        inv_mat = None
                if inv_mat is not None and inv_mat.shape == mat.shape:
                    report[name]["inverse_error"] = _normalised_frobenius(
                        mat @ inv_mat - identity
                    )
        return report

    def nearest(self, op, names=None) -> tuple[str, float] | None:
        """Return the registered operator closest to ``op`` in Frobenius distance.

        ``op`` may be an operator or a square matrix.  Distances are normalised
        by ``2 * d``.  Returns ``(name, distance)`` or ``None`` when no
        registered operator has a comparable matrix.
        """
        target = _matrix_of(op) if isinstance(op, Operator) else None
        if target is None:
            try:
                candidate = np.asarray(op, dtype=np.float64)
            except (TypeError, ValueError):
                return None
            if candidate.ndim != 2 or candidate.shape[0] != candidate.shape[1]:
                return None
            target = candidate

        best: tuple[str, float] | None = None
        for name in (self._operators if names is None else names):
            op_i = self._operators.get(name)
            if op_i is None:
                continue
            mat = _matrix_of(op_i)
            if mat is None or mat.shape != target.shape:
                continue
            distance = _normalised_frobenius(mat - target)
            if best is None or distance < best[1]:
                best = (name, distance)
        return best

    # -- persistence -----------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist the bank to an ``.npz`` file with JSON metadata.

        Only signed-permutation and low-rank operators round-trip.  Composite
        operators (and unknown families) raise ``ValueError`` rather than being
        dropped silently.
        """
        path = Path(path)
        unsupported = [
            name
            for name, op in self._operators.items()
            if not isinstance(op, (SignedPermutationOperator, LowRankOperator))
        ]
        if unsupported:
            raise ValueError(
                "can only persist signed-permutation and low-rank operators; "
                f"unsupported: {', '.join(sorted(unsupported))}"
            )

        arrays: dict[str, np.ndarray] = {}
        entries = []
        for index, (name, op) in enumerate(self._operators.items()):
            if isinstance(op, SignedPermutationOperator):
                arrays[f"op{index}_perm"] = op.perm
                arrays[f"op{index}_signs"] = op.signs
                family = "signed_perm"
            else:
                arrays[f"op{index}_U"] = op.U
                arrays[f"op{index}_V"] = op.V
                family = "low_rank"
            entries.append(
                {"name": name, "kind": op.kind, "type": family, "index": index}
            )
        metadata = {"version": 1, "operators": entries}
        arrays["meta"] = np.array(json.dumps(metadata, ensure_ascii=False))

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            np.savez(handle, **arrays)

    @classmethod
    def load(cls, path: str | Path, closure=None) -> "RelationOperatorBank":
        """Load a bank written by :meth:`save`."""
        bank = cls(closure=closure)
        with np.load(Path(path), allow_pickle=False) as data:
            metadata = json.loads(str(data["meta"].item()))
            for entry in metadata["operators"]:
                index = int(entry["index"])
                name = str(entry["name"])
                kind = str(entry["kind"])
                family = str(entry["type"])
                if family == "signed_perm":
                    op: Operator = SignedPermutationOperator(
                        perm=data[f"op{index}_perm"],
                        signs=data[f"op{index}_signs"],
                        name=name,
                        kind=kind,
                    )
                elif family == "low_rank":
                    op = LowRankOperator(
                        U=data[f"op{index}_U"],
                        V=data[f"op{index}_V"],
                        name=name,
                        kind=kind,
                    )
                else:  # pragma: no cover - guarded by save
                    raise ValueError(f"unknown persisted operator type {family!r}")
                bank.register(name, op)
        return bank
