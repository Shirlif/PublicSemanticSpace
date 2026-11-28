"""Monte Carlo relation beliefs from the linearised 3D scene posterior.

The scene solver returns a position and a linearised covariance per object.
This module draws ``N`` scene samples from the product of the per-object
Gaussians and evaluates the 3D relation queries on every sample, turning a
hard predicate into a probability:

    P(axis_aligned(a, b)) ~= mean_k[ axis_aligned(p_a^k, p_b^k) ]

The covariance is the linearised ``sigma^2 (J^T J)^-1`` of the least-squares
solution, not an exact posterior.  Systematic error (model mismatch, robust
loss reweighting, assumed object association) is not represented; the
probabilities are therefore *conditional on the estimator being correct and
near-linear*.  Degenerate covariances are detected, replaced by an isotropic
fallback, and reported in ``SamplingInfo`` instead of silently producing
point masses.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field

import numpy as np

from .bench_api import ArmOutput, BenchConfig
from .metrics import axis_aligned, collision, contact, same_level

N_SAMPLES = 200
MIN_EIG = 1e-10
MAX_SIGMA = 1.0  # metres; a per-axis std above this is treated as degenerate
DEFAULT_FALLBACK_SIGMA = 0.05  # metres; observed 3-view pose std is ~0.045


@dataclass
class SamplingInfo:
    n_samples: int
    n_objects: int
    degraded_objects: list[str] = field(default_factory=list)
    clipped_objects: list[str] = field(default_factory=list)
    max_sigma: float = 0.0

    def as_dict(self) -> dict:
        return {
            "n_samples": int(self.n_samples),
            "n_objects": int(self.n_objects),
            "degraded_objects": list(self.degraded_objects),
            "clipped_objects": list(self.clipped_objects),
            "max_sigma": float(self.max_sigma),
            "n_degraded": len(self.degraded_objects),
        }


@dataclass
class RelationBelief:
    """Per-pair relation probabilities and per-object support probabilities."""

    axis_aligned: dict[tuple[str, str], float] = field(default_factory=dict)
    same_level: dict[tuple[str, str], float] = field(default_factory=dict)
    contact: dict[tuple[str, str], float] = field(default_factory=dict)
    collision: dict[tuple[str, str], float] = field(default_factory=dict)
    support: dict[str, float] = field(default_factory=dict)
    n_samples: int = N_SAMPLES

    def probabilities(self, relation: str = "axis_aligned") -> dict:
        if relation not in {
            "axis_aligned", "same_level", "contact", "collision", "support",
        }:
            raise KeyError(
                f"unknown relation {relation!r}; use axis_aligned / same_level / "
                "contact / collision / support"
            )
        return getattr(self, relation)


def stable_seed(*parts: object) -> int:
    """Stable 32-bit seed from config values and a scene id."""

    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return int(zlib.crc32(payload) & 0x7FFFFFFF)


def _fallback_covariance(sigma: float) -> np.ndarray:
    return np.eye(3, dtype=np.float64) * float(sigma) ** 2


def _safe_covariance(
    cov,
    *,
    fallback_sigma: float = DEFAULT_FALLBACK_SIGMA,
    max_sigma: float = MAX_SIGMA,
) -> tuple[np.ndarray, bool, bool]:
    """Return ``(psd_cov, degraded, clipped)``.

    ``degraded`` is set when the input is missing, non-finite or not a 3x3
    matrix, in which case the isotropic fallback is used.  ``clipped`` is set
    when eigenvalues had to be clipped to the PSD / max-sigma range.
    """

    if cov is None:
        return _fallback_covariance(fallback_sigma), True, False
    arr = np.asarray(cov, dtype=np.float64)
    if arr.shape != (3, 3) or not np.isfinite(arr).all():
        return _fallback_covariance(fallback_sigma), True, False
    arr = 0.5 * (arr + arr.T)
    try:
        eigvals, eigvecs = np.linalg.eigh(arr)
    except np.linalg.LinAlgError:
        return _fallback_covariance(fallback_sigma), True, False
    if not np.isfinite(eigvals).all() or eigvals.max() <= 0.0:
        # A zero/negative-definite matrix carries no usable uncertainty.
        return _fallback_covariance(fallback_sigma), True, False
    upper = float(max_sigma) ** 2
    clipped = bool(eigvals.min() < 0.0 or eigvals.max() > upper)
    eigvals = np.clip(eigvals, MIN_EIG, upper)
    psd = (eigvecs * eigvals) @ eigvecs.T
    psd = 0.5 * (psd + psd.T)
    return psd, False, clipped


def sample_scene_positions(
    positions: dict[str, np.ndarray],
    covariances: dict[str, np.ndarray] | None = None,
    *,
    n_samples: int = N_SAMPLES,
    rng: np.random.Generator | None = None,
    fallback_sigma: float = DEFAULT_FALLBACK_SIGMA,
    max_sigma: float = MAX_SIGMA,
) -> tuple[dict[str, np.ndarray], SamplingInfo]:
    """Draw ``n_samples`` scenes from ``N(p_i, Sigma_i)`` per object.

    Objects are sampled independently, matching the per-object least-squares
    solver (``solve_scene`` fits each object separately).  Sampling is done via
    the eigendecomposition so a numerically indefinite matrix cannot raise.
    """

    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    rng = np.random.default_rng(0) if rng is None else rng
    covariances = covariances or {}
    samples: dict[str, np.ndarray] = {}
    info = SamplingInfo(n_samples=int(n_samples), n_objects=len(positions))
    for oid, mean in positions.items():
        mean = np.asarray(mean, dtype=np.float64).reshape(3)
        cov, degraded, clipped = _safe_covariance(
            covariances.get(oid),
            fallback_sigma=fallback_sigma,
            max_sigma=max_sigma,
        )
        if degraded:
            info.degraded_objects.append(oid)
        if clipped:
            info.clipped_objects.append(oid)
        eigvals, eigvecs = np.linalg.eigh(cov)
        factor = eigvecs * np.sqrt(np.clip(eigvals, 0.0, None))
        z = rng.standard_normal((int(n_samples), 3))
        samples[oid] = mean[None, :] + z @ factor.T
        info.max_sigma = max(
            info.max_sigma, float(np.sqrt(max(float(eigvals.max()), 0.0)))
        )
    return samples, info


def _pairs(ids):
    return [(a, b) for i, a in enumerate(ids) for b in ids[i + 1 :]]


def relation_probabilities(
    samples: dict[str, np.ndarray],
    radii: dict[str, float],
    *,
    eps_xy: float,
    eps_z: float,
    eps_c: float,
    table_z: float = 0.0,
    eps_s: float = 0.05,
) -> RelationBelief:
    """Evaluate 3D relation queries on every sampled scene.

    ``samples`` maps object id -> ``(N, 3)`` positions.  Every query is the
    sample mean of the corresponding boolean predicate, so a probability of
    0.0 / 1.0 only means "all samples agree", not "certain".
    """

    ids = [oid for oid in samples if oid in radii]
    n_samples = (
        int(next(iter(samples.values())).shape[0]) if samples else N_SAMPLES
    )
    belief = RelationBelief(n_samples=n_samples)
    for oid in ids:
        r = float(radii[oid])
        bottom = samples[oid][:, 2] - r
        belief.support[oid] = float(np.mean(np.abs(bottom - table_z) <= eps_s))
    for a, b in _pairs(ids):
        sa, sb = samples[a], samples[b]
        ra, rb = float(radii[a]), float(radii[b])
        d = sb - sa
        belief.axis_aligned[(a, b)] = float(
            np.mean((np.abs(d[:, 0]) <= eps_xy) | (np.abs(d[:, 1]) <= eps_xy))
        )
        belief.same_level[(a, b)] = float(
            np.mean(np.abs((sb[:, 2] - rb) - (sa[:, 2] - ra)) <= eps_z)
        )
        dist = np.linalg.norm(d, axis=1)
        belief.contact[(a, b)] = float(np.mean(dist <= ra + rb + eps_c))
        belief.collision[(a, b)] = float(np.mean(dist < ra + rb))
    return belief


def belief_from_output(
    output: ArmOutput,
    radii: dict[str, float],
    config: BenchConfig,
    *,
    n_samples: int = N_SAMPLES,
    seed: int = 0,
) -> tuple[RelationBelief, SamplingInfo, dict[str, np.ndarray]]:
    """Convenience wrapper: ``ArmOutput`` -> samples -> relation beliefs."""

    rng = np.random.default_rng(int(seed))
    samples, info = sample_scene_positions(
        output.positions, output.covariances, n_samples=n_samples, rng=rng
    )
    belief = relation_probabilities(
        samples,
        radii,
        eps_xy=config.eps_xy,
        eps_z=config.eps_z,
        eps_c=config.eps_c,
        table_z=config.table_z,
        eps_s=config.eps_s,
    )
    return belief, info, samples


# ---------------------------------------------------------------------------
# Risk / coverage and abstention


def _as_arrays(probabilities, labels) -> tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(list(probabilities), dtype=np.float64)
    labels = np.asarray(list(labels), dtype=np.float64)
    if probs.shape != labels.shape:
        raise ValueError("probabilities and labels must have the same length")
    return probs, labels


def risk_coverage_curve(
    probabilities,
    labels,
    *,
    n_points: int = 11,
) -> dict[str, list[float]]:
    """Selective-prediction curve for a binary relation belief.

    Pairs are ranked by confidence ``max(p, 1-p)``.  For every coverage level
    the retained accuracy and risk (1 - accuracy) are reported.  A calibrated
    belief should show risk falling as coverage shrinks.
    """

    probs, labels = _as_arrays(probabilities, labels)
    n = len(probs)
    if n == 0:
        return {"coverage": [], "accuracy": [], "risk": [], "confidence": []}
    confidence = np.maximum(probs, 1.0 - probs)
    correct = ((probs >= 0.5) == labels.astype(bool)).astype(np.float64)
    order = np.argsort(-confidence, kind="stable")
    correct = correct[order]
    confidence = confidence[order]
    points = sorted(
        {max(1, min(n, int(round(k)))) for k in np.linspace(1, n, n_points)}
    )
    coverage, accuracy, risk, conf = [], [], [], []
    for k in points:
        coverage.append(float(k / n))
        accuracy.append(float(correct[:k].mean()))
        risk.append(float(1.0 - correct[:k].mean()))
        conf.append(float(confidence[:k].min()))
    return {
        "coverage": coverage,
        "accuracy": accuracy,
        "risk": risk,
        "confidence": conf,
    }


def abstention_summary(
    probabilities,
    labels,
    *,
    max_risk: float = 0.05,
) -> dict:
    """Lowest-confidence threshold whose retained risk stays under ``max_risk``.

    Returns the maximal-coverage operating point satisfying the risk budget,
    or ``{"feasible": False}`` when even the single most confident pair misses
    the budget.
    """

    probs, labels = _as_arrays(probabilities, labels)
    n = len(probs)
    if n == 0:
        return {"feasible": False, "n": 0}
    confidence = np.maximum(probs, 1.0 - probs)
    correct = ((probs >= 0.5) == labels.astype(bool)).astype(np.float64)
    order = np.argsort(-confidence, kind="stable")
    correct = correct[order]
    confidence = confidence[order]
    best = None
    min_risk = float("inf")
    # Only thresholds at observed confidences can change the retained set.
    # Cumulative risk is not strictly monotone, so scan every prefix and keep
    # the largest one that still satisfies the budget.
    for k in range(1, n + 1):
        risk = float(1.0 - correct[:k].mean())
        min_risk = min(min_risk, risk)
        if risk > max_risk:
            continue
        best = {
            "feasible": True,
            "threshold": float(confidence[k - 1]),
            "coverage": float(k / n),
            "accuracy": float(correct[:k].mean()),
            "risk": risk,
            "n_retained": int(k),
            "n": int(n),
            "max_risk": float(max_risk),
        }
    if best is None:
        return {
            "feasible": False,
            "n": int(n),
            "max_risk": float(max_risk),
            "best_risk": float(min_risk),
        }
    return best


__all__ = [
    "DEFAULT_FALLBACK_SIGMA",
    "MAX_SIGMA",
    "MIN_EIG",
    "N_SAMPLES",
    "RelationBelief",
    "SamplingInfo",
    "abstention_summary",
    "belief_from_output",
    "relation_probabilities",
    "risk_coverage_curve",
    "sample_scene_positions",
    "stable_seed",
]
