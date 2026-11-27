"""Calibration robustness pipeline.

Implements the five stages frozen in Wave 0 (see
``docs/plans/PSS-校准抗干扰融合设计.md``):

    source_health → score_samples → fit_params → bound_step → gate

Design invariants
-----------------
- **One weight vector.** ``score_samples`` returns the only mask; ``fit_params``
  consumes it.  There is no second consensus definition.
- **Prediction-level audit.** Screening signals use predictions and labels, not
  calibrated probabilities, so changing the estimator cannot move the audit.
- **One aggregated parameter set.** Sources are a detection signal, never a
  parameter axis, so the step bound is always well defined.
- **Relative gate.** The gate compares candidates on the same data with the same
  estimator; it never uses an absolute ECE threshold.

Determinism: no RNG in any stage; modalities iterate in ``sorted`` order; ties
break on the lowest class index; splits are caller-provided, never shuffled here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum

import numpy as np

from .calibration import (
    TEMPERATURE_LOG_BOUNDS,
    expected_calibration_error,
    fit_temperature_weighted,
    temperature_at_bound,
)
from .confidence_map import (
    ConfidenceMap,
    apply_confidence_map,
    fit_confidence_map,
)
from .cusum import CusumBank, CusumDetector, PerClassCusum

__all__ = [
    "PIPELINE_ORDER",
    "CalibrationBatch",
    "Evidence",
    "GateDecision",
    "GateThresholds",
    "Params",
    "RobustCalibrationPipeline",
    "SampleWeights",
    "State",
    "TransitionThresholds",
    "bound_step",
    "bootstrap_params",
    "classify_state",
    "evaluate_params",
    "fit_confidence_maps",
    "fit_params",
    "fused_probabilities",
    "gate",
    "score_samples",
    "source_health",
]

#: Fixed stage order.  ``gate`` validates the candidate that would ship, so it
#: must run after ``bound_step``.
PIPELINE_ORDER: tuple[str, ...] = (
    "source_health",
    "score_samples",
    "fit_params",
    "bound_step",
    "gate",
)

# Step policy (design doc §4, decision B).  Log-space for temperature so the
# limit is scale-free.
_STEP_SMALL_LOG = float(np.log(1.2))
_STEP_DRIFT_LOG = float(np.log(3.0))
_REL_STEP_SMALL = 0.15
_REL_STEP_DRIFT = 0.40

# Screening constants.  Calibrate on the clean stream in a follow-up; they are
# deliberately conservative so the pipeline under-fits rather than over-fits a
# poisoned window.
_SOURCE_EXCESS_TOL = 0.25
_DUP_COSINE = 0.9999
_OUTLIER_Z = 4.0
_DEFAULT_TEMPERATURE = 1.0


class State(str, Enum):
    """Calibration state machine (design doc §6)."""

    NORMAL = "normal"
    QUARANTINE = "quarantine"
    RECALIBRATE = "recalibrate"
    ABSTAIN = "abstain"


@dataclass(frozen=True)
class CalibrationBatch:
    """One window of calibration data, already in the public space."""

    embeddings: Mapping[str, np.ndarray]
    prototypes: Mapping[str, np.ndarray]
    labels: np.ndarray
    sources: Sequence[str]
    sample_ids: Sequence[str]

    def __post_init__(self) -> None:
        n = len(self.labels)
        if len(self.sources) != n or len(self.sample_ids) != n:
            raise ValueError("sources and sample_ids must align with labels")
        for modality, emb in self.embeddings.items():
            if emb.shape[0] != n:
                raise ValueError(
                    f"embedding rows for {modality!r} ({emb.shape[0]}) != labels ({n})"
                )


@dataclass(frozen=True)
class Params:
    """Per-modality calibration parameters (read-only contract).

    ``pinned`` records modalities whose fitted temperature sits on the search
    bound.  A pinned fit is silently uninformative — it looks like "the
    parameters did not change" — so it is carried explicitly for reporting.
    """

    temperature: Mapping[str, float]
    reliability: Mapping[str, float]
    pinned: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Evidence:
    """Signals driving the state machine and the step bound.

    ``accuracy_inflation`` and ``class_agreement_spread`` are the two targeted
    attack signals: the first catches "relabel the model's errors as correct",
    the second catches an attack concentrated on one class (which a global
    agreement average dilutes).
    """

    cross_modal_agreement: float
    label_consensus_agreement: float
    drift_score: float
    suspect_mass: float
    accuracy_inflation: float = 0.0
    class_agreement_spread: float = 0.0
    label_coverage: float = 1.0


@dataclass(frozen=True)
class SampleWeights:
    """Per-sample trust weights shared by audit and fit."""

    per_sample: np.ndarray
    reasons: Mapping[str, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True)
class GateDecision:
    """Result of the two-criterion gate."""

    accept: bool
    integrity_ok: bool
    adaptation_ok: bool
    reason: str


@dataclass(frozen=True)
class GateThresholds:
    """Tolerances for the relative comparison.

    ``acc_tol`` is the decision-quality tolerance: the candidate may not lose
    more than this much **fused accuracy** on either the frozen gate set or the
    recent holdout.

    Calibration error is deliberately *not* a gate criterion.  After the channel
    split the confidence map is refit downstream and is exactly invariant to a
    global temperature (measured: identical out-of-sample ECE for T from 0.01 to
    1.0), so an ECE-based gate would judge every candidate a tie.  The gate
    protects decisions; the map protects calibration.

    ``drift_override`` relaxes *integrity* when the caller has independent
    evidence of real drift (state RECALIBRATE).  Without it a stale gate set
    would block every legitimate adaptation; with it in the wrong state, poison
    would be accepted.  The caller sets it from the state machine, never from
    the data being fitted.
    """

    acc_tol: float = 0.01
    drift_override: bool = False


@dataclass(frozen=True)
class TransitionThresholds:
    """State-machine thresholds.  Defaults are placeholders, to be calibrated."""

    theta_label: float = 0.85
    theta_cross: float = 0.80
    theta_drift: float = 0.06
    theta_inflation: float = 0.15
    theta_spread: float = 0.18
    # The per-class spread test is only meaningful while the global picture is
    # healthy; below this agreement the global signal already handles it.
    theta_global_ok: float = 0.93
    quarantine_recover_windows: int = 3
    gate_fail_abstain: int = 3
    # ABSTAIN is sticky, but not forever: this many consecutive NORMAL windows
    # release it automatically.  Without this an unattended system deadlocks.
    abstain_release_windows: int = 3


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _nll(logits: np.ndarray, labels: np.ndarray) -> float:
    probs = _softmax(logits)
    picked = probs[np.arange(len(labels)), labels].clip(1e-12)
    return float(-np.log(picked).mean())


def _modalities(batch: CalibrationBatch) -> list[str]:
    return sorted(m for m in batch.embeddings if m in batch.prototypes)


def _logits(batch: CalibrationBatch, modality: str) -> np.ndarray:
    return _unit(batch.embeddings[modality]) @ _unit(batch.prototypes[modality]).T


def _valid_labels(batch: CalibrationBatch, n_classes: int) -> np.ndarray:
    labels = np.asarray(batch.labels, dtype=np.int64)
    return (labels >= 0) & (labels < int(n_classes))


def _consensus(batch: CalibrationBatch) -> tuple[np.ndarray | None, np.ndarray]:
    """Majority vote across modalities.

    Ties resolve to the lowest class index, which makes the result deterministic
    without an RNG.
    """
    modalities = _modalities(batch)
    if not modalities:
        return None, np.zeros(len(batch.labels), dtype=np.float64)
    return _consensus_with_weights(batch, {m: 1.0 for m in modalities})


def _vote_weights(
    params: Params,
    modalities: Sequence[str],
) -> dict[str, float]:
    """Reliability-proportional vote weights, dropping clearly weak experts.

    A weak expert (e.g. tactile at 0.20 accuracy) must not drag the consensus:
    on the real expert zoo an unweighted vote drops agreement to ~0.73 and
    flags most clean samples.  Only the floor rule and the ranking matter, not
    the exact reliability values, so a mildly poisoned fit cannot flip the vote
    order on its own.
    """
    rel = {m: max(0.0, float(params.reliability.get(m, 0.0))) for m in modalities}
    top = max(rel.values()) if rel else 0.0
    keep = {m: r for m, r in rel.items() if r > 0.0 and r >= 0.5 * top}
    if not keep:
        return {m: 1.0 for m in modalities}
    total = sum(keep.values())
    return {m: keep[m] / total for m in modalities if m in keep}


def _consensus_with_weights(
    batch: CalibrationBatch,
    vote_weights: Mapping[str, float],
) -> tuple[np.ndarray | None, np.ndarray]:
    modalities = [m for m in _modalities(batch) if vote_weights.get(m, 0.0) > 0.0]
    n = len(batch.labels)
    if not modalities:
        return None, np.zeros(n, dtype=np.float64)
    n_classes = max(int(batch.prototypes[m].shape[0]) for m in modalities)
    votes = np.zeros((n_classes, n), dtype=np.float64)
    rows = np.arange(n)
    for modality in modalities:
        pred = _logits(batch, modality).argmax(axis=1)
        np.add.at(votes, (pred, rows), float(vote_weights[modality]))
    consensus = votes.argmax(axis=0)
    strength = votes.max(axis=0)
    return consensus, strength


def _raw_accuracy(batch: CalibrationBatch, modality: str) -> float:
    logits = _logits(batch, modality)
    valid = _valid_labels(batch, logits.shape[1])
    if not valid.any():
        return float("nan")
    pred = logits.argmax(axis=1)
    return float(np.mean(pred[valid] == np.asarray(batch.labels)[valid]))


def _proto_fit_median(batch: CalibrationBatch, modality: str) -> float:
    """Median best-prototype cosine.

    The **median** is the point: real drift moves every sample, while a subset
    attack (embedding replacement) leaves the majority intact, so the median
    separates the two where the mean would not.
    """
    return float(np.median(_logits(batch, modality).max(axis=1)))


def _fused_top1(
    params: Params,
    batch: CalibrationBatch,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Fused top-1 score and correctness, using vote weights.

    This is the score the system would actually expose as its confidence.  It
    is separate from the per-modality scores used for diagnosis.
    """
    modalities = _modalities(batch)
    weights = _vote_weights(params, modalities)
    labels = np.asarray(batch.labels, dtype=np.int64)
    fused = None
    for modality in modalities:
        w = float(weights.get(modality, 0.0))
        if w <= 0.0:
            continue
        temperature = max(float(params.temperature.get(modality, 1.0)), 1e-3)
        term = w * (_logits(batch, modality) / temperature)
        fused = term if fused is None else fused + term
    if fused is None:
        return None, None
    valid = labels >= 0
    if mask is not None:
        valid = valid & np.asarray(mask, dtype=bool)
    if not valid.any():
        return None, None
    pred = fused.argmax(axis=1)
    return fused.max(axis=1)[valid], (pred[valid] == labels[valid]).astype(np.float64)


def fused_probabilities(
    params: Params,
    batch: CalibrationBatch,
) -> tuple[np.ndarray, np.ndarray]:
    """Fused class probabilities and labels (valid rows only).

    This is the score the system acts on; it is the input to the conformal
    channel, which replaces "fitted confidence" with a coverage guarantee.
    """
    modalities = _modalities(batch)
    weights = _vote_weights(params, modalities)
    labels = np.asarray(batch.labels, dtype=np.int64)
    fused = None
    for modality in modalities:
        w = float(weights.get(modality, 0.0))
        if w <= 0.0:
            continue
        temperature = max(float(params.temperature.get(modality, 1.0)), 1e-3)
        term = w * (_logits(batch, modality) / temperature)
        fused = term if fused is None else fused + term
    if fused is None:
        return np.zeros((0, 0)), np.zeros(0, dtype=np.int64)
    valid = labels >= 0
    return _softmax(fused[valid]), labels[valid]


def fit_confidence_maps(
    params: Params,
    batch: CalibrationBatch,
    *,
    n_bins: int = 6,
    weights: SampleWeights | None = None,
) -> dict[str, ConfidenceMap]:
    """Fit one monotone confidence map per modality plus one for the fused score.

    The maps are fitted on a *labelled* window, so the caller must only refit
    them on a window the audit has accepted; fitting on a quarantined window
    would poison the confidence channel exactly as it would poison the
    temperature.

    ``weights`` additionally down-weights samples the audit distrusts.  They are
    used as bin weights rather than a hard filter: dropping samples would bias
    the map toward easy items and inflate confidence on the test set.
    """
    labels = np.asarray(batch.labels, dtype=np.int64)
    sw = None
    if weights is not None:
        sw = np.asarray(weights.per_sample, dtype=np.float64).reshape(-1)

    maps: dict[str, ConfidenceMap] = {}
    for modality in _modalities(batch):
        logits = _logits(batch, modality)
        valid = _valid_labels(batch, logits.shape[1])
        if not valid.any():
            continue
        pred = logits.argmax(axis=1)
        maps[modality] = fit_confidence_map(
            logits.max(axis=1)[valid],
            (pred[valid] == labels[valid]).astype(np.float64),
            n_bins=n_bins,
            sample_weights=sw[valid] if sw is not None else None,
        )
    valid = labels >= 0
    fused_scores, fused_ok = _fused_top1(params, batch, mask=valid)
    if fused_scores is not None and fused_ok is not None:
        maps["fused"] = fit_confidence_map(
            fused_scores,
            fused_ok,
            n_bins=n_bins,
            sample_weights=sw[valid] if sw is not None else None,
        )
    return maps


def _evaluate(
    params: Params,
    batch: CalibrationBatch,
    weights: Mapping[str, float] | None = None,
) -> tuple[float, float]:
    """Reliability-weighted (mean NLL, mean ECE) over modalities.

    ``weights`` defaults to the same vote weights used for the consensus:
    reliability-proportional, with clearly weak experts excluded.  An unweighted
    mean lets a bad expert dominate (tactile NLL ~2.29 swamps vision ~0.9), so a
    change that helps the good experts can look like a regression.
    """
    modalities = _modalities(batch)
    if weights is None:
        weights = _vote_weights(params, modalities)

    nll_num = ece_num = denom = 0.0
    labels = np.asarray(batch.labels, dtype=np.int64)
    for modality in modalities:
        w = float(weights.get(modality, 0.0))
        if w <= 0.0:
            continue
        logits = _logits(batch, modality)
        valid = _valid_labels(batch, logits.shape[1])
        if not valid.any():
            continue
        temperature = max(float(params.temperature.get(modality, 1.0)), 1e-3)
        scaled = logits[valid] / temperature
        probs = _softmax(scaled)
        conf = probs.max(axis=1)
        ok = (probs.argmax(axis=1) == labels[valid]).astype(np.float64)
        nll_num += w * _nll(scaled, labels[valid])
        ece_num += w * expected_calibration_error(conf, ok)
        denom += w

    if denom <= 0.0:
        return float("nan"), float("nan")
    return float(nll_num / denom), float(ece_num / denom)


# ---------------------------------------------------------------------------
# stage 1: provenance
# ---------------------------------------------------------------------------


def source_health(
    batch: CalibrationBatch,
    *,
    min_source_n: int = 8,
) -> dict[str, float]:
    """Trust weight in ``[0, 1]`` per source.

    A source is down-weighted when its labels disagree with the cross-modal
    consensus more often than the batch average, or when it holds too few
    samples to judge.  Sources never produce their own parameters.
    """
    sources = list(batch.sources)
    if not sources:
        return {}
    consensus, _ = _consensus(batch)
    labels = np.asarray(batch.labels, dtype=np.int64)
    n = len(labels)

    if consensus is None:
        agree = np.ones(n, dtype=np.float64)
    else:
        valid = labels >= 0
        agree = np.zeros(n, dtype=np.float64)
        agree[valid] = (labels[valid] == consensus[valid]).astype(np.float64)
    global_agree = float(agree.mean()) if n else 0.0

    trust: dict[str, float] = {}
    for source in sorted(set(sources)):
        idx = np.array([i for i, s in enumerate(sources) if s == source], dtype=np.int64)
        n_s = int(len(idx))
        if n_s == 0:
            trust[source] = 0.0
            continue
        excess = max(0.0, global_agree - float(agree[idx].mean()))
        value = float(np.clip(1.0 - excess / _SOURCE_EXCESS_TOL, 0.0, 1.0))
        if n_s < int(min_source_n):
            value *= n_s / float(max(1, int(min_source_n)))
        trust[source] = value
    return trust


# ---------------------------------------------------------------------------
# stage 2: per-sample screening
# ---------------------------------------------------------------------------


def score_samples(
    batch: CalibrationBatch,
    params: Params,
    source_trust: Mapping[str, float],
    *,
    state: State = State.NORMAL,
) -> SampleWeights:
    """Continuous per-sample weights.

    Signals use predictions and labels only.  ``params`` is used for one monotone
    decision only: modalities whose previous reliability is at the floor do not
    vote in the consensus.  This prevents a persistently broken expert from
    dragging the vote without letting a poisoned temperature influence
    screening.

    ``state`` breaks the circular dependency between screening and the state
    machine.  Under ``RECALIBRATE`` the labels are trusted and the experts are
    the ones that are wrong, so label/consensus disagreement must **not** be
    penalised: doing so deletes exactly the samples calibration needs to see and
    makes the fitted temperature blind to the model's own errors.  The trade-off
    is explicit: poison arriving *during* a drift window is under-weighted, and
    is caught by the gate's integrity check plus the bounded step instead.
    """
    modalities = _modalities(batch)
    labels = np.asarray(batch.labels, dtype=np.int64)
    n = len(labels)
    if n == 0:
        return SampleWeights(per_sample=np.zeros(0), reasons={})

    trusting_labels = state is State.RECALIBRATE

    voting = _vote_weights(params, modalities)
    strong = [m for m in modalities if voting.get(m, 0.0) > 0.0] or modalities
    consensus, strength = _consensus_with_weights(batch, voting)
    if consensus is None:
        consensus = np.zeros(n, dtype=np.int64)
        strength = np.ones(n, dtype=np.float64)

    valid = labels >= 0
    mismatch = np.zeros(n, dtype=bool)
    mismatch[valid] = labels[valid] != consensus[valid]

    weights = np.ones(n, dtype=np.float64)
    # Strong consensus + disagreeing label -> near-zero weight.  Skipped under
    # drift, where disagreement is the signal we are trying to learn from.
    if not trusting_labels:
        weights *= np.where(mismatch, np.clip(1.0 - strength, 0.0, 1.0), 1.0)

    # Margin violation: the labelled class is not the best or is close to it.
    # Only strong modalities vote: a weak expert would otherwise flag almost
    # every clean sample (the real tactile expert sits at ~0.20 accuracy).
    margin_bad = np.zeros(n, dtype=bool)
    for modality in strong:
        logits = _logits(batch, modality)
        ok = _valid_labels(batch, logits.shape[1])
        if not ok.any():
            continue
        own = logits[np.arange(n), np.clip(labels, 0, logits.shape[1] - 1)]
        other = logits.copy()
        other[np.arange(n), np.clip(labels, 0, logits.shape[1] - 1)] = -np.inf
        margin_bad |= ok & (own - other.max(axis=1) <= 0.0)
    weights *= np.where(margin_bad, 0.5, 1.0) if not trusting_labels else 1.0

    # Near-duplicates across the window.  Reported as an audit reason but NOT
    # down-weighted: F6 showed duplicates are benign, and on real category data
    # same-class views are routinely >0.99 cosine, so penalising them would
    # bias the fit toward atypical samples.
    duplicate = np.zeros(n, dtype=bool)
    if n > 1:
        for modality in strong:
            sims = _unit(batch.embeddings[modality]) @ _unit(batch.embeddings[modality]).T
            np.fill_diagonal(sims, -1.0)
            duplicate |= sims.max(axis=1) >= _DUP_COSINE

    # Centroid outliers within each labelled class.
    outlier = np.zeros(n, dtype=bool)
    for modality in strong:
        emb = _unit(batch.embeddings[modality])
        for cls in np.unique(labels[valid]):
            idx = np.where(labels == cls)[0]
            if len(idx) < 4:
                continue
            centroid = _unit(emb[idx].mean(axis=0))
            dist = 1.0 - emb[idx] @ centroid
            med = float(np.median(dist))
            mad = float(np.median(np.abs(dist - med)))
            z = (dist - med) / (1.4826 * mad + 1e-9)
            outlier[idx] |= z > _OUTLIER_Z
    weights *= np.where(outlier, 0.5, 1.0)

    # Source trust.
    source_suspect = np.zeros(n, dtype=bool)
    for i, source in enumerate(batch.sources):
        trust = float(source_trust.get(source, 1.0))
        weights[i] *= trust
        if trust <= 0.0:
            source_suspect[i] = True

    weights = np.clip(weights, 0.0, 1.0)
    reasons = {
        "label_consensus_mismatch": mismatch,
        "margin_violation": margin_bad,
        "duplicate": duplicate,
        "centroid_outlier": outlier,
        "source_suspect": source_suspect,
    }
    return SampleWeights(per_sample=weights, reasons=reasons)


# ---------------------------------------------------------------------------
# stage 3: robust fit
# ---------------------------------------------------------------------------


def bootstrap_params(batch: CalibrationBatch) -> Params:
    """Initial parameters from raw per-modality accuracy.

    Needed because vote weighting depends on reliability, which normally comes
    from a fitted calibration.  Raw accuracy breaks the cycle without needing a
    consensus, so the first fit does not inherit a weak expert's vote.
    """
    labels = np.asarray(batch.labels, dtype=np.int64)
    temperature: dict[str, float] = {}
    reliability: dict[str, float] = {}
    for modality in _modalities(batch):
        logits = _logits(batch, modality)
        valid = _valid_labels(batch, logits.shape[1])
        temperature[modality] = _DEFAULT_TEMPERATURE
        if not valid.any():
            reliability[modality] = 0.0
            continue
        pred = logits.argmax(axis=1)
        accuracy = float(np.mean(pred[valid] == labels[valid]))
        chance = 1.0 / max(1, int(logits.shape[1]))
        reliability[modality] = float(
            np.clip((accuracy - chance) / max(1e-9, 1.0 - chance), 0.0, 1.0)
        )
    return Params(temperature=temperature, reliability=reliability)


def fit_params(batch: CalibrationBatch, weights: SampleWeights) -> Params:
    """Weighted robust fit of temperature and reliability.

    Zero-weight rows contribute nothing; ``fit_temperature_weighted`` returns a
    fallback when the effective mass collapses, which the gate is expected to
    reject.
    """
    w = np.clip(np.asarray(weights.per_sample, dtype=np.float64).reshape(-1), 0.0, None)
    labels = np.asarray(batch.labels, dtype=np.int64)
    temperature: dict[str, float] = {}
    reliability: dict[str, float] = {}

    for modality in _modalities(batch):
        logits = _logits(batch, modality)
        valid = _valid_labels(batch, logits.shape[1])
        if not valid.any():
            temperature[modality] = _DEFAULT_TEMPERATURE
            reliability[modality] = 0.0
            continue

        wv = w[valid]
        temperature[modality] = fit_temperature_weighted(
            logits[valid],
            labels[valid],
            wv,
            fallback=_DEFAULT_TEMPERATURE,
            # Measured: widening to (-8, 3) unpins the fit but *hurts* fused
            # accuracy (0.974 -> 0.964 clean) and the confidence channel
            # (ECE 0.0151 -> 0.0296).  The bound acts as a regulariser for
            # fusion, so it stays.  Pinning is reported via ``Params.pinned``.
            log_bounds=TEMPERATURE_LOG_BOUNDS,
        )

        total = float(wv.sum())
        if total > 0.0:
            pred = logits.argmax(axis=1)[valid]
            accuracy = float((wv * (pred == labels[valid])).sum() / total)
        else:
            accuracy = 0.0
        chance = 1.0 / max(1, int(logits.shape[1]))
        reliability[modality] = float(
            np.clip((accuracy - chance) / max(1e-9, 1.0 - chance), 0.0, 1.0)
        )

    pinned = frozenset(
        modality
        for modality, temperature in temperature.items()
        if temperature_at_bound(temperature, log_bounds=TEMPERATURE_LOG_BOUNDS)
    )
    return Params(temperature=temperature, reliability=reliability, pinned=pinned)


# ---------------------------------------------------------------------------
# stage 4: bounded step
# ---------------------------------------------------------------------------


def bound_step(old: Params, new: Params, evidence: Evidence) -> Params:
    """Limit the single-step move according to evidence strength.

    The bound is derived from ``classify_state`` so the step policy and the
    state machine share one source of truth.
    """
    state = classify_state(evidence)
    if state in (State.QUARANTINE, State.ABSTAIN):
        return Params(
            temperature=dict(old.temperature),
            reliability=dict(old.reliability),
            pinned=old.pinned,
        )

    if state is State.RECALIBRATE:
        log_step, rel_step = _STEP_DRIFT_LOG, _REL_STEP_DRIFT
    else:
        log_step, rel_step = _STEP_SMALL_LOG, _REL_STEP_SMALL

    temperature: dict[str, float] = {}
    for modality in sorted(set(old.temperature) | set(new.temperature)):
        old_t = float(old.temperature.get(modality, _DEFAULT_TEMPERATURE))
        new_t = float(new.temperature.get(modality, old_t))
        if old_t <= 0.0 or not np.isfinite(old_t) or new_t <= 0.0:
            temperature[modality] = old_t if old_t > 0.0 else _DEFAULT_TEMPERATURE
            continue
        delta = float(np.clip(np.log(new_t / old_t), -log_step, log_step))
        temperature[modality] = float(old_t * np.exp(delta))

    reliability: dict[str, float] = {}
    for modality in sorted(set(old.reliability) | set(new.reliability)):
        old_r = float(old.reliability.get(modality, 0.0))
        new_r = float(new.reliability.get(modality, old_r))
        reliability[modality] = float(np.clip(new_r, old_r - rel_step, old_r + rel_step))

    return Params(
        temperature=temperature, reliability=reliability, pinned=new.pinned
    )


# ---------------------------------------------------------------------------
# stage 5: gate
# ---------------------------------------------------------------------------


def _per_class_mismatch(
    batch: CalibrationBatch,
    params: Params,
    *,
    min_group: int = 3,
) -> dict[int, float]:
    """Mismatch rate per **consensus** class.

    Grouping by consensus rather than label is what makes a single-class attack
    visible: the attack relabels its victims, so by label they disappear into
    the classes they were flipped into.
    """
    modalities = _modalities(batch)
    weights = _vote_weights(params, modalities)
    consensus, _ = _consensus_with_weights(batch, weights)
    if consensus is None:
        return {}
    labels = np.asarray(batch.labels, dtype=np.int64)
    valid = labels >= 0
    out: dict[int, float] = {}
    for cls in np.unique(consensus[valid]):
        idx = valid & (consensus == cls)
        if int(idx.sum()) < int(min_group):
            continue
        out[int(cls)] = float(np.mean(labels[idx] != consensus[idx]))
    return out


def _fused_accuracy(params: Params, batch: CalibrationBatch) -> float:
    """Accuracy of the fused decision (the ranking channel's real quality)."""
    modalities = _modalities(batch)
    weights = _vote_weights(params, modalities)
    labels = np.asarray(batch.labels, dtype=np.int64)
    valid = labels >= 0
    if not valid.any():
        return float("nan")
    fused = None
    for modality in modalities:
        w = float(weights.get(modality, 0.0))
        if w <= 0.0:
            continue
        temperature = max(float(params.temperature.get(modality, 1.0)), 1e-3)
        term = w * (_logits(batch, modality) / temperature)
        fused = term if fused is None else fused + term
    if fused is None:
        return float("nan")
    pred = fused.argmax(axis=1)
    return float(np.mean(pred[valid] == labels[valid]))


def gate(
    old: Params,
    new: Params,
    gate_set: CalibrationBatch,
    window_holdout: CalibrationBatch,
    *,
    thresholds: GateThresholds = GateThresholds(),
) -> GateDecision:
    """Two-criterion acceptance on **fused decision accuracy**.

    ``integrity_ok``  — ``new`` does not lose accuracy on the frozen gate set.
    ``adaptation_ok`` — ``new`` does not lose accuracy on the recent holdout.

    Both candidates are scored with the incumbent's vote weights, so the
    comparison is on a fixed scale; letting each candidate choose its own
    weighting would let it game the metric.

    Calibration error is not used here: the confidence map is invariant to a
    global temperature, so an ECE gate cannot separate candidates.  See
    :class:`GateThresholds`.
    """
    old_acc_g = _fused_accuracy(old, gate_set)
    new_acc_g = _fused_accuracy(new, gate_set)
    integrity_ok = _finite_at_least(new_acc_g, old_acc_g - thresholds.acc_tol)

    old_acc_w = _fused_accuracy(old, window_holdout)
    new_acc_w = _fused_accuracy(new, window_holdout)
    adaptation_ok = _finite_at_least(new_acc_w, old_acc_w - thresholds.acc_tol)

    accept = adaptation_ok and (integrity_ok or thresholds.drift_override)
    parts = []
    if not adaptation_ok:
        parts.append("no adaptation gain")
    if not integrity_ok:
        parts.append("integrity regression")
    if thresholds.drift_override and not integrity_ok:
        parts.append("drift override")
    return GateDecision(
        accept=accept,
        integrity_ok=integrity_ok,
        adaptation_ok=adaptation_ok,
        reason="; ".join(parts) if parts else "ok",
    )


def _finite_at_least(value: float, bound: float) -> bool:
    """``value >= bound`` treating NaN as failure."""
    if not np.isfinite(value):
        return False
    return bool(value >= bound)


def evaluate_params(
    params: Params,
    batch: CalibrationBatch,
    *,
    confidence_maps: Mapping[str, ConfidenceMap] | None = None,
) -> dict[str, float]:
    """Reliability-weighted prediction and calibration metrics.

    ``signed_gap = mean_confidence - accuracy``.  Reporting it (rather than ECE
    alone) makes the *direction* of miscalibration visible: negative means
    underconfident, positive means overconfident.

    When ``confidence_maps`` is supplied (fitted on a *different*, labelled
    window), ``ece_map`` and ``ece_map_fused`` report the separated confidence
    channel.  The temperature-based ``ece`` is kept for comparison; it is the
    ranking channel's by-product, not the calibration claim.
    """
    modalities = _modalities(batch)
    weights = _vote_weights(params, modalities)
    labels = np.asarray(batch.labels, dtype=np.int64)

    nll_num = ece_num = acc_num = conf_num = denom = 0.0
    plain_eces: list[float] = []
    map_num = map_denom = map_conf_num = map_acc_num = 0.0

    for modality in modalities:
        logits = _logits(batch, modality)
        valid = _valid_labels(batch, logits.shape[1])
        if not valid.any():
            continue
        temperature = max(float(params.temperature.get(modality, 1.0)), 1e-3)
        scaled = logits[valid] / temperature
        probs = _softmax(scaled)
        conf = probs.max(axis=1)
        ok = (probs.argmax(axis=1) == labels[valid]).astype(np.float64)
        ece = expected_calibration_error(conf, ok)
        plain_eces.append(ece)

        w = float(weights.get(modality, 0.0))
        if w <= 0.0:
            continue
        nll_num += w * _nll(scaled, labels[valid])
        ece_num += w * ece
        acc_num += w * float(ok.mean())
        conf_num += w * float(conf.mean())
        denom += w

        if confidence_maps is not None and modality in confidence_maps:
            mapped = apply_confidence_map(
                logits.max(axis=1)[valid], confidence_maps[modality]
            )
            map_num += w * expected_calibration_error(mapped, ok)
            map_denom += w
            map_conf_num += w * float(mapped.mean())
            map_acc_num += w * float(ok.mean())

    if denom <= 0.0:
        return {
            "nll": float("nan"),
            "ece": float("nan"),
            "ece_unweighted": float("nan"),
            "ece_map": float("nan"),
            "ece_map_fused": float("nan"),
            "signed_gap_map": float("nan"),
            "signed_gap_map_fused": float("nan"),
            "fused_accuracy": float("nan"),
            "accuracy": float("nan"),
            "mean_confidence": float("nan"),
            "signed_gap": float("nan"),
        }

    accuracy = float(acc_num / denom)
    confidence = float(conf_num / denom)

    ece_map = float(map_num / map_denom) if map_denom > 0.0 else float("nan")
    signed_gap_map = (
        float((map_conf_num - map_acc_num) / map_denom)
        if map_denom > 0.0
        else float("nan")
    )
    ece_map_fused = float("nan")
    signed_gap_map_fused = float("nan")
    fused_accuracy = float("nan")
    if confidence_maps is not None and "fused" in confidence_maps:
        fused_scores, fused_ok = _fused_top1(params, batch)
        if fused_scores is not None and fused_ok is not None:
            mapped = apply_confidence_map(fused_scores, confidence_maps["fused"])
            ece_map_fused = float(expected_calibration_error(mapped, fused_ok))
            signed_gap_map_fused = float(mapped.mean() - fused_ok.mean())
            fused_accuracy = float(fused_ok.mean())

    return {
        "nll": float(nll_num / denom),
        "ece": float(ece_num / denom),
        "ece_unweighted": float(np.mean(plain_eces)) if plain_eces else float("nan"),
        "ece_map": ece_map,
        "ece_map_fused": ece_map_fused,
        "signed_gap_map": signed_gap_map,
        "signed_gap_map_fused": signed_gap_map_fused,
        "fused_accuracy": fused_accuracy,
        "accuracy": accuracy,
        "mean_confidence": confidence,
        "signed_gap": float(confidence - accuracy),
    }


# ---------------------------------------------------------------------------
# state machine
# ---------------------------------------------------------------------------


def classify_state(
    evidence: Evidence,
    thresholds: TransitionThresholds = TransitionThresholds(),
) -> State:
    """Map evidence to the state machine.

    ABSTAIN is sticky at the pipeline level: this function may enter it but
    ``RobustCalibrationPipeline`` will not leave it without an external release.
    """
    labels_bad = evidence.label_consensus_agreement < thresholds.theta_label
    experts_bad = evidence.cross_modal_agreement < thresholds.theta_cross
    drifting = evidence.drift_score >= thresholds.theta_drift

    if evidence.label_coverage <= 0.0:
        # No valid label at all: nothing can be fitted or judged.
        return State.QUARANTINE
    if drifting and labels_bad:
        # The experts are confused *and* the feature reference moved: the world
        # changed, so adapt rather than freeze.  This is what separates heavy
        # drift from label poisoning.
        return State.RECALIBRATE
    if labels_bad and experts_bad:
        return State.ABSTAIN
    if labels_bad:
        return State.QUARANTINE
    if evidence.drift_score >= 0.5 * thresholds.theta_drift:
        # Any noticeable feature shift with intact labels is real drift, not
        # poisoning: poison changes labels, not the feature distribution.
        return State.RECALIBRATE
    if experts_bad:
        return State.RECALIBRATE
    # A class-concentrated attack: one class' agreement collapses while the
    # global average still looks healthy.
    if (
        evidence.label_consensus_agreement >= thresholds.theta_global_ok
        and evidence.class_agreement_spread >= thresholds.theta_spread
    ):
        return State.QUARANTINE
    # Too-good-to-be-true accuracy is the signature of "agree with the model"
    # poisoning: relabelling wrong predictions to the model's own answer.
    if evidence.accuracy_inflation >= thresholds.theta_inflation:
        return State.QUARANTINE
    return State.NORMAL


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


@dataclass
class RobustCalibrationPipeline:
    """Runs the five stages in order and owns the state machine.

    ``gate_set`` must be frozen, human-audited and independent of any fit.
    ``window_holdout`` is the verification slice of the recent window; it must
    not overlap the fit samples.
    """

    params: Params
    gate_set: CalibrationBatch | None = None
    thresholds: TransitionThresholds = field(default_factory=TransitionThresholds)
    gate_thresholds: GateThresholds = field(default_factory=GateThresholds)
    gate_every: int = 32
    state: State = State.NORMAL
    cusum: "CusumBank | None" = None
    # Per-class detectors catch single-class attacks that the global agreement
    # dilutes to ~0.4 sd per window.
    per_class_cusum: "PerClassCusum | None" = None
    # Slow-degradation guard: the per-window gate tolerance (1pp) cannot see a
    # steady 0.9pp/window slide.  A CUSUM on fused accuracy can.
    accuracy_cusum: "CusumDetector | None" = None
    steps: int = 0
    gate_failures: int = 0
    _recover: int = field(default=0, init=False, repr=False)
    history: list[dict] = field(default_factory=list)

    def step(
        self,
        batch: CalibrationBatch,
        window_holdout: CalibrationBatch | None = None,
        *,
        force_gate: bool = False,
    ) -> dict:
        trust = source_health(batch)

        # Pass 1 — shadow evidence with neutral weights.  None of the state
        # signals depend on the sample weights, so classifying here is exact,
        # not an approximation.
        shadow = SampleWeights(per_sample=np.ones(len(batch.labels)))
        pre = self.evidence(batch, shadow)
        observed = classify_state(pre, self.thresholds)
        if self.state is State.ABSTAIN:
            # Auto-release after a run of NORMAL windows; manual release stays
            # available and is audited the same way.
            if observed is State.NORMAL:
                self._recover += 1
            else:
                self._recover = 0
            if self._recover >= int(self.thresholds.abstain_release_windows):
                self.state = State.NORMAL
                self._recover = 0
                self.history.append(
                    {"event": "auto_release_abstain", "after_windows": int(self._recover)}
                )
        else:
            self.state = observed

        # Pass 2 — screen with the state.  Under RECALIBRATE the labels are
        # trusted, so drift no longer deletes the model's own errors from the
        # fit (the circular dependency that blocked adaptation).
        weights = score_samples(batch, self.params, trust, state=self.state)
        evidence = replace(
            pre,
            suspect_mass=float(1.0 - np.mean(weights.per_sample))
            if len(weights.per_sample)
            else 0.0,
        )

        # A single window cannot see a small targeted attack; the CUSUM can.
        # Drift is excluded: it moves the label/consensus relation for a
        # legitimate reason, and feeding it to a poison detector would alarm on
        # every drift and block adaptation.
        cusum_reasons: list[str] = []
        drifting = evidence.drift_score >= 0.5 * self.thresholds.theta_drift
        if self.cusum is not None and not drifting:
            cusum_reasons = self.cusum.update(evidence.label_consensus_agreement)
            if cusum_reasons and self.state is not State.ABSTAIN:
                self.state = State.QUARANTINE

        per_class_alarm: list[int] = []
        if self.per_class_cusum is not None and not drifting:
            mixups = _per_class_mismatch(batch, self.params)
            per_class_alarm = self.per_class_cusum.update(mixups)
            if per_class_alarm and self.state is not State.ABSTAIN:
                self.state = State.QUARANTINE

        # Slow decision-quality decay, invisible to the per-window gate.
        accuracy_alarm = False
        if self.accuracy_cusum is not None:
            accuracy_alarm = self.accuracy_cusum.update(
                _fused_accuracy(self.params, batch)
            )
            if accuracy_alarm and self.state is State.NORMAL:
                self.state = State.RECALIBRATE

        candidate = bound_step(self.params, fit_params(batch, weights), evidence)
        decision = GateDecision(True, True, True, "gate not due")
        frozen = (
            dict(candidate.temperature) == dict(self.params.temperature)
            and dict(candidate.reliability) == dict(self.params.reliability)
        )

        if self.state in (State.QUARANTINE, State.ABSTAIN):
            # No update is accepted while parameters are frozen.  Reporting the
            # gate's "accept" here would claim an update that never happened.
            decision = GateDecision(
                False, False, False, f"frozen by {self.state.value}"
            )
        elif self._gate_due(force_gate):
            holdout = window_holdout if window_holdout is not None else batch
            gate_set = self.gate_set if self.gate_set is not None else holdout
            thresholds = replace(
                self.gate_thresholds,
                drift_override=self.state is State.RECALIBRATE,
            )
            decision = gate(self.params, candidate, gate_set, holdout, thresholds=thresholds)
            if decision.accept:
                self.params = candidate
                self.gate_failures = 0
            else:
                self.gate_failures += 1
                if self.gate_failures >= self.thresholds.gate_fail_abstain:
                    self.state = State.ABSTAIN
        else:
            if frozen:
                decision = GateDecision(False, True, True, "no change")
            else:
                self.params = candidate

        self.steps += 1
        record = {
            "step": int(self.steps),
            "state": self.state.value,
            "suspect_mass": float(evidence.suspect_mass),
            "cross_modal_agreement": float(evidence.cross_modal_agreement),
            "label_consensus_agreement": float(evidence.label_consensus_agreement),
            "drift_score": float(evidence.drift_score),
            "accuracy_inflation": float(evidence.accuracy_inflation),
            "class_agreement_spread": float(evidence.class_agreement_spread),
            "label_coverage": float(evidence.label_coverage),
            "gate": {
                "accept": bool(decision.accept),
                "integrity_ok": bool(decision.integrity_ok),
                "adaptation_ok": bool(decision.adaptation_ok),
                "reason": decision.reason,
            },
            "temperature": dict(self.params.temperature),
            "reliability": dict(self.params.reliability),
            "pinned": sorted(self.params.pinned),
            "n_effective": float(np.sum(weights.per_sample)),
            "reason_rates": {
                reason: float(np.mean(mask)) for reason, mask in weights.reasons.items()
            },
            "cusum_alarm": list(cusum_reasons),
            "per_class_alarm": list(per_class_alarm),
            "accuracy_alarm": bool(accuracy_alarm),
        }
        self.history.append(record)
        return record

    def release_abstain(self, *, reason: str) -> None:
        """Leave ABSTAIN.  Only external signals may call this."""
        if self.state is State.ABSTAIN:
            self.state = State.NORMAL
            self.gate_failures = 0
            self.history.append({"event": "release_abstain", "reason": str(reason)})

    def evidence(self, batch: CalibrationBatch, weights: SampleWeights) -> Evidence:
        consensus, strength = _consensus_with_weights(
            batch, _vote_weights(self.params, _modalities(batch))
        )
        labels = np.asarray(batch.labels, dtype=np.int64)
        valid = labels >= 0 if consensus is not None else np.zeros(len(labels), dtype=bool)

        cross = float(strength.mean()) if len(strength) else 0.0
        label_agree = (
            float(np.mean(labels[valid] == consensus[valid]))
            if consensus is not None and valid.any()
            else 0.0
        )

        drift = 0.0
        inflation = 0.0
        if self.gate_set is not None:
            shifts: list[float] = []
            rises: list[float] = []
            for modality in _modalities(batch):
                ref_fit = _proto_fit_median(self.gate_set, modality)
                cur_fit = _proto_fit_median(batch, modality)
                if np.isfinite(ref_fit) and np.isfinite(cur_fit):
                    shifts.append(max(0.0, ref_fit - cur_fit))
                ref = _raw_accuracy(self.gate_set, modality)
                cur = _raw_accuracy(batch, modality)
                if np.isfinite(ref) and np.isfinite(cur):
                    rises.append(max(0.0, cur - ref))
            # Class-coverage drop catches label shift, where the feature
            # distribution barely moves but the observed class set shrinks.
            ref_classes = {
                int(v) for v in np.asarray(self.gate_set.labels) if int(v) >= 0
            }
            cur_classes = {int(v) for v in labels if int(v) >= 0}
            if ref_classes:
                coverage_keep = len(ref_classes & cur_classes) / float(len(ref_classes))
                shifts.append(1.0 - coverage_keep)
            drift = float(max(shifts)) if shifts else 0.0
            inflation = float(max(rises)) if rises else 0.0

        valid = labels >= 0
        label_coverage = float(valid.mean()) if len(labels) else 0.0

        spread = 0.0
        if consensus is not None and valid.any():
            per_class = []
            # Group by *consensus* class, not by label class: a single-class
            # attack relabels its victims, so grouping by label would hide the
            # attack inside the classes it spills into.
            for cls in np.unique(consensus[valid]):
                idx = valid & (consensus == cls)
                if int(idx.sum()) < 10:
                    # Small groups make the rate too noisy to test.
                    continue
                per_class.append(float(np.mean(labels[idx] == consensus[idx])))
            if len(per_class) >= 2:
                spread = float(np.mean(per_class) - np.min(per_class))

        return Evidence(
            cross_modal_agreement=cross,
            label_consensus_agreement=label_agree,
            drift_score=drift,
            suspect_mass=float(1.0 - np.mean(weights.per_sample))
            if len(weights.per_sample)
            else 0.0,
            accuracy_inflation=inflation,
            class_agreement_spread=spread,
            label_coverage=label_coverage,
        )

    def _gate_due(self, force: bool) -> bool:
        if force or self.state is State.RECALIBRATE:
            return True
        return self.steps > 0 and self.steps % max(1, int(self.gate_every)) == 0
