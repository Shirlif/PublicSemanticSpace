"""Properties evaluated against any :class:`~pss_eval.protocol.SystemUnderTest`.

Every property receives a *factory* (not an instance), so it can build two
independent systems for determinism checks and a fresh system after a reset.
A property returns a :class:`PropertyResult` with status ``pass``, ``fail`` or
``skip``.  ``skip`` means the system does not implement a required optional
capability; it is never an implicit pass.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .fixtures import LabeledSample, Stream
from .protocol import (
    Prediction,
    SystemUnderTest,
    capabilities_of,
    iter_arrays,
    predictions_equal,
    state_digest,
    state_size_bytes,
)

__all__ = [
    "BudgetProperty",
    "ContractProperty",
    "DeterminismProperty",
    "IdentityRetentionProperty",
    "LabelAccuracyProperty",
    "LatencyProperty",
    "Property",
    "PropertyResult",
    "ResetExactnessProperty",
    "StateHygieneProperty",
    "SwapRollbackProperty",
    "default_properties",
    "extended_properties",
]

SystemFactory = Callable[[], SystemUnderTest]

PASS = "pass"
FAIL = "fail"
SKIP = "skip"

MAX_STATE_BYTES = 256 * 1024


@dataclass(frozen=True)
class PropertyResult:
    name: str
    status: str
    metrics: dict[str, float]
    notes: str = ""
    smoke: bool = True

    def to_dict(self) -> dict:
        return {
            "property": self.name,
            "status": self.status,
            "smoke": bool(self.smoke),
            "metrics": {k: float(v) for k, v in self.metrics.items()},
            "notes": self.notes,
        }


class Property(Protocol):
    name: str
    smoke: bool

    def evaluate(
        self,
        factory: SystemFactory,
        stream: Stream,
        *,
        seed: int,
    ) -> PropertyResult:
        ...


def _replay(
    system: SystemUnderTest,
    samples: tuple[LabeledSample, ...],
    *,
    seed: int,
) -> list[Prediction]:
    system.reset(seed=seed)
    predictions = []
    for sample in samples:
        system.ingest(sample.observation)
        predictions.append(system.step())
    return predictions


def _ratio(flags: list[bool]) -> float:
    return float(sum(1 for flag in flags if flag)) / float(len(flags)) if flags else 0.0


# ── smoke properties ───────────────────────────────────────────────────────


class DeterminismProperty:
    """Same seed + same stream => same predictions and same final state."""

    name = "determinism"
    smoke = True

    def evaluate(
        self,
        factory: SystemFactory,
        stream: Stream,
        *,
        seed: int,
    ) -> PropertyResult:
        if len(stream) == 0:
            return PropertyResult(self.name, FAIL, {"agreement": 0.0}, "empty stream")

        left = factory()
        right = factory()
        preds_left = _replay(left, stream.samples, seed=seed)
        preds_right = _replay(right, stream.samples, seed=seed)

        agreement = _ratio(
            [
                predictions_equal(a, b)
                for a, b in zip(preds_left, preds_right)
            ]
        )
        id_ok = _ratio(
            [
                pred.observation_id == sample.observation.observation_id
                for pred, sample in zip(preds_left, stream.samples)
            ]
        )
        state_match = float(state_digest(left.state()) == state_digest(right.state()))
        passed = agreement == 1.0 and id_ok == 1.0 and state_match == 1.0
        return PropertyResult(
            self.name,
            PASS if passed else FAIL,
            {
                "agreement": agreement,
                "observation_id_match": id_ok,
                "final_state_match": state_match,
            },
        )


class ResetExactnessProperty:
    """``reset`` returns to the same state as a freshly constructed system."""

    name = "reset_exactness"
    smoke = True

    def evaluate(
        self,
        factory: SystemFactory,
        stream: Stream,
        *,
        seed: int,
    ) -> PropertyResult:
        prefix = stream.samples[: min(8, len(stream.samples))]
        if not prefix:
            return PropertyResult(self.name, FAIL, {"reset_exact": 0.0}, "empty stream")

        used = factory()
        _replay(used, prefix, seed=seed)
        used.reset(seed=seed)

        fresh = factory()
        fresh.reset(seed=seed)

        reset_exact = float(state_digest(used.state()) == state_digest(fresh.state()))

        preds_used = _replay(used, prefix, seed=seed)
        preds_fresh = _replay(fresh, prefix, seed=seed)
        replay_agreement = _ratio(
            [predictions_equal(a, b) for a, b in zip(preds_used, preds_fresh)]
        )
        passed = reset_exact == 1.0 and replay_agreement == 1.0
        return PropertyResult(
            self.name,
            PASS if passed else FAIL,
            {"reset_exact": reset_exact, "replay_agreement": replay_agreement},
        )


class StateHygieneProperty:
    """State is serialisable, finite, and not absurdly large."""

    name = "state_hygiene"
    smoke = True

    def evaluate(
        self,
        factory: SystemFactory,
        stream: Stream,
        *,
        seed: int,
    ) -> PropertyResult:
        system = factory()
        _replay(system, stream.samples, seed=seed)
        state = system.state()
        if not isinstance(state, dict):
            return PropertyResult(
                self.name, FAIL, {"finite_arrays": 0.0}, "state() did not return a dict"
            )

        arrays = list(iter_arrays(state))
        finite = (
            _ratio([bool(np.all(np.isfinite(array))) for array in arrays])
            if arrays
            else 1.0
        )

        try:
            digest_ok = float(len(state_digest(state)) == 64)
            size = state_size_bytes(state)
            size_ok = float(size <= MAX_STATE_BYTES)
        except ValueError as exc:  # NaN/Infinity in a scalar
            digest_ok = 0.0
            size_ok = 0.0
            size = -1
            note = f"non-finite state: {exc}"
        else:
            note = ""

        passed = finite == 1.0 and digest_ok == 1.0 and size_ok == 1.0
        return PropertyResult(
            self.name,
            PASS if passed else FAIL,
            {
                "finite_arrays": finite,
                "digest_ok": digest_ok,
                "within_size_bound": size_ok,
                "state_bytes": float(size),
            },
            note,
        )


# ── contract property (expected to fail for contract-free systems) ─────────


class ContractProperty:
    """Snapshot/rollback, invalid-module rejection, and audit presence.

    The property is only applicable when the system implements all three
    optional capabilities.  A contract-free system is expected to fail it: it
    accepts invalid modules, cannot roll back, and records no audit.
    """

    name = "contract"
    smoke = False

    def evaluate(
        self,
        factory: SystemFactory,
        stream: Stream,
        *,
        seed: int,
    ) -> PropertyResult:
        system = factory()
        missing = sorted({"snapshot", "swap", "audit"} - set(capabilities_of(system)))
        if missing:
            return PropertyResult(
                self.name,
                SKIP,
                {},
                f"missing capabilities: {', '.join(missing)}",
                smoke=False,
            )

        prefix = stream.samples[: min(4, len(stream.samples))]
        _replay(system, prefix, seed=seed)
        before = state_digest(system.state())

        token = system.snapshot()
        accepted = bool(
            system.swap(
                {
                    "modality": "vision",
                    "kind": "quality",
                    "quality": "Q2",
                    "version": "eval-probe-v1",
                    "valid": True,
                }
            )
        )

        before_invalid = state_digest(system.state())
        invalid_accepted = bool(
            system.swap(
                {
                    "modality": "vision",
                    "kind": "invalid_dim",
                    "version": "eval-invalid-v1",
                    "valid": False,
                }
            )
        )
        after_invalid = state_digest(system.state())
        invalid_keeps_state = float(after_invalid == before_invalid)

        rolled = bool(system.rollback(token))
        after_rollback = state_digest(system.state())
        rollback_exact = float(rolled and after_rollback == before)

        entries = system.audit()
        audit_present = float(
            isinstance(entries, list) and len(entries) > 0
            and all(isinstance(entry, dict) for entry in entries)
        )

        metrics = {
            "swap_accepted": float(accepted),
            "invalid_rejected": float(not invalid_accepted),
            "invalid_keeps_state": invalid_keeps_state,
            "rollback_exact": rollback_exact,
            "audit_present": audit_present,
        }
        passed = all(value == 1.0 for value in metrics.values())
        return PropertyResult(
            self.name,
            PASS if passed else FAIL,
            metrics,
            smoke=False,
        )


# ── swap / rollback ────────────────────────────────────────────────────────


class SwapRollbackProperty:
    """Repeated valid swaps, exact rollback, and a swap after rollback.

    ``contract`` is the admission probe (valid/invalid modules, audit); this
    property stresses the swap *stack*: three distinct valid modules are
    installed, then one rollback to the snapshot must restore both the state
    digest and the active module versions, and a further swap must still work.
    """

    name = "swap_rollback"
    smoke = False

    def evaluate(
        self,
        factory: SystemFactory,
        stream: Stream,
        *,
        seed: int,
    ) -> PropertyResult:
        system = factory()
        missing = sorted({"snapshot", "swap"} - set(capabilities_of(system)))
        if missing:
            return PropertyResult(
                self.name,
                SKIP,
                {},
                f"missing capabilities: {', '.join(missing)}",
                smoke=False,
            )

        prefix = stream.samples[: min(4, len(stream.samples))]
        _replay(system, prefix, seed=seed)
        before = state_digest(system.state())
        token = system.snapshot()

        accepted = [
            bool(
                system.swap(
                    {
                        "modality": "vision",
                        "kind": "quality",
                        "quality": "Q2",
                        "version": f"eval-swap-{index}",
                        "valid": True,
                    }
                )
            )
            for index in range(3)
        ]
        swapped_digest = state_digest(system.state())
        rolled = bool(system.rollback(token))
        after = state_digest(system.state())
        post_rollback = bool(
            system.swap(
                {
                    "modality": "vision",
                    "kind": "quality",
                    "quality": "Q3",
                    "version": "eval-post-rollback",
                    "valid": True,
                }
            )
        )

        metrics = {
            "swap_success_rate": _ratio(accepted),
            "swap_changed_state": float(swapped_digest != before),
            "rollback_exact": float(rolled and after == before),
            "post_rollback_swap": float(post_rollback),
        }
        passed = all(value == 1.0 for value in metrics.values())
        return PropertyResult(
            self.name, PASS if passed else FAIL, metrics, smoke=False
        )


# ── identity retention / label accuracy ────────────────────────────────────


class IdentityRetentionProperty:
    """Detections of the same physical instance keep the same track id.

    Requires the system's ``identity_assignments()`` capability and a stream
    carrying harness-side ``identity_labels`` (detection ref -> instance id).
    Unassigned detections count against retention.  A stream without identity
    labels, or a system without the capability, is skipped -- never passed.
    """

    name = "identity_retention"
    smoke = False

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = float(threshold)

    def evaluate(
        self,
        factory: SystemFactory,
        stream: Stream,
        *,
        seed: int,
    ) -> PropertyResult:
        if "identity" not in capabilities_of(factory()):
            return PropertyResult(
                self.name, SKIP, {}, "missing capability: identity", smoke=False
            )
        labels = dict(getattr(stream, "identity_labels", {}) or {})
        if not labels:
            return PropertyResult(
                self.name, SKIP, {}, "stream has no identity labels", smoke=False
            )

        system = factory()
        _replay(system, stream.samples, seed=seed)
        assignments = dict(system.identity_assignments())

        by_instance: dict[str, list[str | None]] = defaultdict(list)
        for ref, instance in labels.items():
            by_instance[str(instance)].append(assignments.get(str(ref)))

        consistent = 0
        total = 0
        per_instance: list[float] = []
        for tracks in by_instance.values():
            assigned = [track for track in tracks if track is not None]
            if not assigned:
                continue
            modal = Counter(assigned).most_common(1)[0][1]
            consistent += int(modal)
            total += len(tracks)  # unassigned detections count against retention
            per_instance.append(float(modal) / float(len(tracks)))

        if total == 0:
            return PropertyResult(
                self.name,
                SKIP,
                {},
                "no detections with an entity assignment",
                smoke=False,
            )

        retention = float(consistent) / float(total)
        instance_purity = float(np.mean(per_instance)) if per_instance else 0.0
        passed = retention >= self.threshold
        return PropertyResult(
            self.name,
            PASS if passed else FAIL,
            {
                "identity_retention": retention,
                "instance_purity": instance_purity,
                "n_instances": float(len(per_instance)),
                "n_detections": float(total),
                "threshold": self.threshold,
            },
            smoke=False,
        )


class LabelAccuracyProperty:
    """Prediction-label accuracy against the harness's scalar labels.

    Skips unless *every* sample has a scalar label and every prediction emits
    a label.  Accuracy must beat the uniform-chance baseline.
    """

    name = "label_accuracy"
    smoke = False

    def evaluate(
        self,
        factory: SystemFactory,
        stream: Stream,
        *,
        seed: int,
    ) -> PropertyResult:
        if len(stream) == 0:
            return PropertyResult(self.name, SKIP, {}, "empty stream", smoke=False)
        if any(sample.label is None for sample in stream.samples):
            return PropertyResult(
                self.name, SKIP, {}, "stream has no scalar labels", smoke=False
            )

        system = factory()
        predictions = _replay(system, stream.samples, seed=seed)
        if any(prediction.label is None for prediction in predictions):
            return PropertyResult(
                self.name, SKIP, {}, "system emits no labels", smoke=False
            )

        correct = sum(
            1
            for prediction, sample in zip(predictions, stream.samples)
            if prediction.label == sample.label
        )
        accuracy = float(correct) / float(len(predictions))
        n_classes = max(1, len(set(sample.label for sample in stream.samples)))
        chance = 1.0 / float(n_classes)
        passed = accuracy >= chance
        return PropertyResult(
            self.name,
            PASS if passed else FAIL,
            {"accuracy": accuracy, "chance": chance, "n": float(len(predictions))},
            smoke=False,
        )


# ── budget / anytime and latency ───────────────────────────────────────────


class BudgetProperty:
    """Anytime deadline budget: caller latency stays bounded, timeouts counted.

    Requires the system's ``budget()`` / ``budget_stats()`` capability with a
    positive configured budget.  The guard is allowed to degrade (return early)
    on a miss; what is checked is that the caller's p99 stays within the
    budget and the timeout accounting is internally consistent.
    """

    name = "budget"
    smoke = False

    def __init__(self, tolerance_ms: float = 2.0) -> None:
        self.tolerance_ms = float(tolerance_ms)

    def evaluate(
        self,
        factory: SystemFactory,
        stream: Stream,
        *,
        seed: int,
    ) -> PropertyResult:
        if "budget" not in capabilities_of(factory()):
            return PropertyResult(
                self.name, SKIP, {}, "missing capability: budget", smoke=False
            )
        probe = factory()
        config = dict(probe.budget() or {})
        budget_ms = float(config.get("budget_ms", 0.0) or 0.0)
        reserve_ms = float(config.get("reserve_ms", 0.0) or 0.0)
        if budget_ms <= 0.0:
            return PropertyResult(
                self.name, SKIP, {}, "budget is not enabled", smoke=False
            )

        system = factory()
        _replay(system, stream.samples, seed=seed)
        stats = dict(system.budget_stats() or {})
        elapsed = dict(stats.get("elapsed_ms", {}) or {})
        frames = int(stats.get("frames", 0))
        timeouts = int(stats.get("timeouts", 0))
        errors = int(stats.get("errors", 0))
        p99 = float(elapsed.get("p99", float("inf")))
        timeout_rate = float(stats.get("timeout_rate", 0.0))

        bounded = bool(np.isfinite(p99)) and p99 <= budget_ms + self.tolerance_ms
        frames_ok = frames == len(stream)
        rate_ok = abs(timeout_rate - (timeouts / max(1, frames))) <= 1e-9
        config_ok = budget_ms > reserve_ms
        passed = bounded and frames_ok and rate_ok and config_ok and errors == 0
        return PropertyResult(
            self.name,
            PASS if passed else FAIL,
            {
                "budget_ms": budget_ms,
                "elapsed_p50_ms": float(elapsed.get("p50", 0.0)),
                "elapsed_p99_ms": p99,
                "timeout_rate": timeout_rate,
                "frames": float(frames),
                "errors": float(errors),
            },
            smoke=False,
        )


class LatencyProperty:
    """Per-observation wall time, reported as p50 / p99.

    This property is intentionally a loose ceiling: it checks that latency is
    finite and bounded, and carries the actual p50 / p99 in its metrics.  The
    strict deadline guarantee is the ``budget`` property.
    """

    name = "latency"
    smoke = False

    def __init__(self, ceiling_ms: float = 1000.0) -> None:
        self.ceiling_ms = float(ceiling_ms)

    def evaluate(
        self,
        factory: SystemFactory,
        stream: Stream,
        *,
        seed: int,
    ) -> PropertyResult:
        system = factory()
        system.reset(seed=seed)
        latencies: list[float] = []
        for sample in stream.samples:
            started = time.perf_counter()
            system.ingest(sample.observation)
            system.step()
            latencies.append((time.perf_counter() - started) * 1000.0)
        if not latencies:
            return PropertyResult(self.name, SKIP, {}, "empty stream", smoke=False)

        values = np.asarray(latencies, dtype=np.float64)
        p50 = float(np.percentile(values, 50))
        p99 = float(np.percentile(values, 99))
        maximum = float(values.max())
        finite = bool(np.all(np.isfinite(values)))
        passed = finite and p99 <= self.ceiling_ms
        return PropertyResult(
            self.name,
            PASS if passed else FAIL,
            {
                "p50_ms": p50,
                "p99_ms": p99,
                "max_ms": maximum,
                "n": float(values.size),
            },
            smoke=False,
        )


def default_properties() -> list[Property]:
    """The original F5 suite: three smoke properties + one contract probe."""

    return [
        DeterminismProperty(),
        ResetExactnessProperty(),
        StateHygieneProperty(),
        ContractProperty(),
    ]


def extended_properties() -> list[Property]:
    """The full protocol suite used for the three-way comparison.

    Keeps the original four properties first so reports stay comparable, then
    adds swap-stack, identity, label-accuracy, budget and latency probes.
    """

    return [
        DeterminismProperty(),
        ResetExactnessProperty(),
        StateHygieneProperty(),
        ContractProperty(),
        SwapRollbackProperty(),
        IdentityRetentionProperty(),
        LabelAccuracyProperty(),
        BudgetProperty(),
        LatencyProperty(),
    ]
