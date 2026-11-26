"""Properties evaluated against any :class:`~pss_eval.protocol.SystemUnderTest`.

Every property receives a *factory* (not an instance), so it can build two
independent systems for determinism checks and a fresh system after a reset.
A property returns a :class:`PropertyResult` with status ``pass``, ``fail`` or
``skip``.  ``skip`` means the system does not implement a required optional
capability; it is never an implicit pass.
"""

from __future__ import annotations

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
    "ContractProperty",
    "DeterminismProperty",
    "Property",
    "PropertyResult",
    "ResetExactnessProperty",
    "StateHygieneProperty",
    "default_properties",
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


def default_properties() -> list[Property]:
    """The suite used by the CLI: three smoke properties + one contract probe."""

    return [
        DeterminismProperty(),
        ResetExactnessProperty(),
        StateHygieneProperty(),
        ContractProperty(),
    ]
