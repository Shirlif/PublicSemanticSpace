"""Bridge ``ContactEvent`` objects into the PSS cognitive runtime (C5).

This is the *opt-in* seam between the contact-event layer and the runtime:

    ContactEvent (model.py, status/confidence/evidence)
        -> Relation       (predicate="contact", origin="contact_event")
        -> CognitiveState (relations + audit entry)
        -> UncertaintyService (low-confidence events -> review / abstain)

Nothing here is a runtime default and no state schema is changed.  The caller
explicitly invokes :func:`apply_events`; the existing symbolic / learned
relation paths are untouched.

Evidence transport
------------------
``Relation`` (``pss/runtime/state.py``) has no ``attributes`` / ``metadata``
field.  To carry provenance without adding a schema field, this layer attaches
the same payload as **instance attributes** on the returned ``Relation``:

    relation.attributes  # dict: event_id, status, onset/offset, evidence, ...
    relation.metadata    # alias of the same dict

``Relation.to_dict()`` still emits only the seven original fields, so the
runtime's serialized view is unchanged.  Callers that need the evidence read
``getattr(relation, "attributes", {})`` (see :func:`relation_metadata`).

Confidence bands
----------------
``confidence`` is the event's fused confidence, not a calibrated probability
(the C2 calibration is the caller's responsibility).  This layer only routes
it:

* ``status`` in ``{"contradicted", "expired"}`` -> rejected, never written;
* ``confidence < abstain_threshold``            -> abstain, never written;
* ``confidence < review_threshold``             -> written, flagged ``review``;
* otherwise                                     -> written, ``accept``.

``review_threshold`` defaults to the runtime's ``UncertaintyService``
relation threshold (0.5).  A ``review`` relation is still queryable (with its
low probability); an ``abstain`` event is kept only in the audit payload.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from pss.reasoning.relations import RelationRecord
from pss.runtime.state import CognitiveState, Relation

NON_ASSERTING_STATUSES = frozenset({"contradicted", "expired"})
DEFAULT_ABSTAIN_THRESHOLD = 0.25
DEFAULT_REVIEW_THRESHOLD = 0.5
AUDIT_ACTION = "contact_event_ingest"


# ── JSON-safe helpers ────────────────────────────────────────────────────────


def _jsonable(value, ndigits: int = 6):
    """Convert numpy / nested evidence payloads to plain JSON-safe values."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return round(value, ndigits)
    if isinstance(value, np.ndarray):
        return [_jsonable(v, ndigits) for v in value.reshape(-1).tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item(), ndigits)
    if isinstance(value, dict):
        return {str(k): _jsonable(v, ndigits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v, ndigits) for v in value]
    if isinstance(value, (set, frozenset)):
        return [_jsonable(v, ndigits) for v in sorted(value, key=str)]
    return str(value)


def _evidence_rows(event) -> list[dict]:
    rows: list[dict] = []
    for item in getattr(event, "evidence", None) or []:
        rows.append(
            {
                "source": str(getattr(item, "source", "unknown")),
                "timestamp": _jsonable(getattr(item, "timestamp", None)),
                "score": _jsonable(getattr(item, "score", None)),
                "intensity": _jsonable(getattr(item, "intensity", None)),
                "location": _jsonable(getattr(item, "location", None)),
                "reliability": _jsonable(getattr(item, "reliability", None)),
                "metadata": _jsonable(dict(getattr(item, "metadata", {}) or {})),
            }
        )
    return rows


def _event_source(event) -> str:
    """Best-effort source label: provenance first, then evidence channels."""
    provenance = dict(getattr(event, "provenance", {}) or {})
    source = provenance.get("source") or provenance.get("channel")
    if source:
        return str(source)
    channels = sorted(
        {str(getattr(item, "source", "unknown")) for item in getattr(event, "evidence", None) or []}
    )
    return "+".join(channels) if channels else "unknown"


# ── event -> relation ────────────────────────────────────────────────────────


def event_to_relation(event) -> Relation:
    """Map a ``ContactEvent`` to a runtime ``Relation``.

    ``predicate="contact"``, ``probability=event.confidence``,
    ``origin="contact_event"``.  The event's evidence/provenance and timing
    fields are attached as ``relation.attributes`` / ``relation.metadata``;
    the state schema is not extended.
    """

    confidence = float(np.clip(float(getattr(event, "confidence", 0.0)), 0.0, 1.0))
    onset = getattr(event, "onset", 0.0)
    offset = getattr(event, "offset", None)
    duration = max(0.0, float(offset if offset is not None else onset) - float(onset))
    metadata = {
        "event_id": str(getattr(event, "event_id", "")),
        "status": str(getattr(event, "status", "predicted")),
        "onset": _jsonable(onset),
        "offset": _jsonable(offset),
        "duration": _jsonable(duration),
        "location": _jsonable(getattr(event, "location", None)),
        "intensity": _jsonable(getattr(event, "intensity", None)),
        "event_version": int(getattr(event, "version", 1)),
        "evidence": _evidence_rows(event),
        "provenance": _jsonable(dict(getattr(event, "provenance", {}) or {})),
    }
    relation = Relation(
        source=str(event.subject),
        target=str(event.object),
        relation_type="contact_event",
        predicate="contact",
        probability=confidence,
        origin="contact_event",
        frame="world",
    )
    # Instance-level carrier, not a schema change.  Relation has no
    # attributes/metadata field and to_dict() still returns the 7 base fields.
    relation.attributes = metadata
    relation.metadata = metadata
    return relation


def relation_metadata(relation: Relation) -> dict:
    """Return the contact-event payload attached by :func:`event_to_relation`."""
    payload = getattr(relation, "metadata", None)
    if payload is None:
        payload = getattr(relation, "attributes", None)
    return dict(payload or {})


# ── decision / uncertainty ───────────────────────────────────────────────────


def _review_threshold(runtime, override: float | None) -> float:
    if override is not None:
        return float(override)
    service = getattr(runtime, "uncertainty", None)
    threshold = getattr(service, "relation_threshold", DEFAULT_REVIEW_THRESHOLD)
    return float(threshold)


def _decide(event, review_threshold: float, abstain_threshold: float) -> str:
    status = str(getattr(event, "status", "predicted"))
    confidence = float(np.clip(float(getattr(event, "confidence", 0.0)), 0.0, 1.0))
    if status in NON_ASSERTING_STATUSES:
        return "rejected"
    if confidence < abstain_threshold:
        return "abstain"
    if confidence < review_threshold:
        return "review"
    return "accept"


def _remember_in_layer(runtime, relation: Relation) -> bool:
    """Mirror the edge in ``RelationFactLayer``'s record cache.

    ``add_symbolic`` normalises ``origin`` to symbolic/geometry, so the layer
    writes the runtime relation directly and only mirrors the record here.
    ``_records`` is not part of the layer snapshot (existing runtime
    behaviour); rollback restores ``CognitiveState.relations`` only.
    """

    layer = getattr(runtime, "relations", None)
    remember = getattr(layer, "_remember", None)
    if remember is None:
        return False
    try:
        remember(
            RelationRecord(
                subject=relation.source,
                predicate=relation.predicate,
                object=relation.target,
                confidence=float(relation.probability),
                source="contact_event",
                frame=relation.frame,
            )
        )
    except Exception:
        return False
    return True


def _assess_uncertainty(
    runtime,
    state: CognitiveState,
    *,
    review_count: int,
    abstain_count: int,
):
    service = getattr(runtime, "uncertainty", None)
    if service is None or not bool(getattr(runtime, "uncertainty_enabled", True)):
        return None
    conflicts = {}
    layer = getattr(runtime, "relations", None)
    if layer is not None and hasattr(layer, "conflicts"):
        try:
            conflicts = layer.conflicts()
        except Exception:
            conflicts = {}
    report = service.assess(state, fact_conflicts=conflicts)
    if abstain_count:
        if "contact_event_abstain" not in report.flags:
            report.flags.append("contact_event_abstain")
        report.recommendation = "abstain"
    elif review_count:
        if "contact_event_review" not in report.flags:
            report.flags.append("contact_event_review")
        if report.recommendation == "accept":
            report.recommendation = "review"
    return report


# ── runtime ingress ──────────────────────────────────────────────────────────


def apply_events(
    runtime,
    events,
    *,
    commit: bool = True,
    review_threshold: float | None = None,
    abstain_threshold: float = DEFAULT_ABSTAIN_THRESHOLD,
    actor: str = "contact_event",
    reason: str = "contact events -> runtime",
) -> dict:
    """Write contact events into ``runtime.state`` (and the relation cache).

    ``commit=True`` appends one ``contact_event_ingest`` audit entry and bumps
    the runtime version, so ``runtime.rollback(before_version)`` restores the
    pre-ingest state.  ``commit=False`` writes the relations without an audit
    entry or version bump (preview/tests only; not rollback-safe).

    Returns a compact summary (decisions, status/source histograms, audit
    version, per-event rows).
    """

    events = list(events or [])
    state: CognitiveState = runtime.state
    before_version = int(state.version)
    review_threshold = _review_threshold(runtime, review_threshold)
    abstain_threshold = float(abstain_threshold)
    if not 0.0 <= abstain_threshold <= review_threshold <= 1.0:
        raise ValueError(
            "expected 0 <= abstain_threshold <= review_threshold <= 1, got "
            f"{abstain_threshold!r} / {review_threshold!r}"
        )

    if not events:
        return {
            "events": 0,
            "written": 0,
            "decisions": {},
            "statuses": {},
            "sources": {},
            "recommendation": None,
            "uncertainty": None,
            "before_version": before_version,
            "version": before_version,
            "audit_action": None,
            "relations": [],
            "entries": [],
        }

    # Permission check happens before any mutation.  PSSRuntime audits the
    # denial itself through ``_require``; duck-typed runtimes skip the check.
    require = getattr(runtime, "_require", None)
    if require is not None:
        require("ingest", f"contact_events:{len(events)}")

    written: list[Relation] = []
    entries: list[dict] = []
    decisions: Counter = Counter()
    statuses: Counter = Counter()
    sources: Counter = Counter()
    evidence_sources: Counter = Counter()

    before = {
        "version": before_version,
        "relations": len(state.relations),
        "contact_relations": sum(
            1 for relation in state.relations if relation.origin == "contact_event"
        ),
    }

    for event in events:
        relation = event_to_relation(event)
        decision = _decide(event, review_threshold, abstain_threshold)
        status = str(getattr(event, "status", "predicted"))
        source = _event_source(event)
        evidence_items = getattr(event, "evidence", None) or []
        event_evidence_sources = sorted(
            {str(getattr(item, "source", "unknown")) for item in evidence_items}
        )
        decisions[decision] += 1
        statuses[status] += 1
        sources[source] += 1
        for item in evidence_items:
            evidence_sources[str(getattr(item, "source", "unknown"))] += 1

        if decision in ("accept", "review"):
            state.add_relation(relation)
            _remember_in_layer(runtime, relation)
            written.append(relation)

        entries.append(
            {
                "event_id": str(getattr(event, "event_id", "")),
                "subject": str(event.subject),
                "object": str(event.object),
                "confidence": _jsonable(relation.probability),
                "status": status,
                "decision": decision,
                "source": source,
                "evidence_sources": event_evidence_sources,
            }
        )

    review_count = decisions.get("review", 0)
    abstain_count = decisions.get("abstain", 0)
    contact_recommendation = (
        "abstain" if abstain_count else ("review" if review_count else "accept")
    )
    report = _assess_uncertainty(
        runtime,
        state,
        review_count=review_count,
        abstain_count=abstain_count,
    )
    if report is not None:
        recommendation = report.recommendation
    elif not bool(getattr(runtime, "uncertainty_enabled", True)):
        recommendation = "disabled"
    else:
        recommendation = contact_recommendation

    after = {
        "events": len(events),
        "written": len(written),
        "decisions": dict(sorted(decisions.items())),
        "statuses": dict(sorted(statuses.items())),
        "sources": dict(sorted(sources.items())),
        "evidence_sources": dict(sorted(evidence_sources.items())),
        "recommendation": recommendation,
        "before_version": before_version,
        "entries": entries,
    }

    version = before_version
    audit_action = None
    if commit and bool(getattr(runtime, "audit_enabled", True)):
        commit_fn = getattr(runtime, "_commit", None)
        if commit_fn is not None:
            version = int(
                commit_fn(
                    AUDIT_ACTION,
                    f"contact_events:{len(events)}",
                    before=before,
                    after=after,
                    actor=actor,
                    reason=reason,
                )
            )
        else:
            version = int(
                state.commit(
                    AUDIT_ACTION,
                    f"contact_events:{len(events)}",
                    before=before,
                    after=after,
                    actor=actor,
                    reason=reason,
                )
            )
        audit_action = AUDIT_ACTION

    return {
        "events": len(events),
        "written": len(written),
        "decisions": dict(sorted(decisions.items())),
        "statuses": dict(sorted(statuses.items())),
        "sources": dict(sorted(sources.items())),
        "evidence_sources": dict(sorted(evidence_sources.items())),
        "recommendation": recommendation,
        "uncertainty": report.to_dict() if report is not None else None,
        "before_version": before_version,
        "version": version,
        "audit_action": audit_action,
        "relations": [relation.to_dict() for relation in written],
        "entries": entries,
    }


def contact_relations(state: CognitiveState) -> list[Relation]:
    """Contact relations currently asserted in ``state``."""
    return [relation for relation in state.relations if relation.origin == "contact_event"]


__all__ = [
    "AUDIT_ACTION",
    "DEFAULT_ABSTAIN_THRESHOLD",
    "DEFAULT_REVIEW_THRESHOLD",
    "NON_ASSERTING_STATUSES",
    "apply_events",
    "contact_relations",
    "event_to_relation",
    "relation_metadata",
]
