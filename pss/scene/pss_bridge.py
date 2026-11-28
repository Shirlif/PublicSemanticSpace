"""Bridge the 3D scene model into the PSS cognitive runtime (S4).

The scene model is the geometric base layer; PSS is the semantic/relation
layer.  This module is the *opt-in* seam between them:

    SceneModel + solved positions/covariances
        -> Entity          (pose, covariance, provenance, confidence)
        -> RelationRecord  (aligned_x/aligned_y/same_level/contact/...)
        -> CognitiveState  (entities + relations + audit entry)

Nothing here is a runtime default.  The caller explicitly invokes
``apply_scene_to_runtime``; the historical 1D axis predicates
(``pss.reasoning.relations.bayes_predicate``) remain the fallback path.

Confidences are **heuristics**, not calibrated probabilities.  They decrease
as the measured quantity approaches its ``eps`` boundary (and are capped by
the positional covariance when one is supplied).  Calibration, if wanted, is
the caller's responsibility (see the S3 experiments).
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from pss.reasoning.relations import RelationRecord
from pss.runtime.state import CognitiveState, Entity, Relation

# Trace scale for the covariance -> confidence map.  Typical linearised
# position covariances on the CLEVR simulator are ~1e-4..1e-3 m^2, so a trace
# of TRACE_REF maps to confidence 0.5.  This is a monotone heuristic only.
TRACE_REF = 3e-3

# Minimum confidence used for penetration-based collision records, so an
# existing relation is never indistinguishable from a missing one.
COLLISION_MIN_CONF = 0.05


def _as_positions(scene, positions) -> dict[str, np.ndarray]:
    if positions is None:
        return scene.positions()
    out: dict[str, np.ndarray] = {}
    for oid, value in positions.items():
        arr = np.asarray(value, dtype=np.float64).reshape(3)
        out[str(oid)] = arr
    return out


def _yaw_quat(yaw: float) -> np.ndarray:
    half = 0.5 * float(yaw)
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)


def _covariance_matrix(cov) -> np.ndarray | None:
    if cov is None:
        return None
    arr = np.asarray(cov, dtype=np.float64)
    if arr.shape != (3, 3):
        raise ValueError(f"covariance must be 3x3, got {arr.shape}")
    return arr


def positional_confidence(cov, *, trace_ref: float = TRACE_REF) -> float:
    """Map a 3x3 position covariance to a [0, 1] heuristic confidence.

    ``trace`` is the summed per-axis variance.  ``1 / (1 + trace/ref)`` is
    monotone and bounded; it is not a calibrated probability.  ``None`` (no
    covariance supplied) maps to 1.0 because no uncertainty evidence exists.
    """

    if cov is None:
        return 1.0
    arr = _covariance_matrix(cov)
    assert arr is not None
    if not np.isfinite(arr).all():
        return 0.0
    trace = float(np.trace(arr))
    if trace <= 0.0:
        return 1.0
    return float(np.clip(1.0 / (1.0 + trace / max(float(trace_ref), 1e-12)), 0.0, 1.0))


def scene_to_entities(
    scene,
    positions,
    covariances=None,
    *,
    trace_ref: float = TRACE_REF,
) -> list[Entity]:
    """Convert solved scene positions into runtime ``Entity`` objects.

    ``attributes["covariance"]`` is always a 3x3 list; if no covariance was
    supplied it is the zero matrix and ``attributes["covariance_source"]`` is
    ``"missing"`` so downstream code can tell the two cases apart.  The state
    schema is unchanged: covariance travels only in ``attributes``.
    """

    resolved = _as_positions(scene, positions)
    covariances = covariances or {}
    entities: list[Entity] = []
    for oid, obj in scene.objects.items():
        if oid not in resolved:
            continue
        cov = _covariance_matrix(covariances.get(oid))
        cov_list = (
            cov.tolist()
            if cov is not None
            else [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        )
        entities.append(
            Entity(
                id=oid,
                category=str(obj.shape),
                pos=np.asarray(resolved[oid], dtype=np.float64).copy(),
                quat=_yaw_quat(obj.yaw),
                attributes={
                    "covariance": cov_list,
                    "covariance_source": "linearised" if cov is not None else "missing",
                    "radius": float(obj.radius),
                    "size": str(obj.size),
                    "shape": str(obj.shape),
                    "scene_id": str(scene.scene_id),
                },
                confidence=positional_confidence(cov, trace_ref=trace_ref),
                provenance={
                    "source": "scene_model",
                    "scene_id": str(scene.scene_id),
                    "object_id": oid,
                },
                status="active",
            )
        )
    return entities


def _one_minus_ratio(value: float, eps: float) -> float:
    if eps <= 0.0:
        return 1.0
    return float(np.clip(1.0 - abs(float(value)) / float(eps), 0.0, 1.0))


def scene_relation_records(
    scene,
    positions,
    covariances=None,
    eps_xy: float = 0.12,
    eps_z: float = 0.05,
    eps_c: float = 0.02,
    table_z: float = 0.0,
    *,
    eps_s: float = 0.05,
    probabilities: dict[tuple[str, str, str], float] | None = None,
    trace_ref: float = TRACE_REF,
) -> list[RelationRecord]:
    """Query the 3D scene state and emit runtime ``RelationRecord`` objects.

    Predicates
    ----------
    aligned_x / aligned_y
        ``|dx| <= eps_xy`` / ``|dy| <= eps_xy`` (the two halves of
        ``metrics.axis_aligned``).  Emitted for both directions.
    same_level
        ``|bottom_z_a - bottom_z_b| <= eps_z`` (``metrics.same_level``).
    contact
        centre distance ``<= r_a + r_b + eps_c`` (``metrics.contact``).
    collision
        centre distance ``< r_a + r_b`` (``metrics.collision``).
    support
        ``|bottom_z - table_z| <= eps_s`` (``metrics.supported``), target
        ``"table"``.

    Confidence is ``1 - |quantity| / eps`` clipped to [0, 1] (contact uses the
    surface gap, collision uses normalised penetration); when covariances are
    supplied the record confidence is additionally capped by the positional
    confidence of the two endpoints.  ``probabilities`` (keyed by
    ``(subject, predicate, object)``) overrides the heuristic and is applied
    last.  None of these are calibrated probabilities.

    ``covariances`` is accepted for interface parity with the solver output
    and for the confidence cap; it is not otherwise interpreted here.
    """

    resolved = _as_positions(scene, positions)
    covariances = covariances or {}
    ids = [oid for oid in scene.objects if oid in resolved]
    caps = {
        oid: positional_confidence(covariances.get(oid), trace_ref=trace_ref)
        for oid in ids
    }
    records: list[RelationRecord] = []

    def emit(subject: str, predicate: str, obj: str, confidence: float) -> None:
        if probabilities is not None:
            override = probabilities.get((subject, predicate, obj))
            if override is not None:
                confidence = float(override)
        confidence = float(np.clip(confidence, 0.0, 1.0))
        records.append(
            RelationRecord(
                subject=subject,
                predicate=predicate,
                object=obj,
                confidence=confidence,
                source="scene_model",
                frame="world",
            )
        )

    for i, a in enumerate(ids):
        pa = resolved[a]
        ra = float(scene.objects[a].radius)
        bottom_a = float(pa[2] - ra)
        gap_table = abs(bottom_a - float(table_z))
        if gap_table <= eps_s:
            emit(
                a,
                "support",
                "table",
                _one_minus_ratio(gap_table, eps_s) * caps[a],
            )
        for b in ids[i + 1 :]:
            pb = resolved[b]
            rb = float(scene.objects[b].radius)
            cap = min(caps[a], caps[b])
            dx = float(pb[0] - pa[0])
            dy = float(pb[1] - pa[1])
            bottom_b = float(pb[2] - rb)
            dz = bottom_a - bottom_b

            if eps_xy > 0.0 and abs(dx) <= eps_xy:
                conf = _one_minus_ratio(dx, eps_xy) * cap
                emit(a, "aligned_x", b, conf)
                emit(b, "aligned_x", a, conf)
            if eps_xy > 0.0 and abs(dy) <= eps_xy:
                conf = _one_minus_ratio(dy, eps_xy) * cap
                emit(a, "aligned_y", b, conf)
                emit(b, "aligned_y", a, conf)
            if eps_z > 0.0 and abs(dz) <= eps_z:
                conf = _one_minus_ratio(dz, eps_z) * cap
                emit(a, "same_level", b, conf)
                emit(b, "same_level", a, conf)

            distance = float(np.linalg.norm(pa - pb))
            surface_gap = distance - (ra + rb)
            if eps_c >= 0.0 and surface_gap <= eps_c:
                conf = (
                    1.0
                    if surface_gap <= 0.0
                    else _one_minus_ratio(surface_gap, eps_c)
                )
                conf *= min(cap, caps[a], caps[b])
                emit(a, "contact", b, conf)
                emit(b, "contact", a, conf)

            penetration = (ra + rb) - distance
            if penetration > 0.0:
                scale = max(ra + rb, 1e-9)
                conf = float(np.clip(penetration / scale, COLLISION_MIN_CONF, 1.0))
                conf *= cap
                emit(a, "collision", b, conf)
                emit(b, "collision", a, conf)

    return records


def apply_scene_to_runtime(
    runtime,
    scene,
    positions,
    covariances=None,
    *,
    eps_xy: float = 0.12,
    eps_z: float = 0.05,
    eps_c: float = 0.02,
    table_z: float | None = None,
    eps_s: float = 0.05,
    probabilities: dict[tuple[str, str, str], float] | None = None,
    actor: str = "scene_model",
    commit: bool = True,
    trace_ref: float = TRACE_REF,
) -> dict:
    """Write a solved scene into ``runtime.state`` and append one audit entry.

    Entities go through ``CognitiveState.add_entity``; relations go through
    ``CognitiveState.add_relation`` with ``origin="scene_model"`` and
    ``relation_type="scene_geometry"`` so they are distinguishable from the
    1D symbolic fallback.  No runtime schema or default is modified.

    Returns a compact summary (counts, predicate histogram, audit version).
    """

    state: CognitiveState = runtime.state
    if table_z is None:
        table_z = float(scene.table_z)

    entities = scene_to_entities(
        scene, positions, covariances, trace_ref=trace_ref
    )
    for entity in entities:
        state.add_entity(entity)

    records = scene_relation_records(
        scene,
        positions,
        covariances,
        eps_xy=eps_xy,
        eps_z=eps_z,
        eps_c=eps_c,
        table_z=table_z,
        eps_s=eps_s,
        probabilities=probabilities,
        trace_ref=trace_ref,
    )
    for record in records:
        state.add_relation(
            Relation(
                source=record.subject,
                target=record.object or "world",
                relation_type="scene_geometry",
                predicate=record.predicate,
                probability=float(record.confidence),
                origin="scene_model",
                frame=record.frame,
            )
        )

    predicates = Counter(record.predicate for record in records)
    after = {
        "scene_id": str(scene.scene_id),
        "source": "scene_model",
        "entities": len(entities),
        "relations": len(records),
        "predicates": dict(sorted(predicates.items())),
        "eps_xy": float(eps_xy),
        "eps_z": float(eps_z),
        "eps_c": float(eps_c),
        "eps_s": float(eps_s),
        "table_z": float(table_z),
    }
    version = int(state.version)
    if commit:
        version = int(
            state.commit(
                "scene_ingest",
                str(scene.scene_id),
                before=None,
                after=after,
                actor=actor,
                reason="3D scene model -> runtime",
            )
        )

    return {
        "scene_id": str(scene.scene_id),
        "version": version,
        "entities": len(entities),
        "relations": len(records),
        "predicates": dict(sorted(predicates.items())),
        "entity_ids": [entity.id for entity in entities],
        "relation_origins": sorted({str(record.source) for record in records}),
        "audit_action": "scene_ingest" if commit else None,
    }


__all__ = [
    "apply_scene_to_runtime",
    "positional_confidence",
    "scene_relation_records",
    "scene_to_entities",
]
