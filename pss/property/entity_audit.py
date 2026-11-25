"""Entity-state audit properties (claim-gap E1).

The cognitive-runtime claim says the entity layer keeps an explicit,
auditable state.  The KB suite measures ``AUD`` for facts; this module measures
the equivalent properties for *entity* state:

- provenance completeness: every state change has entity/op/source/frame;
- retraction correctness: a retraction removes only the target attribute;
- identity rollback exactness: snapshot/rollback is bit-exact;
- rebind identity: re-binding to a new working-memory slot preserves id/raw;
- tamper detection: the hash-chained audit log detects an edited entry.

Everything runs on structured L0 observations and the real ``Entity`` /
``build_entity`` / ``rebuild_for_slot`` code paths.  No perception is involved.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from pss.binding.entity_ops import build_entity, rebuild_for_slot
from pss.contracts import Entity


def _round(value: Any, digits: int = 6) -> Any:
    if isinstance(value, np.ndarray):
        return [round(float(x), digits) for x in value.reshape(-1)]
    if isinstance(value, (list, tuple)):
        return [_round(v, digits) for v in value]
    if isinstance(value, dict):
        return {str(k): _round(v, digits) for k, v in sorted(value.items())}
    if isinstance(value, (np.floating, float)):
        return round(float(value), digits)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _hash(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def entity_state(entity: Entity) -> dict:
    return {
        "id": entity.id,
        "slot": entity.slot,
        "raw": _round(entity.raw),
        "conf": _round(entity.conf),
        "state": _round(entity.state),
        "provenance": _round(entity.provenance),
        "status": entity.status,
        "parent": entity.parent,
    }


@dataclass
class AuditEntry:
    seq: int
    entity_id: str
    op: str
    attribute: str | None
    old: Any
    new: Any
    source: str
    frame: int
    parent: str | None
    prev_hash: str
    entry_hash: str = ""

    def payload(self) -> dict:
        return {
            "seq": self.seq,
            "entity_id": self.entity_id,
            "op": self.op,
            "attribute": self.attribute,
            "old": _round(self.old),
            "new": _round(self.new),
            "source": self.source,
            "frame": self.frame,
            "parent": self.parent,
            "prev_hash": self.prev_hash,
        }

    def complete(self) -> bool:
        return (
            bool(self.entity_id)
            and bool(self.op)
            and bool(self.source)
            and self.frame >= 0
        )


class EntityLedger:
    """Entity store with a hash-chained audit log.

    ``audit``/``chain``/``deep_snapshot`` exist so the property suite can run
    negative controls: disable the audit log, disable the hash chain, or make
    snapshots shallow. A correct ledger leaves all three enabled.
    """

    GENESIS = "genesis"

    def __init__(
        self,
        *,
        audit: bool = True,
        chain: bool = True,
        deep_snapshot: bool = True,
    ) -> None:
        self.entities: dict[str, Entity] = {}
        self.log: list[AuditEntry] = []
        self.audit = bool(audit)
        self.chain = bool(chain)
        self.deep_snapshot = bool(deep_snapshot)
        self.operation_count = 0

    # ── log ─────────────────────────────────────────────────────────────

    def _append(
        self,
        entity_id: str,
        op: str,
        attribute: str | None,
        old: Any,
        new: Any,
        source: str,
        frame: int,
        parent: str | None = None,
    ) -> AuditEntry | None:
        self.operation_count += 1
        if not self.audit:
            return None
        prev = self.log[-1].entry_hash if self.log else self.GENESIS
        entry = AuditEntry(
            seq=len(self.log),
            entity_id=str(entity_id),
            op=str(op),
            attribute=attribute,
            old=old,
            new=new,
            source=str(source),
            frame=int(frame),
            parent=parent,
            prev_hash=prev,
        )
        entry.entry_hash = _hash(entry.payload())
        self.log.append(entry)
        return entry

    def verify_chain(self) -> bool:
        if not self.chain:
            return True
        prev = self.GENESIS
        for entry in self.log:
            if entry.prev_hash != prev:
                return False
            if entry.entry_hash != _hash(entry.payload()):
                return False
            prev = entry.entry_hash
        return True

    def provenance_completeness(self) -> float:
        if self.operation_count == 0:
            return 1.0
        return float(min(1.0, len(self.log) / self.operation_count))

    # ── entity operations ───────────────────────────────────────────────

    def create(self, entity_id: str, raw: dict, slot: np.ndarray, source: str, frame: int) -> Entity:
        entity = build_entity(
            entity_id,
            raw,
            slot,
            slot_index=int(frame % max(1, len(slot))),
            provenance={"created_by": source, "created_frame": frame},
            state={"observed": True},
            status="active",
        )
        self.entities[entity.id] = entity
        self._append(
            entity.id,
            "create",
            None,
            None,
            {"slot": entity.slot, "modalities": sorted(entity.raw)},
            source,
            frame,
        )
        return entity

    def witness(self, entity_id: str, observation: Any, source: str, frame: int) -> None:
        entity = self.entities[entity_id]
        old = entity.state.get("observed")
        entity.state["observed"] = observation
        entity.provenance["last_witness"] = {"source": source, "frame": frame}
        entity.last_seen = float(frame)
        self._append(entity_id, "witness", "observed", old, observation, source, frame)

    def revise(self, entity_id: str, attribute: str, value: Any, source: str, frame: int) -> None:
        entity = self.entities[entity_id]
        old = entity.state.get(attribute)
        entity.state[attribute] = value
        entity.provenance[f"revise:{attribute}"] = {"source": source, "frame": frame}
        self._append(entity_id, "revise", attribute, old, value, source, frame)

    def retract(self, entity_id: str, attribute: str, source: str, frame: int) -> Any:
        entity = self.entities[entity_id]
        old = entity.state.pop(attribute, None)
        entity.provenance[f"retract:{attribute}"] = {"source": source, "frame": frame}
        self._append(entity_id, "retract", attribute, old, None, source, frame)
        return old

    def rebind(
        self,
        entity_id: str,
        new_slot: np.ndarray,
        frame: int,
        source: str = "rebind",
    ) -> Entity:
        old = self.entities[entity_id]
        new = rebuild_for_slot(
            old,
            new_slot,
            slot_index=int(frame % max(1, len(new_slot))),
        )
        self.entities[entity_id] = new
        self._append(entity_id, "rebind", "slot", old.slot, new.slot, source, frame)
        return new

    def evict(self, entity_id: str, frame: int, source: str = "evict") -> None:
        entity = self.entities[entity_id]
        old = entity.status
        entity.status = "dormant"
        self._append(entity_id, "evict", "status", old, entity.status, source, frame)

    def reload(
        self,
        entity_id: str,
        new_slot: np.ndarray,
        frame: int,
        source: str = "reload",
    ) -> Entity:
        old = self.entities[entity_id]
        new = rebuild_for_slot(
            old,
            new_slot,
            slot_index=int(frame % max(1, len(new_slot))),
        )
        new.status = "active"
        self.entities[entity_id] = new
        self._append(entity_id, "reload", "status", old.status, new.status, source, frame)
        return new

    # ── snapshots ───────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        entities = {
            k: copy.deepcopy(v) if self.deep_snapshot else v
            for k, v in sorted(self.entities.items())
        }
        return {
            "entities": entities,
            "log_len": len(self.log),
            "operation_count": self.operation_count,
        }

    def snapshot_hash(self) -> str:
        return _hash(
            {
                "entities": {
                    k: entity_state(v) for k, v in sorted(self.entities.items())
                },
                "log_len": len(self.log),
                "operation_count": self.operation_count,
            }
        )

    def rollback(self, snapshot: dict) -> None:
        if not self.deep_snapshot:
            return
        self.entities = {k: copy.deepcopy(v) for k, v in snapshot["entities"].items()}
        self.log = self.log[: int(snapshot["log_len"])]
        self.operation_count = int(snapshot["operation_count"])


def _unit(rng: np.random.Generator, dim: int) -> np.ndarray:
    v = rng.normal(size=dim)
    return v / (np.linalg.norm(v) + 1e-12)


def _one_seed(
    seed: int,
    dim: int = 128,
    n_entities: int = 12,
    n_slots: int = 4,
    **ledger_kwargs,
) -> dict:
    rng = np.random.default_rng(seed)
    ledger = EntityLedger(**ledger_kwargs)
    slots = [_unit(rng, dim) for _ in range(n_slots)]
    ids = [f"obj_{i:03d}" for i in range(n_entities)]

    for i, entity_id in enumerate(ids):
        raw = {
            "vision": _unit(rng, dim),
            "geometry": _unit(rng, dim),
            "tactile": _unit(rng, dim),
        }
        ledger.create(entity_id, raw, slots[i % len(slots)], source="init", frame=0)

    # ── retraction correctness ──────────────────────────────────────────
    retraction_ok = []
    for i, entity_id in enumerate(ids[:6]):
        ledger.revise(entity_id, "material", f"m{i}", source="vision", frame=10 + i)
        ledger.revise(entity_id, "color", f"c{i}", source="vision", frame=10 + i)
        before = copy.deepcopy(ledger.entities[entity_id].state)
        ledger.retract(entity_id, "material", source="review", frame=20 + i)
        after = ledger.entities[entity_id].state
        unrelated = {
            k: v for k, v in before.items() if k != "material"
        }
        retraction_ok.append(
            ("material" not in after)
            and all(after.get(k) == v for k, v in unrelated.items())
        )

    # ── rebind identity ─────────────────────────────────────────────────
    rebind_ok = []
    for i, entity_id in enumerate(ids[:6]):
        before = ledger.entities[entity_id]
        raw_before = copy.deepcopy(before.raw)
        new_slot = slots[(i + 2) % len(slots)]
        after = ledger.rebind(entity_id, new_slot, frame=40 + i)
        rebind_ok.append(
            after.id == before.id
            and set(after.raw) == set(raw_before)
            and all(np.allclose(after.raw[k], raw_before[k]) for k in raw_before)
        )

    # ── snapshot / rollback exactness ───────────────────────────────────
    snap = ledger.snapshot()
    hash_before = ledger.snapshot_hash()
    for i, entity_id in enumerate(ids[:6]):
        ledger.witness(entity_id, f"obs_{i}", source="sensor", frame=60 + i)
        ledger.revise(entity_id, "pose", [i, i + 1, i + 2], source="slam", frame=70 + i)
        ledger.evict(entity_id, frame=80 + i)
        ledger.reload(entity_id, slots[(i + 1) % len(slots)], frame=90 + i)
    ledger.rollback(snap)
    hash_after = ledger.snapshot_hash()

    # ── tamper detection on a copy ──────────────────────────────────────
    chain_ok = ledger.verify_chain()
    tampered = copy.deepcopy(ledger)
    if tampered.log:
        tampered.log[0].source = "forged"
    tamper_detected = not tampered.verify_chain()

    return {
        "seed": int(seed),
        "provenance_completeness": ledger.provenance_completeness(),
        "retraction_correctness": float(np.mean(retraction_ok)) if retraction_ok else 1.0,
        "id_rollback_exactness": float(hash_before == hash_after),
        "rebind_identity": float(np.mean(rebind_ok)) if rebind_ok else 1.0,
        "tamper_detection": float(tamper_detected),
        "audit_chain_valid": float(chain_ok),
        "stale_provenance_rate": float(1.0 - ledger.provenance_completeness()),
        "n_entities": len(ledger.entities),
        "n_audit_entries": len(ledger.log),
    }


def run_entity_audit(seeds: int = 5, dim: int = 128) -> dict:
    rows = [_one_seed(seed, dim=dim) for seed in range(seeds)]
    keys = (
        "provenance_completeness",
        "retraction_correctness",
        "id_rollback_exactness",
        "rebind_identity",
        "tamper_detection",
        "audit_chain_valid",
        "stale_provenance_rate",
    )
    summary = {}
    for key in keys:
        vals = [float(r[key]) for r in rows]
        summary[key] = {
            "mean": float(np.mean(vals)),
            "min": float(np.min(vals)),
            "std": float(np.std(vals)),
        }
    passes = {
        "provenance_completeness": summary["provenance_completeness"]["min"] >= 1.0,
        "retraction_correctness": summary["retraction_correctness"]["min"] >= 1.0,
        "id_rollback_exactness": summary["id_rollback_exactness"]["min"] >= 1.0,
        "rebind_identity": summary["rebind_identity"]["min"] >= 1.0,
        "tamper_detection": summary["tamper_detection"]["min"] >= 1.0,
        "audit_chain_valid": summary["audit_chain_valid"]["min"] >= 1.0,
    }
    controls = {
        "no_audit": {
            "kwargs": {"audit": False},
            "metric": "provenance_completeness",
        },
        "no_chain": {
            "kwargs": {"chain": False},
            "metric": "tamper_detection",
        },
        "shallow_rollback": {
            "kwargs": {"deep_snapshot": False},
            "metric": "id_rollback_exactness",
        },
    }
    control_rows = {}
    for name, spec in controls.items():
        vals = [
            float(_one_seed(seed, dim=dim, **spec["kwargs"])[spec["metric"]])
            for seed in range(seeds)
        ]
        mean = float(np.mean(vals))
        # A negative control must fail the property it targets.
        control_rows[name] = {
            "metric": spec["metric"],
            "mean": mean,
            "expected": "fail",
            "detected": bool(mean < 1.0),
        }

    return {
        "seeds": int(seeds),
        "dim": int(dim),
        "rows": rows,
        "summary": summary,
        "thresholds": {
            "provenance_completeness": 1.0,
            "retraction_correctness": 1.0,
            "id_rollback_exactness": 1.0,
            "rebind_identity": 1.0,
            "tamper_detection": 1.0,
            "audit_chain_valid": 1.0,
        },
        "pass": passes,
        "all_pass": bool(all(passes.values())),
        "controls": control_rows,
        "controls_valid": bool(all(v["detected"] for v in control_rows.values())),
    }
