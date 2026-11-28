"""C5 demo: MuJoCo drop -> force contact events -> PSSRuntime.

Builds a ``drop`` scenario, runs the ``force`` channel, writes the resulting
``ContactEvent`` objects into an opt-in ``PSSRuntime`` through
``pss.events.layer.apply_events``, and prints entity / relation / audit
summaries.  The demo also verifies the C5 invariants:

  * relations have ``predicate="contact"`` / ``origin="contact_event"``;
  * evidence/provenance travel in the relation's instance attributes;
  * exactly one ``contact_event_ingest`` audit entry is appended;
  * ``runtime.rollback(before_version)`` removes the contact relations while
    the audit log stays append-only.

Example
-------
python -u -m pss.eval.scripts.run_contact_event_runtime
python -u -m pss.eval.scripts.run_contact_event_runtime --seed 1 --max-steps 500
"""

from __future__ import annotations

import argparse

import numpy as np

from pss._console import ensure_utf8
from pss.events import ContactConfig, build_scenario, run_channel
from pss.events.layer import apply_events, contact_relations, relation_metadata
from pss.runtime import Entity, PSSRuntime


def _seed_participants(runtime: PSSRuntime, scenario, participant_ids) -> list[str]:
    """Seed demo-only entity placeholders from the scenario's frame 0 poses."""

    frame = scenario.frames[0]
    scene = frame.scene_model
    seeded: list[str] = []
    for oid in sorted(participant_ids):
        if oid not in frame.poses:
            continue
        obj = scene.objects.get(oid) if scene is not None else None
        runtime.state.add_entity(
            Entity(
                id=oid,
                category=None if obj is None else str(obj.shape),
                pos=np.asarray(frame.poses[oid], dtype=float).copy(),
                quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
                attributes={
                    "role": "contact_participant",
                    "radius": None if obj is None else float(obj.radius),
                },
                confidence=1.0,
                provenance={
                    "source": "contact_event_demo",
                    "scenario": scenario.scenario_id,
                },
                status="active",
            )
        )
        seeded.append(oid)
    if seeded:
        runtime.state.commit(
            "seed_contact_participants",
            scenario.scenario_id,
            after={"entities": seeded, "source": "contact_event_demo"},
            actor="contact_event_demo",
            reason="demo fixture: participant poses from scenario frame 0",
        )
    return seeded


def _validate(runtime: PSSRuntime, summary: dict, before_version: int) -> list[str]:
    errors: list[str] = []
    contacts = contact_relations(runtime.state)
    if summary["written"] and not contacts:
        errors.append("apply_events reported writes but no contact relations exist")
    for relation in contacts:
        if relation.predicate != "contact":
            errors.append(f"{relation.source}->{relation.target}: predicate != contact")
        if relation.origin != "contact_event":
            errors.append(f"{relation.source}->{relation.target}: origin != contact_event")
        if not 0.0 <= float(relation.probability) <= 1.0:
            errors.append(f"{relation.source}->{relation.target}: probability out of [0,1]")
        metadata = relation_metadata(relation)
        if not metadata.get("event_id"):
            errors.append(f"{relation.source}->{relation.target}: missing event_id metadata")
        if "evidence" not in metadata:
            errors.append(f"{relation.source}->{relation.target}: missing evidence metadata")

    audit = [e for e in runtime.state.audit if e.action == "contact_event_ingest"]
    if len(audit) != 1:
        errors.append(f"expected 1 contact_event_ingest audit entry, got {len(audit)}")
    else:
        entry = audit[0]
        after = entry.after or {}
        if after.get("events") != summary["events"]:
            errors.append("audit events count mismatch")
        if after.get("written") != summary["written"]:
            errors.append("audit written count mismatch")
        if "statuses" not in after or "sources" not in after:
            errors.append("audit missing statuses/sources")
        if entry.actor != "contact_event":
            errors.append(f"audit actor {entry.actor!r} != 'contact_event'")

    rolled = runtime.rollback(before_version)
    if rolled != before_version:
        errors.append(f"rollback version {rolled} != {before_version}")
    if contact_relations(runtime.state):
        errors.append("rollback left contact relations in state")
    if not any(e.action == "contact_event_ingest" for e in runtime.state.audit):
        errors.append("rollback truncated the contact_event_ingest audit entry")
    return errors


def run(args) -> dict:
    config = ContactConfig(
        max_steps=args.max_steps,
        capture_every=args.capture_every,
        threshold=args.threshold,
        seed=args.seed,
    )
    scenario = build_scenario("drop", config)
    events = run_channel(scenario, config, channel="force")
    print(
        f"[scenario] {scenario.scenario_id} frames={len(scenario.frames)} "
        f"gt_events={len(scenario.gt_events)} force_events={len(events)}",
        flush=True,
    )
    if not events:
        print("[events] force channel produced no events; demo cannot proceed", flush=True)
        raise SystemExit(1)
    for event in events:
        source = (event.provenance or {}).get("source") or (event.provenance or {}).get("channel")
        print(
            f"  - {event.event_id} {event.subject}->{event.object} "
            f"onset={event.onset:.4f} offset={event.offset} "
            f"conf={event.confidence:.3f} intensity={event.intensity:.3f} "
            f"status={event.status} source={source}",
            flush=True,
        )

    runtime = PSSRuntime()
    participants = {event.subject for event in events} | {event.object for event in events}
    seeded = _seed_participants(runtime, scenario, participants)
    before_version = int(runtime.state.version)
    summary = apply_events(runtime, events, actor="contact_event")

    print(
        f"\n[apply] events={summary['events']} written={summary['written']} "
        f"decisions={summary['decisions']} statuses={summary['statuses']} "
        f"sources={summary['sources']} recommendation={summary['recommendation']} "
        f"before_v={summary['before_version']} after_v={summary['version']}",
        flush=True,
    )

    print(f"\n[entities] n={len(runtime.state.entities)} seeded={seeded}", flush=True)
    for entity in runtime.state.entities.values():
        print(
            f"  - {entity.id} category={entity.category} "
            f"pos=[{', '.join(f'{float(x):.3f}' for x in entity.pos[:3])}] "
            f"confidence={entity.confidence:.3f} "
            f"provenance={entity.provenance.get('source')}",
            flush=True,
        )

    contacts = contact_relations(runtime.state)
    print(f"\n[relations] contact={len(contacts)} total={len(runtime.state.relations)}", flush=True)
    for relation in contacts:
        metadata = relation_metadata(relation)
        evidence = sorted({row["source"] for row in metadata.get("evidence", [])})
        print(
            f"  - {relation.source} -{relation.predicate}-> {relation.target} "
            f"p={relation.probability:.3f} origin={relation.origin} "
            f"type={relation.relation_type} event={metadata.get('event_id')} "
            f"evidence={evidence}",
            flush=True,
        )
    records = [r for r in runtime.relations.records() if r.predicate == "contact"]
    print(f"  relation_layer_records(predicate=contact)={len(records)}", flush=True)

    print(f"\n[audit] n={len(runtime.state.audit)}", flush=True)
    for entry in runtime.state.audit[-5:]:
        after = entry.after or {}
        print(
            f"  - v{entry.version} {entry.action} target={entry.target} "
            f"actor={entry.actor} events={after.get('events')} "
            f"statuses={after.get('statuses')} sources={after.get('sources')} "
            f"recommendation={after.get('recommendation')}",
            flush=True,
        )

    errors = _validate(runtime, summary, before_version)
    print("\n[rollback]", flush=True)
    print(
        f"  to v{before_version}: contact_relations={len(contact_relations(runtime.state))} "
        f"entities={len(runtime.state.entities)} audit={len(runtime.state.audit)} "
        f"(audit stays append-only)",
        flush=True,
    )
    provenance_ok = not any("evidence" in e or "metadata" in e for e in errors)
    origin_ok = not any("origin" in e or "predicate" in e for e in errors)
    audit_ok = not any("audit" in e for e in errors)
    rollback_ok = not any("rollback" in e for e in errors)
    print(
        "  checks: "
        f"provenance={'ok' if provenance_ok else 'FAIL'} "
        f"origin={'ok' if origin_ok else 'FAIL'} "
        f"audit={'ok' if audit_ok else 'FAIL'} "
        f"rollback={'ok' if rollback_ok else 'FAIL'}",
        flush=True,
    )
    if errors:
        for error in errors:
            print(f"  ERROR {error}", flush=True)
        raise SystemExit(1)
    return {"scenario": scenario, "events": events, "summary": summary, "errors": errors}


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--capture-every", type=int, default=2)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
