"""U5: audit completeness for the cognitive runtime.

Executable form of ``docs/plans/PSS-架构上界复核与下一阶段.md`` §6 (U5),
sub-item 3.  ``run_runtime_audit.py`` only reports the permission matrix; it
does not quantify whether every expected mutation actually left an audit
entry.  This script builds one scripted episode and measures:

``audit_completeness``
    canonical audit events retained in ``runtime.state.audit`` at the end of
    the episode divided by the expected event count (fact_write /
    permission_denied / rollback / plan / act).

``rollback_exactness``
    after ``rollback`` every state field (entities, relations, facts, goals,
    plans, uncertainty) equals the snapshot captured at the rollback target.

``permission_violations``
    denied actions that did not append a ``permission_denied`` audit entry at
    the time they were denied (target 0).

The runtime's ``CognitiveState.rollback`` restores the *whole* snapshot,
including its audit list.  Any audit entry newer than the rollback target is
therefore dropped.  The primary arm rolls back to the version just before the
denied write, so this truncation is visible; a control arm rolls back to the
latest version and should retain every event.

Example
-------
python -u -m pss.eval.scripts.run_upper_bound_u5 \
  --out eval/results/upper_bound/u5/audit_completeness.json
"""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pss._console import ensure_utf8
from pss.eval.scripts.run_runtime import _move_library
from pss.runtime import (
    CognitiveState,
    Entity,
    Goal,
    PSSRuntime,
    PermissionDenied,
    RuntimePlanner,
)

EXPECTED = {
    "fact_write": 1,
    "permission_denied": 1,
    "rollback": 1,
    "plan": 1,
    "act": 1,
}
EXPECTED_TOTAL = sum(EXPECTED.values())

# audit action -> canonical event kind.  ``set_goal`` is the auxiliary entry
# emitted by ``runtime.set_goal`` and is not one of the five expected events.
ACTION_MAP = {
    "assert_fact": "fact_write",
    "fact_rejected": "fact_write",
    "permission_denied": "permission_denied",
    "rollback": "rollback",
    "plan": "plan",
    "act": "act",
}


def _state(profile: str) -> PSSRuntime:
    state = CognitiveState()
    state.add_entity(
        Entity("cup", "cup", np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    )
    planner = RuntimePlanner(_move_library(["cup"]))
    return PSSRuntime(state=state, planner=planner, permissions=profile)


def _fingerprint(state: CognitiveState) -> dict:
    """Round-trip the mutable state fields through their serializable views."""

    return {
        "entities": {
            key: entity.to_dict() for key, entity in sorted(state.entities.items())
        },
        "relations": [relation.to_dict() for relation in state.relations],
        "facts": {
            f"{key[0]}.{key[1]}": fact.to_dict()
            for key, fact in sorted(state.facts.items())
        },
        "goals": {key: goal.to_dict() for key, goal in sorted(state.goals.items())},
        "plans": {key: plan.to_dict() for key, plan in sorted(state.plans.items())},
        "uncertainty": state.uncertainty.to_dict() if state.uncertainty else None,
    }


def _kinds(entries) -> list[str]:
    kinds = []
    for entry in entries:
        kind = ACTION_MAP.get(entry.action)
        if kind is not None:
            kinds.append(kind)
    return kinds


def _tally(kinds: list[str]) -> dict:
    counts = {kind: 0 for kind in EXPECTED}
    for kind in kinds:
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def _arm(rollback_to_latest: bool) -> dict:
    """Run one scripted episode; return the audit metrics for that arm."""

    runtime = _state("admin")

    # 1. fact_write: seed/update the fact through the runtime (audited).
    runtime.assert_fact("cup", "location", "A", source="script")
    baseline_version = int(runtime.state.version)
    seen: set[str] = set(_kinds(runtime.state.audit))

    # 2. permission_denied: a read_only runtime shares the same state, so its
    # denied write is audited into the episode's audit list.
    denials_attempted = 1
    denials_audited = 0
    denied_runtime = _state("read_only")
    denied_runtime.state = runtime.state
    before_denied = sum(
        1 for entry in runtime.state.audit if entry.action == "permission_denied"
    )
    try:
        denied_runtime.assert_fact("cup", "location", "B", source="script")
    except PermissionDenied:
        after_denied = sum(
            1 for entry in runtime.state.audit if entry.action == "permission_denied"
        )
        denials_audited = after_denied - before_denied
    seen.update(_kinds(runtime.state.audit))

    # 3. rollback.  The primary arm targets the pre-denial version, which is
    # the natural "undo" point; the control arm targets the latest version.
    target = int(runtime.state.version) if rollback_to_latest else baseline_version
    target_snapshot = copy.deepcopy(runtime.state.snapshots[target])
    runtime.rollback(target)
    restored = _fingerprint(runtime.state)
    # The snapshot stores dataclasses; serialize it with the same shape.
    expected_state = _fingerprint_from_snapshot(target_snapshot)
    checks = {
        field: expected_state[field] == restored[field]
        for field in ("entities", "relations", "facts", "goals", "plans", "uncertainty")
    }
    seen.update(_kinds(runtime.state.audit))

    # 4. plan (set_goal + plan) and 5. act.
    runtime.set_goal(
        Goal("g", frozenset({("cup", "location", "G")}), "cup to G")
    )
    runtime.plan("g")
    seen.update(_kinds(runtime.state.audit))
    runtime.act({"kind": "push", "target_is_human": False})
    seen.update(_kinds(runtime.state.audit))

    final_kinds = _kinds(runtime.state.audit)
    final_counts = _tally(final_kinds)
    cumulative_counts = _tally(sorted(seen))
    retained = {kind: final_counts[kind] for kind in EXPECTED}
    missing = {
        kind: EXPECTED[kind] - final_counts[kind]
        for kind in EXPECTED
        if final_counts[kind] < EXPECTED[kind]
    }
    denied_retained = final_counts["permission_denied"]
    return {
        "rollback_target_version": target,
        "rollback_target": "latest" if rollback_to_latest else "pre_denial",
        "baseline_version": baseline_version,
        "final_version": int(runtime.state.version),
        "audit_completeness": sum(
            min(final_counts[kind], EXPECTED[kind]) for kind in EXPECTED
        )
        / EXPECTED_TOTAL,
        "audit_completeness_cumulative": sum(
            min(cumulative_counts[kind], EXPECTED[kind]) for kind in EXPECTED
        )
        / EXPECTED_TOTAL,
        "retained_event_counts": retained,
        "cumulative_event_counts": cumulative_counts,
        "missing_event_counts": missing,
        "final_audit_actions": [entry.action for entry in runtime.state.audit],
        "final_audit_versions": [int(entry.version) for entry in runtime.state.audit],
        "n_final_audit_entries": len(runtime.state.audit),
        "rollback_exactness": bool(all(checks.values())),
        "rollback_field_checks": checks,
        "permission_violations": int(denials_attempted - denials_audited),
        "denials_attempted": int(denials_attempted),
        "denials_audited_at_time": int(denials_audited),
        "denied_entries_retained_after_rollback": int(denied_retained),
        "rollback_truncated_audit": bool(missing),
    }


def _fingerprint_from_snapshot(snapshot: dict) -> dict:
    """Serialize a raw snapshot payload with the same shape as _fingerprint."""

    return {
        "entities": {
            key: entity.to_dict()
            for key, entity in sorted(snapshot["entities"].items())
        },
        "relations": [relation.to_dict() for relation in snapshot["relations"]],
        "facts": {
            f"{key[0]}.{key[1]}": fact.to_dict()
            for key, fact in sorted(snapshot["facts"].items())
        },
        "goals": {
            key: goal.to_dict() for key, goal in sorted(snapshot["goals"].items())
        },
        "plans": {
            key: plan.to_dict() for key, plan in sorted(snapshot["plans"].items())
        },
        "uncertainty": (
            snapshot["uncertainty"].to_dict() if snapshot["uncertainty"] else None
        ),
    }


def run() -> dict:
    primary = _arm(rollback_to_latest=False)
    control = _arm(rollback_to_latest=True)
    return {
        "stage": "u5",
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "expected_events": dict(EXPECTED),
        "expected_total": EXPECTED_TOTAL,
        "action_map": dict(ACTION_MAP),
        "note": (
            "audit_completeness counts canonical events retained in "
            "runtime.state.audit at the end of the episode. rollback restores "
            "the snapshot's audit list, so the primary arm (rollback to the "
            "pre-denial version) drops the permission_denied entry; "
            "audit_completeness_cumulative counts events that were recorded at "
            "some point. The control arm rolls back to the latest version and "
            "retains every event."
        ),
        "primary": primary,
        "control_rollback_to_latest": control,
        # Same flat keys the task asks for, taken from the primary arm.
        "audit_completeness": primary["audit_completeness"],
        "rollback_exactness": primary["rollback_exactness"],
        "permission_violations": primary["permission_violations"],
    }


def _render(payload: dict) -> str:
    p = payload["primary"]
    c = payload["control_rollback_to_latest"]
    lines = [
        "# U5 audit completeness",
        "",
        f"> generated: {payload['generated']}",
        "",
        f"- expected events: {payload['expected_events']} (total {payload['expected_total']})",
        f"- primary audit_completeness: **{p['audit_completeness']:.3f}** "
        f"(cumulative {p['audit_completeness_cumulative']:.3f})",
        f"- primary rollback_exactness: **{p['rollback_exactness']}**",
        f"- primary permission_violations: **{p['permission_violations']}**",
        f"- control (rollback to latest) audit_completeness: "
        f"**{c['audit_completeness']:.3f}**",
        "",
        "## Per-arm",
        "",
        "| arm | rollback target | completeness | cumulative | missing | rollback exact | violations |",
        "|---|---|---:|---:|---|---|---:|",
    ]
    for name, row in (("primary", p), ("control", c)):
        lines.append(
            f"| {name} | v{row['rollback_target_version']} "
            f"({row['rollback_target']}) | {row['audit_completeness']:.3f} | "
            f"{row['audit_completeness_cumulative']:.3f} | "
            f"{row['missing_event_counts']} | {row['rollback_exactness']} | "
            f"{row['permission_violations']} |"
        )
    lines += [
        "",
        "## Primary final audit",
        "",
        f"- actions: {p['final_audit_actions']}",
        f"- versions: {p['final_audit_versions']}",
        "",
        "## Finding",
        "",
        "- `rollback` restores the snapshot's audit list, so entries newer than",
        "  the rollback target are dropped. Denied actions before the target",
        "  disappear from the final audit even though they were audited when",
        "  they happened.",
        "- A control rollback to the latest version retains all five events,",
        "  which confirms the loss is caused by the rollback target, not by",
        "  missing instrumentation.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        default="eval/results/upper_bound/u5/audit_completeness.json",
    )
    args = ap.parse_args()
    payload = run()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md = out.with_suffix(".md")
    md.write_text(_render(payload), encoding="utf-8")
    p = payload["primary"]
    c = payload["control_rollback_to_latest"]
    print(
        f"primary: completeness={p['audit_completeness']:.3f} "
        f"cumulative={p['audit_completeness_cumulative']:.3f} "
        f"rollback_exact={p['rollback_exactness']} "
        f"violations={p['permission_violations']} "
        f"missing={p['missing_event_counts']}"
    )
    print(
        f"control: completeness={c['audit_completeness']:.3f} "
        f"rollback_exact={c['rollback_exactness']}"
    )
    print(f"output: {out}")


if __name__ == "__main__":
    main()
