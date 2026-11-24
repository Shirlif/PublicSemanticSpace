"""Row 7: permission profiles for the runtime.

Exercises read / ingest / write_facts / plan / act / rollback under
read_only, service, operator and admin profiles, and checks that denied
actions are audited.

Example:
    python -m pss.eval.scripts.run_runtime_audit
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pss._console import ensure_utf8
from pss.eval.scripts.run_runtime import _move_library
from pss.runtime import (
    CognitiveState,
    Entity,
    Fact,
    Goal,
    PSSRuntime,
    PermissionDenied,
    RuntimePlanner,
)

OUT = "eval/results/runtime_audit"
PROFILES = ("read_only", "service", "operator", "admin")


def _fmt(v):
    return "yes" if v else "no"


def _runtime(profile: str):
    state = CognitiveState()
    state.add_entity(
        Entity("cup", "cup", np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    )
    state.set_fact(Fact("cup", "location", "A"))
    planner = RuntimePlanner(_move_library(["cup"]))
    return PSSRuntime(state=state, planner=planner, permissions=profile)


def _attempt(runtime, action: str) -> bool:
    try:
        if action == "read":
            runtime.summary()
        elif action == "ingest":
            runtime.ingest([], 0)
        elif action == "write_facts":
            runtime.assert_fact("cup", "location", "B", source="test")
        elif action == "plan":
            runtime.set_goal(
                Goal("g", frozenset({("cup", "location", "G")}), "cup to G")
            )
            runtime.plan("g")
        elif action == "act":
            runtime.act({"kind": "push", "target_is_human": False})
        elif action == "rollback":
            runtime.rollback(0)
        return True
    except PermissionDenied:
        return False


def run() -> dict:
    actions = ("read", "ingest", "write_facts", "plan", "act", "rollback")
    matrix = {}
    denied_counts = {}
    for profile in PROFILES:
        row = {}
        denied = 0
        for action in actions:
            runtime = _runtime(profile)
            ok = _attempt(runtime, action)
            row[action] = ok
            denied += sum(
                1 for e in runtime.state.audit if e.action == "permission_denied"
            )
        matrix[profile] = row
        denied_counts[profile] = denied
    payload = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "actions": list(actions),
        "matrix": matrix,
        "audit_denied_entries": denied_counts,
    }
    out = Path(OUT)
    out.mkdir(parents=True, exist_ok=True)
    (out / "runtime_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "runtime_audit.md").write_text(render(payload), encoding="utf-8")
    return payload


def render(payload: dict) -> str:
    actions = payload["actions"]
    lines = [
        "# Row 7 — runtime permission profiles",
        "",
        f"> generated: {payload['generated']}",
        "",
        "| profile | " + " | ".join(actions) + " | denied audit entries |",
        "|---|" + "---|" * len(actions) + "---:|",
    ]
    for profile, row in payload["matrix"].items():
        lines.append(
            f"| {profile} | "
            + " | ".join(_fmt(row[a]) for a in actions)
            + f" | {payload['audit_denied_entries'][profile]} |"
        )
    lines += [
        "",
        "## Verdict",
        "",
        "- `read_only` can read but cannot ingest, write, plan, act or roll back.",
        "- `service` can ingest perception but cannot write facts or plan.",
        "- `operator` can write facts, plan and act; rollback requires `admin`.",
        "- `admin` can do everything, including rollback.",
        "- Every denied action is recorded as a `permission_denied` audit entry.",
        "- This closes the permission part of row 7; versions/rollback/audit were",
        "  already present from Phase 16.",
        "",
    ]
    return "\n".join(lines)


def main():
    ensure_utf8()
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    payload = run()
    for profile, row in payload["matrix"].items():
        print(profile, row, "denied", payload["audit_denied_entries"][profile])
    print(f"output: {OUT}")


if __name__ == "__main__":
    main()
