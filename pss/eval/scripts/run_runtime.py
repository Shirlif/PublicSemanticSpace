"""Phase 16: cross-layer PSS cognitive runtime benchmark.

A tabletop task exercises the full layering:

    observations
      -> EntityLayer        (persistent IDs / object slots)
      -> RelationFactLayer  (typed relations / versioned facts)
      -> UncertaintyService (conflict / review)
      -> RuntimePlanner     (goal / plan)
      -> OutputRouter       (language / safety)
      -> CognitiveState     (versions / audit / rollback)

Ablations remove one layer at a time. The task deliberately occludes the
object that the goal needs, so persistent identity matters; it also injects a
conflicting fact and a bad write for the uncertainty/audit checks.

Example:
    python -m pss.eval.scripts.run_runtime
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pss._console import ensure_utf8
from pss.memory.perception import Observation
from pss.reasoning.relations import geometric_relations
from pss.runtime import (
    CognitiveState,
    EntityLayer,
    Goal,
    OutputRouter,
    PSSRuntime,
    RelationFactLayer,
    RuntimePlanner,
    UncertaintyService,
)
from pss.world_model.actions import Operator, OperatorLibrary

OUT = "eval/results/runtime"
REGIONS = {
    "A": (-0.24, 0.00),
    "B": (0.24, 0.00),
    "C": (0.00, 0.18),
    "D": (0.00, -0.18),
    "G": (0.00, 0.00),
}
NAMES = list(REGIONS)
ENTITIES = (("cup", "A"), ("block", "B"), ("bottle", "D"))


def _fmt(v, digits=3):
    if v is None:
        return "n/a"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    return "n/a" if math.isnan(v) else f"{v:.{digits}f}"


def _observation(i, gt, region, frame, confidence=0.9):
    pos = np.array([*REGIONS[region], 0.0], dtype=float)
    appearance = np.eye(len(ENTITIES))[i]
    vision = np.zeros(128, dtype=float)
    vision[i] = 1.0
    raw = {
        "vision": vision,
        "geometry": np.zeros(128, dtype=float),
        "tactile": np.zeros(128, dtype=float),
    }
    return Observation(
        gt_id=gt,
        pos=pos,
        quat=np.array([1.0, 0.0, 0.0, 0.0]),
        appearance=appearance,
        raw=raw,
        confidence=confidence,
        category=gt,
        frame=frame,
        gt_category=gt,
    )


def _move_library(track_ids, regions=NAMES) -> OperatorLibrary:
    ops = []
    for tid in track_ids:
        for src in regions:
            for dst in regions:
                if src == dst:
                    continue
                ops.append(
                    Operator(
                        f"move_{tid}_{src}_{dst}",
                        frozenset({(tid, "location", src)}),
                        frozenset({(tid, "location", dst)}),
                        frozenset({(tid, "location", src)}),
                    )
                )
    return OperatorLibrary(ops)


def _apply_action(world, action_name):
    prefix, dst = action_name.rsplit("_", 1)
    tid = prefix[len("move_") :].rsplit("_", 1)[0]
    world[tid] = dst


def _build_runtime(
    use_entity,
    use_relations,
    use_uncertainty,
    use_audit,
    use_learned_relations=False,
    use_active_policy=False,
    use_write_policy=False,
):
    return PSSRuntime(
        entity_layer=EntityLayer(
            use_default_relations=use_learned_relations,
            use_default_active=use_active_policy,
        ) if use_entity else None,
        relation_layer=RelationFactLayer(use_default_write=use_write_policy),
        planner=None,
        outputs=OutputRouter(),
        uncertainty=UncertaintyService(object_threshold=0.5),
        audit=use_audit,
        uncertainty_enabled=use_uncertainty,
    )


def run_variant(
    name: str,
    use_entity: bool = True,
    use_relations: bool = True,
    use_uncertainty: bool = True,
    use_audit: bool = True,
    use_language: bool = True,
    use_learned_relations: bool = False,
    use_active_policy: bool = False,
    use_write_policy: bool = False,
    noisy_fact: bool = False,
    distractor: bool = False,
) -> dict:
    world = {gt: region for gt, region in ENTITIES}
    runtime = _build_runtime(
        use_entity,
        use_relations,
        use_uncertainty,
        use_audit,
        use_learned_relations,
        use_active_policy,
        use_write_policy,
    )
    last_observations = []
    occluded = {"bottle"} if use_entity else {"bottle"}
    gt_to_track = {gt: gt for gt, _ in ENTITIES}

    if use_entity:
        for frame in range(3):
            observations = [
                _observation(i, gt, world[gt], frame)
                for i, (gt, _) in enumerate(ENTITIES)
                if not (frame == 2 and gt in occluded)
            ]
            instances = {
                gt: {"pos": np.array([*REGIONS[world[gt]], 0.0]), "radius": 0.04}
                for gt, _ in ENTITIES
            }
            records = geometric_relations(instances, near_dist=0.5) if use_relations else None
            last_observations = observations
            runtime.ingest(observations, frame, relation_records=records)
        gt_to_track = {
            str(track.category): track_id
            for track_id, track in runtime.state.entities.items()
        }
        if distractor:
            from pss.memory.scene_graph import Track

            raw = {
                "vision": np.zeros(128, dtype=float),
                "geometry": np.zeros(128, dtype=float),
                "tactile": np.zeros(128, dtype=float),
            }
            raw["vision"][-1] = 1.0
            runtime.entities.graph.add(
                Track(
                    id="obj_distractor",
                    pos=np.array([2.0, 2.0, 0.0]),
                    quat=np.array([1.0, 0.0, 0.0, 0.0]),
                    category="toy",
                    confidence=1.0,
                    raw=raw,
                )
            )
            runtime.entities.sync(
                runtime.state, active_ids=runtime.entities.active_ids(
                    observations=last_observations
                )
            )
        # Ingested tracks use generated ids; map by category (distinct here).
    else:
        # No entity layer: entities are the raw observations of the last frame.
        observations = [
            _observation(i, gt, world[gt], 2)
            for i, (gt, _) in enumerate(ENTITIES)
            if gt not in occluded
        ]
        for obs in observations:
            from pss.runtime import Entity

            runtime.state.add_entity(
                Entity(
                    id=obs.gt_id,
                    category=obs.category,
                    pos=np.asarray(obs.pos),
                    quat=np.asarray(obs.quat),
                    confidence=obs.confidence,
                    active=True,
                )
            )
        gt_to_track = {gt: gt for gt, _ in ENTITIES if gt not in occluded}

    # Facts: location for each tracked entity.
    for gt, region in world.items():
        tid = gt_to_track.get(gt)
        if tid is None:
            continue
        runtime.assert_fact(tid, "location", region, source="perception", priority=0)
    if use_relations:
        instances = {
            gt_to_track.get(gt, gt): {
                "pos": np.array([*REGIONS[region], 0.0]),
                "radius": 0.04,
            }
            for gt, region in world.items()
            if gt in gt_to_track
        }
        runtime.relations.add_symbolic(geometric_relations(instances, near_dist=0.5), runtime.state)
        runtime.relations.sync(runtime.state)

    # Conflicting fact: a human source disagrees about the bottle.
    bottle_tid = gt_to_track.get("bottle")
    if bottle_tid is not None:
        runtime.assert_fact(
            bottle_tid, "location", "A", source="human", priority=2, reason="conflict test"
        )

    noisy_rejected = False
    if noisy_fact and bottle_tid is not None:
        before_len = len(runtime.state.audit)
        runtime.assert_fact(
            bottle_tid,
            "location",
            "C",
            source="noisy",
            priority=0,
            reason="low-reliability source",
            reliability=0.3,
        )
        noisy_rejected = any(
            e.action == "fact_rejected" and e.actor == "noisy"
            for e in runtime.state.audit[before_len:]
        )

    conflict_flagged = bool(
        runtime.state.uncertainty and "fact_conflict" in runtime.state.uncertainty.flags
    )
    recommendation = (
        runtime.state.uncertainty.recommendation if runtime.state.uncertainty else "disabled"
    )

    # Goal needs the occluded bottle: only persistent identity can supply it.
    goal_facts = set()
    cup_tid = gt_to_track.get("cup")
    if cup_tid:
        goal_facts.add((cup_tid, "location", "G"))
    goal_facts.add((bottle_tid or "bottle", "location", "C"))
    goal = Goal(id="goal_tabletop", facts=frozenset(goal_facts), description="cup to G, bottle to C")
    runtime.set_goal(goal)
    library = _move_library(sorted(set(gt_to_track.values()) | {bottle_tid or "bottle"}))
    runtime.planner = RuntimePlanner(library)
    plan = runtime.plan(goal.id)

    # Execute the plan in the world (kinematic).
    task_success = False
    if plan.actions:
        for action in plan.actions:
            _apply_action(world, action)
        final = {gt_to_track.get(gt, gt): region for gt, region in world.items()}
        task_success = all(
            final.get(fact[0]) == fact[2] for fact in goal_facts
        )
    elif set(goal_facts) <= runtime.state.fact_set():
        task_success = True

    # Explanation + citations.
    citations = []
    explanation = ""
    if use_language:
        result = runtime.explain(
            f"{bottle_tid or 'bottle'} 的 location 是什么？" if bottle_tid else "列出所有对象"
        )
        explanation = result.language
        citations = result.citations

    # Audit / rollback: write a bad fact, then restore the previous version.
    rollback_ok = False
    version_before = runtime.state.version
    cup_before = (
        runtime.state.get_fact(cup_tid, "location").value if cup_tid else None
    )
    if cup_tid:
        try:
            runtime.assert_fact(cup_tid, "location", "B", source="bad", priority=3)
            runtime.rollback(version_before)
            restored = runtime.state.get_fact(cup_tid, "location")
            rollback_ok = bool(restored and restored.value == cup_before)
        except Exception:
            rollback_ok = False

    summary = runtime.summary()
    slots = runtime.entities.slots(2, observations=last_observations) if use_entity else []
    return {
        "variant": name,
        "learned_relations": sum(1 for r in runtime.state.relations if r.origin == "learned"),
        "rejected_facts": sum(1 for e in runtime.state.audit if e.action == "fact_rejected"),
        "noisy_fact_rejected": noisy_rejected,
        "slots": [s["id"] for s in slots],
        "components": summary.get("components"),
        "entity_persistence": bottle_tid is not None,
        "plan_success": plan is not None and bool(plan.actions),
        "plan_actions": len(plan.actions) if plan else 0,
        "task_success": task_success,
        "conflict_flagged": conflict_flagged,
        "recommendation": recommendation,
        "rollback_ok": rollback_ok,
        "explanation_has_location": "location" in explanation,
        "explanation_citations": len(citations),
        "n_entities": summary["entities"],
        "n_relations": summary["relations"],
        "n_facts": summary["facts"],
        "n_audit": summary["audit"],
        "version": summary["version"],
    }


def run() -> dict:
    variants = {
        "full": run_variant("full"),
        "no_entity": run_variant("no_entity", use_entity=False),
        "no_relations": run_variant("no_relations", use_relations=False),
        "no_uncertainty": run_variant("no_uncertainty", use_uncertainty=False),
        "no_audit": run_variant("no_audit", use_audit=False),
        "no_language": run_variant("no_language", use_language=False),
    }
    payload = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "variants": variants,
    }
    out = Path(OUT)
    out.mkdir(parents=True, exist_ok=True)
    (out / "runtime.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "runtime.md").write_text(render(payload), encoding="utf-8")
    return payload


def render(payload: dict) -> str:
    lines = [
        "# PSS cognitive runtime: cross-layer closed loop",
        "",
        f"> generated: {payload['generated']}",
        "",
        "| variant | entity persistence | plan | task success | conflict flagged | recommendation | rollback | explanation citations |",
        "|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for name, r in payload["variants"].items():
        lines.append(
            f"| {name} | {'yes' if r['entity_persistence'] else 'no'} | "
            f"{'yes' if r['plan_success'] else 'no'} ({r['plan_actions']}) | "
            f"{'yes' if r['task_success'] else 'no'} | "
            f"{'yes' if r['conflict_flagged'] else 'no'} | {r['recommendation']} | "
            f"{'yes' if r['rollback_ok'] else 'no'} | {r['explanation_citations']} |"
        )
    lines += [
        "",
        "## State size",
        "",
        "| variant | entities | relations | facts | audit | version |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, r in payload["variants"].items():
        lines.append(
            f"| {name} | {r['n_entities']} | {r['n_relations']} | {r['n_facts']} | "
            f"{r['n_audit']} | {r['version']} |"
        )
    lines += [
        "",
        "## What the ablations show",
        "",
        "- `no_entity`: the occluded bottle is missing, so the goal that needs it cannot be planned.",
        "- `no_relations`: the plan still succeeds from facts, but the relation layer is empty.",
        "- `no_uncertainty`: the conflicting fact is not flagged and no review is recommended.",
        "- `no_audit`: the bad write cannot be rolled back.",
        "- `no_language`: no grounded explanation/citations.",
        "",
    ]
    return "\n".join(lines)


def main():
    ensure_utf8()
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    payload = run()
    for name, r in payload["variants"].items():
        print(
            f"{name}: entity={r['entity_persistence']} plan={r['plan_success']} "
            f"task={r['task_success']} conflict={r['conflict_flagged']} "
            f"rollback={r['rollback_ok']} citations={r['explanation_citations']}"
        )
    print(f"output: {OUT}")


if __name__ == "__main__":
    main()
