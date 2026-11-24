"""Phase 7 S1 closed loop: perception -> scene graph -> planner -> push -> verify.

A block must be pushed to a goal region. An obstacle blocks the direct route,
so the planner has to detour. After the block reaches the detour waypoint the
obstacle is relocated, which invalidates the current plan; the next perception
step sees the new layout and replans.

Each outer iteration performs exactly one push pulse, then re-renders and
re-tracks. This keeps the entity ID stable while the block moves and makes the
loop closed at the perception level, not only at the planner level.

No text is produced anywhere in the loop.

Example:
    python -m pss.reasoning.scripts.run_closed_loop
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pss._console import ensure_utf8
from pss.heads.action import ActionHead
from pss.memory import SceneGraph, Tracker, TrackerConfig
from pss.memory.perception import PoseNoise, extract_frame, load_perception_models
from pss.reasoning.grounding import (
    REGIONS,
    build_tabletop_library,
    parse_move_action,
    segment_clear,
    tabletop_facts,
    tabletop_goal,
)
from pss.reasoning.planner import AStarPlanner
from pss.sim.generator import capture_scene_sample
from pss.sim.scene import SceneConfig, build_scene
from pss.sim.sensors import SensorRig


def _push_once(scene, target_index: int, goal_xy, head: ActionHead) -> int:
    pos, _ = scene.get_object_pose(target_index)
    goal = np.asarray(goal_xy, dtype=float)
    if float(np.linalg.norm(pos[:2] - goal)) < 0.035:
        return 0
    proposal = head.propose_push_to(pos[:2], goal, target=target_index)
    if proposal is None:
        return 0
    direction = np.asarray(proposal.params["direction"], dtype=float)
    force = float(proposal.params["force"])
    body_id = scene.model.body(scene.objects[target_index].body_name).id
    scene.data.xfrc_applied[body_id, :3] = [direction[0] * force, direction[1] * force, 0.0]
    scene.step(6)
    scene.data.xfrc_applied[body_id, :3] = 0.0
    scene.step(10)
    return 1


def run_closed_loop(
    expert_dir: str | Path,
    projection_dir: str | Path,
    prototype_dir: str | Path | None = None,
    out_dir: str | Path = "eval/results/closed_loop",
    seed: int = 0,
    anomaly: bool = True,
    max_actions: int = 80,
) -> dict:
    cfg = SceneConfig(
        n_objects=2,
        seed=seed,
        width=96,
        height=96,
        bins=0,
        min_separation=0.05,
    )
    scene = build_scene(cfg, categories=["block", "brick"], instance_indices=[0, 0])
    block_rest = scene.objects[0].rest_height
    brick_rest = scene.objects[1].rest_height
    scene.set_object_pose(0, [-0.22, 0.0, block_rest + 0.002])
    scene.set_object_pose(1, [0.0, 0.0, brick_rest + 0.002])
    scene.forward()

    rig = SensorRig(scene, 96, 96)
    rig.set_camera(azimuth=0.0, elevation=-48.0, distance=1.05)
    models = load_perception_models(
        expert_dir,
        projection_dir,
        prototype_dir,
        device="cpu",
        weights_mode="reliability",
    )
    tracker = Tracker(TrackerConfig(gate_radius=0.60, w_category=0.0), SceneGraph())
    head = ActionHead(gain=10.0, max_force=5.0, min_force=1.5, tolerance=0.03)

    target_track_id: str | None = None
    move_sequence: list[str] = []
    pushes = 0
    replans = 0
    anomaly_injected = False
    success = False
    failure = ""
    final_distance = None
    trace: list[dict] = []
    active_action: str | None = None
    active_dst: str | None = None

    for step in range(max_actions):
        if anomaly_injected:
            # The relocated obstacle is held by an external actor; otherwise
            # table physics slowly drifts it into the remaining route.
            scene.set_object_pose(1, [0.10, 0.105, brick_rest + 0.002])
            scene.forward()
        sample = capture_scene_sample(scene, rig, seed=seed + step)
        observations = extract_frame(
            sample,
            0,
            models,
            PoseNoise(seed=seed + step, enabled=True),
        )
        tracker.step(observations, step)
        active = tracker.graph.active()
        entry = {
            "step": step,
            "active": [
                {"id": t.id, "pos": t.pos[:2].round(4).tolist()} for t in active
            ],
        }
        trace.append(entry)
        if not active:
            failure = "no active tracks"
            entry["failure"] = failure
            break

        if target_track_id is None:
            target = min(
                active,
                key=lambda t: float(np.linalg.norm(t.pos[:2] - REGIONS["A"])),
            )
            target_track_id = target.id
        target = tracker.graph.get(target_track_id)
        if target is None:
            failure = "target track lost"
            entry["failure"] = failure
            break
        entry["target_track"] = target_track_id
        entry["target_pos"] = target.pos[:2].round(4).tolist()
        entry["block_gt"] = scene.get_object_pose(0)[0][:2].round(4).tolist()
        entry["obstacle_gt"] = scene.get_object_pose(1)[0][:2].round(4).tolist()

        final_distance = float(np.linalg.norm(target.pos[:2] - REGIONS["G"]))
        if final_distance < 0.06:
            success = True
            entry["success"] = True
            break

        obstacles = [t.pos for t in active if t.id != target_track_id]
        facts = tabletop_facts(target_track_id, target.pos[:2], max_dist=0.20)
        entry["facts"] = [list(f) for f in facts]

        if (
            anomaly
            and not anomaly_injected
            and float(np.linalg.norm(target.pos[:2] - REGIONS["B"])) < 0.08
        ):
            scene.set_object_pose(1, [0.10, 0.105, brick_rest + 0.002])
            scene.forward()
            anomaly_injected = True
            replans += 1
            active_action = None
            active_dst = None
            entry["anomaly"] = "obstacle moved to (0.10, 0.105)"
            continue

        # Commit to the current move until its waypoint is reached or the
        # segment becomes blocked. Re-planning every push makes the block
        # oscillate between neighboring regions.
        if active_action is not None and active_dst is not None:
            reached = float(np.linalg.norm(target.pos[:2] - REGIONS[active_dst])) < 0.06
            blocked = not segment_clear(
                target.pos[:2], REGIONS[active_dst], obstacles, 0.09
            )
            if reached:
                active_action = None
                active_dst = None
            elif blocked:
                entry["blocked"] = active_action
                replans += 1
                active_action = None
                active_dst = None

        if active_action is None:
            if not facts:
                failure = "target is not near a waypoint"
                entry["failure"] = failure
                break
            library = build_tabletop_library(target_track_id, obstacles)
            plan = AStarPlanner(library).plan(facts, tabletop_goal(target_track_id))
            entry["plan"] = plan.actions
            if not plan.success or not plan.actions:
                failure = plan.reason or "no plan"
                entry["failure"] = failure
                break
            active_action = plan.actions[0]
            _, active_dst = parse_move_action(active_action)

        action = active_action
        dst = active_dst
        entry["action"] = action
        entry["dst"] = dst
        pushes += _push_once(scene, 0, REGIONS[dst], head)
        if not move_sequence or move_sequence[-1] != action:
            move_sequence.append(action)
        entry["block_after"] = scene.get_object_pose(0)[0][:2].round(4).tolist()
        entry["obstacle_after"] = scene.get_object_pose(1)[0][:2].round(4).tolist()

    metrics = {
        "success": success,
        "failure": failure,
        "actions": move_sequence,
        "pushes": pushes,
        "replans": replans,
        "final_distance": final_distance,
        "anomaly_injected": anomaly_injected,
        "text_used": False,
        "target_track": target_track_id,
        "goal_region": REGIONS["G"].tolist(),
        "trace": trace,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Phase 7 S1 closed loop",
        "",
        f"- success: **{success}**",
        f"- moves: {' → '.join(move_sequence) or '—'}",
        f"- pushes: {pushes}",
        f"- replans: {replans}",
        f"- final distance: {final_distance:.3f} m" if final_distance is not None else "",
        f"- anomaly injected: {anomaly_injected}",
        "- text used: False",
        "",
        "Loop: render -> segmentation/tracking -> planner facts -> A* -> one push pulse -> re-render.",
        "The obstacle relocation invalidates the plan; the next perception step replans.",
        "",
    ]
    (out / "REPORT.md").write_text(
        "\n".join(line for line in lines if line is not None), encoding="utf-8"
    )
    return metrics


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", default="checkpoints/category_pretrain_experts")
    ap.add_argument("--projections", default="checkpoints/phase45_projection")
    ap.add_argument("--prototypes", default="checkpoints/phase45_prototypes")
    ap.add_argument("--out", default="eval/results/closed_loop")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-anomaly", action="store_true")
    ap.add_argument("--max-actions", type=int, default=80)
    args = ap.parse_args()
    metrics = run_closed_loop(
        args.experts,
        args.projections,
        args.prototypes,
        args.out,
        seed=args.seed,
        anomaly=not args.no_anomaly,
        max_actions=args.max_actions,
    )
    print("closed loop:")
    print(f"  success: {metrics['success']}")
    print(f"  moves: {' -> '.join(metrics['actions']) or '-'}")
    print(f"  pushes: {metrics['pushes']}  replans: {metrics['replans']}")
    print(f"  final distance: {metrics['final_distance']}")
    print(f"  output: {args.out}")


if __name__ == "__main__":
    main()
