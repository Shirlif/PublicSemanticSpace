"""PSS-in-the-loop two-object task: two cups into the bin on their side.

The bowl is the spatial reference.  PSS must assign each cup to a bin from the
``left_of`` relation, keep both identities across two pick-place cycles, and
re-observe the scene after the first placement.  The sensor is Isaac GT
boxes/poses plus a colour appearance feature; GT ids are never given to the
runtime.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default="specs/s4_pss_two.yaml")
    parser.add_argument("--out", default="scorecards/S4_pss_two.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--variant",
        default="full",
        help="PSS runtime variant (e.g. full, no_relations); see pss/runtime/variants.py",
    )
    args = parser.parse_args()
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaacsim import SimulationApp

    from pss.isaac.launch import launch_config

    app = SimulationApp(launch_config())
    try:
        return _run(app, args)
    except BaseException:
        import traceback

        Path(r"D:\isaac\s4_pss_two_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        try:
            app.close()
        except Exception:
            pass


def _colour(inst) -> np.ndarray:
    vec = np.asarray(inst.get("color", [0.5, 0.5, 0.5]), dtype=float)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _observations(stage, assets):
    from pss.isaac.runner import _world_pose
    from pss.isaac.visibility import Box, compute_visibility
    from pss.memory.perception import Observation

    poses = {
        aid: _world_pose(stage, f"/World/Assets/{aid}") or [0, 0, 0, 1, 0, 0, 0]
        for aid in assets
    }
    boxes = [
        Box(aid, poses[aid], assets[aid].get("size_m", [0.04, 0.04, 0.04]))
        for aid in assets
    ]
    visibility = compute_visibility(
        boxes, [0.0, 0.0, 1.3, 1, 0, 0, 0], fov_deg=60, aspect=4 / 3
    )
    observations = []
    for aid, inst in assets.items():
        if visibility.get(aid, {}).get("visibility_ratio", 0.0) <= 0.001:
            continue
        observations.append(
            Observation(
                gt_id=aid,
                pos=np.asarray(poses[aid][0:3], dtype=float),
                quat=np.asarray(poses[aid][3:7], dtype=float),
                appearance=_colour(inst),
                raw={"vision": np.zeros(4), "geometry": np.zeros(4), "tactile": np.zeros(4)},
                confidence=1.0,
                category=str(inst["class"]),
            )
        )
    return observations


def _run(app, args) -> int:
    import omni.usd
    from isaacsim.core.api import SimulationContext
    from isaacsim.core.prims import Articulation

    from pss.isaac.runner import _set_asset_translate, _world_pose
    from pss.isaac.scene_builder import build_scene
    from pss.isaac.skills import SkillExecutor
    from pss.relations import RelObject, relation_holds
    from pss.runtime.operators import VersionedOperatorLibrary
    from pss.runtime.runtime import RuntimePlanner
    from pss.runtime.state import Goal
    from pss.runtime.variants import build_runtime
    from pss.spec import load_spec
    from pss.world_model.actions import Operator

    spec = load_spec(args.spec)
    build = build_scene(spec)
    context = omni.usd.get_context()
    context.open_stage(str(Path(build["cache_dir"]) / build["usd"]))
    stage = context.get_stage()
    rng = np.random.default_rng(args.seed)
    for inst in spec.assets:
        pose = list(inst.get("pose", [0.0, 0.0, 0.0, 1, 0, 0, 0]))
        pose[0] += float(rng.uniform(-0.015, 0.015))
        pose[1] += float(rng.uniform(-0.015, 0.015))
        _set_asset_translate(stage, f"/World/Assets/{inst['id']}", pose[:3])
    sim = SimulationContext(
        physics_dt=float(spec.physics["dt"]),
        rendering_dt=float(spec.physics["dt"]),
        physics_prim_path="/World/PhysicsScene",
    )
    sim.play()
    articulation = Articulation("/World/Robot")
    articulation.initialize()
    executor = SkillExecutor(
        articulation,
        base_position=tuple(float(v) for v in spec.robot["pose"][:3]),
        base_orientation=tuple(float(v) for v in spec.robot["pose"][3:7]),
    )
    executor.home(sim, 40)
    assets = {str(a["id"]): a for a in spec.assets}
    runtime = build_runtime(args.variant, tier="frozen_real")

    # Observe the full scene before the first action.
    for frame in range(10):
        runtime.ingest(_observations(stage, assets), frame)
        for _ in range(12):
            executor._step(sim)

    def tracks(category):
        return [t for t in runtime.entities.graph.tracks.values() if t.category == category]

    cups, bins = tracks("cup"), tracks("bin")
    bowl = next(iter(tracks("bowl")), None)
    if len(cups) != 2 or len(bins) != 2 or bowl is None:
        report = {
            "task_success": False,
            "reason": "missing tracks",
            "counts": {"cups": len(cups), "bins": len(bins), "bowl": bowl is not None},
            "tracks": [(t.id, t.category) for t in runtime.entities.graph.tracks.values()],
        }
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        sim.stop()
        return 1

    def side(track):
        rel = RelObject(track.id, track.pos, [0.04, 0.04, 0.04])
        ref = RelObject(bowl.id, bowl.pos, [0.06, 0.06, 0.04])
        if relation_holds(rel, "left_of", ref, tolerance_m=0.02):
            return "left"
        if relation_holds(rel, "right_of", ref, tolerance_m=0.02):
            return "right"
        return "unknown"

    cup_by_side = {side(c): c for c in cups}
    bin_by_side = {side(b): b for b in bins}
    if set(cup_by_side) != {"left", "right"} or set(bin_by_side) != {"left", "right"}:
        report = {
            "task_success": False,
            "reason": "relations did not separate both cups and bins",
            "cup_sides": [(c.id, side(c)) for c in cups],
            "bin_sides": [(b.id, side(b)) for b in bins],
        }
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        sim.stop()
        return 1

    assignments = {
        cup_by_side[s].id: bin_by_side[s].id for s in ("left", "right")
    }
    goals = frozenset(
        {(cup_id, "in", bin_id) for cup_id, bin_id in assignments.items()}
    )

    for cup in cups:
        runtime.assert_fact(cup.id, "on", "table", source="perception")
    goal = Goal(id="g_two_object", facts=goals, description="put each cup in the bin on its side")
    runtime.set_goal(goal)

    operators = []
    for cup_id in assignments:
        operators.append(
            Operator(
                f"pick|{cup_id}",
                frozenset({(cup_id, "on", "table")}),
                frozenset({(cup_id, "held")}),
                frozenset({(cup_id, "on", "table")}),
            )
        )
    for cup_id, bin_id in assignments.items():
        operators.append(
            Operator(
                f"place|{cup_id}|{bin_id}",
                frozenset({(cup_id, "held")}),
                frozenset({(cup_id, "in", bin_id)}),
                frozenset({(cup_id, "held")}),
            )
        )
    runtime.planner = RuntimePlanner(VersionedOperatorLibrary(operators))
    plan = runtime.plan(goal.id)

    def nearest_asset(class_name: str, pos):
        best = None
        best_dist = 1e9
        for aid, inst in assets.items():
            if str(inst["class"]) != class_name:
                continue
            world = _world_pose(stage, f"/World/Assets/{aid}") or [0, 0, 0]
            distance = float(np.linalg.norm(np.asarray(world[0:3]) - np.asarray(pos[0:3])))
            if distance < best_dist:
                best, best_dist = aid, distance
        return f"/World/Assets/{best}" if best else None

    cup_prim = {cup.id: nearest_asset("cup", cup.pos) for cup in cups}
    bin_prim = {
        b.id: nearest_asset("bin", b.pos) for b in bins
    }
    if any(v is None for v in cup_prim.values()) or any(v is None for v in bin_prim.values()):
        report = {"task_success": False, "reason": "no physical prim for a selected track"}
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        sim.stop()
        return 1

    started_ids = {cup.id: cup.id for cup in cups}
    actions = []
    id_stable = True
    wall_start = time.time()
    for index, action_name in enumerate(plan.actions):
        parts = action_name.split("|")
        if parts[0] == "pick" and len(parts) == 2:
            cup_id = parts[1]
            track = runtime.entities.graph.tracks.get(cup_id)
            if track is None:
                actions.append({"skill": "pick", "status": "lost", "detail": cup_id})
                break
            target = np.asarray(track.pos[:3]) + np.array([0.0, 0.0, 0.02])
            reach = executor.reach(target, sim)
            actions.append({"skill": f"reach:{cup_id}", "status": reach.status, "detail": reach.detail})
            if reach.status != "ok":
                break
            grasp = executor.grasp(cup_prim[cup_id], sim)
            actions.append({"skill": f"grasp:{cup_id}", "status": grasp.status, "detail": grasp.detail})
            if grasp.status != "ok":
                break
        elif parts[0] == "place" and len(parts) == 3:
            cup_id, bin_id = parts[1], parts[2]
            track = runtime.entities.graph.tracks.get(bin_id)
            if track is None:
                actions.append({"skill": "place", "status": "lost", "detail": bin_id})
                break
            target = np.asarray(track.pos[:3]) + np.array([0.0, 0.0, 0.08])
            place = executor.place(target, cup_prim[cup_id], sim, tolerance=0.05)
            actions.append({"skill": f"place:{cup_id}->{bin_id}", "status": place.status, "detail": place.detail})
            if place.status != "ok":
                break

            # Re-observe after the first placement: PSS must still see the same
            # cup identities before the second cycle starts.
            for frame in range(10):
                runtime.ingest(_observations(stage, assets), 100 + index * 10 + frame)
                for _ in range(6):
                    executor._step(sim)
            for cup_track in cups:
                current = runtime.entities.graph.tracks.get(cup_track.id)
                if current is None or current.category != "cup":
                    id_stable = False
        else:
            actions.append({"skill": action_name, "status": "unsupported", "detail": "bad action"})
            break

    for _ in range(60):
        executor._step(sim)

    result = {}
    physical = True
    for cup_id, bin_id in assignments.items():
        cup_pos = _world_pose(stage, cup_prim[cup_id]) or [0, 0, 0, 1, 0, 0, 0]
        bin_pos = _world_pose(stage, bin_prim[bin_id]) or [0, 0, 0, 1, 0, 0, 0]
        ok = bool(
            abs(cup_pos[0] - bin_pos[0]) < 0.09
            and abs(cup_pos[1] - bin_pos[1]) < 0.09
            and 0.75 < cup_pos[2] < 0.90
        )
        physical = physical and ok
        result[cup_id] = {
            "bin": bin_id,
            "cup_xyz": [round(float(v), 3) for v in cup_pos[:3]],
            "bin_xyz": [round(float(v), 3) for v in bin_pos[:3]],
            "ok": ok,
        }

    report = {
        "spec": spec.id,
        "seed": args.seed,
        "variant": args.variant,
        "plan_actions": list(plan.actions),
        "plan_confidence": float(plan.confidence),
        "bowl_track": bowl.id,
        "assignments": assignments,
        "started_ids": started_ids,
        "id_stable_after_first": id_stable,
        "actions": actions,
        "placements": result,
        "physical_success": bool(physical),
        "actions_ok": bool(actions) and all(a["status"] == "ok" for a in actions),
        "task_success": bool(
            physical
            and id_stable
            and len(plan.actions) == 4
            and actions
            and all(a["status"] == "ok" for a in actions)
        ),
        "steps": executor.steps,
        "wall_s": round(time.time() - wall_start, 2),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    sim.stop()
    return 0 if report["task_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
