"""PSS-in-the-loop pick-and-place on Isaac (S4-PSS).

PSS ingests Isaac GT boxes/poses (oracle sensor, class labels only), selects the
cup/bin tracks, asserts goal facts, runs the symbolic planner, and the G1 skill
layer executes the plan.  Success is checked from GT after execution.
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
    parser.add_argument("--spec", default="specs/s4_pss_pick_place.yaml")
    parser.add_argument("--out", default="scorecards/S4_pss.json")
    args = parser.parse_args()
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

    from isaacsim import SimulationApp

    from pss.isaac.launch import launch_config

    app = SimulationApp(launch_config())
    try:
        return _run(app, args)
    except BaseException:
        import traceback

        Path(r"D:\isaac\s4_pss_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        try:
            app.close()
        except Exception:
            pass


def _observations(stage, assets):
    from pss.isaac.runner import _world_pose
    from pss.memory.perception import Observation

    observations = []
    for aid, inst in assets.items():
        pose = _world_pose(stage, f"/World/Assets/{aid}") or [0, 0, 0, 1, 0, 0, 0]
        observations.append(
            Observation(
                gt_id=aid,
                pos=np.asarray(pose[0:3], dtype=float),
                quat=np.asarray(pose[3:7], dtype=float),
                appearance=np.zeros(4),
                raw={"vision": np.zeros(4), "geometry": np.zeros(4), "tactile": np.zeros(4)},
                confidence=1.0,
                category=str(inst["class"]),
                frame=0,
            )
        )
    return observations


def _run(app, args) -> int:
    import omni.usd
    from isaacsim.core.api import SimulationContext
    from isaacsim.core.prims import Articulation

    from pss.isaac.runner import _world_pose
    from pss.isaac.scene_builder import build_scene
    from pss.isaac.skills import SkillExecutor
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
    runtime = build_runtime("full", tier="frozen_real")

    # Let PSS ingest a few frames and build entity tracks.
    for frame in range(12):
        runtime.ingest(_observations(stage, assets), frame)
        for _ in range(12):
            executor._step(sim)

    tracks = list(runtime.entities.graph.tracks.values())
    cup = next((t for t in tracks if t.category == "cup"), None)
    bin_track = next((t for t in tracks if t.category == "bin"), None)
    if cup is None or bin_track is None:
        report = {"task_success": False, "reason": "PSS did not form cup/bin tracks",
                  "tracks": [(t.id, t.category) for t in tracks]}
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report))
        sim.stop()
        return 1

    runtime.assert_fact(cup.id, "on", "table", source="perception")
    goal = Goal(id="g_pick_place", facts=frozenset({(cup.id, "in", bin_track.id)}),
                description="put the cup in the bin")
    runtime.set_goal(goal)

    library = VersionedOperatorLibrary(
        [
            Operator(
                "pick",
                frozenset({(cup.id, "on", "table")}),
                frozenset({(cup.id, "held")}),
                frozenset({(cup.id, "on", "table")}),
            ),
            Operator(
                "place",
                frozenset({(cup.id, "held")}),
                frozenset({(cup.id, "in", bin_track.id)}),
                frozenset({(cup.id, "held")}),
            ),
        ]
    )
    runtime.planner = RuntimePlanner(library)
    plan = runtime.plan(goal.id)

    cup_prim = "/World/Assets/cup_01"
    actions = []
    wall_start = time.time()
    for action_name in plan.actions:
        if action_name == "pick":
            target = np.asarray(cup.pos[:3]) + np.array([0.0, 0.0, 0.02])
            reach = executor.reach(target, sim)
            actions.append({"skill": "reach", "status": reach.status, "detail": reach.detail})
            if reach.status != "ok":
                break
            grasp = executor.grasp(cup_prim, sim)
            actions.append({"skill": "grasp", "status": grasp.status, "detail": grasp.detail})
            if grasp.status != "ok":
                break
        elif action_name == "place":
            target = np.asarray(bin_track.pos[:3]) + np.array([0.0, 0.0, 0.08])
            place = executor.place(target, cup_prim, sim)
            actions.append({"skill": "place", "status": place.status, "detail": place.detail})
            if place.status != "ok":
                break
        else:
            actions.append({"skill": action_name, "status": "unsupported", "detail": "unknown operator"})
            break

    for _ in range(60):
        executor._step(sim)

    cup_pos = _world_pose(stage, cup_prim) or [0, 0, 0, 1, 0, 0, 0]
    bin_pos = _world_pose(stage, "/World/Assets/bin_a") or [0, 0, 0, 1, 0, 0, 0]
    success = bool(
        abs(cup_pos[0] - bin_pos[0]) < 0.09
        and abs(cup_pos[1] - bin_pos[1]) < 0.09
        and 0.75 < cup_pos[2] < 0.90
    )
    report = {
        "spec": spec.id,
        "plan_actions": list(plan.actions),
        "plan_confidence": float(plan.confidence),
        "goal": [cup.id, "in", bin_track.id],
        "cup_track": cup.id,
        "bin_track": bin_track.id,
        "actions": actions,
        "task_success": success,
        "cup_xyz": [round(float(v), 3) for v in cup_pos[:3]],
        "bin_xyz": [round(float(v), 3) for v in bin_pos[:3]],
        "steps": executor.steps,
        "wall_s": round(time.time() - wall_start, 2),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    sim.stop()
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
