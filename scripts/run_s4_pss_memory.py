"""PSS-in-the-loop memory task: target occluded, relocated, then picked.

The sensor is Isaac GT boxes/poses with an RGB color feature (not GT ids).
During occlusion the cup is removed from observations and moved; PSS must
re-identify the same track from color/pose and then execute the pick-place.
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
    parser.add_argument("--spec", default="specs/s4_pss_memory.yaml")
    parser.add_argument("--out", default="scorecards/S4_pss_memory.json")
    args = parser.parse_args()
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaacsim import SimulationApp

    from pss.isaac.launch import launch_config

    app = SimulationApp(launch_config())
    try:
        return _run(app, args)
    except BaseException:
        import traceback

        Path(r"D:\isaac\s4_pss_memory_error.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
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


def _observations(stage, assets, *, hidden=()):
    from pss.isaac.runner import _world_pose
    from pss.isaac.visibility import Box, compute_visibility
    from pss.memory.perception import Observation

    poses = {}
    for aid in assets:
        poses[aid] = _world_pose(stage, f"/World/Assets/{aid}") or [0, 0, 0, 1, 0, 0, 0]
    boxes = [
        Box(aid, poses[aid], assets[aid].get("size_m", [0.04, 0.04, 0.04]))
        for aid in assets
        if aid not in hidden
    ]
    visibility = compute_visibility(
        boxes, [0.0, 0.0, 1.3, 1, 0, 0, 0], fov_deg=60, aspect=4 / 3
    )
    observations = []
    for aid, inst in assets.items():
        if aid in hidden or visibility.get(aid, {}).get("visibility_ratio", 0.0) <= 0.001:
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

    from pss.isaac.runner import _world_pose
    from pss.isaac.scene_builder import _add_cube, build_scene
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

    # Phase 1: observe the cup before occlusion.
    for frame in range(8):
        runtime.ingest(_observations(stage, assets), frame)
        for _ in range(12):
            executor._step(sim)
    cup_before = next((t for t in runtime.entities.graph.tracks.values() if t.category == "cup"), None)
    if cup_before is None:
        Path(args.out).write_text(json.dumps({"task_success": False, "reason": "no cup track before occlusion"}) + "\n", encoding="utf-8")
        sim.stop()
        return 1
    cup_before_id = cup_before.id

    # Phase 2: occlude the cup for 2 s and relocate it while hidden.
    cup_pose = _world_pose(stage, "/World/Assets/cup_01") or [0.2, -0.18, 0.79]
    board_name = "memory_board"
    _add_cube(
        stage,
        f"/World/Faults/{board_name}",
        (0.16, 0.16, 0.01),
        [float(cup_pose[0]), float(cup_pose[1]), 0.90, 1, 0, 0, 0],
        color=(0.05, 0.05, 0.05),
        rigid=False,
    )
    for _ in range(4):
        executor._step(sim)
    for frame in range(8, 28):
        runtime.ingest(_observations(stage, assets, hidden={"cup_01"}), frame)
        for _ in range(12):
            executor._step(sim)
    from pss.isaac.runner import _set_asset_translate

    _set_asset_translate(stage, "/World/Assets/cup_01", [0.24, -0.10, 0.79])
    _set_asset_translate(stage, f"/World/Faults/{board_name}", [0.0, 0.0, -10.0])
    for _ in range(4):
        executor._step(sim)

    # Phase 3: reveal, re-identify, and execute.
    for frame in range(28, 40):
        runtime.ingest(_observations(stage, assets), frame)
        for _ in range(12):
            executor._step(sim)
    cup_after = next(
        (t for t in runtime.entities.graph.tracks.values() if t.category == "cup"),
        None,
    )
    bin_track = next(
        (t for t in runtime.entities.graph.tracks.values() if t.category == "bin"),
        None,
    )
    if cup_after is None or bin_track is None:
        Path(args.out).write_text(json.dumps({"task_success": False, "reason": "missing track after reveal"}) + "\n", encoding="utf-8")
        sim.stop()
        return 1
    same_identity = cup_after.id == cup_before_id

    runtime.assert_fact(cup_after.id, "on", "table", source="perception")
    goal = Goal(
        id="g_memory_pick",
        facts=frozenset({(cup_after.id, "in", bin_track.id)}),
        description="remember the cup, then put it in the bin",
    )
    runtime.set_goal(goal)
    library = VersionedOperatorLibrary(
        [
            Operator(
                "pick",
                frozenset({(cup_after.id, "on", "table")}),
                frozenset({(cup_after.id, "held")}),
                frozenset({(cup_after.id, "on", "table")}),
            ),
            Operator(
                "place",
                frozenset({(cup_after.id, "held")}),
                frozenset({(cup_after.id, "in", bin_track.id)}),
                frozenset({(cup_after.id, "held")}),
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
            target = np.asarray(cup_after.pos[:3]) + np.array([0.0, 0.0, 0.02])
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
            place = executor.place(target, cup_prim, sim, tolerance=0.05)
            actions.append({"skill": "place", "status": place.status, "detail": place.detail})
            if place.status != "ok":
                break
    for _ in range(60):
        executor._step(sim)

    cup_pos = _world_pose(stage, cup_prim) or [0, 0, 0, 1, 0, 0, 0]
    bin_pos = _world_pose(stage, "/World/Assets/bin_a") or [0, 0, 0, 1, 0, 0, 0]
    physical_success = bool(
        abs(cup_pos[0] - bin_pos[0]) < 0.09
        and abs(cup_pos[1] - bin_pos[1]) < 0.09
        and 0.75 < cup_pos[2] < 0.90
    )
    report = {
        "spec": spec.id,
        "plan_actions": list(plan.actions),
        "cup_before": cup_before_id,
        "cup_after": cup_after.id,
        "same_identity": same_identity,
        "goal": [cup_after.id, "in", bin_track.id],
        "actions": actions,
        "physical_success": physical_success,
        "task_success": bool(physical_success and same_identity),
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
    return 0 if report["task_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
