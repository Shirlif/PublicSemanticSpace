"""S4 scripted execution baseline (T31/T33).

Loads the G1 pick-place spec, runs home -> reach -> grasp -> place with the
suction-proxy skill layer, optionally injects one S5 fault, and writes a bundle
plus a success flag.  This is a **scripted baseline**, not PSS in the loop.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default="specs/s4_g1_pick_place.yaml")
    parser.add_argument("--fault", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="bundles/dev")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

    from isaacsim import SimulationApp

    from pss.isaac.launch import launch_config

    app = SimulationApp(launch_config())
    try:
        return _run(app, args)
    finally:
        try:
            app.close()
        except Exception:
            pass


def _run(app, args) -> int:
    import omni.usd
    from isaacsim.core.api import SimulationContext
    from isaacsim.core.prims import Articulation

    from pss.isaac.bundle import write_bundle
    from pss.isaac.faults import FaultInjector
    from pss.isaac.runner import _asset_contacts, _bbox, _git_commit, _gpu_info, _world_pose
    from pss.isaac.scene_builder import build_scene
    from pss.isaac.skills import SkillExecutor
    from pss.spec import load_spec

    spec = load_spec(args.spec)
    build = build_scene(spec)
    usd_path = Path(build["cache_dir"]) / build["usd"]
    fault_name = args.fault or "none"
    bundle_dir = (
        Path(args.out)
        / spec.id.rsplit("_v", 1)[0]
        / spec.short_hash
        / fault_name
        / f"seed{args.seed}"
    )
    if bundle_dir.exists() and args.force:
        shutil.rmtree(bundle_dir)

    context = omni.usd.get_context()
    context.open_stage(str(usd_path))
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
    assets = {str(a["id"]): a for a in spec.assets}
    injector = FaultInjector(stage, assets, executor)
    cup = "/World/Assets/cup_01"
    bin_prim = "/World/Assets/bin_a"

    actions: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    wall_start = time.time()

    def record(result, payload: dict[str, Any]) -> None:
        actions.append(
            {
                "step": executor.steps,
                "t_s": executor.steps * float(spec.physics["dt"]),
                "action_type": result.name,
                "payload_json": json.dumps(payload),
                "executor_status": result.status,
                "rejected_reason": result.detail if result.status != "ok" else None,
            }
        )

    record(executor.home(sim, 40), {})
    # Default arm pose can nudge tabletop objects during play(); reset the task
    # objects to their spec poses before measuring/reaching.
    from isaacsim.core.prims import RigidPrim

    for aid in ("cup_01", "bin_a"):
        view = RigidPrim(f"/World/Assets/{aid}")
        view.initialize()
        pose = assets[aid]["pose"]
        view.set_world_poses([pose[0:3]], [pose[3:7]])
    for _ in range(5):
        executor._step(sim)
    if args.fault:
        detail = injector.inject(args.fault, _fault_params(args.fault, spec))
        events.append(
            {
                "t_s": executor.steps * float(spec.physics["dt"]),
                "kind": "fault_injected",
                "source": "fault_engine",
                "detail_json": json.dumps(detail),
            }
        )

    cup_pos = np.asarray(_world_pose(stage, cup)[0:3])
    bin_pos = np.asarray(_world_pose(stage, bin_prim)[0:3])
    reach_target = cup_pos + np.array([0.0, 0.0, 0.02])
    place_target = bin_pos + np.array([0.0, 0.0, 0.08])

    for attempt in range(2):
        reach = executor.reach(reach_target, sim)
        record(reach, {"target": reach_target.tolist(), "attempt": attempt})
        if reach.status != "ok":
            continue
        grasp = executor.grasp(cup, sim)
        record(grasp, {"object": "cup_01", "attempt": attempt})
        if grasp.status != "ok":
            continue
        if args.fault == "grasp_slip" and attempt == 0:
            detail = injector.inject("grasp_slip", {"target": "cup_01"})
            events.append(
                {
                    "t_s": executor.steps * float(spec.physics["dt"]),
                    "kind": "fault_injected",
                    "source": "fault_engine",
                    "detail_json": json.dumps(detail),
                }
            )
        place = executor.place(place_target, cup, sim)
        record(place, {"target": place_target.tolist(), "attempt": attempt})
        if place.status == "ok":
            break
        cup_pos = np.asarray(_world_pose(stage, cup)[0:3])
        reach_target = cup_pos + np.array([0.0, 0.0, 0.02])

    for _ in range(60):
        executor._step(sim)

    cup_pos = np.asarray(_world_pose(stage, cup)[0:3])
    bin_pos = np.asarray(_world_pose(stage, bin_prim)[0:3])
    success = bool(
        abs(cup_pos[0] - bin_pos[0]) < 0.09
        and abs(cup_pos[1] - bin_pos[1]) < 0.09
        and 0.75 < cup_pos[2] < 0.90
    )

    # GT at 10 Hz (poses + contacts; visibility is 1.0: no occlusion in S4).
    gt_rows = []
    obs_rows = []
    steps = executor.steps
    for step in range(0, steps, 12):
        poses = {aid: _world_pose(stage, f"/World/Assets/{aid}") for aid in assets}
        poses = {aid: pose or [0, 0, 0, 1, 0, 0, 0] for aid, pose in poses.items()}
        contacts = _asset_contacts(poses, assets, table_top=0.77)
        for aid, inst in assets.items():
            bmin, bmax, bflat = _bbox(poses[aid], inst.get("size_m", [0.04, 0.04, 0.04]))
            gt_rows.append(
                {
                    "physics_step": step,
                    "t_s": step * float(spec.physics["dt"]),
                    "instance_id": aid,
                    "class": str(inst["class"]),
                    "twin_group": inst.get("twin_group"),
                    "pose_xyz": poses[aid][0:3],
                    "pose_quat": poses[aid][3:7],
                    "bbox_min": bmin,
                    "bbox_max": bmax,
                    "bbox_3d": bflat,
                    "visibility_ratio": 1.0,
                    "occluded_by": [],
                    "occluder_of": [],
                    "contact_points": list(contacts[aid]["contact_points"]),
                    "net_force_n": float(contacts[aid]["net_force_n"]),
                    "in_contact_with": list(contacts[aid]["in_contact_with"]),
                    "grasped_by": "g1" if aid == "cup_01" and cup in executor.attachments else None,
                    "q": [],
                    "qd": [],
                    "tau": [],
                    "relations": [],
                }
            )
        obs_rows.append(
            {
                "step": step,
                "t_s": step * float(spec.physics["dt"]),
                "camera": str(spec.cameras[0]["name"]),
                "image_ref": None,
                "depth_ref": None,
                "robot_q": None,
                "ee_pose": None,
                "camera_pose": [float(v) for v in spec.cameras[0]["pose"]],
            }
        )

    wall_s = time.time() - wall_start
    gpu = _gpu_info()
    events.append(
        {
            "t_s": steps * float(spec.physics["dt"]),
            "kind": "task_end",
            "source": "scripted_policy",
            "detail_json": json.dumps({"success": success, "cup": cup_pos.tolist()}),
        }
    )
    write_bundle(
        bundle_dir,
        episode_id=f"s4__{fault_name}__seed{args.seed}",
        spec_id=spec.id,
        spec_hash=spec.sha256,
        device_class="dev",
        seed=args.seed,
        stack={
            "isaac_sim": "5.1.0.0",
            "isaac_lab": "not_used",
            "physx": "107.3.26",
            "gpu": f"{gpu['name']} {gpu['memory']}",
            "driver": gpu["driver"],
            "pss_commit": _git_commit(),
            "detector": {"name": "none", "sha256": None},
        },
        tables={"actions": actions, "events": events, "gt": gt_rows, "obs_frames": obs_rows},
        variant="full",
        detector_tier="oracle",
        outcome={"task_success": success, "steps": steps, "wall_s": wall_s},
        faults_injected=1 if args.fault else 0,
    )
    print(json.dumps({"fault": fault_name, "success": success, "steps": steps,
                      "wall_s": round(wall_s, 1), "bundle": str(bundle_dir)}), flush=True)
    sim.stop()
    return 0


def _fault_params(kind: str, spec) -> dict:
    if kind == "object_relocate":
        return {"target": "cup_01", "pose": [0.18, -0.02, 0.79]}
    if kind == "object_remove":
        return {"target": "cup_01"}
    if kind == "twin_swap":
        return {"targets": ["cup_01", "bin_a"]}
    if kind == "joint_limit":
        return {"command": 3.0}
    if kind == "lighting_shock":
        return {"intensity": 2500.0}
    return {}


if __name__ == "__main__":
    raise SystemExit(main())
