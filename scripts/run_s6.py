"""S6 continuous memory session (T35).

One Isaac app process, one stage, N sub-episodes.  Objects are physically
relocated between sub-episodes; each sub-episode is written as its own bundle
so replay can keep PSS state across the session without mixing episodes.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default="specs/s6_memory.yaml")
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

    from pss.isaac.bundle import write_bundle
    from pss.isaac.runner import (
        _asset_contacts,
        _bbox,
        _git_commit,
        _gpu_info,
        _set_asset_translate,
        _world_pose,
    )
    from pss.isaac.scene_builder import build_scene
    from pss.spec import load_spec

    spec = load_spec(args.spec)
    session = spec.raw["task"]["session"]
    sub_episodes = int(session["sub_episodes"])
    duration = float(session["sub_episode_s"])
    relocations = {
        int(r["after"]): r for r in session.get("relocations", [])
    }
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
    dt = float(spec.physics["dt"])
    steps_per_episode = int(round(duration / dt))
    render_every = 12
    assets = {str(a["id"]): a for a in spec.assets}
    root = (
        Path(args.out)
        / spec.id.rsplit("_v", 1)[0]
        / spec.short_hash
        / f"seed{args.seed}"
    )
    gpu = _gpu_info()
    global_step = 0
    wall_start = time.time()

    for ep in range(sub_episodes):
        if ep in relocations:
            reloc = relocations[ep]
            _set_asset_translate(
                stage, f"/World/Assets/{reloc['target']}", reloc["pose"][0:3]
            )
            for _ in range(5):
                sim.step(render=False)
        bundle_dir = root / f"ep{ep}"
        obs_rows = []
        gt_rows = []
        for _ in range(steps_per_episode):
            if global_step % render_every == 0:
                t_s = global_step * dt
                poses = {
                    aid: _world_pose(stage, f"/World/Assets/{aid}")
                    or [0, 0, 0, 1, 0, 0, 0]
                    for aid in assets
                }
                contacts = _asset_contacts(poses, assets, table_top=0.77)
                for aid, inst in assets.items():
                    bmin, bmax, bflat = _bbox(
                        poses[aid], inst.get("size_m", [0.04, 0.04, 0.04])
                    )
                    gt_rows.append(
                        {
                            "physics_step": global_step,
                            "t_s": t_s,
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
                            "grasped_by": None,
                            "q": [],
                            "qd": [],
                            "tau": [],
                            "relations": [],
                        }
                    )
                obs_rows.append(
                    {
                        "step": global_step,
                        "t_s": t_s,
                        "camera": str(spec.cameras[0]["name"]),
                        "image_ref": None,
                        "depth_ref": None,
                        "robot_q": None,
                        "ee_pose": None,
                        "camera_pose": [float(v) for v in spec.cameras[0]["pose"]],
                    }
                )
            sim.step(render=False)
            global_step += 1

        write_bundle(
            bundle_dir,
            episode_id=f"s6__ep{ep}__seed{args.seed}",
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
                "detector": {"name": "oracle", "sha256": None},
            },
            tables={"obs_frames": obs_rows, "gt": gt_rows, "events": []},
            detector_tier="oracle",
            outcome={"task_success": None, "steps": steps_per_episode, "wall_s": None},
        )
        print(f"[s6] episode {ep} -> {bundle_dir}", flush=True)

    sim.stop()
    (root / "session.json").write_text(
        json.dumps(
            {
                "spec": spec.id,
                "hash": spec.sha256,
                "sub_episodes": sub_episodes,
                "sub_episode_s": duration,
                "relocations": list(relocations.values()),
                "wall_s": time.time() - wall_start,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
