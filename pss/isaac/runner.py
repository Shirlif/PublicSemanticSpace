"""Run one or more episodes in Isaac Sim and write bundles (T10).

Usage (through the top-level CLI)::

    pss sim run specs/s1_occlusion.yaml --episodes 1 --seed 0 --device dev
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence

from pss.spec import WorldSpec, load_spec

from .bundle import write_bundle

REPO_ROOT = Path(__file__).resolve().parents[2]


def _gpu_info() -> dict[str, str]:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        name, memory, driver = [part.strip() for part in out.split(",")]
        return {"name": name, "memory": memory, "driver": driver}
    except Exception:
        return {"name": "unknown", "memory": "unknown", "driver": "unknown"}


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _physx_version() -> str:
    try:
        import omni.kit.app

        manager = omni.kit.app.get_app().get_extension_manager()
        ext_id = manager.get_enabled_extension_id("omni.physx")
        if ext_id:
            return ext_id.split("omni.physx-", 1)[-1].split("+", 1)[0]
    except Exception:
        pass
    try:
        import omni.physx

        interface = omni.physx.get_physx_interface()
        for attr in ("get_physx_version", "get_version"):
            fn = getattr(interface, attr, None)
            if fn is not None:
                return str(fn())
    except Exception:
        pass
    return "unknown"


def _isaac_version() -> str:
    try:
        return importlib.metadata.version("isaacsim")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _interp_motion(motion: Sequence[dict[str, Any]], t_s: float) -> list[float]:
    points = sorted(
        (float(k["t_s"]), [float(v) for v in k["pos"]]) for k in motion
    )
    if t_s <= points[0][0]:
        return points[0][1]
    if t_s >= points[-1][0]:
        return points[-1][1]
    for (t0, p0), (t1, p1) in zip(points, points[1:]):
        if t0 <= t_s <= t1:
            alpha = 0.0 if t1 == t0 else (t_s - t0) / (t1 - t0)
            return [p0[i] + (p1[i] - p0[i]) * alpha for i in range(3)]
    return points[-1][1]


def _set_occluders(stage: Any, spec: WorldSpec, t_s: float) -> dict[str, list[float]]:
    """Return occluder poses at ``t_s``.

    Animated occluders are already authored as USD time samples; the timeline
    drives them.  We only write a static pose for occluders without motion.
    The returned poses are what the analytic visibility sampler uses, keeping
    GT independent of the renderer.
    """
    from pxr import Gf, UsdGeom

    poses: dict[str, list[float]] = {}
    for occ in spec.scene.get("occluders", []):
        name = str(occ["name"])
        pose = list(occ.get("pose", [0.0, 0.0, 0.0, 1, 0, 0, 0]))
        motion = occ.get("motion")
        if motion:
            pose[0:3] = _interp_motion(motion, t_s)
        prim = stage.GetPrimAtPath(f"/World/Occluders/{name}")
        if prim and prim.IsValid() and not motion:
            xformable = UsdGeom.Xformable(prim)
            for op in xformable.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    op.Set(Gf.Vec3d(pose[0], pose[1], pose[2]))
                    break
        poses[name] = pose
    return poses


def _world_pose(stage: Any, prim_path: str) -> list[float] | None:
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = matrix.ExtractTranslation()
    q = matrix.ExtractRotationQuat()
    real = float(q.GetReal())
    imag = [float(v) for v in q.GetImaginary()]
    norm = math.sqrt(real * real + sum(v * v for v in imag))
    if norm > 1e-12:
        real = real / norm
        imag = [v / norm for v in imag]
    return [
        float(t[0]),
        float(t[1]),
        float(t[2]),
        real,
        imag[0],
        imag[1],
        imag[2],
    ]


def _bbox(pose: Sequence[float], size: Sequence[float]) -> tuple[list[float], list[float], list[float]]:
    from .visibility import transform_point

    hx, hy, hz = (float(size[0]) / 2, float(size[1]) / 2, float(size[2]) / 2)
    corners = [
        transform_point(pose, (sx * hx, sy * hy, sz * hz))
        for sx in (-1, 1)
        for sy in (-1, 1)
        for sz in (-1, 1)
    ]
    mins = [min(c[i] for c in corners) for i in range(3)]
    maxs = [max(c[i] for c in corners) for i in range(3)]
    flat = [coord for corner in corners for coord in corner]
    return mins, maxs, flat


def _asset_contacts(
    poses: dict[str, list[float]],
    assets: dict[str, Any],
    *,
    table_top: float,
    tolerance: float = 0.003,
) -> dict[str, dict[str, Any]]:
    """W2 analytic contact GT for support and box-box touching.

    Real contact sensors land with S4/S5.  For rigid boxes on a table this is
    exact enough to exercise the GT schema and the contact-dependent metrics.
    """
    boxes: dict[str, tuple[list[float], list[float]]] = {}
    for aid, inst in assets.items():
        mins, maxs, _ = _bbox(poses[aid], inst.get("size_m", [0.04, 0.04, 0.04]))
        boxes[aid] = (mins, maxs)

    out: dict[str, dict[str, Any]] = {
        aid: {"in_contact_with": [], "contact_points": [], "net_force_n": 0.0}
        for aid in assets
    }
    for aid, (mins, maxs) in boxes.items():
        mass = float(assets[aid].get("mass_kg") or 0.0)
        if mins[2] <= table_top + tolerance:
            out[aid]["in_contact_with"].append("table")
            out[aid]["contact_points"].append("table")
            out[aid]["net_force_n"] += mass * 9.81
        for other, (omins, omaxs) in boxes.items():
            if other == aid:
                continue
            overlap = all(
                mins[i] <= omaxs[i] + tolerance and omins[i] <= maxs[i] + tolerance
                for i in range(3)
            )
            if overlap:
                out[aid]["in_contact_with"].append(other)
                out[aid]["contact_points"].append(other)
    return out


def _flat(value: Any) -> list[float]:
    try:
        array = value.detach().cpu().numpy() if hasattr(value, "detach") else value
        import numpy as np

        return np.asarray(array, dtype=float).reshape(-1).tolist()
    except Exception:
        return []


def _sample_layout(spec: WorldSpec, seed: int, table_top: float) -> dict[str, list[float]]:
    """Stratified per-seed placement inside the spec workspace."""
    policy = spec.raw.get("seed_policy", {}).get("pose_sampling", "fixed")
    if policy == "fixed":
        return {}
    import numpy as np

    workspace = spec.scene.get("workspace", {}).get("xy", [[-0.3, -0.25], [0.3, 0.25]])
    (x0, y0), (x1, y1) = workspace
    assets = list(spec.assets)
    n = len(assets)
    nx = max(1, int(math.ceil(math.sqrt(n))))
    ny = max(1, int(math.ceil(n / nx)))
    cells = [(i, j) for i in range(nx) for j in range(ny)]
    rng = np.random.default_rng(seed)
    rng.shuffle(cells)
    layout: dict[str, list[float]] = {}
    for inst, (i, j) in zip(assets, cells):
        cell_w = (x1 - x0) / nx
        cell_h = (y1 - y0) / ny
        cx = x0 + (i + float(rng.uniform(0.15, 0.85))) * cell_w
        cy = y0 + (j + float(rng.uniform(0.15, 0.85))) * cell_h
        size_z = float(inst.get("size_m", [0.04, 0.04, 0.04])[2])
        layout[str(inst["id"])] = [cx, cy, table_top + size_z / 2.0]
    return layout


def _set_asset_translate(stage: Any, prim_path: str, pos: list[float]) -> None:
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return
    xformable = UsdGeom.Xformable(prim)
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
            return
    xformable.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))


def _run_one(
    spec: WorldSpec,
    seed: int,
    *,
    device: str,
    out_root: Path,
    suite: str,
    variant: str,
    cache_dir: Path | None,
    force: bool,
    bus: bool = False,
    obs_addr: str = "tcp://127.0.0.1:5555",
    gt_addr: str = "tcp://127.0.0.1:5556",
) -> Path:
    from .scene_builder import build_scene

    build = build_scene(spec, cache_dir=cache_dir, force=force)
    usd_path = Path(build["cache_dir"]) / build["usd"]
    hash8 = spec.short_hash
    bundle_dir = out_root / device / suite / hash8 / f"seed{seed}"
    if (bundle_dir / "manifest.json").exists() and not force:
        print(f"[skip] {bundle_dir} already exists (use --force to overwrite)")
        return bundle_dir
    if bundle_dir.exists() and force:
        shutil.rmtree(bundle_dir)

    if spec.recording.get("store_images") is not False:
        raise NotImplementedError(
            "store_images=true is not implemented in W1; set recording.store_images: false "
            "in the spec until the camera renderer lands in W2"
        )

    import omni.usd
    from isaacsim.core.api import SimulationContext

    context = omni.usd.get_context()
    context.open_stage(str(usd_path))
    stage = context.get_stage()

    physics_dt = float(spec.physics["dt"])
    obs_rate = float(spec.recording["obs_rate_hz"])
    render_every = max(1, round(1.0 / (physics_dt * obs_rate)))
    time_limit = float(spec.raw.get("task", {}).get("time_limit_s", 10.0))
    total_steps = int(round(time_limit / physics_dt))
    camera = spec.cameras[0]
    cam_name = str(camera["name"])
    cam_pose = [float(v) for v in camera.get("pose", [0, 0, 0, 1, 0, 0, 0])]
    fov = float(camera.get("fov_deg", 60.0))
    aspect = float(camera["width"]) / float(camera["height"])

    table_cfg = spec.scene.get("table", {})
    table_size = [float(v) for v in table_cfg.get("size", [1.2, 0.8, 0.04])]
    table_top = float(table_cfg.get("height_m", 0.75)) + table_size[2] / 2.0

    asset_rows = {str(a["id"]): a for a in spec.assets}
    asset_paths = {aid: f"/World/Assets/{aid}" for aid in asset_rows}

    layout = _sample_layout(spec, seed, table_top)
    for aid, pos in layout.items():
        _set_asset_translate(stage, asset_paths[aid], pos)

    sim = SimulationContext(
        physics_dt=physics_dt,
        rendering_dt=physics_dt,
        stage_units_in_meters=1.0,
        physics_prim_path="/World/PhysicsScene",
    )
    wall_start = time.time()
    sim.play()

    publisher = None
    if bus:
        from .bus import BusPublisher

        publisher = BusPublisher(obs_addr, gt_addr)
        print(f"[bus] obs={obs_addr} gt={gt_addr}")

    from .visibility import Box, compute_visibility

    fault_specs = sorted(spec.faults, key=lambda f: int(f.get("at_step", 0)))
    active_faults: dict[str, dict[str, Any]] = {}
    fault_events: list[dict[str, Any]] = []
    if layout:
        fault_events.append(
            {
                "t_s": 0.0,
                "kind": "layout_sampled",
                "source": "runner",
                "detail_json": json.dumps(layout),
            }
        )
    faults_injected = 0

    robot_view = None
    robot_prim_path = None
    if spec.robot:
        robot_prim_path = str(spec.robot.get("prim_path", "/World/Robot"))
        try:
            from isaacsim.core.prims import Articulation

            robot_view = Articulation(robot_prim_path)
            robot_view.initialize()
        except Exception as exc:  # articulation metadata is optional in W2
            print(f"[warn] articulation view init failed: {exc}")

    gt_rows: list[dict[str, Any]] = []
    obs_rows: list[dict[str, Any]] = []
    robot_rows: list[dict[str, Any]] = []
    last_visibility: dict[str, dict[str, object]] = {}
    last_occluder_poses: dict[str, list[float]] = {}

    for step in range(total_steps):
        t_s = step * physics_dt

        for fault in fault_specs:
            if step != int(fault.get("at_step", -1)):
                continue
            kind = str(fault.get("kind"))
            params = dict(fault.get("params") or {})
            if kind != "occlusion_insert":
                raise NotImplementedError(f"fault kind not implemented in W2: {kind}")
            name = str(params.get("occluder", f"occluder_{step}"))
            region = params.get("region", [0.0, 0.0])
            pose = list(params.get("pose") or [float(region[0]), float(region[1]), 0.90, 1, 0, 0, 0])
            size = list(params.get("size", [0.25, 0.25, 0.01]))
            from .scene_builder import _add_cube

            _add_cube(
                stage,
                f"/World/Faults/{name}",
                tuple(float(v) for v in size),
                pose,
                color=(0.08, 0.08, 0.08),
                rigid=False,
            )
            duration = int(params.get("duration_steps", 120))
            active_faults[name] = {
                "pose": pose,
                "size": size,
                "end": step + duration,
            }
            faults_injected += 1
            fault_events.append(
                {
                    "t_s": t_s,
                    "kind": "fault_injected",
                    "source": "fault_engine",
                    "detail_json": json.dumps({"kind": kind, "name": name, **params}),
                }
            )

        for name, fault in list(active_faults.items()):
            if step >= int(fault["end"]):
                try:
                    stage.RemovePrim(f"/World/Faults/{name}")
                except Exception:
                    pass
                active_faults.pop(name, None)
                fault_events.append(
                    {
                        "t_s": t_s,
                        "kind": "fault_removed",
                        "source": "fault_engine",
                        "detail_json": json.dumps({"name": name}),
                    }
                )

        last_occluder_poses = _set_occluders(stage, spec, t_s)

        poses: dict[str, list[float]] = {}
        for aid, path in asset_paths.items():
            pose = _world_pose(stage, path)
            if pose is not None:
                poses[aid] = pose
            else:
                poses[aid] = [float(v) for v in asset_rows[aid].get("pose", [0, 0, 0, 1, 0, 0, 0])]

        if step % render_every == 0:
            boxes = [
                Box(
                    name=aid,
                    pose=poses[aid],
                    size=asset_rows[aid].get("size_m", [0.04, 0.04, 0.04]),
                )
                for aid in asset_rows
            ]
            for occ in spec.scene.get("occluders", []):
                name = str(occ["name"])
                boxes.append(
                    Box(
                        name=name,
                        pose=last_occluder_poses.get(name, occ.get("pose")),
                        size=occ["size"],
                    )
                )
            for name, fault in active_faults.items():
                boxes.append(
                    Box(
                        name=f"fault:{name}",
                        pose=fault["pose"],
                        size=fault["size"],
                    )
                )
            last_visibility = compute_visibility(
                boxes, cam_pose, fov_deg=fov, aspect=aspect, per_axis=3
            )
            obs_rows.append(
                {
                    "step": step,
                    "t_s": t_s,
                    "camera": cam_name,
                    "image_ref": None,
                    "depth_ref": None,
                    "robot_q": None,
                    "ee_pose": None,
                    "camera_pose": cam_pose,
                }
            )
            if publisher is not None:
                publisher.publish_obs(
                    {
                        "kind": "obs",
                        "step": step,
                        "t_s": t_s,
                        "camera": cam_name,
                        "camera_pose": cam_pose,
                        "width": int(camera["width"]),
                        "height": int(camera["height"]),
                        "fov_deg": fov,
                    }
                )

        contacts = _asset_contacts(poses, asset_rows, table_top=table_top)

        for aid, inst in asset_rows.items():
            pose = poses[aid]
            vis = last_visibility.get(aid, {})
            contact = contacts.get(aid, {})
            bbox_min, bbox_max, bbox_flat = _bbox(
                pose, inst.get("size_m", [0.04, 0.04, 0.04])
            )
            gt_rows.append(
                {
                    "physics_step": step,
                    "t_s": t_s,
                    "instance_id": aid,
                    "class": str(inst["class"]),
                    "twin_group": inst.get("twin_group"),
                    "pose_xyz": pose[0:3],
                    "pose_quat": pose[3:7],
                    "bbox_min": bbox_min,
                    "bbox_max": bbox_max,
                    "bbox_3d": bbox_flat,
                    "visibility_ratio": float(vis.get("visibility_ratio", 0.0)),
                    "occluded_by": list(vis.get("occluded_by", [])),
                    "occluder_of": list(vis.get("occluder_of", [])),
                    "contact_points": list(contact.get("contact_points", [])),
                    "net_force_n": float(contact.get("net_force_n", 0.0)),
                    "in_contact_with": list(contact.get("in_contact_with", [])),
                    "grasped_by": None,
                    "q": [],
                    "qd": [],
                    "tau": [],
                    "relations": [],
                }
            )

        if publisher is not None:
            publisher.publish_gt(
                {
                    "kind": "gt",
                    "step": step,
                    "t_s": t_s,
                    "instances": [
                        {
                            "instance_id": aid,
                            "class": str(inst["class"]),
                            "pose_xyz": poses[aid][0:3],
                            "pose_quat": poses[aid][3:7],
                            "visibility_ratio": float(
                                last_visibility.get(aid, {}).get("visibility_ratio", 0.0)
                            ),
                            "occluded_by": list(
                                last_visibility.get(aid, {}).get("occluded_by", [])
                            ),
                        }
                        for aid, inst in asset_rows.items()
                    ],
                }
            )

        sim.step(render=False)

        if robot_view is not None:
            try:
                q = _flat(robot_view.get_joint_positions())
                qd = _flat(robot_view.get_joint_velocities())
                tau = _flat(robot_view.get_applied_joint_efforts())
            except Exception:
                q, qd, tau = [], [], []
            base = _world_pose(stage, robot_prim_path) if robot_prim_path else None
            robot_rows.append(
                {
                    "physics_step": step,
                    "t_s": t_s,
                    "base_pose": base or [],
                    "q": q,
                    "qd": qd,
                    "tau": tau,
                }
            )

    if publisher is not None:
        publisher.close()

    sim_time = float(sim.current_time)
    # ``SimulationContext.play()`` performs two internal physics updates before
    # the loop starts (one initial update plus the play commit).  Those frames
    # are part of the simulated clock but are not recorded as GT rows; expect
    # them explicitly instead of pretending the loop starts at t=0 exactly.
    expected_time = time_limit + 2.0 * physics_dt
    if abs(sim_time - expected_time) > 0.5 * physics_dt:
        raise RuntimeError(
            f"timebase mismatch: simulated {sim_time:.4f}s for {total_steps} steps, "
            f"expected {expected_time:.4f}s (physics_dt={physics_dt})"
        )
    sim.stop()
    wall_s = time.time() - wall_start
    try:
        context.close_stage()
    except Exception:
        pass

    gpu = _gpu_info()
    stack = {
        "isaac_sim": _isaac_version(),
        "isaac_lab": "not_used_w1",
        "physx": _physx_version(),
        "gpu": f"{gpu['name']} {gpu['memory']}",
        "driver": gpu["driver"],
        "pss_commit": _git_commit(),
        "detector": {
            "name": str(spec.perception.get("detector_tier", "oracle")),
            "sha256": None,
        },
        "llm": {"name": None, "revision": None},
    }

    episode_id = f"{suite}__seed{seed:02d}__{spec.perception.get('detector_tier', 'oracle')}__{variant}"
    markers = [
        {"t_s": 0.0, "kind": "sim_start", "source": "runner", "detail_json": "{}"},
        {"t_s": sim_time, "kind": "sim_end", "source": "runner", "detail_json": "{}"},
    ]
    write_bundle(
        bundle_dir,
        episode_id=episode_id,
        spec_id=spec.id,
        spec_hash=spec.sha256,
        device_class=device,
        seed=seed,
        stack=stack,
        tables={
            "obs_frames": obs_rows,
            "gt": gt_rows,
            "robot_state": robot_rows,
            "events": [markers[0], *fault_events, markers[1]],
        },
        variant=variant,
        detector_tier=str(spec.perception.get("detector_tier", "oracle")),
        faults_injected=faults_injected,
        outcome={
            "task_success": None,
            "steps": total_steps,
            "wall_s": wall_s,
            "sim_time_s": sim_time,
        },
    )
    print(f"[ok] seed {seed}: {bundle_dir}  steps={total_steps}  wall={wall_s:.1f}s")
    return bundle_dir


def run(
    spec_path: str | Path,
    *,
    episodes: int = 1,
    seed: int = 0,
    device: str = "dev",
    out_root: str | Path | None = None,
    suite: str | None = None,
    variant: str = "full",
    cache_dir: str | Path | None = None,
    force: bool = False,
    bus: bool = False,
    obs_addr: str = "tcp://127.0.0.1:5555",
    gt_addr: str = "tcp://127.0.0.1:5556",
) -> list[Path]:
    if device not in ("dev", "official"):
        raise ValueError("device must be 'dev' or 'official'")
    spec = load_spec(spec_path)
    gpu = _gpu_info()
    if device == "official" and "3090" not in gpu["name"]:
        raise RuntimeError(
            f"device=official requires a 3090 Ti, found {gpu['name']!r}; "
            "official numbers may not come from the dev machine"
        )
    suite_name = suite or spec.id.rsplit("_v", 1)[0]
    root = Path(out_root) if out_root is not None else REPO_ROOT / "bundles"
    cache = Path(cache_dir) if cache_dir is not None else None

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaacsim import SimulationApp

    from .launch import launch_config

    app = SimulationApp(launch_config())
    outputs: list[Path] = []
    try:
        for k in range(episodes):
            outputs.append(
                _run_one(
                    spec,
                    seed + k,
                    device=device,
                    out_root=root,
                    suite=suite_name,
                    variant=variant,
                    cache_dir=cache,
                    force=force,
                    bus=bus,
                    obs_addr=obs_addr,
                    gt_addr=gt_addr,
                )
            )
    except BaseException:
        import sys
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        # ``app.close()`` can spin forever after a mid-episode exception (open
        # stage + playing timeline); leave the process immediately so the
        # traceback survives.
        os._exit(1)
    finally:
        app.close()
    return outputs
