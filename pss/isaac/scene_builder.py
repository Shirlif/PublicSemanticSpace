"""Compile a world spec into a deterministic USD scene (A2).

The compiled scene is cached under ``~/.cache/pss_usd/<spec_sha256>/``.  A
``build.json`` records the spec hash, builder version and the hash of the prim
tree, so two builds of the same spec can be compared without diffing binary
USD files.

This module requires ``pxr`` (supplied by Isaac Sim's Python environment).  It
is imported lazily by the CLI so the PSS environment does not need Isaac.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

from pss.spec import WorldSpec

BUILDER_VERSION = "1.0"


def default_cache_dir() -> Path:
    return Path.home() / ".cache" / "pss_usd"


def _spec_key(spec: WorldSpec) -> str:
    return spec.sha256.split(":", 1)[-1]


def _jsonable(value: Any) -> Any:
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Gf.Vec2f) or isinstance(value, Gf.Vec2d):
        return [float(v[0]), float(v[1])]
    if isinstance(value, Gf.Vec3f) or isinstance(value, Gf.Vec3d):
        return [float(v[0]), float(v[1]), float(v[2])]
    if isinstance(value, Gf.Vec4f) or isinstance(value, Gf.Vec4d):
        return [float(v[i]) for i in range(4)]
    if isinstance(value, Gf.Quatf) or isinstance(value, Gf.Quatd):
        return [float(value.GetReal())] + [float(x) for x in value.GetImaginary()]
    if isinstance(value, Gf.Matrix4d):
        return [[float(value[r][c]) for c in range(4)] for r in range(4)]
    if isinstance(value, Sdf.AssetPath):
        return value.path
    return str(value)


# ---------------------------------------------------------------------------
# prim tree hash


def prim_tree_records(stage: Usd.Stage) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for prim in stage.Traverse():
        attrs: dict[str, Any] = {}
        for attr in prim.GetAttributes():
            name = attr.GetName()
            samples = attr.GetTimeSamples()
            if samples:
                attrs[name] = {
                    "samples": [[float(t), _jsonable(attr.Get(t))] for t in samples]
                }
            elif attr.HasAuthoredValue():
                attrs[name] = _jsonable(attr.Get())
        records.append(
            {
                "path": str(prim.GetPath()),
                "type": str(prim.GetTypeName()),
                "schemas": sorted(prim.GetAppliedSchemas()),
                "attrs": {k: attrs[k] for k in sorted(attrs)},
            }
        )
    records.sort(key=lambda r: r["path"])
    return records


def hash_prim_tree(stage: Usd.Stage) -> str:
    payload = json.dumps(
        prim_tree_records(stage), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# helpers


def _apply_pose(prim: Usd.Prim, pose: Iterable[float] | None) -> None:
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    if pose is None:
        return
    values = [float(v) for v in pose]
    translate = xformable.AddTranslateOp()
    translate.Set(Gf.Vec3d(values[0], values[1], values[2]))
    if len(values) >= 7:
        w, x, y, z = values[3:7]
        xformable.AddOrientOp().Set(Gf.Quatf(w, x, y, z))


def _add_scale(prim: Usd.Prim, scale: Iterable[float]) -> None:
    values = [float(v) for v in scale]
    UsdGeom.Xformable(prim).AddScaleOp().Set(Gf.Vec3f(*values))


def _make_material(
    stage: Usd.Stage,
    path: str,
    color: tuple[float, float, float],
    friction: dict[str, float] | None,
) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    if friction:
        physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics_material.CreateStaticFrictionAttr().Set(float(friction.get("static", 0.8)))
        physics_material.CreateDynamicFrictionAttr().Set(float(friction.get("dynamic", 0.6)))
    return material


def _bind_material(prim: Usd.Prim, material: UsdShade.Material) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def _add_rigid_body(
    prim: Usd.Prim, mass_kg: float | None, *, kinematic: bool = False
) -> None:
    rigid = UsdPhysics.RigidBodyAPI.Apply(prim)
    if kinematic:
        rigid.CreateKinematicEnabledAttr().Set(True)
    if mass_kg is not None:
        mass = UsdPhysics.MassAPI.Apply(prim)
        mass.CreateMassAttr().Set(float(mass_kg))


def _add_collider(prim: Usd.Prim) -> None:
    UsdPhysics.CollisionAPI.Apply(prim)


def _add_cube(
    stage: Usd.Stage,
    path: str,
    size: tuple[float, float, float],
    pose: Iterable[float] | None,
    *,
    color: tuple[float, float, float] = (0.5, 0.5, 0.5),
    friction: dict[str, float] | None = None,
    mass_kg: float | None = None,
    rigid: bool = True,
    kinematic: bool = False,
    material_path: str | None = None,
) -> UsdGeom.Cube:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    _apply_pose(cube.GetPrim(), pose)
    _add_scale(cube.GetPrim(), size)
    _add_collider(cube.GetPrim())
    if rigid or kinematic:
        _add_rigid_body(cube.GetPrim(), mass_kg, kinematic=kinematic)
    material = _make_material(
        stage,
        material_path or f"{path}/Material",
        color,
        friction,
    )
    _bind_material(cube.GetPrim(), material)
    return cube


# ---------------------------------------------------------------------------
# build


def build_scene(
    spec: WorldSpec,
    *,
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Build (or reuse) the USD scene for ``spec`` and return its manifest."""
    root = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    out_dir = root / _spec_key(spec)
    manifest_path = out_dir / "build.json"

    if manifest_path.exists() and not force:
        try:
            cached = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cached = {}
        if (
            cached.get("spec_hash") == spec.sha256
            and cached.get("builder_version") == BUILDER_VERSION
        ):
            cached["_cache_hit"] = True
            return cached

    out_dir.mkdir(parents=True, exist_ok=True)
    stage_path = out_dir / "scene.usd"
    if stage_path.exists():
        stage_path.unlink()

    stage = Usd.Stage.CreateNew(str(stage_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetTimeCodesPerSecond(60.0)

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    physics_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    physics_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    physics_scene.CreateGravityMagnitudeAttr().Set(9.81)
    if spec.physics.get("solver") == "TGS":
        try:
            from pxr import PhysxSchema

            physx_scene = PhysxSchema.PhysxSceneAPI.Apply(physics_scene.GetPrim())
            physx_scene.CreateSolverTypeAttr().Set("TGS")
        except Exception:
            # PhysxSchema is only available inside Kit; core USD remains valid.
            pass

    # --- table ---------------------------------------------------------
    table_cfg = spec.scene.get("table", {})
    table_size = [float(v) for v in table_cfg.get("size", [1.2, 0.8, 0.04])]
    # ``height_m`` is the slab centre height: 0.75 m with a 0.04 m slab puts the
    # top at 0.77 m, matching workspace.table_z and the cube poses in the spec.
    table_center_z = float(table_cfg.get("height_m", 0.75))
    table_thickness = table_size[2]
    table_friction = {"static": 0.7, "dynamic": 0.7}
    if "friction" in table_cfg:
        table_friction = {"static": float(table_cfg["friction"]), "dynamic": float(table_cfg["friction"])}
    _add_cube(
        stage,
        "/World/Table",
        (table_size[0], table_size[1], table_thickness),
        (0.0, 0.0, table_center_z, 1, 0, 0, 0),
        color=(0.42, 0.42, 0.42),
        friction=table_friction,
        rigid=False,
    )

    # --- assets --------------------------------------------------------
    for inst in spec.assets:
        instance_id = str(inst["id"])
        usd_ref = str(inst["usd"])
        pose = inst.get("pose", [0.0, 0.0, 0.0, 1, 0, 0, 0])
        size = inst.get("size_m", [0.04, 0.04, 0.04])
        color = tuple(float(c) for c in inst.get("color", [0.6, 0.6, 0.6]))
        path = f"/World/Assets/{instance_id}"
        if usd_ref != "builtin://cube":
            raise NotImplementedError(
                f"asset {instance_id}: only builtin://cube is supported in W1 "
                f"(got {usd_ref!r}); external USD assets arrive in W4"
            )
        _add_cube(
            stage,
            path,
            tuple(float(v) for v in size),
            pose,
            color=color,  # type: ignore[arg-type]
            friction=inst.get("friction"),
            mass_kg=inst.get("mass_kg"),
            rigid=True,
        )
        # tag the instance for the GT writer
        cube_prim = stage.GetPrimAtPath(path)
        cube_prim.SetCustomDataByKey("pss_instance_id", instance_id)
        cube_prim.SetCustomDataByKey("pss_class", str(inst["class"]))
        if inst.get("twin_group"):
            cube_prim.SetCustomDataByKey("pss_twin_group", str(inst["twin_group"]))

    # --- occluders -----------------------------------------------------
    for occ in spec.scene.get("occluders", []):
        name = str(occ["name"])
        path = f"/World/Occluders/{name}"
        cube = _add_cube(
            stage,
            path,
            tuple(float(v) for v in occ["size"]),
            occ.get("pose", [0.0, 0.0, 0.0, 1, 0, 0, 0]),
            color=(0.15, 0.15, 0.15),
            rigid=False,
        )
        motion = occ.get("motion")
        if motion:
            translate = None
            xformable = UsdGeom.Xformable(cube.GetPrim())
            for op in xformable.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    translate = op
                    break
            if translate is None:
                translate = xformable.AddTranslateOp()
            for key in motion:
                t_s = float(key["t_s"])
                pos = [float(v) for v in key["pos"]]
                translate.Set(Gf.Vec3d(*pos), Usd.TimeCode(t_s * stage.GetTimeCodesPerSecond()))
        cube.GetPrim().SetCustomDataByKey("pss_occluder", True)

    # --- lights --------------------------------------------------------
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.CreateIntensityAttr().Set(600.0)
    distant = UsdLux.DistantLight.Define(stage, "/World/Lights/Key")
    distant.CreateIntensityAttr().Set(1200.0)
    distant.CreateAngleAttr().Set(2.0)
    UsdGeom.Xformable(distant.GetPrim()).AddRotateXYZOp().Set(
        Gf.Vec3f(-45.0, 15.0, 0.0)
    )

    # --- cameras -------------------------------------------------------
    for cam in spec.cameras:
        name = str(cam["name"])
        path = f"/World/Cameras/{name}"
        camera = UsdGeom.Camera.Define(stage, path)
        aperture = 20.955
        fov_deg = float(cam.get("fov_deg", 60.0))
        focal = (aperture / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
        camera.CreateHorizontalApertureAttr().Set(aperture)
        camera.CreateFocalLengthAttr().Set(focal)
        camera.CreateClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))
        _apply_pose(camera.GetPrim(), cam.get("pose"))

    stage.GetRootLayer().customLayerData = {
        "pss:spec_id": spec.id,
        "pss:spec_hash": spec.sha256,
        "pss:builder_version": BUILDER_VERSION,
    }
    stage.Save()

    prim_hash = hash_prim_tree(stage)
    records = prim_tree_records(stage)
    (out_dir / "prim_tree.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    manifest = {
        "spec_id": spec.id,
        "spec_hash": spec.sha256,
        "builder_version": BUILDER_VERSION,
        "usd": "scene.usd",
        "prim_hash": prim_hash,
        "prim_count": len(records),
        "cache_dir": str(out_dir),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    manifest["_cache_hit"] = False
    return manifest
