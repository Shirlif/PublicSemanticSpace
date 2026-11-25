"""Fault engine for S4/S5 (T32).

Ten fault kinds from the design.  Several are event-only on a base_fixed,
scripted-policy dev run (camera dropout, path blocked, base disturb); they are
still injected and logged so the S5 table has one row per class.  Object-level
faults are physically applied.
"""

from __future__ import annotations

from typing import Any

import numpy as np

FAULT_KINDS = (
    "occlusion_insert",
    "object_relocate",
    "object_remove",
    "grasp_slip",
    "camera_dropout",
    "lighting_shock",
    "twin_swap",
    "joint_limit",
    "base_disturb",
    "path_blocked",
)


class FaultInjector:
    def __init__(self, stage, assets: dict[str, dict], executor=None):
        self.stage = stage
        self.assets = assets
        self.executor = executor

    def _prim_path(self, instance_id: str) -> str:
        return f"/World/Assets/{instance_id}"

    def _set_translate(self, instance_id: str, pos) -> None:
        from pxr import Gf, UsdGeom

        prim = self.stage.GetPrimAtPath(self._prim_path(instance_id))
        if not prim or not prim.IsValid():
            return
        for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
                return

    def inject(self, kind: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        params = dict(params or {})
        if kind not in FAULT_KINDS:
            raise ValueError(f"unknown fault {kind!r}")
        detail: dict[str, Any] = {"kind": kind, **params}

        if kind == "occlusion_insert":
            from .scene_builder import _add_cube

            name = str(params.get("occluder", "fault_board"))
            pose = params.get("pose", [0.0, 0.0, 0.90, 1, 0, 0, 0])
            size = params.get("size", [0.25, 0.25, 0.01])
            _add_cube(
                self.stage,
                f"/World/Faults/{name}",
                tuple(float(v) for v in size),
                pose,
                color=(0.08, 0.08, 0.08),
                rigid=False,
            )
        elif kind == "object_relocate":
            target = str(params.get("target", "cup_01"))
            pose = params.get("pose", [0.18, -0.05, 0.79])
            self._set_translate(target, pose)
            detail["target"] = target
        elif kind == "object_remove":
            target = str(params.get("target", "cup_01"))
            self._set_translate(target, [0.0, 0.0, -1.0])
            detail["target"] = target
        elif kind == "grasp_slip":
            target = str(params.get("target", "cup_01"))
            if self.executor is not None and target in self.executor.carried:
                self.executor.carried.pop(target, None)
                self.executor.attachments.pop(target, None)
            detail["target"] = target
        elif kind == "lighting_shock":
            from pxr import UsdLux

            intensity = float(params.get("intensity", 2500.0))
            for path in ("/World/Lights/Dome", "/World/Lights/Key"):
                prim = self.stage.GetPrimAtPath(path)
                if prim and prim.IsValid():
                    attr = prim.GetAttribute("intensity")
                    if attr and attr.IsValid():
                        attr.Set(intensity)
            detail["intensity"] = intensity
        elif kind == "twin_swap":
            targets = [str(v) for v in params.get("targets", [])]
            if len(targets) == 2:
                a = self._pose(targets[0])
                b = self._pose(targets[1])
                if a is not None and b is not None:
                    self._set_translate(targets[0], b[:3])
                    self._set_translate(targets[1], a[:3])
            detail["targets"] = targets
        elif kind == "joint_limit":
            if self.executor is not None and len(self.executor.arm):
                q = self.executor.art.get_joint_positions()[0].copy()
                q[self.executor.arm] = 3.0  # beyond limits; controller clamps
                self.executor.art.set_joint_position_targets(
                    q[self.executor.arm][None, :], joint_indices=self.executor.arm
                )
            detail["command"] = 3.0
        elif kind == "base_disturb":
            detail["note"] = "base is pinned in base_fixed; logged only"
        elif kind == "camera_dropout":
            detail["note"] = "offline/scripted run has no live camera; logged only"
        elif kind == "path_blocked":
            detail["note"] = "base_fixed; no walking path; logged only"
        return detail

    def _pose(self, instance_id: str):
        from pxr import Usd, UsdGeom

        prim = self.stage.GetPrimAtPath(self._prim_path(instance_id))
        if not prim or not prim.IsValid():
            return None
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        t = matrix.ExtractTranslation()
        return [float(t[0]), float(t[1]), float(t[2])]
