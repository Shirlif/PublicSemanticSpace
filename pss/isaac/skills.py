"""G1 skill library (T29/T30).

Base is pinned every physics step (``base_fixed``).  Arm motion uses a damped
least-squares position IK on the right arm Jacobian.  Grasping is a **suction
proxy**: on grasp the object is snapped to the hand and a fixed joint is
created; release removes it.  The plan explicitly allows a simplified gripper
when contact grasping is unreliable; this proxy is labeled as such in S4/S5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

STATUSES = ("ok", "timeout", "slipped", "collision", "unreachable", "unsupported")


@dataclass
class SkillResult:
    name: str
    status: str
    steps: int
    detail: str = ""


class SkillExecutor:
    def __init__(
        self,
        articulation,
        *,
        hand_prim_path: str = "/World/Robot/right_wrist_yaw_link",
        base_position=(0.0, -0.42, 0.77),
        base_orientation=(0.70710678, 0.0, 0.0, 0.70710678),
    ):
        from isaacsim.core.prims import RigidPrim

        self.art = articulation
        self.hand = RigidPrim(hand_prim_path)
        self.hand.initialize()
        self.hand_path = hand_prim_path
        self.base_position = np.asarray(base_position, dtype=float)[None, :]
        self.base_orientation = np.asarray(base_orientation, dtype=float)[None, :]
        self.arm = [self.art.get_dof_index(name) for name in ARM_JOINTS]
        self.hand_link = self.art.get_link_index("right_wrist_yaw_link")
        self.attachments: dict[str, str] = {}
        self.carried: dict[str, Any] = {}
        self.steps = 0

    # -- low level -----------------------------------------------------
    def _pin_base(self) -> None:
        try:
            self.art.set_world_poses(self.base_position, self.base_orientation)
        except Exception:
            pass

    def _hand_position(self) -> np.ndarray:
        return np.asarray(self.hand.get_world_poses()[0][0], dtype=float)

    def _step(self, sim) -> None:
        self._pin_base()
        sim.step(render=False, update_fabric=True)
        self.steps += 1
        # Suction proxy: carry the grasped object kinematically with the hand.
        if self.carried:
            pos, quat = self.hand.get_world_poses()
            for path, view in self.carried.items():
                try:
                    view.set_world_poses(pos, quat)
                except Exception:
                    pass
                # USD fallback: the RigidPrim view can silently fail after a
                # physics reset; authoring the transform directly keeps the
                # object attached in that case.
                try:
                    self._set_prim_pose(path, pos[0], quat[0])
                except Exception:
                    pass

    @staticmethod
    def _set_prim_pose(prim_path: str, pos, quat) -> None:
        from pxr import Gf, Sdf, UsdGeom

        stage = SkillExecutor._stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return
        for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
            elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                if op.GetAttr().GetTypeName() == Sdf.ValueTypeNames.Quatd:
                    op.Set(Gf.Quatd(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])))
                else:
                    op.Set(Gf.Quatf(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])))

    def _ik_tick(self, target: np.ndarray, sim, gain: float = 0.5, clip: float = 0.05) -> float:
        self._step(sim)
        pos = self._hand_position()
        error = target - pos
        jacobians = self.art.get_jacobians()[0]
        # Jacobian columns are [root 6, dofs...]; joint i is column i+6.
        jac = jacobians[self.hand_link, 0:3][:, [i + 6 for i in self.arm]]
        damping = 0.05
        dq = jac.T @ np.linalg.solve(jac @ jac.T + damping * np.eye(3), error)
        dq = np.clip(dq * gain, -clip, clip)
        q = self.art.get_joint_positions()[0].copy()
        q[self.arm] += dq
        self.art.set_joint_position_targets(q[self.arm][None, :], joint_indices=self.arm)
        return float(np.linalg.norm(error))

    # -- skills --------------------------------------------------------
    def home(self, sim, steps: int = 60) -> SkillResult:
        q = self.art.get_joint_positions()[0].copy()
        q[self.arm] = 0.0
        self.art.set_joint_position_targets(q[self.arm][None, :], joint_indices=self.arm)
        for _ in range(steps):
            self._step(sim)
        return SkillResult("home", "ok", steps)

    def reach(self, target, sim, *, tolerance: float = 0.03, max_steps: int = 320) -> SkillResult:
        target = np.asarray(target, dtype=float)
        error = float("inf")
        for step in range(max_steps):
            error = self._ik_tick(target, sim)
            if error <= tolerance:
                return SkillResult("reach", "ok", step + 1, f"err={error:.4f}")
        return SkillResult("reach", "timeout", max_steps, f"err={error:.4f}")

    def grasp(self, object_prim_path: str, sim, *, tolerance: float = 0.06) -> SkillResult:
        object_pos = self._prim_position(object_prim_path)
        hand_pos = self._hand_position()
        if object_pos is None:
            return SkillResult("grasp", "unreachable", 0, "missing object")
        distance = float(np.linalg.norm(object_pos - hand_pos))
        if distance > tolerance:
            return SkillResult("grasp", "unreachable", 0, f"dist={distance:.3f}")
        # Suction proxy: carry the object kinematically with the hand.
        from isaacsim.core.prims import RigidPrim

        pos, quat = self.hand.get_world_poses()
        view = RigidPrim(object_prim_path)
        view.initialize()
        view.set_world_poses(pos, quat)
        self.carried[object_prim_path] = view
        self.attachments[object_prim_path] = "carried"
        for _ in range(5):
            self._step(sim)
        carried_pos = self._prim_position(object_prim_path)
        if carried_pos is None or float(np.linalg.norm(carried_pos - self._hand_position())) > 0.06:
            self.carried.pop(object_prim_path, None)
            self.attachments.pop(object_prim_path, None)
            return SkillResult("grasp", "slipped", 5, "object did not follow hand")
        return SkillResult("grasp", "ok", 5, f"dist={distance:.3f} suction")

    def release(self, object_prim_path: str, sim) -> SkillResult:
        if object_prim_path not in self.carried:
            return SkillResult("release", "unreachable", 0, "not attached")
        self.carried.pop(object_prim_path, None)
        self.attachments.pop(object_prim_path, None)
        for _ in range(10):
            self._step(sim)
        return SkillResult("release", "ok", 10)

    def place(self, target, object_prim_path: str, sim, *, tolerance: float = 0.04) -> SkillResult:
        target = np.asarray(target, dtype=float)
        reached = self.reach(target, sim, tolerance=tolerance)
        if reached.status != "ok":
            return SkillResult("place", reached.status, reached.steps, reached.detail)
        released = self.release(object_prim_path, sim)
        return SkillResult("place", released.status, reached.steps + released.steps, released.detail)

    def walk_to(self, *args, **kwargs) -> SkillResult:
        return SkillResult("walk_to", "unsupported", 0, "base_mobile policy not installed")

    def turn(self, *args, **kwargs) -> SkillResult:
        return SkillResult("turn", "unsupported", 0, "base_fixed")

    # -- helpers -------------------------------------------------------
    @staticmethod
    def _stage():
        import omni.usd

        return omni.usd.get_context().get_stage()

    def _prim_position(self, prim_path: str):
        from pxr import Usd, UsdGeom

        prim = self._stage().GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return None
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        return np.asarray(matrix.ExtractTranslation(), dtype=float)
