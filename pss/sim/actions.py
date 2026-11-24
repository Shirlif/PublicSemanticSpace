"""Scripted action primitives for the tabletop data pipeline.

These are **proxies**, not a learned policy:
- ``push`` applies a real external force and simulates the response;
- ``lift`` holds the object kinematically (a stand-in for grasping);
- ``place`` teleports to a target and lets it settle;
- ``wait`` advances the simulation.

The goal is to produce action-state transitions with ground-truth poses, not to
claim physical fidelity for grasping.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .scene import Scene


@dataclass
class Action:
    kind: str
    target: int | None = None
    params: dict = field(default_factory=dict)


def _body_id(scene: Scene, index: int) -> int:
    return int(scene.model.body(scene.objects[index].body_name).id)


def push(scene: Scene, target: int, rng: np.random.Generator, force: float | None = None,
         steps: int = 15, settle: int = 20) -> dict:
    direction = rng.normal(size=2)
    direction /= max(np.linalg.norm(direction), 1e-9)
    magnitude = float(force if force is not None else rng.uniform(1.5, 4.5))
    body_id = _body_id(scene, target)
    scene.data.xfrc_applied[body_id, :3] = [direction[0] * magnitude, direction[1] * magnitude, 0.0]
    scene.step(steps)
    scene.data.xfrc_applied[body_id, :3] = 0.0
    scene.step(settle)
    return {
        "kind": "push",
        "target": target,
        "force": magnitude,
        "direction": direction.tolist(),
        "steps": steps,
    }


def lift(scene: Scene, target: int, rng: np.random.Generator, height: float = 0.12,
         hold_steps: int = 10) -> dict:
    pos, _ = scene.get_object_pose(target)
    target_pos = pos + np.array([0.0, 0.0, float(height)])
    for _ in range(hold_steps):
        scene.set_object_pose(target, target_pos)
        mujoco.mj_step(scene.model, scene.data)
    final_pos, _ = scene.get_object_pose(target)
    return {
        "kind": "lift",
        "target": target,
        "height": float(height),
        "hold_steps": hold_steps,
        "final_pos": final_pos.tolist(),
    }


def place(scene: Scene, target: int, rng: np.random.Generator, target_xy=None,
          settle: int = 25) -> dict:
    obj = scene.objects[target]
    if target_xy is None:
        target_xy = [float(rng.uniform(-0.30, 0.16)), float(rng.uniform(-0.20, 0.20))]
    target_xy = [float(target_xy[0]), float(target_xy[1])]
    target_pos = np.array([target_xy[0], target_xy[1], obj.rest_height + 0.08])
    scene.set_object_pose(target, target_pos)
    scene.step(settle)
    return {"kind": "place", "target": target, "target_xy": target_xy}


def wait(scene: Scene, steps: int = 10) -> dict:
    scene.step(int(steps))
    return {"kind": "wait", "steps": int(steps)}


def apply_action(scene: Scene, action: Action, rng: np.random.Generator) -> dict:
    if action.kind == "push":
        return push(scene, int(action.target), rng, **action.params)
    if action.kind == "lift":
        return lift(scene, int(action.target), rng, **action.params)
    if action.kind == "place":
        return place(scene, int(action.target), rng, **action.params)
    if action.kind == "wait":
        return wait(scene, **action.params)
    raise ValueError(f"unknown action kind: {action.kind}")


def _bin_target(scene: Scene, rng: np.random.Generator) -> list[float]:
    if not scene.bins:
        return [float(rng.uniform(-0.30, 0.16)), float(rng.uniform(-0.20, 0.20))]
    b = scene.bins[int(rng.integers(0, len(scene.bins)))]
    jitter = rng.uniform(-0.02, 0.02, size=2)
    return [float(b["pos"][0] + jitter[0]), float(b["pos"][1] + jitter[1])]


def scripted_actions(
    scene: Scene, rng: np.random.Generator, n_actions: int = 4
) -> list[Action]:
    """A coherent push/lift/place/wait sequence for one episode."""
    actions: list[Action] = []
    held: int | None = None
    pattern = ["push", "lift", "place", "wait"]
    for k in range(n_actions):
        kind = pattern[k % len(pattern)]
        if kind == "push":
            target = int(rng.integers(0, len(scene.objects)))
            actions.append(
                Action("push", target, {"force": float(rng.uniform(1.5, 4.5)), "steps": 15})
            )
        elif kind == "lift":
            held = int(rng.integers(0, len(scene.objects)))
            actions.append(
                Action("lift", held, {"height": float(rng.uniform(0.10, 0.16)), "hold_steps": 10})
            )
        elif kind == "place":
            target = held if held is not None and rng.random() < 0.7 else int(
                rng.integers(0, len(scene.objects))
            )
            actions.append(Action("place", target, {"target_xy": _bin_target(scene, rng)}))
            held = None
        else:
            actions.append(Action("wait", None, {"steps": 10}))
    return actions
