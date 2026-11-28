"""C4: minimal active perception for contact events (re-observe / wait).

The policy runs on a **live** MuJoCo tabletop scene.  The simulation is stepped
one tick at a time; two observers watch the *same* physical trajectory:

- **passive**: a fixed low-rate observer, one frame every ``base_every`` steps;
- **active**: the same base frames plus extra frames requested by the policy.

The policy triggers when force and geometry evidence disagree, or when
geometry predicts an imminent/ongoing contact:

- ``reobserve``: inside the nominal horizon, switch to a dense observation
  window (``dense_every``) around the suspected event;
- ``wait``: at the nominal horizon, if the suspicion is unresolved, extend the
  run (up to ``max_wait_steps``) and keep observing densely until the contact
  has been seen and has ended.

The active arm is **observation-only**.  No physical push/tap is applied, so the
physics trajectory, and therefore the ground truth, is identical for both
observers.  A physical probe is deliberately deferred: applying a force would
change the trajectory and make the passive/active comparison depend on
probe-induced contacts.  C2 fusion is not integrated here, so the uncertainty
signal is the simple force/geometry agreement described above (not a calibrated
fusion posterior).

Ground truth is built from MuJoCo contact forces over a **fixed reference
trajectory** (``nominal + max_wait_steps``), independent of the policy, so the
active arm cannot choose its own GT horizon.  Events are re-estimated on the
observed frame set after the run (``run_channel("force")``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pss.scene.model import ObjectState, SceneModel
from pss.sim.scene import SceneConfig, build_scene
from pss.sim.sensors import SensorRig

from .bench_api import (
    ContactConfig,
    ContactScenario,
    all_pairs,
    evaluate_events,
    run_channel,
)
from .gt import events_from_contacts
from .model import ContactFrame


@dataclass
class ActiveConfig:
    """Configuration for one live passive-vs-active run."""

    kind: str = "drop"
    seed: int = 0
    width: int = 96
    height: int = 72

    # Observation budget.
    nominal_steps: int = 400
    base_every: int = 10  # passive period (sim steps); 10 * 2 ms = 20 ms
    dense_every: int = 2  # active re-observe period
    reobserve_steps: int = 100
    cooldown_steps: int = 40
    max_episodes: int = 3
    max_extra_frames: int = 200
    max_wait_steps: int = 150
    end_margin_steps: int = 30

    # Force/geometry agreement (the C2-fusion stand-in).
    wake_gap: float = 0.06
    wait_gap: float = 0.06
    conflict_delta: float = 0.35
    borderline_lo: float = 0.15
    borderline_hi: float = 0.85
    force_f0: float = 1.0
    eps_contact: float = 0.02
    geom_tau: float = 0.01

    # Event/GT thresholds (aligned with pss.events.bench_api defaults).
    threshold: float = 0.5
    gt_threshold: float = 1.0
    min_duration: float = 0.02
    max_gap: float = 0.05
    hysteresis: float = 0.0


@dataclass
class ProbeRecord:
    """One active action (observation request)."""

    kind: str  # "reobserve" | "wait"
    step: int
    t: float
    reason: str  # near | conflict | borderline
    pair: tuple[str, str] | None = None
    extra_frames: int = 0
    extra_steps: int = 0


# ---------------------------------------------------------------------------
# Scene / frame helpers (mirror pss.events.scenarios without modifying it)
# ---------------------------------------------------------------------------


def _radius_for(geom: str, size: tuple) -> float:
    size = tuple(float(v) for v in size)
    if geom == "sphere":
        return float(size[0])
    if geom == "box":
        return float(max(size))
    if geom == "cylinder":
        return float(max(size[0], 2.0 * size[1]))
    if geom == "capsule":
        return float(size[0] + size[1])
    if geom == "ellipsoid":
        return float(max(size))
    return float(max(size))


def _scene_model(scene, rig: SensorRig, scene_id: str) -> SceneModel:
    poses = rig.object_poses()
    objects = {}
    for obj in scene.objects:
        pos = np.asarray(poses[obj.instance_id]["pos"], dtype=np.float64)
        objects[obj.instance_id] = ObjectState(
            id=obj.instance_id,
            pos=pos,
            shape=obj.geom_type,
            size=str(obj.size),
            radius=_radius_for(obj.geom_type, obj.size),
            provenance={"source": "mujoco", "category": str(obj.category)},
        )
    return SceneModel(scene_id=scene_id, objects=objects)


def _capture(scene, rig: SensorRig, scene_id: str, t: float) -> ContactFrame:
    return ContactFrame(
        t=float(t),
        contacts=rig.contacts(),
        tactile=rig.tactile_maps(),
        poses={
            oid: np.asarray(p["pos"], dtype=np.float64)
            for oid, p in rig.object_poses().items()
        },
        scene_model=_scene_model(scene, rig, scene_id),
    )


def _setup(kind: str, cfg: ActiveConfig):
    """Build the same primitive scenes as ``pss.events.scenarios``."""
    scene = build_scene(
        SceneConfig(
            n_objects=2,
            seed=cfg.seed,
            width=cfg.width,
            height=cfg.height,
            bins=0,
        ),
        categories=["box", "ball"],
        instance_indices=[0, 0],
    )
    rig = SensorRig(scene, cfg.width, cfg.height)
    external = None
    if kind == "drop":
        target, dropped = scene.objects[0], scene.objects[1]
        scene.set_object_pose(0, [0.0, 0.0, target.rest_height + 0.002])
        scene.set_object_pose(1, [0.0, 0.0, dropped.rest_height + 0.35])
        scene.forward()
    elif kind == "collide":
        target, mover = scene.objects[0], scene.objects[1]
        scene.set_object_pose(0, [0.0, 0.0, target.rest_height + 0.002])
        scene.set_object_pose(1, [-0.18, 0.0, mover.rest_height + 0.002])
        scene.forward()
        external = (1, np.array([2.5, 0.0, 0.0]), 60)
    elif kind == "near_miss":
        a, b = scene.objects[0], scene.objects[1]
        scene.set_object_pose(0, [-0.15, 0.15, a.rest_height + 0.002])
        scene.set_object_pose(1, [0.05, -0.15, b.rest_height + 0.002])
        scene.forward()
    else:
        raise ValueError(f"unknown live scenario {kind!r}")
    return scene, rig, external


# ---------------------------------------------------------------------------
# Evidence / agreement signal
# ---------------------------------------------------------------------------


def _pair_of(c: dict, known: set[str]) -> tuple[str, str] | None:
    a, b = str(c.get("instance1")), str(c.get("instance2"))
    if a not in known or b not in known or a == b:
        return None
    return tuple(sorted((a, b)))


def _pair_gap(frame: ContactFrame, pair: tuple[str, str]) -> float | None:
    scene = frame.scene_model
    if scene is None or pair[0] not in scene.objects or pair[1] not in scene.objects:
        return None
    a = scene.objects[pair[0]]
    b = scene.objects[pair[1]]
    dist = float(np.linalg.norm(np.asarray(a.pos) - np.asarray(b.pos)))
    return dist - float(a.radius) - float(b.radius)


def _pair_force(frame: ContactFrame, pair: tuple[str, str]) -> float:
    best = 0.0
    known = set(frame.poses)
    for c in frame.contacts:
        if _pair_of(c, known) != pair:
            continue
        best = max(best, float(c.get("normal_force", 0.0)))
    return best


def _evidence(
    frame: ContactFrame, pair: tuple[str, str], cfg: ActiveConfig
) -> tuple[float, float, float | None]:
    force = _pair_force(frame, pair)
    f = force / (force + cfg.force_f0) if force > 0 else 0.0
    gap = _pair_gap(frame, pair)
    if gap is None:
        g = 0.0
    else:
        g = float(1.0 / (1.0 + np.exp((gap - cfg.eps_contact) / cfg.geom_tau)))
    return float(f), float(g), gap


def _reason(f: float, g: float, gap: float | None, cfg: ActiveConfig, gap_threshold: float) -> str | None:
    if gap is not None and gap < gap_threshold:
        return "near"
    if abs(f - g) > cfg.conflict_delta:
        return "conflict"
    if cfg.borderline_lo < f < cfg.borderline_hi or cfg.borderline_lo < g < cfg.borderline_hi:
        return "borderline"
    return None


def _policy_hit(
    frame: ContactFrame, cfg: ActiveConfig, gap_threshold: float
) -> tuple[str, tuple[str, str], float, float, float | None] | None:
    """Return the most severe triggered pair, if any."""
    severity = {"near": 0, "conflict": 1, "borderline": 2}
    best = None
    for pair in all_pairs(frame):
        f, g, gap = _evidence(frame, pair, cfg)
        reason = _reason(f, g, gap, cfg, gap_threshold)
        if reason is None:
            continue
        rank = severity[reason]
        if best is None or rank < best[0]:
            best = (rank, reason, pair, f, g, gap)
    if best is None:
        return None
    _, reason, pair, f, g, gap = best
    return reason, pair, f, g, gap


def _max_force(frame: ContactFrame) -> float:
    known = set(frame.poses)
    best = 0.0
    for c in frame.contacts:
        if _pair_of(c, known) is None:
            continue
        best = max(best, float(c.get("normal_force", 0.0)))
    return best


# ---------------------------------------------------------------------------
# Live passive vs active run
# ---------------------------------------------------------------------------


def run_live_scenario(kind: str, cfg: ActiveConfig) -> dict:
    """Run one live scene and return passive/active event metrics."""
    scene, rig, external = _setup(kind, cfg)
    dt = float(scene.model.opt.timestep)
    scene_id = f"{kind}{cfg.seed}"
    nominal = int(cfg.nominal_steps)

    dense: list[ContactFrame] = []
    passive: list[ContactFrame] = []
    active: list[ContactFrame] = []
    probes: list[ProbeRecord] = []
    probe_extra: dict[int, int] = {}

    reference_steps = nominal + int(cfg.max_wait_steps)
    active_stop_step = nominal  # last step the active observer is allowed to see
    reobserve_until = -1
    waiting = False
    last_probe_step = -10**9
    episodes = 0
    extra_frames = 0
    current_probe: int | None = None
    confirmed = False
    settled = False
    last_contact_step = -1
    step = 0

    # The reference trajectory always runs to ``reference_steps`` so that GT is
    # independent of the active policy.  The policy only decides how long/how
    # densely it observes that fixed trajectory.
    while step < reference_steps:
        step += 1
        if external is not None:
            body_index, force, until = external
            body_id = int(scene.model.body(scene.objects[body_index].body_name).id)
            scene.data.xfrc_applied[body_id, :3] = (
                force if (step - 1) < until else 0.0
            )
        scene.step(1)
        t = step * dt
        frame = _capture(scene, rig, scene_id, t)
        dense.append(frame)

        base = step <= nominal and (step % cfg.base_every == 0)
        if base:
            passive.append(frame)

        # ---- settle: contact seen and finished -------------------------
        if _max_force(frame) >= cfg.gt_threshold:
            confirmed = True
            last_contact_step = step
        if (
            confirmed
            and last_contact_step > 0
            and step >= last_contact_step + cfg.end_margin_steps
        ):
            settled = True
            if reobserve_until > step:
                reobserve_until = step
            if waiting and active_stop_step > step:
                active_stop_step = step

        # ---- policy: request more observations -------------------------
        if (
            step < nominal
            and base
            and not settled
            and step > reobserve_until
            and (step - last_probe_step) > cfg.cooldown_steps
        ):
            hit = _policy_hit(frame, cfg, cfg.wake_gap)
            if (
                hit is not None
                and episodes < cfg.max_episodes
                and extra_frames < cfg.max_extra_frames
            ):
                reason, pair, _f, _g, _gap = hit
                end = min(step + cfg.reobserve_steps, nominal)
                if end > step:
                    reobserve_until = end
                    episodes += 1
                    last_probe_step = step
                    probes.append(ProbeRecord("reobserve", step, t, reason, pair))
                    current_probe = len(probes) - 1
        elif step == nominal and not waiting and not settled:
            hit = _policy_hit(frame, cfg, cfg.wait_gap)
            if hit is not None:
                reason, pair, _f, _g, _gap = hit
                waiting = True
                active_stop_step = reference_steps
                probes.append(ProbeRecord("wait", step, t, reason, pair))
                current_probe = len(probes) - 1

        # ---- which active frames are observed --------------------------
        in_reobs = step <= reobserve_until
        in_wait = waiting and step > nominal and step <= active_stop_step
        observe_active = base or (
            (in_reobs or in_wait) and (step % cfg.dense_every == 0)
        )
        if observe_active:
            active.append(frame)
            if not base:
                extra_frames += 1
                if current_probe is not None:
                    probe_extra[current_probe] = probe_extra.get(current_probe, 0) + 1

    gt = events_from_contacts(
        dense,
        threshold=cfg.gt_threshold,
        min_duration=cfg.min_duration,
        max_gap=cfg.max_gap,
    )
    event_cfg = ContactConfig(
        threshold=cfg.threshold,
        gt_threshold=cfg.gt_threshold,
        hysteresis=cfg.hysteresis,
        min_duration=cfg.min_duration,
        max_gap=cfg.max_gap,
        seed=cfg.seed,
    )
    passive_scenario = ContactScenario(
        scenario_id=f"{scene_id}-passive",
        frames=passive,
        gt_events=gt,
        metadata={"kind": kind, "seed": cfg.seed, "arm": "passive"},
    )
    active_scenario = ContactScenario(
        scenario_id=f"{scene_id}-active",
        frames=active,
        gt_events=gt,
        metadata={"kind": kind, "seed": cfg.seed, "arm": "active"},
    )
    pred_passive = run_channel(passive_scenario, event_cfg, channel="force")
    pred_active = run_channel(active_scenario, event_cfg, channel="force")
    metrics_passive = evaluate_events(pred_passive, gt)
    metrics_active = evaluate_events(pred_active, gt)

    def _event_dict(event) -> dict:
        return {
            "event_id": event.event_id,
            "pair": list(event.pair),
            "onset": event.onset,
            "offset": event.offset,
            "confidence": event.confidence,
            "intensity": event.intensity,
        }

    extra_steps = max(0, int(active_stop_step - nominal))
    for i, record in enumerate(probes):
        record.extra_frames = int(probe_extra.get(i, 0))
        if record.kind == "wait":
            record.extra_steps = extra_steps
    rig.close()

    return {
        "kind": kind,
        "seed": cfg.seed,
        "scenario": scene_id,
        "dt": dt,
        "nominal_steps": nominal,
        "reference_steps": reference_steps,
        "active_stop_step": int(active_stop_step),
        "total_steps": step,
        "dense_frames": len(dense),
        "passive_frames": len(passive),
        "active_frames": len(active),
        "extra_frames": int(len(active) - len(passive)),
        "extra_steps": int(extra_steps),
        "probes": [
            {
                "kind": p.kind,
                "step": p.step,
                "t": round(p.t, 6),
                "reason": p.reason,
                "pair": list(p.pair) if p.pair else None,
                "extra_frames": p.extra_frames,
                "extra_steps": p.extra_steps,
            }
            for p in probes
        ],
        "gt_events": [
            {"onset": e.onset, "offset": e.offset, "intensity": e.intensity}
            for e in gt
        ],
        "passive_events": [_event_dict(e) for e in pred_passive],
        "active_events": [_event_dict(e) for e in pred_active],
        "passive": metrics_passive,
        "active": metrics_active,
    }


__all__ = [
    "ActiveConfig",
    "ProbeRecord",
    "run_live_scenario",
]
