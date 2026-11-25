"""World spec: typed views, validation and canonical hashing.

A world spec is a single YAML document describing one scene family.  Its
canonical SHA-256 is the primary key for every experiment artifact (USD cache,
bundle directory, manifest, scorecard), so the hash must depend only on the
document content -- never on defaults injected at runtime.

The schema follows ``docs/plans/PSS-IsaacSim-评测平台设计.md`` §3.1.  Unknown keys are
preserved (and hashed) but not interpreted; sections that are not needed yet
(e.g. ``robot``) may be omitted and are marked TBD in ``notes``.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

SPEC_VERSION = 1

ID_RE = re.compile(r"^[a-z][a-z0-9_]{2,63}$")

SPLITS = ("train", "val", "test")
TASK_FAMILIES = ("manipulation", "tracking", "navigation")
BASE_MODES = ("base_fixed", "base_mobile")
SOLVERS = ("TGS", "PGS")
DETECTOR_TIERS = ("oracle", "noisy", "frozen_real", "frozen_real_fast")
PERCEPTION_MODES = ("offline", "in_loop")
POSE_SAMPLING = ("latin_hypercube", "uniform", "fixed")
COMPUTE_FROM = ("gt_poses", "gt_contacts")
BUNDLE_FORMATS = ("parquet",)

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

PREDICATES = (
    "left_of",
    "right_of",
    "in_front_of",
    "behind",
    "on_top_of",
    "inside",
    "touching",
    "near",
    "above",
    "below",
)

DEFAULT_PREDICATES = list(PREDICATES)
DEFAULT_METRICS = {
    "primary": ["idf1", "id_retention", "id_switches", "duplicate_rate", "reappearance_rate"],
    "leading": ["slot_purity", "provenance_completeness"],
    "guardrail": ["safety_violation_rate", "permission_violation_count"],
}


class SpecValidationError(ValueError):
    """Raised when a spec has one or more hard validation errors."""

    def __init__(self, errors: list[str], warnings: list[str] | None = None):
        self.errors = errors
        self.warnings = warnings or []
        super().__init__("; ".join(errors))


@dataclass(frozen=True)
class LintResult:
    spec_id: str
    sha256: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class WorldSpec:
    """A validated world spec plus its canonical hash."""

    raw: Mapping[str, Any]
    sha256: str
    source: Path | None = None

    @property
    def id(self) -> str:
        return str(self.raw["id"])

    @property
    def short_hash(self) -> str:
        return self.sha256.split(":", 1)[-1][:8]

    @property
    def split(self) -> str:
        return str(self.raw["split"])

    @property
    def task_family(self) -> str:
        return str(self.raw["task_family"])

    @property
    def scene(self) -> Mapping[str, Any]:
        return self.raw.get("scene", {})

    @property
    def recording(self) -> Mapping[str, Any]:
        return self.raw.get("recording", {})

    @property
    def physics(self) -> Mapping[str, Any]:
        return self.raw.get("physics", {})

    @property
    def perception(self) -> Mapping[str, Any]:
        return self.raw.get("perception", {"mode": "offline", "detector_tier": "oracle"})

    @property
    def assets(self) -> list[Mapping[str, Any]]:
        return list(self.raw.get("assets", {}).get("instances", []))

    @property
    def cameras(self) -> list[Mapping[str, Any]]:
        return list(self.raw.get("cameras", []))

    @property
    def robot(self) -> Mapping[str, Any] | None:
        return self.raw.get("robot")

    @property
    def faults(self) -> list[Mapping[str, Any]]:
        return list(self.raw.get("faults", []))

    @property
    def relations(self) -> Mapping[str, Any]:
        return self.raw.get("relations_gt", {})

    def asset(self, instance_id: str) -> Mapping[str, Any] | None:
        for item in self.assets:
            if item.get("id") == instance_id:
                return item
        return None


# ---------------------------------------------------------------------------
# canonical hash


def canonical_json(raw: Mapping[str, Any]) -> str:
    """Stable JSON rendering used both for hashing and for registry diffs."""
    return json.dumps(
        raw,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def hash_spec(raw: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(raw).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# validation helpers


def _fail(errors: list[str], path: str, msg: str) -> None:
    errors.append(f"{path}: {msg}")


def _warn(warnings: list[str], path: str, msg: str) -> None:
    warnings.append(f"{path}: {msg}")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_vec(value: Any, n: int, *, positive: bool = False) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != n:
        return False
    if not all(_is_number(v) for v in value):
        return False
    return not positive or all(v > 0 for v in value)


def _check_keys(
    errors: list[str], obj: Mapping[str, Any], path: str, allowed: tuple[str, ...]
) -> None:
    extra = sorted(set(obj) - set(allowed))
    for key in extra:
        _fail(errors, f"{path}.{key}", "unknown key (typo or unsupported field)")


TOP_LEVEL_KEYS = (
    "spec_version",
    "id",
    "task_family",
    "split",
    "notes",
    "seed_policy",
    "assets",
    "scene",
    "cameras",
    "robot",
    "physics",
    "task",
    "faults",
    "perception",
    "relations_gt",
    "metrics",
    "recording",
)

ASSET_KEYS = (
    "id",
    "class",
    "usd",
    "mass_kg",
    "friction",
    "restitution",
    "twin_group",
    "split",
    "size_m",
    "pose",
    "color",
    "notes",
)


def validate(raw: Any) -> LintResult:
    """Validate a parsed YAML document.  Returns errors and warnings."""
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(raw, Mapping):
        return LintResult("<unknown>", hash_spec({}), ["<root>: spec must be a YAML mapping"])

    _check_keys(errors, raw, "<root>", TOP_LEVEL_KEYS)

    spec_id = raw.get("id")
    if not isinstance(spec_id, str) or not ID_RE.match(spec_id):
        _fail(errors, "id", "must match ^[a-z][a-z0-9_]{2,63}$")
        spec_id = str(spec_id or "<unknown>")

    version = raw.get("spec_version")
    if version != SPEC_VERSION:
        _fail(errors, "spec_version", f"must be {SPEC_VERSION}")

    if raw.get("task_family") not in TASK_FAMILIES:
        _fail(errors, "task_family", f"must be one of {TASK_FAMILIES}")

    if raw.get("split") not in SPLITS:
        _fail(errors, "split", f"must be one of {SPLITS}")

    _validate_seed_policy(raw.get("seed_policy"), errors, warnings)
    _validate_assets(raw.get("assets"), errors, warnings)
    _validate_scene(raw.get("scene"), errors, warnings)
    _validate_cameras(raw.get("cameras"), raw.get("robot"), errors, warnings)
    _validate_robot(raw.get("robot"), errors, warnings)
    _validate_physics(raw.get("physics"), raw.get("recording"), errors, warnings)
    _validate_task(raw.get("task"), raw.get("robot"), errors, warnings)
    _validate_faults(raw.get("faults"), errors, warnings)
    _validate_perception(raw.get("perception"), errors, warnings)
    _validate_relations(raw.get("relations_gt"), errors, warnings)
    _validate_metrics(raw.get("metrics"), errors, warnings)
    _validate_recording(raw.get("recording"), raw.get("physics"), errors, warnings)

    return LintResult(
        spec_id=spec_id,
        sha256=hash_spec(raw),
        errors=errors,
        warnings=warnings,
    )


def _validate_seed_policy(policy: Any, errors: list[str], warnings: list[str]) -> None:
    if policy is None:
        return
    if not isinstance(policy, Mapping):
        _fail(errors, "seed_policy", "must be a mapping")
        return
    _check_keys(errors, policy, "seed_policy", ("episode_seeds", "pose_sampling"))
    seeds = policy.get("episode_seeds")
    if seeds is not None:
        if not isinstance(seeds, list) or not seeds:
            _fail(errors, "seed_policy.episode_seeds", "must be a non-empty list")
        elif not all(isinstance(s, int) and s >= 0 for s in seeds):
            _fail(errors, "seed_policy.episode_seeds", "must be non-negative integers")
        elif len(set(seeds)) != len(seeds):
            _fail(errors, "seed_policy.episode_seeds", "must not contain duplicates")
    sampling = policy.get("pose_sampling")
    if sampling is not None and sampling not in POSE_SAMPLING:
        _fail(errors, "seed_policy.pose_sampling", f"must be one of {POSE_SAMPLING}")


def _validate_assets(assets: Any, errors: list[str], warnings: list[str]) -> None:
    if not isinstance(assets, Mapping):
        _fail(errors, "assets", "must be a mapping with an instances list")
        return
    _check_keys(errors, assets, "assets", ("instances", "catalog_exclude"))
    instances = assets.get("instances")
    if not isinstance(instances, list) or not instances:
        _fail(errors, "assets.instances", "must be a non-empty list")
        return

    seen: set[str] = set()
    twin_members: dict[str, list[str]] = {}
    for i, inst in enumerate(instances):
        path = f"assets.instances[{i}]"
        if not isinstance(inst, Mapping):
            _fail(errors, path, "must be a mapping")
            continue
        _check_keys(errors, inst, path, ASSET_KEYS)
        instance_id = inst.get("id")
        if not isinstance(instance_id, str) or not ID_RE.match(instance_id):
            _fail(errors, f"{path}.id", "must match ^[a-z][a-z0-9_]{2,63}$")
        elif instance_id in seen:
            _fail(errors, f"{path}.id", f"duplicate instance id {instance_id!r}")
        else:
            seen.add(instance_id)

        if not isinstance(inst.get("class"), str) or not inst.get("class"):
            _fail(errors, f"{path}.class", "must be a non-empty string")
        if not isinstance(inst.get("usd"), str) or not inst.get("usd"):
            _fail(errors, f"{path}.usd", "must be a non-empty string")

        if "mass_kg" in inst and not (_is_number(inst["mass_kg"]) and inst["mass_kg"] > 0):
            _fail(errors, f"{path}.mass_kg", "must be a positive number")
        if "restitution" in inst and not _is_number(inst["restitution"]):
            _fail(errors, f"{path}.restitution", "must be a number")
        if "friction" in inst:
            friction = inst["friction"]
            if not isinstance(friction, Mapping):
                _fail(errors, f"{path}.friction", "must be a mapping")
            else:
                _check_keys(errors, friction, f"{path}.friction", ("static", "dynamic"))
                for key in ("static", "dynamic"):
                    if key in friction and not (
                        _is_number(friction[key]) and friction[key] >= 0
                    ):
                        _fail(errors, f"{path}.friction.{key}", "must be non-negative")
        if "size_m" in inst and not _is_vec(inst["size_m"], 3, positive=True):
            _fail(errors, f"{path}.size_m", "must be 3 positive numbers")
        if "pose" in inst and not _is_vec(inst["pose"], 7):
            _fail(errors, f"{path}.pose", "must be 7 numbers (xyz + quat wxyz)")
        if "split" in inst and inst["split"] not in SPLITS:
            _fail(errors, f"{path}.split", f"must be one of {SPLITS}")
        if "twin_group" in inst:
            if not isinstance(inst["twin_group"], str) or not inst["twin_group"]:
                _fail(errors, f"{path}.twin_group", "must be a non-empty string")
            elif isinstance(instance_id, str):
                twin_members.setdefault(inst["twin_group"], []).append(instance_id)

    for group, members in twin_members.items():
        if len(members) < 2:
            _warn(warnings, f"assets.instances[{group}]", "twin_group has fewer than 2 members")
        splits = {inst.get("split") for inst in instances if inst.get("twin_group") == group}
        if len(splits) > 1:
            _warn(warnings, f"assets[{group}]", "twin group spans multiple splits")

    exclude = assets.get("catalog_exclude")
    if exclude is not None and not (
        isinstance(exclude, list) and all(isinstance(x, str) for x in exclude)
    ):
        _fail(errors, "assets.catalog_exclude", "must be a list of glob strings")


def _validate_scene(scene: Any, errors: list[str], warnings: list[str]) -> None:
    if not isinstance(scene, Mapping):
        _fail(errors, "scene", "must be a mapping")
        return
    _check_keys(
        errors,
        scene,
        "scene",
        ("table", "workspace", "bins", "lighting", "occluders", "props"),
    )

    table = scene.get("table")
    if not isinstance(table, Mapping):
        _fail(errors, "scene.table", "must be a mapping")
    else:
        _check_keys(errors, table, "scene.table", ("size", "height_m", "friction"))
        if not _is_vec(table.get("size"), 3, positive=True):
            _fail(errors, "scene.table.size", "must be 3 positive numbers")
        height = table.get("height_m")
        if height is not None and not (_is_number(height) and height >= 0):
            _fail(errors, "scene.table.height_m", "must be a non-negative number")

    workspace = scene.get("workspace")
    if workspace is not None:
        if not isinstance(workspace, Mapping):
            _fail(errors, "scene.workspace", "must be a mapping")
        else:
            _check_keys(errors, workspace, "scene.workspace", ("xy", "table_z"))
            xy = workspace.get("xy")
            if xy is not None and not (
                isinstance(xy, list)
                and len(xy) == 2
                and all(_is_vec(point, 2) for point in xy)
            ):
                _fail(errors, "scene.workspace.xy", "must be [[x0,y0],[x1,y1]]")

    bins = scene.get("bins")
    if bins is not None:
        if not isinstance(bins, list):
            _fail(errors, "scene.bins", "must be a list")
        else:
            for i, b in enumerate(bins):
                if not isinstance(b, Mapping) or not b.get("name"):
                    _fail(errors, f"scene.bins[{i}]", "must be a mapping with name")

    occluders = scene.get("occluders")
    if occluders is not None:
        if not isinstance(occluders, list):
            _fail(errors, "scene.occluders", "must be a list")
        else:
            for i, occ in enumerate(occluders):
                path = f"scene.occluders[{i}]"
                if not isinstance(occ, Mapping):
                    _fail(errors, path, "must be a mapping")
                    continue
                _check_keys(
                    errors,
                    occ,
                    path,
                    ("name", "size", "pose", "motion", "material", "notes"),
                )
                if not isinstance(occ.get("name"), str) or not occ.get("name"):
                    _fail(errors, f"{path}.name", "must be a non-empty string")
                if not _is_vec(occ.get("size"), 3, positive=True):
                    _fail(errors, f"{path}.size", "must be 3 positive numbers")
                if "pose" in occ and not _is_vec(occ["pose"], 7):
                    _fail(errors, f"{path}.pose", "must be 7 numbers (xyz + quat wxyz)")
                motion = occ.get("motion")
                if motion is not None:
                    if not isinstance(motion, list) or len(motion) < 2:
                        _fail(errors, f"{path}.motion", "must have at least 2 keyframes")
                    else:
                        for j, key in enumerate(motion):
                            kp = f"{path}.motion[{j}]"
                            if not isinstance(key, Mapping):
                                _fail(errors, kp, "must be a mapping")
                            elif not _is_vec(key.get("pos"), 3) or not (
                                _is_number(key.get("t_s")) and key["t_s"] >= 0
                            ):
                                _fail(errors, kp, "requires t_s >= 0 and pos [x,y,z]")

    lighting = scene.get("lighting")
    if lighting is not None and not isinstance(lighting, Mapping):
        _fail(errors, "scene.lighting", "must be a mapping")


def _validate_cameras(
    cameras: Any, robot: Any, errors: list[str], warnings: list[str]
) -> None:
    if not isinstance(cameras, list) or not cameras:
        _fail(errors, "cameras", "must be a non-empty list")
        return
    names: set[str] = set()
    for i, cam in enumerate(cameras):
        path = f"cameras[{i}]"
        if not isinstance(cam, Mapping):
            _fail(errors, path, "must be a mapping")
            continue
        _check_keys(
            errors,
            cam,
            path,
            ("name", "mount", "pose", "fov_deg", "width", "height", "rate_hz", "motion"),
        )
        name = cam.get("name")
        if not isinstance(name, str) or not name:
            _fail(errors, f"{path}.name", "must be a non-empty string")
        elif name in names:
            _fail(errors, f"{path}.name", f"duplicate camera name {name!r}")
        else:
            names.add(name)

        for key in ("width", "height"):
            if not (isinstance(cam.get(key), int) and cam[key] > 0):
                _fail(errors, f"{path}.{key}", "must be a positive integer")
        if not (_is_number(cam.get("rate_hz")) and cam["rate_hz"] > 0):
            _fail(errors, f"{path}.rate_hz", "must be a positive number")
        if "fov_deg" in cam and not (
            _is_number(cam["fov_deg"]) and 0 < cam["fov_deg"] < 180
        ):
            _fail(errors, f"{path}.fov_deg", "must be in (0, 180)")
        if "pose" in cam and not _is_vec(cam["pose"], 7):
            _fail(errors, f"{path}.pose", "must be 7 numbers (xyz + quat wxyz)")
        if "mount" in cam:
            if not isinstance(cam["mount"], str) or not cam["mount"]:
                _fail(errors, f"{path}.mount", "must be a non-empty string")
            elif not robot:
                _fail(errors, f"{path}.mount", "camera mount requires a robot section")
        if "pose" not in cam and "mount" not in cam:
            _fail(errors, path, "camera requires pose or mount")
        motion = cam.get("motion")
        if motion is not None:
            if not isinstance(motion, list) or len(motion) < 2:
                _fail(errors, f"{path}.motion", "must have at least 2 keyframes")
            else:
                for j, key in enumerate(motion):
                    kp = f"{path}.motion[{j}]"
                    if not isinstance(key, Mapping):
                        _fail(errors, kp, "must be a mapping")
                        continue
                    if not (
                        _is_number(key.get("t_s")) and float(key["t_s"]) >= 0
                    ):
                        _fail(errors, f"{kp}.t_s", "must be >= 0")
                    if not _is_vec(key.get("pos"), 3):
                        _fail(errors, f"{kp}.pos", "must be [x, y, z]")
                    if "ori" in key and not _is_vec(key["ori"], 4):
                        _fail(errors, f"{kp}.ori", "must be [w, x, y, z]")


def _validate_robot(robot: Any, errors: list[str], warnings: list[str]) -> None:
    if robot is None:
        return
    if not isinstance(robot, Mapping):
        _fail(errors, "robot", "must be a mapping")
        return
    _check_keys(
        errors,
        robot,
        "robot",
        (
            "model",
            "base_mode",
            "locomotion_policy",
            "arms_controller",
            "end_effector",
            "skill_library",
            "workspace_limits_m",
            "usd",
            "prim_path",
            "variants",
            "pose",
            "visible",
        ),
    )
    if not isinstance(robot.get("model"), str) or not robot.get("model"):
        _fail(errors, "robot.model", "must be a non-empty string")
    if "usd" in robot and not (
        isinstance(robot["usd"], str) and robot["usd"]
    ):
        _fail(errors, "robot.usd", "must be a non-empty path")
    if "prim_path" in robot and not (
        isinstance(robot["prim_path"], str) and robot["prim_path"].startswith("/")
    ):
        _fail(errors, "robot.prim_path", "must be an absolute USD path")
    if "variants" in robot and not (
        isinstance(robot["variants"], Mapping)
        and all(isinstance(k, str) and isinstance(v, str) for k, v in robot["variants"].items())
    ):
        _fail(errors, "robot.variants", "must map variant-set names to selections")
    if "pose" in robot and not _is_vec(robot["pose"], 7):
        _fail(errors, "robot.pose", "must be 7 numbers (xyz + quat wxyz)")
    if "visible" in robot and not isinstance(robot["visible"], bool):
        _fail(errors, "robot.visible", "must be a boolean")
    base_mode = robot.get("base_mode")
    if base_mode not in BASE_MODES:
        _fail(errors, "robot.base_mode", f"must be one of {BASE_MODES}")
    if base_mode == "base_mobile" and not robot.get("locomotion_policy"):
        _fail(errors, "robot.locomotion_policy", "required when base_mode=base_mobile")
    limits = robot.get("workspace_limits_m")
    if limits is not None and not (
        isinstance(limits, list)
        and len(limits) == 2
        and all(_is_vec(point, 3) for point in limits)
    ):
        _fail(errors, "robot.workspace_limits_m", "must be [min_xyz, max_xyz]")
    for key in ("arms_controller", "end_effector"):
        if key in robot and not isinstance(robot[key], Mapping):
            _fail(errors, f"robot.{key}", "must be a mapping")


def _validate_physics(physics: Any, recording: Any, errors: list[str], warnings: list[str]) -> None:
    if not isinstance(physics, Mapping):
        _fail(errors, "physics", "must be a mapping")
        return
    _check_keys(
        errors,
        physics,
        "physics",
        ("dt", "substeps", "solver", "contact_report", "determinism"),
    )
    dt = physics.get("dt")
    if not (_is_number(dt) and 0 < dt <= 1):
        _fail(errors, "physics.dt", "must be in (0, 1] seconds")
    substeps = physics.get("substeps", 1)
    if not (isinstance(substeps, int) and substeps >= 1):
        _fail(errors, "physics.substeps", "must be an integer >= 1")
    if physics.get("solver") not in SOLVERS:
        _fail(errors, "physics.solver", f"must be one of {SOLVERS}")

    if isinstance(recording, Mapping) and _is_number(dt) and dt > 0:
        gt_rate = recording.get("gt_rate_hz")
        if _is_number(gt_rate) and gt_rate > (1.0 / dt) + 1e-9:
            _warn(
                warnings,
                "recording.gt_rate_hz",
                f"gt rate {gt_rate} Hz exceeds physics rate {1.0 / dt:.2f} Hz",
            )


def _validate_task(task: Any, robot: Any, errors: list[str], warnings: list[str]) -> None:
    if task is None:
        return
    if not isinstance(task, Mapping):
        _fail(errors, "task", "must be a mapping")
        return
    _check_keys(
        errors,
        task,
        "task",
        ("id", "initial_facts", "goal_facts", "success_check", "time_limit_s", "session"),
    )
    if not isinstance(task.get("id"), str) or not task.get("id"):
        _fail(errors, "task.id", "must be a non-empty string")
    limit = task.get("time_limit_s")
    if limit is not None and not (_is_number(limit) and limit > 0):
        _fail(errors, "task.time_limit_s", "must be a positive number")
    for key in ("initial_facts", "goal_facts"):
        facts = task.get(key)
        if facts is not None and not (
            isinstance(facts, list) and all(isinstance(f, str) and f for f in facts)
        ):
            _fail(errors, f"task.{key}", "must be a list of fact strings")
    if task.get("goal_facts") and not robot:
        _warn(warnings, "task.goal_facts", "goal facts present but no robot section (W1 tracking)")
    session = task.get("session")
    if session is not None:
        if not isinstance(session, Mapping):
            _fail(errors, "task.session", "must be a mapping")
        else:
            _check_keys(
                errors,
                session,
                "task.session",
                ("sub_episodes", "sub_episode_s", "relocations"),
            )
            count = session.get("sub_episodes")
            if not (isinstance(count, int) and count >= 1):
                _fail(errors, "task.session.sub_episodes", "must be an integer >= 1")
            duration = session.get("sub_episode_s")
            if not (_is_number(duration) and duration > 0):
                _fail(errors, "task.session.sub_episode_s", "must be > 0")
            relocations = session.get("relocations", [])
            if not isinstance(relocations, list):
                _fail(errors, "task.session.relocations", "must be a list")
            else:
                for i, reloc in enumerate(relocations):
                    path = f"task.session.relocations[{i}]"
                    if not isinstance(reloc, Mapping):
                        _fail(errors, path, "must be a mapping")
                        continue
                    _check_keys(errors, reloc, path, ("after", "target", "pose"))
                    if not (isinstance(reloc.get("after"), int) and reloc["after"] >= 0):
                        _fail(errors, f"{path}.after", "must be an integer >= 0")
                    if not isinstance(reloc.get("target"), str) or not reloc.get("target"):
                        _fail(errors, f"{path}.target", "must be a non-empty string")
                    if not _is_vec(reloc.get("pose"), 7):
                        _fail(errors, f"{path}.pose", "must be 7 numbers (xyz + quat wxyz)")


def _validate_faults(faults: Any, errors: list[str], warnings: list[str]) -> None:
    if faults is None:
        return
    if not isinstance(faults, list):
        _fail(errors, "faults", "must be a list")
        return
    for i, fault in enumerate(faults):
        path = f"faults[{i}]"
        if not isinstance(fault, Mapping):
            _fail(errors, path, "must be a mapping")
            continue
        _check_keys(errors, fault, path, ("at_step", "kind", "params"))
        if not (isinstance(fault.get("at_step"), int) and fault["at_step"] >= 0):
            _fail(errors, f"{path}.at_step", "must be an integer >= 0")
        if fault.get("kind") not in FAULT_KINDS:
            _fail(errors, f"{path}.kind", f"must be one of {FAULT_KINDS}")
        if "params" in fault and not isinstance(fault["params"], Mapping):
            _fail(errors, f"{path}.params", "must be a mapping")


def _validate_perception(perception: Any, errors: list[str], warnings: list[str]) -> None:
    if perception is None:
        return
    if not isinstance(perception, Mapping):
        _fail(errors, "perception", "must be a mapping")
        return
    _check_keys(
        errors,
        perception,
        "perception",
        ("mode", "detector_tier", "detector", "in_loop_detector", "noise_profile"),
    )
    if perception.get("mode") not in PERCEPTION_MODES:
        _fail(errors, "perception.mode", f"must be one of {PERCEPTION_MODES}")
    tier = perception.get("detector_tier")
    if tier not in DETECTOR_TIERS:
        _fail(errors, "perception.detector_tier", f"must be one of {DETECTOR_TIERS}")
    detector = perception.get("detector")
    if tier in ("frozen_real", "frozen_real_fast", "noisy"):
        if not isinstance(detector, Mapping) or not detector.get("name"):
            _fail(errors, "perception.detector", "required for non-oracle tiers")
    if tier == "noisy" and not isinstance(perception.get("noise_profile"), Mapping):
        _fail(errors, "perception.noise_profile", "required when detector_tier=noisy")
    if perception.get("mode") == "in_loop" and not isinstance(
        perception.get("in_loop_detector"), Mapping
    ):
        _warn(warnings, "perception.in_loop_detector", "in_loop mode without a fast detector")


def _validate_relations(relations: Any, errors: list[str], warnings: list[str]) -> None:
    if relations is None:
        return
    if not isinstance(relations, Mapping):
        _fail(errors, "relations_gt", "must be a mapping")
        return
    _check_keys(
        errors,
        relations,
        "relations_gt",
        ("predicates", "compute_from", "tolerance_m", "ood_combinations"),
    )
    predicates = relations.get("predicates")
    if predicates is not None:
        if not isinstance(predicates, list) or not predicates:
            _fail(errors, "relations_gt.predicates", "must be a non-empty list")
        else:
            unknown = [p for p in predicates if p not in PREDICATES]
            if unknown:
                _fail(errors, "relations_gt.predicates", f"unknown predicates {unknown}")
    if relations.get("compute_from") not in COMPUTE_FROM:
        _fail(errors, "relations_gt.compute_from", f"must be one of {COMPUTE_FROM}")
    tol = relations.get("tolerance_m")
    if tol is not None and not (_is_number(tol) and tol >= 0):
        _fail(errors, "relations_gt.tolerance_m", "must be >= 0")
    ood = relations.get("ood_combinations")
    if ood is not None:
        if not (
            isinstance(ood, list)
            and all(
                isinstance(item, list)
                and len(item) == 2
                and all(isinstance(p, str) and p in PREDICATES for p in item)
                for item in ood
            )
        ):
            _fail(errors, "relations_gt.ood_combinations", "must be pairs of known predicates")


def _validate_metrics(metrics: Any, errors: list[str], warnings: list[str]) -> None:
    if metrics is None:
        return
    if not isinstance(metrics, Mapping):
        _fail(errors, "metrics", "must be a mapping")
        return
    _check_keys(errors, metrics, "metrics", ("primary", "leading", "guardrail"))
    for key in ("primary", "leading", "guardrail"):
        value = metrics.get(key)
        if value is not None and not (
            isinstance(value, list) and all(isinstance(v, str) and v for v in value)
        ):
            _fail(errors, f"metrics.{key}", "must be a list of metric names")


def _validate_recording(recording: Any, physics: Any, errors: list[str], warnings: list[str]) -> None:
    if not isinstance(recording, Mapping):
        _fail(errors, "recording", "must be a mapping")
        return
    _check_keys(
        errors,
        recording,
        "recording",
        ("obs_rate_hz", "gt_rate_hz", "bundle_format", "store_images"),
    )
    for key in ("obs_rate_hz", "gt_rate_hz"):
        value = recording.get(key)
        if not (_is_number(value) and value > 0):
            _fail(errors, f"recording.{key}", "must be a positive number")
    if recording.get("bundle_format") not in BUNDLE_FORMATS:
        _fail(errors, "recording.bundle_format", f"must be one of {BUNDLE_FORMATS}")
    store = recording.get("store_images")
    if store not in (True, False, "per_suite"):
        _fail(errors, "recording.store_images", "must be true, false or 'per_suite'")
    if isinstance(physics, Mapping):
        dt = physics.get("dt")
        obs_rate = recording.get("obs_rate_hz")
        if _is_number(dt) and dt > 0 and _is_number(obs_rate):
            ratio = (1.0 / dt) / obs_rate
            if abs(ratio - round(ratio)) > 1e-6:
                _warn(
                    warnings,
                    "recording.obs_rate_hz",
                    f"physics rate {1.0 / dt:.2f} Hz is not an integer multiple of obs rate",
                )


# ---------------------------------------------------------------------------
# JSON Schema export (A1)


def json_schema() -> dict[str, Any]:
    """Hand-maintained JSON Schema for the parts that are stable in W1.

    The validator above is the source of truth for semantic rules; this schema
    documents the shape for external tools and for future editor support.
    """
    vec = lambda n: {"type": "array", "minItems": n, "maxItems": n, "items": {"type": "number"}}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://pss.local/schemas/world_spec_v1.json",
        "title": "PSS World Spec v1",
        "type": "object",
        "required": [
            "spec_version",
            "id",
            "task_family",
            "split",
            "assets",
            "scene",
            "cameras",
            "physics",
            "recording",
        ],
        "additionalProperties": True,
        "properties": {
            "spec_version": {"const": SPEC_VERSION},
            "id": {"type": "string", "pattern": ID_RE.pattern},
            "task_family": {"enum": list(TASK_FAMILIES)},
            "split": {"enum": list(SPLITS)},
            "notes": {"type": ["string", "array", "null"]},
            "seed_policy": {
                "type": "object",
                "properties": {
                    "episode_seeds": {"type": "array", "items": {"type": "integer", "minimum": 0}},
                    "pose_sampling": {"enum": list(POSE_SAMPLING)},
                },
                "additionalProperties": False,
            },
            "assets": {
                "type": "object",
                "required": ["instances"],
                "properties": {
                    "instances": {"type": "array", "minItems": 1, "items": {"$ref": "#/$defs/asset"}},
                    "catalog_exclude": {"type": "array", "items": {"type": "string"}},
                },
                "additionalProperties": False,
            },
            "scene": {
                "type": "object",
                "required": ["table"],
                "properties": {
                    "table": {
                        "type": "object",
                        "required": ["size", "height_m"],
                        "properties": {"size": vec(3), "height_m": {"type": "number", "minimum": 0}},
                        "additionalProperties": False,
                    },
                    "workspace": {"type": "object"},
                    "bins": {"type": "array"},
                    "lighting": {"type": "object"},
                    "occluders": {"type": "array", "items": {"$ref": "#/$defs/occluder"}},
                },
                "additionalProperties": False,
            },
            "cameras": {
                "type": "array",
                "minItems": 1,
                "items": {"$ref": "#/$defs/camera"},
            },
            "robot": {"type": "object"},
            "physics": {
                "type": "object",
                "required": ["dt", "solver"],
                "properties": {
                    "dt": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
                    "substeps": {"type": "integer", "minimum": 1},
                    "solver": {"enum": list(SOLVERS)},
                    "contact_report": {"type": "object"},
                    "determinism": {"type": "object"},
                },
                "additionalProperties": False,
            },
            "task": {"type": "object"},
            "faults": {"type": "array", "items": {"$ref": "#/$defs/fault"}},
            "perception": {"type": "object"},
            "relations_gt": {"type": "object"},
            "metrics": {"type": "object"},
            "recording": {
                "type": "object",
                "required": ["obs_rate_hz", "gt_rate_hz", "bundle_format", "store_images"],
                "properties": {
                    "obs_rate_hz": {"type": "number", "exclusiveMinimum": 0},
                    "gt_rate_hz": {"type": "number", "exclusiveMinimum": 0},
                    "bundle_format": {"enum": list(BUNDLE_FORMATS)},
                    "store_images": {"enum": [True, False, "per_suite"]},
                },
                "additionalProperties": False,
            },
        },
        "$defs": {
            "asset": {
                "type": "object",
                "required": ["id", "class", "usd"],
                "properties": {
                    "id": {"type": "string", "pattern": ID_RE.pattern},
                    "class": {"type": "string", "minLength": 1},
                    "usd": {"type": "string", "minLength": 1},
                    "mass_kg": {"type": "number", "exclusiveMinimum": 0},
                    "friction": {"type": "object"},
                    "restitution": {"type": "number"},
                    "twin_group": {"type": "string", "minLength": 1},
                    "split": {"enum": list(SPLITS)},
                    "size_m": vec(3),
                    "pose": vec(7),
                    "color": vec(3),
                },
                "additionalProperties": False,
            },
            "camera": {
                "type": "object",
                "required": ["name", "fov_deg", "width", "height", "rate_hz"],
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "mount": {"type": "string", "minLength": 1},
                    "pose": vec(7),
                    "fov_deg": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 180},
                    "width": {"type": "integer", "minimum": 1},
                    "height": {"type": "integer", "minimum": 1},
                    "rate_hz": {"type": "number", "exclusiveMinimum": 0},
                    "motion": {
                        "type": "array",
                        "minItems": 2,
                        "items": {
                            "type": "object",
                            "required": ["t_s", "pos"],
                            "properties": {
                                "t_s": {"type": "number", "minimum": 0},
                                "pos": vec(3),
                                "ori": vec(4),
                            },
                        },
                    },
                },
                "additionalProperties": False,
            },
            "occluder": {
                "type": "object",
                "required": ["name", "size"],
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "size": vec(3),
                    "pose": vec(7),
                    "motion": {
                        "type": "array",
                        "minItems": 2,
                        "items": {
                            "type": "object",
                            "required": ["t_s", "pos"],
                            "properties": {
                                "t_s": {"type": "number", "minimum": 0},
                                "pos": vec(3),
                            },
                        },
                    },
                },
                "additionalProperties": False,
            },
            "fault": {
                "type": "object",
                "required": ["at_step", "kind"],
                "properties": {
                    "at_step": {"type": "integer", "minimum": 0},
                    "kind": {"enum": list(FAULT_KINDS)},
                    "params": {"type": "object"},
                },
                "additionalProperties": False,
            },
        },
    }
