"""S4 demo: 3D scene model -> PSS runtime (explicit, opt-in).

Loads CLEVR scenes, solves 3D positions with ``solve_scene``, and writes each
scene into a fresh ``PSSRuntime`` through ``pss.scene.pss_bridge``.  Prints an
entity / relation / audit summary and validates the bridge invariants:

  * entity provenance is ``source=scene_model`` with the scene id;
  * entity covariance is a finite 3x3 matrix;
  * relations have ``origin=scene_model`` / ``relation_type=scene_geometry``;
  * an audit entry ``scene_ingest`` exists for the scene;
  * the 1D axis fallback still works and the runtime axis mode is still "off".

Example
-------
python -u -m pss.eval.scripts.run_scene_runtime
python -u -m pss.eval.scripts.run_scene_runtime --scenes 5 --n-views 3
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from pss._console import ensure_utf8
from pss.reasoning.relations.bayes_predicate import PREDICATES, AxisPredicate
from pss.runtime import PSSRuntime
from pss.scene.bench_api import BenchConfig, build_examples
from pss.scene.clevr import load_scenes
from pss.scene.metrics import pose_errors
from pss.scene.optimizer import solve_scene
from pss.scene.pss_bridge import apply_scene_to_runtime


def _validate_scene(runtime: PSSRuntime, scene_id: str) -> list[str]:
    """Check the invariants the bridge promises for one scene."""

    errors: list[str] = []
    if not runtime.state.entities:
        errors.append(f"{scene_id}: no entities written")
    for entity in runtime.state.entities.values():
        if entity.provenance.get("source") != "scene_model":
            errors.append(f"{entity.id}: provenance.source != scene_model")
        if entity.provenance.get("scene_id") != scene_id:
            errors.append(f"{entity.id}: provenance.scene_id mismatch")
        cov = np.asarray(entity.attributes.get("covariance"), dtype=np.float64)
        if cov.shape != (3, 3):
            errors.append(f"{entity.id}: covariance shape {cov.shape} != (3, 3)")
        elif not np.isfinite(cov).all():
            errors.append(f"{entity.id}: covariance is not finite")
        if not 0.0 <= float(entity.confidence) <= 1.0:
            errors.append(f"{entity.id}: confidence out of [0, 1]")
    if not runtime.state.relations:
        errors.append(f"{scene_id}: no relations written")
    for relation in runtime.state.relations:
        if relation.origin != "scene_model":
            errors.append(
                f"{relation.source}->{relation.target}: origin "
                f"{relation.origin!r} != 'scene_model'"
            )
        if relation.relation_type != "scene_geometry":
            errors.append(
                f"{relation.source}->{relation.target}: relation_type "
                f"{relation.relation_type!r} != 'scene_geometry'"
            )
        if not 0.0 <= float(relation.probability) <= 1.0:
            errors.append(
                f"{relation.source}->{relation.target}: probability out of [0, 1]"
            )
    audit = [
        entry
        for entry in runtime.state.audit
        if entry.action == "scene_ingest" and entry.target == scene_id
    ]
    if len(audit) != 1:
        errors.append(f"{scene_id}: expected 1 scene_ingest audit entry, got {len(audit)}")
    return errors


def _fallback_check() -> dict:
    """The 1D axis predicate must remain usable, and opt-in only."""

    predicate = AxisPredicate(mode="hard", eps=0.02)
    records = predicate.records(
        {
            "a": {"pos": np.array([0.0, 0.0, 0.0]), "radius": 0.05},
            "b": {"pos": np.array([0.5, 0.0, 0.0]), "radius": 0.05},
        }
    )
    runtime = PSSRuntime()
    return {
        "records": len(records),
        "predicates": sorted({record.predicate for record in records}),
        "valid": bool(records)
        and all(record.predicate in PREDICATES for record in records),
        "runtime_axis_mode": runtime.relations.rule_set.axis_mode,
    }


def run(args) -> dict:
    config = BenchConfig(
        n_views=args.n_views,
        sigma_px=args.sigma_px,
        sigma_depth=args.sigma_depth,
        sigma_pose_yaw=args.sigma_pose_yaw,
        sigma_pose_pitch=args.sigma_pose_pitch,
        outlier_frac=args.outlier_frac,
        eps_xy=args.eps_xy,
        eps_z=args.eps_z,
        eps_c=args.eps_c,
        eps_s=args.eps_s,
        table_z=args.table_z,
        seed=args.seed,
    )
    scenes = load_scenes(
        args.clevr_root,
        args.split,
        max_scenes=args.scenes,
        seed=args.seed,
    )
    if not scenes:
        raise RuntimeError(f"no scenes loaded from {args.clevr_root!r} split {args.split!r}")
    examples = build_examples(scenes, config)
    print(
        f"[data] {args.clevr_root} split={args.split} scenes={len(scenes)} "
        f"n_views={config.n_views} eps_xy={config.eps_xy} eps_z={config.eps_z} "
        f"eps_c={config.eps_c}",
        flush=True,
    )

    rows: list[dict] = []
    errors: list[str] = []
    predicate_totals: dict[str, int] = {}
    for index, example in enumerate(examples):
        scene = example.true_scene
        output = solve_scene(example, config)
        # Explicit opt-in: a fresh runtime per scene.  Object ids are
        # scene-local; reusing one runtime across scenes would collide unless
        # ids were namespaced (documented open item).
        runtime = PSSRuntime()
        summary = apply_scene_to_runtime(
            runtime,
            scene,
            output.positions,
            output.covariances,
            eps_xy=config.eps_xy,
            eps_z=config.eps_z,
            eps_c=config.eps_c,
            table_z=config.table_z,
            eps_s=config.eps_s,
            actor="scene_model",
        )
        scene_errors = _validate_scene(runtime, scene.scene_id)
        errors.extend(scene_errors)
        pose = pose_errors(
            output.positions,
            {oid: obj.pos for oid, obj in scene.objects.items()},
        )
        for predicate, count in summary["predicates"].items():
            predicate_totals[predicate] = predicate_totals.get(predicate, 0) + count
        row = {
            "scene_id": scene.scene_id,
            "entities": summary["entities"],
            "relations": summary["relations"],
            "predicates": summary["predicates"],
            "pose_median": pose["median"],
            "audit_version": summary["version"],
            "errors": scene_errors,
        }
        rows.append(row)
        print(
            f"[{index + 1}/{len(examples)}] {scene.scene_id} "
            f"entities={summary['entities']} relations={summary['relations']} "
            f"predicates={summary['predicates']} "
            f"pose_med={pose['median']:.4f} audit_v={summary['version']} "
            f"{'ok' if not scene_errors else 'FAIL'}",
            flush=True,
        )

    fallback = _fallback_check()
    print("\n[summary]", flush=True)
    print(
        f"  scenes={len(rows)} entities={sum(r['entities'] for r in rows)} "
        f"relations={sum(r['relations'] for r in rows)} "
        f"audit_entries={sum(r['audit_version'] for r in rows)}",
        flush=True,
    )
    print(f"  predicates={dict(sorted(predicate_totals.items()))}", flush=True)
    provenance_ok = not any("provenance" in e for e in errors)
    covariance_ok = not any("covariance" in e for e in errors)
    origin_ok = not any("origin" in e or "relation_type" in e for e in errors)
    audit_ok = not any("audit" in e for e in errors)
    print(
        f"  provenance={'ok' if provenance_ok else 'FAIL'} "
        f"covariance={'ok' if covariance_ok else 'FAIL'} "
        f"relation_origin={'ok' if origin_ok else 'FAIL'} "
        f"audit={'ok' if audit_ok else 'FAIL'}",
        flush=True,
    )
    print(
        f"  1D fallback: records={fallback['records']} "
        f"predicates={fallback['predicates']} "
        f"runtime_axis_mode={fallback['runtime_axis_mode']!r} "
        f"{'ok' if fallback['valid'] else 'FAIL'}",
        flush=True,
    )

    if errors:
        for error in errors:
            print(f"  ERROR {error}", flush=True)
        raise SystemExit(1)
    if not fallback["valid"]:
        print("  ERROR 1D axis fallback produced no valid records", flush=True)
        raise SystemExit(1)
    if fallback["runtime_axis_mode"] != "off":
        print("  ERROR scene bridge changed the runtime axis mode default", flush=True)
        raise SystemExit(1)
    return {"scenes": rows, "fallback": fallback, "predicates": predicate_totals}


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--clevr-root",
        default="data/public/clevr_cogent/CLEVR_CoGenT_v1.0",
    )
    ap.add_argument("--split", default="vala")
    ap.add_argument("--scenes", type=int, default=5)
    ap.add_argument("--n-views", type=int, default=3)
    ap.add_argument("--sigma-px", type=float, default=2.0)
    ap.add_argument("--sigma-depth", type=float, default=0.01)
    ap.add_argument("--sigma-pose-yaw", type=float, default=0.0)
    ap.add_argument("--sigma-pose-pitch", type=float, default=0.0)
    ap.add_argument("--outlier-frac", type=float, default=0.0)
    ap.add_argument("--eps-xy", type=float, default=0.12)
    ap.add_argument("--eps-z", type=float, default=0.05)
    ap.add_argument("--eps-c", type=float, default=0.02)
    ap.add_argument("--eps-s", type=float, default=0.05)
    ap.add_argument("--table-z", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
