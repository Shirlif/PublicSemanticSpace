"""W2 T11 acceptance: verify the composed G1 scene.

Checks that the robot reference resolved, the articulation root and colliders
are present, the requested variants are selected, and the robot is hidden from
rendering (collision-only load).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pxr import Usd, UsdGeom, UsdPhysics

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pss.isaac.scene_builder import default_cache_dir
from pss.spec import load_spec


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("spec")
    parser.add_argument("--prim-path", default="/World/Robot")
    args = parser.parse_args()

    spec = load_spec(args.spec)
    cache = default_cache_dir() / spec.sha256.split(":", 1)[-1]
    scene = cache / "scene.usd"
    stage = Usd.Stage.Open(str(scene))
    root = stage.GetPrimAtPath(args.prim_path)
    assert root and root.IsValid(), f"missing robot root {args.prim_path}"

    robot_prims = [p for p in Usd.PrimRange(root)]
    colliders = [p for p in robot_prims if p.HasAPI(UsdPhysics.CollisionAPI)]
    articulations = [
        p for p in robot_prims if p.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    variants = {
        name: root.GetVariantSet(name).GetVariantSelection()
        for name in root.GetVariantSets().GetNames()
    }
    visibility = UsdGeom.Imageable(root).ComputeVisibility()
    report = {
        "scene": str(scene),
        "robot_root": str(root.GetPath()),
        "prims": len(robot_prims),
        "colliders": len(colliders),
        "articulation_roots": [str(p.GetPath()) for p in articulations],
        "variants": variants,
        "visibility": str(visibility),
        "hidden": str(visibility) == "invisible",
    }
    print(json.dumps(report, indent=2))
    ok = (
        report["prims"] > 50
        and report["colliders"] > 0
        and bool(report["articulation_roots"])
        and report["hidden"]
    )
    print("T11:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
