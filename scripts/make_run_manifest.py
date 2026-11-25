#!/usr/bin/env python
"""Write the per-experiment run manifest (P5).

Covers L0/L1/interactions/twins/ANOVA/L2 and records git commit, environment,
asset hashes, output hashes, seeds and the fresh Isaac bundle/spec hash.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pss.property.manifest import capture_environment, sha256_file, write_manifest  # noqa: E402

SCORECARDS = Path("scorecards/property")


def _hashes(paths) -> dict:
    out = {}
    for path in paths:
        path = Path(path)
        if path.exists():
            out[path.name] = {"sha256": sha256_file(path), "size": path.stat().st_size}
    return out


def _bundle_info() -> dict | None:
    manifests = sorted(Path("bundles/property_l2_s4").rglob("manifest.json"))
    if not manifests:
        return None
    path = manifests[0]
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "bundle_dir": str(path.parent),
        "manifest": str(path),
        "manifest_sha256": sha256_file(path),
        "spec_id": payload.get("spec", {}).get("id"),
        "spec_hash": payload.get("spec", {}).get("hash"),
        "pss_commit": payload.get("stack", {}).get("pss_commit"),
        "seed": payload.get("seed"),
        "detector_tier": payload.get("detector_tier"),
    }


def main() -> None:
    expert_manifest = SCORECARDS / "expert_manifest.json"
    payload = {
        "environment": capture_environment(ROOT),
        "expert_manifest": {
            "path": str(expert_manifest),
            "sha256": sha256_file(expert_manifest) if expert_manifest.exists() else None,
        },
        "runs": [
            {
                "name": "L0 property suite",
                "script": "scripts/run_property_l0.py",
                "seeds": [0, 1, 2, 3, 4],
                "inputs": ["pss/observations/synthetic.py", "pss/experts/stubs.py"],
                "outputs": _hashes(sorted(SCORECARDS.glob("l0_seed*.json"))),
            },
            {
                "name": "L1 real-expert suite",
                "script": "scripts/run_property_l1.py",
                "seeds": [0, 1, 2],
                "inputs": [
                    "checkpoints/expert_v1",
                    "checkpoints/projection_v1",
                    "checkpoints/prototypes_v1",
                    "checkpoints/category_pretrain_experts",
                ],
                "outputs": _hashes(sorted(SCORECARDS.glob("l1_seed*.json"))),
            },
            {
                "name": "interactions + negative controls",
                "script": "scripts/run_property_interactions.py",
                "seeds": [0, 1, 2, 3, 4],
                "outputs": _hashes([SCORECARDS / "interactions.json"]),
            },
            {
                "name": "twin pairing",
                "script": "scripts/run_property_twins.py",
                "seeds": [0, 1, 2, 3, 4],
                "outputs": _hashes([SCORECARDS / "twins.json"]),
            },
            {
                "name": "two-way ANOVA",
                "script": "scripts/run_property_anova.py",
                "inputs": sorted(str(p) for p in SCORECARDS.glob("l1_seed*.json")),
                "outputs": _hashes([SCORECARDS / "anova.json"]),
            },
            {
                "name": "L2 bundle traces",
                "script": "scripts/run_property_l2.py",
                "inputs": ["bundles/dev/**/pss_trace.parquet"],
                "outputs": _hashes([SCORECARDS / "l2_bundles.json"]),
            },
            {
                "name": "L2 fresh Isaac S4-PSS",
                "script": "scripts/run_s4_pss.py",
                "env": "D:/isaac/env/Scripts/python.exe (Python 3.11)",
                "seeds": [0, 1, 2, 3, 4],
                "outputs": _hashes(sorted(SCORECARDS.glob("l2_s4_pss_seed*.json"))),
            },
            {
                "name": "§15 report",
                "script": "scripts/run_property_section15.py",
                "outputs": _hashes(
                    [
                        Path("reports/property_validation/SECTION15_RESULTS.md"),
                        Path("reports/property_validation/figures/fig1_invariance.png"),
                        Path("reports/property_validation/figures/fig2_kb_state.png"),
                        Path("reports/property_validation/figures/fig3_budget_quality.png"),
                    ]
                ),
            },
        ],
        "isaac_bundle": _bundle_info(),
    }
    path = write_manifest(SCORECARDS / "run_manifest.json", payload)
    print(f"wrote {path}")
    print(f"runs: {len(payload['runs'])}")


if __name__ == "__main__":
    main()
