#!/usr/bin/env python
"""Write the expert/checkpoint manifest and scorecard hashes (P2/P5).

Usage:
    python scripts/make_asset_manifest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pss.property.manifest import (  # noqa: E402
    build_asset_manifest,
    capture_environment,
    sha256_file,
    write_manifest,
)

ASSETS = {
    "expert_v1.vision": "checkpoints/expert_v1/vision.pt",
    "expert_v1.tactile": "checkpoints/expert_v1/tactile.pt",
    "expert_v1.metrics": "checkpoints/expert_v1/metrics.json",
    "category_pretrain.vision": "checkpoints/category_pretrain_experts/vision.pt",
    "category_pretrain.tactile": "checkpoints/category_pretrain_experts/tactile.pt",
    "category_pretrain.metrics": "checkpoints/category_pretrain_experts/metrics.json",
    "projection_v1.vision": "checkpoints/projection_v1/vision.pt",
    "projection_v1.tactile": "checkpoints/projection_v1/tactile.pt",
    "projection_v1.geometry": "checkpoints/projection_v1/geometry.pt",
    "projection_v1.metrics": "checkpoints/projection_v1/metrics.json",
    "prototypes_v1.bank": "checkpoints/prototypes_v1/prototypes.npz",
    "prototypes_v1.manifest": "checkpoints/prototypes_v1/manifest.json",
    "prototypes_v1.metrics": "checkpoints/prototypes_v1/metrics.json",
    "expert_pretrained_v1.vision": "checkpoints/expert_pretrained_v1/vision_pretrained.pt",
    "expert_resnet_smoke.vision": "checkpoints/expert_resnet_smoke/vision_resnet.pt",
    "tactile_wtmm": "checkpoints/tactile_wtmm/tactile_wtmm.pt",
    "data.expert_v1.index": "data/expert_v1/index.json",
    "data.category_pretrain.index": "data/category_pretrain/index.json",
}


def main() -> None:
    scorecards = [
        p
        for p in sorted(Path("scorecards/property").glob("*.json"))
        if p.name not in ("expert_manifest.json", "run_manifest.json")
    ]
    payload = {
        "environment": capture_environment(ROOT),
        "assets": build_asset_manifest(ASSETS),
        "scorecards": {
            p.name: {"sha256": sha256_file(p), "size": p.stat().st_size}
            for p in scorecards
        },
    }
    path = write_manifest("scorecards/property/expert_manifest.json", payload)
    missing = [name for name, row in payload["assets"].items() if not row.get("exists")]
    print(f"wrote {path}")
    print(f"assets: {len(payload['assets'])}  missing: {missing or 'none'}")
    print(f"scorecards: {len(payload['scorecards'])}")


if __name__ == "__main__":
    main()
