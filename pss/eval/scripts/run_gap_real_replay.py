"""Claim-gap F10 spike: ObjectFolder 2.0 real-data replay.

ObjectFolder 2.0 is the only public multisensory object dataset already present
locally.  This script asks the narrow feasibility question:

* can its rendered vision/tactile features be extracted with the existing
  ``pss.rd2.objectfolder`` helpers;
* can they enter the ``pss_eval`` Stream/property path (PSS runtime adapter +
  a passthrough control);
* what object-level cross-modal binding (R@1) do they actually support?

It does not claim L2 identity/tracking or sim2real.  ObjectFolder has no poses,
depth, segmentation, scenes, or synchronized multi-frame stream.  See
``reports/claim_gaps/real_replay/SPIKE.md`` for the full go/no-go.

Example (reuses the renders already produced in the main checkout)::

    python -m pss.eval.scripts.run_gap_real_replay \
        --renders D:/PublicSemanticSpace/eval/results/objectfolder/renders \
        --objects-csv D:/PublicSemanticSpace/_external/ObjectFolder/objects.csv \
        --limit 20 --out reports/claim_gaps/real_replay
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pss._console import ensure_utf8
from pss.rd2.objectfolder_replay import run_objectfolder_replay

DEFAULT_RENDERS = "D:/PublicSemanticSpace/eval/results/objectfolder/renders"
DEFAULT_OBJECTS_CSV = "D:/PublicSemanticSpace/_external/ObjectFolder/objects.csv"


def _print_summary(results: dict) -> None:
    dataset = results["dataset"]
    binding = results["binding"]
    print(f"objects: {dataset['objects']} (license {dataset['license']})")
    print(
        f"features: {results['features']['dim']}-d DINOv2, "
        f"vision/object={results['features']['vision_images_per_object']}, "
        f"touch/object={results['features']['touch_images_per_object']}"
    )
    for protocol in ("zero_shot", "ridge_half_split"):
        for direction, metric in binding.get(protocol, {}).items():
            print(
                f"  {protocol:16s} {direction:16s} "
                f"R@1={metric['r1']:.3f} R@5={metric['r5']:.3f} "
                f"chance={metric['chance']:.3f}"
            )
    permutation = binding["permutation"]
    print(
        f"  permutation: observed={permutation['observed_r1']:.3f} "
        f"null_mean={permutation['null_mean']:.3f} p={permutation['p_value']:.4f}"
    )
    print("pss_eval:")
    for system, summary in results["pss_eval"]["summary"]["systems"].items():
        statuses = ", ".join(
            f"{prop}={status}" for prop, status in summary["statuses"].items()
        )
        print(f"  {system}: {statuses}")


def main() -> None:
    ensure_utf8()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renders", default=DEFAULT_RENDERS)
    parser.add_argument("--objects-csv", default=DEFAULT_OBJECTS_CSV)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--device", default=None, help="auto / cpu / cuda")
    parser.add_argument("--public-dim", type=int, default=128)
    parser.add_argument("--budget-ms", type=float, default=50.0)
    parser.add_argument(
        "--cache",
        default=None,
        help="optional .npz feature cache path (reused across runs)",
    )
    parser.add_argument("--out", default="reports/claim_gaps/real_replay")
    args = parser.parse_args()

    results = run_objectfolder_replay(
        renders_root=args.renders,
        out_dir=args.out,
        objects_csv=args.objects_csv,
        limit=args.limit,
        seed=args.seed,
        permutations=args.permutations,
        device=args.device,
        public_dim=args.public_dim,
        budget_ms=args.budget_ms,
        cache_path=args.cache,
    )
    _print_summary(results)
    print(f"output: {Path(args.out)}")


if __name__ == "__main__":
    main()
