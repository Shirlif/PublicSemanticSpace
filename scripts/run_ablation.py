"""Run preliminary dev ablations for S1 and S6."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pss.detect.oracle import detect_oracle, write_detections
from pss.metrics.ablation import ablate_entity_suite, ablate_memory_session
from pss.metrics.scorecard import discover_bundles


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundles-root", default="bundles")
    parser.add_argument("--s1-suite", default="s1_occlusion_20")
    parser.add_argument("--s6-session", default="bundles/dev/s6_memory/ae566231/seed0")
    parser.add_argument("--out", default="scorecards")
    args = parser.parse_args()

    bundles = discover_bundles(args.bundles_root, args.s1_suite, "dev")
    for bundle in bundles:
        rows = detect_oracle(bundle)
        write_detections(bundle, rows)
    report = ablate_entity_suite(
        args.bundles_root, args.s1_suite, device="dev", out=args.out
    )
    print(
        "S1 ablation:",
        {v: round(m.get("idf1", {}).get("mean", 0.0), 4) for v, m in report["metrics"].items()},
    )
    if Path(args.s6_session).exists():
        memory = ablate_memory_session(args.s6_session, out=args.out)
        print(
            "S6 ablation:",
            {
                v: round(m.get("edited_accuracy", 0.0), 3)
                for v, m in memory["metrics"].items()
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
