from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pss.metrics.cost import cost_suite, write_costcard


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundles-root", default="bundles/dev")
    parser.add_argument("--suite", default="s1_occlusion_oracle")
    parser.add_argument("--out", default="scorecards")
    args = parser.parse_args()
    report = cost_suite(args.bundles_root, args.suite)
    write_costcard(report, args.out)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
