"""S7 calibration scorecard."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pss.metrics.calibration import calibration_suite


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundles-root", default="bundles/dev")
    parser.add_argument("--suite", default="s1_occlusion_real")
    parser.add_argument("--out", default="scorecards")
    args = parser.parse_args()
    report = calibration_suite(args.bundles_root, args.suite)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "S7.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# S7 calibration (dev)",
        "",
        f"- suite: `{args.suite}`  episodes: {report['episodes']}  samples: {report['samples']}",
        f"- accuracy: {report['accuracy']['mean']:.4f}",
        f"- mean confidence: {report['mean_confidence']['mean']:.4f}",
        f"- Brier: {report['brier']['mean']:.4f}",
        f"- ECE: {report['ece']['mean']:.4f}",
        f"- selective accuracy @50% coverage: {report['selective_accuracy_50']['mean']:.4f}",
        f"- selective accuracy @80% coverage: {report['selective_accuracy_80']['mean']:.4f}",
        f"- conflict F1: {report['conflict_f1']}",
        "",
        "S7 here measures detector score calibration; conflict F1 needs PSS uncertainty traces.",
    ]
    (out / "S7.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
