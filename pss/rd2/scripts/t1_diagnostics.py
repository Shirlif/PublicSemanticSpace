"""CLI: run T1 diagnostics on expert_v1 or a T1 npz.

Examples:
    python -m pss.rd2.scripts.t1_diagnostics --source expert_v1 --data data/expert_v1
    python -m pss.rd2.scripts.t1_diagnostics --source npz --path data/t1/my_t1.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pss._console import ensure_utf8
from pss.rd2.t1_diagnostics import diagnose, format_report, load_expert_v1, load_npz


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["expert_v1", "npz"], default="expert_v1")
    ap.add_argument("--data", default="data/expert_v1")
    ap.add_argument("--path", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-samples", type=int, default=5000)
    args = ap.parse_args()

    if args.source == "expert_v1":
        data = load_expert_v1(args.data)
    else:
        if not args.path:
            raise SystemExit("--path is required for --source npz")
        data = load_npz(args.path)
    result = diagnose(data, max_samples=args.max_samples)
    print(format_report(result))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
