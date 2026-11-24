"""CLI: validate an RD2 dataset directory."""

from __future__ import annotations

import argparse
import json

from pss._console import ensure_utf8
from pss.rd2.validate import validate_dataset


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="data/rd2_mock")
    ap.add_argument("--max-sync-ms", type=float, default=5.0)
    args = ap.parse_args()
    summary = validate_dataset(args.path, max_sync_ns=int(args.max_sync_ms * 1e6))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
