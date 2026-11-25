"""Standalone GT writer: subscribes only to the GT socket and appends JSONL.

Started as a separate process from the simulator, so the PSS process never has
a GT endpoint in its address space.

    python scripts/gt_writer.py --addr tcp://127.0.0.1:5556 --out gt_bus.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pss.isaac.bus import BusSubscriber


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--addr", default="tcp://127.0.0.1:5556")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seconds", type=float, default=60.0)
    args = parser.parse_args()

    subscriber = BusSubscriber(args.addr, kind="pull")
    deadline = time.time() + args.seconds
    count = 0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        while time.time() < deadline:
            message = subscriber.recv(timeout_ms=500)
            if message is None:
                continue
            handle.write(json.dumps(message, ensure_ascii=False) + "\n")
            handle.flush()
            count += 1
    subscriber.close()
    print(f"gt_writer: {count} messages -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
