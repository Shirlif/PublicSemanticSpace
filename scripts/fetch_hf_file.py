"""Parallel HTTP Range downloader for large model files.

HF/NVIDIA CDNs throttle a single connection badly from this network; many
small ranges restore throughput.  Resumable via per-chunk files.

    python scripts/fetch_hf_file.py --url <url> --out <file> [--workers 64]
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--chunk-mb", type=int, default=2)
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    parts = out.parent / (out.name + ".parts")
    parts.mkdir(parents=True, exist_ok=True)

    response = requests.head(args.url, allow_redirects=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers["Content-Length"])
    chunk = args.chunk_mb * 1024 * 1024
    n = (total + chunk - 1) // chunk
    print(f"size={total} chunks={n} workers={args.workers}", flush=True)

    def fetch(i: int) -> int:
        start = i * chunk
        end = min(start + chunk, total) - 1
        expected = end - start + 1
        path = parts / f"{i:05d}.part"
        if path.exists() and path.stat().st_size == expected:
            return i
        tmp = path.with_suffix(".tmp")
        for attempt in range(8):
            try:
                r = requests.get(
                    args.url,
                    headers={"Range": f"bytes={start}-{end}"},
                    timeout=(20, 600),
                    stream=True,
                    allow_redirects=True,
                )
                r.raise_for_status()
                written = 0
                with open(tmp, "wb") as handle:
                    for block in r.iter_content(128 * 1024):
                        handle.write(block)
                        written += len(block)
                r.close()
                if written == expected:
                    os.replace(tmp, path)
                    return i
            except Exception as exc:  # noqa: BLE001
                if attempt == 7:
                    raise RuntimeError(f"chunk {i}: {exc}") from exc
                time.sleep(min(2 + attempt * 2, 20))
        raise RuntimeError(f"chunk {i} failed")

    started = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch, i) for i in range(n)]
        for future in as_completed(futures):
            future.result()
            done += 1
            if done % 20 == 0 or done == n:
                elapsed = time.time() - started
                print(f"[{done}/{n}] {elapsed:.0f}s", flush=True)

    with open(out, "wb") as handle:
        for i in range(n):
            with open(parts / f"{i:05d}.part", "rb") as part:
                shutil.copyfileobj(part, handle)
    size = out.stat().st_size
    if size != total:
        print(f"size mismatch {size} != {total}", file=sys.stderr)
        return 1
    shutil.rmtree(parts, ignore_errors=True)
    print(f"OK {out} {size}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
