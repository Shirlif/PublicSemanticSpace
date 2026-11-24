"""Mirror an Isaac Sim asset folder from the public NVIDIA S3 bucket.

Assets are not in the pip packages; Isaac Sim resolves them over HTTPS at
runtime.  Evaluation runs must not depend on a live CDN, so this script copies
a prefix into ``_external/isaac_assets`` (gitignored) before a run.

Usage::

    python scripts/fetch_isaac_assets.py \
        --prefix Assets/Isaac/5.1/Isaac/Robots/Unitree/G1/ \
        --dest _external/isaac_assets/G1
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.parse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

BUCKET = "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def list_keys(prefix: str, session: requests.Session) -> list[str]:
    keys: list[str] = []
    token = None
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if token:
            params["continuation-token"] = token
        response = session.get(BUCKET + "/", params=params, timeout=60)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        for content in root.findall("s3:Contents", NS):
            key = content.findtext("s3:Key", namespaces=NS)
            if key:
                keys.append(key)
        truncated = root.findtext("s3:IsTruncated", namespaces=NS) == "true"
        token = root.findtext("s3:NextContinuationToken", namespaces=NS)
        if not truncated or not token:
            return keys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--include-thumbs", action="store_true")
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    keys = list_keys(args.prefix, session)
    if not args.include_thumbs:
        keys = [k for k in keys if "/.thumbs/" not in k]
    print(f"{len(keys)} objects under {args.prefix}", flush=True)

    def fetch(key: str) -> tuple[str, int]:
        relative = key[len(args.prefix) :] if key.startswith(args.prefix) else key
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        url = BUCKET + "/" + urllib.parse.quote(key)
        for attempt in range(5):
            try:
                response = requests.get(url, timeout=(20, 300), stream=True)
                response.raise_for_status()
                tmp = target.with_suffix(target.suffix + ".tmp")
                size = 0
                with open(tmp, "wb") as handle:
                    for block in response.iter_content(256 * 1024):
                        handle.write(block)
                        size += len(block)
                os.replace(tmp, target)
                return relative, size
            except Exception as exc:  # noqa: BLE001
                if attempt == 4:
                    raise RuntimeError(f"{key}: {exc}") from exc
        raise RuntimeError(key)

    total = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, key): key for key in keys}
        for i, future in enumerate(as_completed(futures), 1):
            relative, size = future.result()
            total += size
            print(f"[{i}/{len(keys)}] {relative} {size} B", flush=True)
    print(f"done: {total / 1e6:.1f} MB -> {dest}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
