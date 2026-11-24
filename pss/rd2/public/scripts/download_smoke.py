"""CLI: download a small public dataset shard and run the smoke tests."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

from pss._console import ensure_utf8
from pss.rd2.public.lerobot import LeRobotDataset
from pss.rd2.public.registry import PUBLIC_DATASETS
from pss.rd2.public.smoke import run_smoke


def _download_file(endpoint: str, repo_id: str, filename: str, root: Path) -> tuple[Path, str]:
    """Stream a file from an HF-compatible endpoint (default: hf-mirror.com)."""
    dest = root / filename
    if dest.exists() and dest.stat().st_size > 0:
        return dest, f"exists {dest.stat().st_size}"
    url = f"{endpoint.rstrip('/')}/datasets/{repo_id}/resolve/main/{filename}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    start = time.time()
    try:
        for _attempt in (0, 1):
            offset = tmp.stat().st_size if tmp.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with requests.get(url, headers=headers, stream=True, timeout=120) as response:
                if offset and response.status_code == 416:
                    break  # already complete
                if offset and response.status_code != 206:
                    tmp.unlink()
                    continue  # server ignored Range; restart from zero
                response.raise_for_status()
                with tmp.open("ab" if offset else "wb") as handle:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        if chunk:
                            handle.write(chunk)
                break
        tmp.replace(dest)
        elapsed = max(1e-6, time.time() - start)
        size = dest.stat().st_size
        return dest, f"downloaded {size} bytes in {elapsed:.1f}s ({size/elapsed/1024:.0f} KiB/s)"
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(PUBLIC_DATASETS), default="wt_mm")
    ap.add_argument("--root", default=None)
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--video", default=None)
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--endpoint", default="https://hf-mirror.com")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    spec = PUBLIC_DATASETS[args.dataset]
    root = Path(args.root or f"data/public/{args.dataset}")
    root.mkdir(parents=True, exist_ok=True)

    if not args.no_download:
        for pattern in spec.default_patterns:
            try:
                local, message = _download_file(args.endpoint, spec.repo_id, pattern, root)
                print(f"OK {pattern} {message}")
            except Exception as exc:
                print(f"SKIP {pattern} {type(exc).__name__}: {exc}")

    if spec.format != "lerobot_v3":
        print(f"{spec.repo_id}: format {spec.format} is not LeRobot v3; adapter skipped.")
        return

    dataset = LeRobotDataset(root, repo_id=spec.repo_id)
    metrics = run_smoke(dataset, limit_frames=args.limit, video_path=args.video)
    metrics["dataset"] = {
        "key": spec.key,
        "repo_id": spec.repo_id,
        "license": spec.license,
        "commercial_ok": spec.commercial_ok,
        "modalities": list(spec.modalities),
        "sync_tier": spec.sync_tier,
        "tactile_sensor": spec.tactile_sensor,
        "has_object_pose": spec.has_object_pose,
        "has_instance_ids": spec.has_instance_ids,
        "notes": spec.notes,
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
