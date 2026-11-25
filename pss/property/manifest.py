"""Reproducibility manifests for property experiments.

Records the git revision, environment, expert/checkpoint hashes and scorecard
hashes.  The plan's §5.4 asks for these per run; a single manifest per
experiment family is enough to make the archived scorecards auditable.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _git(root: Path) -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        branch = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--abbrev-ref", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return {"commit": commit, "branch": branch}
    except Exception:
        return {"commit": None, "branch": None}


def capture_environment(root: str | Path = ".") -> dict:
    root = Path(root)
    try:
        import numpy

        numpy_version = numpy.__version__
    except Exception:
        numpy_version = None
    try:
        import torch

        torch_version = torch.__version__
    except Exception:
        torch_version = None
    return {
        "timestamp": time.time(),
        "python": sys.version.split()[0],
        "numpy": numpy_version,
        "torch": torch_version,
        "platform": platform.platform(),
        "git": _git(root),
    }


def build_asset_manifest(paths: dict[str, str | Path]) -> dict:
    out = {}
    for name, path in paths.items():
        path = Path(path)
        if not path.exists():
            out[name] = {"path": str(path), "exists": False}
            continue
        out[name] = {
            "path": str(path),
            "exists": True,
            "size": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
    return out


def write_manifest(path: str | Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
