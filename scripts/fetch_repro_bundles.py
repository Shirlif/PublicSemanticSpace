"""Download, verify, and unpack the public L2 bundle reproduction pack.

The pack is published as a release asset, not committed to git.  This script
checks the archive SHA256, rejects unsafe tar members, verifies every file
against ``PACK_MANIFEST.json``, and places the result under ``bundles/repro``.

Usage::

    python scripts/fetch_repro_bundles.py --url <release asset URL> --out bundles/repro
    python scripts/fetch_repro_bundles.py --archive dist/repro/pss-l2-bundles-v0.1.1.tar.gz \\
        --sha256 <hex> --out bundles/repro

Set ``PSS_BUNDLES_URL`` / ``PSS_BUNDLES_SHA256`` to avoid repeating the
arguments.  If ``--sha256`` is omitted, the script reads the first hash from
``docs/repro/SHA256SUMS``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUT = ROOT / "bundles" / "repro"
DEFAULT_SUMS = ROOT / "docs" / "repro" / "SHA256SUMS"


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _read_expected_sha(args: argparse.Namespace, archive_name: str) -> str | None:
    if args.sha256:
        return args.sha256.strip().lower()
    env = os.environ.get("PSS_BUNDLES_SHA256")
    if env:
        return env.strip().lower()
    sums = Path(args.sums)
    if sums.exists():
        for line in sums.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1].endswith(archive_name):
                return parts[0].strip().lower()
    return None


def _download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response, target.open("wb") as handle:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        next_mark = 5 << 20
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            handle.write(block)
            done += len(block)
            if done >= next_mark:
                if total:
                    print(f"  downloaded {done / 1e6:.1f} / {total / 1e6:.1f} MB", file=sys.stderr)
                else:
                    print(f"  downloaded {done / 1e6:.1f} MB", file=sys.stderr)
                next_mark += 5 << 20
    print(f"  downloaded {done / 1e6:.1f} MB", file=sys.stderr)


def _safe_extract(archive: Path, target: Path) -> None:
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            name = member.name.replace("\\", "/")
            path = Path(name)
            if name.startswith("/") or ".." in path.parts:
                raise SystemExit(f"unsafe tar member: {member.name!r}")
            if path.parts and ":" in path.parts[0]:
                raise SystemExit(f"unsafe tar member: {member.name!r}")
            if member.issym() or member.islnk() or member.isdev():
                raise SystemExit(f"unsupported tar member: {member.name!r}")
        try:
            tar.extractall(target, filter="data")  # Python 3.12+
        except TypeError:
            tar.extractall(target)


def _verify_tree(stage: Path) -> dict:
    manifest_path = stage / "PACK_MANIFEST.json"
    if not manifest_path.exists():
        raise SystemExit(f"PACK_MANIFEST.json missing under {stage}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files", [])
    if not files:
        raise SystemExit("PACK_MANIFEST.json has no file list")
    listed = {entry["path"] for entry in files}
    actual = {
        path.relative_to(stage).as_posix()
        for path in stage.rglob("*")
        if path.is_file() and path.name != "PACK_MANIFEST.json"
    }
    missing = listed - actual
    extra = actual - listed
    if missing:
        raise SystemExit(f"manifest lists missing files: {sorted(missing)[:5]}")
    if extra:
        raise SystemExit(f"pack contains unlisted files: {sorted(extra)[:5]}")
    for entry in files:
        path = stage / entry["path"]
        if not path.exists():
            raise SystemExit(f"missing file: {entry['path']}")
        if path.stat().st_size != int(entry["bytes"]):
            raise SystemExit(f"size mismatch: {entry['path']}")
        if _sha256(path) != entry["sha256"]:
            raise SystemExit(f"sha256 mismatch: {entry['path']}")
    return manifest


def main() -> None:
    from pss._console import ensure_utf8

    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("PSS_BUNDLES_URL", ""))
    ap.add_argument("--archive", default="")
    ap.add_argument("--sha256", default="")
    ap.add_argument("--sums", default=str(DEFAULT_SUMS))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()) and not args.force:
        raise SystemExit(f"{out} is not empty; pass --force to replace")

    with tempfile.TemporaryDirectory(prefix="pss-bundles-") as tmp:
        tmp_path = Path(tmp)
        if args.archive:
            archive = Path(args.archive).resolve()
            if not archive.exists():
                raise SystemExit(f"archive not found: {archive}")
            archive_name = archive.name
        elif args.url:
            archive_name = Path(args.url.split("?")[0]).name or "pss-l2-bundles.tar.gz"
            archive = tmp_path / archive_name
            print(f"downloading {args.url}", file=sys.stderr)
            _download(args.url, archive)
        else:
            raise SystemExit(
                "no archive or URL.  Set PSS_BUNDLES_URL or pass --archive.  "
                "The public pack is published as a GitHub release asset."
            )

        expected = _read_expected_sha(args, archive_name)
        if not expected:
            raise SystemExit(
                f"no SHA256 for {archive_name}.  Pass --sha256 or set "
                "PSS_BUNDLES_SHA256; docs/repro/SHA256SUMS is the default source."
            )
        actual = _sha256(archive)
        if actual != expected:
            raise SystemExit(f"archive sha256 mismatch: expected {expected}, got {actual}")
        print(f"archive sha256: {actual}")

        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()
        _safe_extract(archive, extract_dir)

        stages = sorted(p.parent for p in extract_dir.glob("*/PACK_MANIFEST.json"))
        if not stages:
            stages = sorted(p.parent for p in extract_dir.glob("PACK_MANIFEST.json"))
        if len(stages) != 1:
            raise SystemExit(f"expected one pack root, found {len(stages)}")
        stage = stages[0]
        manifest = _verify_tree(stage)
        print(
            f"verified {len(manifest.get('files', []))} files, "
            f"{manifest.get('n_bundles', '?')} bundles, "
            f"{len(manifest.get('embedding_bundles', []))} with embeddings"
        )

        if args.force:
            # A different pack version leaves its own stage behind; remove
            # existing packs so the replay does not count both.
            for child in list(out.iterdir()):
                if child.is_dir() and (child / "PACK_MANIFEST.json").exists():
                    shutil.rmtree(child)
                elif (
                    child.is_file()
                    and child.name.startswith("pss-l2-bundles-")
                    and child.name.endswith((".tar.gz", ".zip"))
                ):
                    child.unlink()

        target = out / stage.name
        if target.exists():
            if not args.force:
                raise SystemExit(f"{target} already exists; pass --force to replace")
            shutil.rmtree(target)
        shutil.copytree(stage, target)

    print(f"installed: {target}")
    print(f"next:      python scripts/reproduce_l2.py --root {out} --out reports/repro")


if __name__ == "__main__":
    main()
