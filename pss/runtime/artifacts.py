"""A tiny on-disk artifact registry with checksums and version rollback.

This is intentionally small: it does not replace the training pipeline, it
only makes "which bank/readout/projection did we deploy?" auditable and
reversible.  A record is one file copied under ``root/<name>/<version>/`` plus
an entry in ``root/index.json``.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from pss.contracts.versioning import sha256_file


@dataclass
class ArtifactRecord:
    name: str
    version: str
    path: str
    sha256: str
    created_at: str

    def to_dict(self) -> dict:
        return asdict(self)


class ArtifactRegistry:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.index_path = self.root / "index.json"
        self.root.mkdir(parents=True, exist_ok=True)

    def _load(self) -> list[dict]:
        if not self.index_path.exists():
            return []
        return json.loads(self.index_path.read_text(encoding="utf-8")) or []

    def _save(self, rows: list[dict]) -> None:
        self.index_path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def register(
        self, name: str, source: str | Path, version: str | None = None
    ) -> ArtifactRecord:
        source = Path(source)
        if not source.exists():
            raise FileNotFoundError(source)
        version = version or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        target_dir = self.root / name / version
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / source.name
        shutil.copy2(source, target)
        record = ArtifactRecord(
            name=name,
            version=version,
            path=str(target),
            sha256=sha256_file(target),
            created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        rows = self._load()
        rows.append(record.to_dict())
        self._save(rows)
        return record

    def list(self, name: str | None = None) -> list[dict]:
        rows = self._load()
        if name is None:
            return rows
        return [row for row in rows if row["name"] == name]

    def resolve(self, name: str, version: str | None = None) -> Path | None:
        rows = self.list(name)
        if not rows:
            return None
        if version is not None:
            rows = [row for row in rows if row["version"] == version]
            if not rows:
                return None
            return Path(rows[-1]["path"])
        return Path(rows[-1]["path"])

    def rollback(self, name: str, version: str, target: str | Path | None = None) -> Path:
        """Restore a previous version to ``target`` (default: its own path)."""
        source = self.resolve(name, version)
        if source is None:
            raise FileNotFoundError(f"no artifact {name!r} version {version!r}")
        destination = Path(target) if target is not None else source
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.resolve() != source.resolve():
            shutil.copy2(source, destination)
        return destination
