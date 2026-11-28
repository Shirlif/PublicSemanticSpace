"""Artifact versioning: one manifest, one compatibility rule.

The PSS stores several kinds of artifacts (prototype banks, joint readouts,
projection heads, calibration records).  A compatibility failure should be a
clear, early error with a migration path — not a shape mismatch at inference
time.

Version rules (``contracts/README.md``):

- **major** changes (``PUBLIC_DIM``, normalisation, binding, ``SLOT_COUNT``)
  invalidate every stored vector;
- **minor** changes (new modality, new expert, new projection) keep old
  artifacts valid;
- **patch** changes (thresholds, calibration, metadata) never invalidate.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class IncompatibleArtifactError(RuntimeError):
    """Raised when an artifact cannot be used with the current space version."""


def parse_version(value: str) -> tuple[int, int, int]:
    parts = str(value).strip().lstrip("v").split(".")
    if len(parts) != 3:
        raise ValueError(f"expected MAJOR.MINOR.PATCH, got {value!r}")
    try:
        return tuple(int(part) for part in parts)  # type: ignore[return-value]
    except ValueError as exc:
        raise ValueError(f"non-numeric version {value!r}") from exc


def is_compatible(artifact_version: str, space_version: str) -> bool:
    """Same major, and the artifact is not newer than the space."""
    artifact = parse_version(artifact_version)
    current = parse_version(space_version)
    if artifact[0] != current[0]:
        return False
    return artifact[1] <= current[1]


def assert_compatible(
    artifact_version: str,
    space_version: str,
    *,
    artifact_name: str = "artifact",
    migration: str | None = None,
) -> None:
    if is_compatible(artifact_version, space_version):
        return
    hint = migration or (
        "major changes invalidate stored vectors: rebuild the bank/readout, "
        "or run `python -m pss.runtime.migrate --help` for metadata-only migrations"
    )
    raise IncompatibleArtifactError(
        f"{artifact_name} was built for space {artifact_version}, "
        f"current space is {space_version}. {hint}"
    )


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


@dataclass
class ArtifactManifest:
    artifact: str
    space_version: str
    public_dim: int
    modalities: list[str]
    projection_versions: dict[str, str] = field(default_factory=dict)
    bank_version: str = ""
    readout_version: str = ""
    created_at: str = ""
    checksums: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ArtifactManifest":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {name for name in cls.__dataclass_fields__}
        payload = {k: v for k, v in data.items() if k in known}
        return cls(**payload)

    @classmethod
    def build(
        cls,
        artifact: str,
        space_version: str,
        public_dim: int,
        modalities: list[str],
        *,
        files: dict[str, str | Path] | None = None,
        **kwargs,
    ) -> "ArtifactManifest":
        checksums = {
            name: sha256_file(path) for name, path in (files or {}).items()
        }
        return cls(
            artifact=artifact,
            space_version=space_version,
            public_dim=int(public_dim),
            modalities=list(modalities),
            created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            checksums=checksums,
            **kwargs,
        )

    def check(self, space_version: str) -> None:
        assert_compatible(self.space_version, space_version, artifact_name=self.artifact)
