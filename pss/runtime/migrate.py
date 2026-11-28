"""Metadata-only artifact migration.

Migration does not change vectors.  It stamps the current ``space_version``
and writes a manifest; anything that would invalidate vectors (dimension,
normalisation, binding) is rejected and must be rebuilt.

CLI:
    python -m pss.runtime.migrate --kind bank --path checkpoints/public_space_v1/bank.npz
    python -m pss.runtime.migrate --kind readout --path ...
    python -m pss.runtime.migrate --kind projection --path ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pss._console import ensure_utf8
from pss.contracts.schema import PUBLIC_DIM, SPACE_VERSION
from pss.contracts.versioning import (
    ArtifactManifest,
    IncompatibleArtifactError,
    is_compatible,
)
from pss.prototypes.bank import PrototypeBank
from pss.prototypes.joint_readout import JointReadout
from pss.projection.heads import load_projection, save_projection


def _manifest(artifact: str, path: Path, modalities, **kwargs) -> ArtifactManifest:
    return ArtifactManifest.build(
        artifact=artifact,
        space_version=SPACE_VERSION,
        public_dim=PUBLIC_DIM,
        modalities=list(modalities),
        files={path.stem: path},
        **kwargs,
    )


def migrate_bank(path: str | Path) -> ArtifactManifest:
    path = Path(path)
    bank = PrototypeBank.load(path)
    if not is_compatible(bank.space_version, SPACE_VERSION):
        raise IncompatibleArtifactError(
            f"bank at {path} is {bank.space_version}; current {SPACE_VERSION}"
        )
    bank.space_version = SPACE_VERSION
    bank.save(path)
    return _manifest(
        "prototype_bank",
        path,
        bank.protos,
        bank_version=bank.version,
    )


def migrate_readout(path: str | Path) -> ArtifactManifest:
    path = Path(path)
    readout = JointReadout.load(path)
    if not is_compatible(readout.space_version, SPACE_VERSION):
        raise IncompatibleArtifactError(
            f"readout at {path} is {readout.space_version}; current {SPACE_VERSION}"
        )
    readout.space_version = SPACE_VERSION
    readout.save(path)
    return _manifest(
        "joint_readout",
        path,
        readout.modalities,
        readout_version=readout.version,
    )


def migrate_projection(path: str | Path, device: str = "cpu") -> ArtifactManifest:
    path = Path(path)
    head = load_projection(path, device=device)
    if not is_compatible(head.space_version, SPACE_VERSION):
        raise IncompatibleArtifactError(
            f"projection at {path} is {head.space_version}; current {SPACE_VERSION}"
        )
    head.space_version = SPACE_VERSION
    save_projection(path, head)
    return _manifest("projection_head", path, (), projection_versions={path.stem: SPACE_VERSION})


def check(artifact_version: str, space_version: str) -> bool:
    """Return True when an artifact version can be used with ``space_version``."""
    return is_compatible(artifact_version, space_version)


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kind", choices=["bank", "readout", "projection"], required=True)
    ap.add_argument("--path", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if args.kind == "bank":
        manifest = migrate_bank(args.path)
    elif args.kind == "readout":
        manifest = migrate_readout(args.path)
    else:
        manifest = migrate_projection(args.path, device=args.device)
    print(manifest.to_dict())


if __name__ == "__main__":
    main()
