"""L3/L4 registries: adapters and cognitive rules with versions + rollback.

These are metadata registries in Phase 9; actual weight files are simulated by
hashes. Edge devices only read the active version.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path


def _payload_hash(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class VersionedPackage:
    version: str
    kind: str
    payload: dict
    parent: str | None = None
    sha256: str = field(default="")
    artifacts: dict[str, bytes] = field(default_factory=dict)

    def __post_init__(self):
        if not self.sha256:
            self.sha256 = _payload_hash(self.payload)


class Registry:
    def __init__(self, layer: str, root: str | Path | None = None):
        self.layer = layer
        self.root = None if root is None else Path(root)
        self.packages: dict[str, VersionedPackage] = {}
        self.active: str | None = None
        self.history: list[str] = []

    def _package_dir(self, version: str) -> Path:
        if self.root is None:
            raise RuntimeError("registry has no storage root")
        return self.root / version

    def install(self, package: VersionedPackage) -> None:
        if package.sha256 != _payload_hash(package.payload):
            raise ValueError("package hash mismatch")
        if self.root is not None:
            directory = self._package_dir(package.version)
            (directory / "files").mkdir(parents=True, exist_ok=True)
            (directory / "payload.json").write_text(
                json.dumps(package.payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            file_hashes = {}
            for name, data in package.artifacts.items():
                target = directory / "files" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                file_hashes[name] = _file_hash(target)
            manifest = {
                "version": package.version,
                "kind": package.kind,
                "parent": package.parent,
                "payload_hash": package.sha256,
                "files": file_hashes,
            }
            (directory / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        self.packages[package.version] = package

    def verify(self, version: str) -> bool:
        package = self.packages.get(version)
        if not package or package.sha256 != _payload_hash(package.payload):
            return False
        if self.root is not None:
            directory = self._package_dir(version)
            manifest_path = directory / "manifest.json"
            if not manifest_path.exists():
                return False
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("payload_hash") != package.sha256:
                return False
            for name, digest in manifest.get("files", {}).items():
                target = directory / "files" / name
                if not target.exists() or _file_hash(target) != digest:
                    return False
        return True

    def activate(self, version: str) -> None:
        if not self.verify(version):
            raise ValueError(f"cannot activate unverified package {version}")
        if self.active is not None:
            self.history.append(self.active)
        self.active = version

    def rollback(self) -> str:
        if not self.history:
            raise RuntimeError("no previous version")
        self.active = self.history.pop()
        return self.active

    def active_package(self) -> VersionedPackage | None:
        return self.packages.get(self.active) if self.active else None


class AdapterRegistry(Registry):
    def __init__(self, root: str | Path | None = None):
        super().__init__("L3", root)


class RuleRegistry(Registry):
    def __init__(self, root: str | Path | None = None):
        super().__init__("L4", root)


def adapter_package(
    version: str,
    kind: str,
    files: dict[str, str],
    parent: str | None = None,
    artifacts: dict[str, bytes] | None = None,
):
    return VersionedPackage(
        version=version,
        kind=kind,
        payload={"files": files},
        parent=parent,
        artifacts=artifacts or {},
    )


def rule_package(
    version: str,
    value_vector: dict,
    blacklist: list[str],
    scenario_weights: dict | None = None,
    parent: str | None = None,
    artifacts: dict[str, bytes] | None = None,
):
    return VersionedPackage(
        version=version,
        kind="rules",
        payload={
            "value_vector": value_vector,
            "blacklist": blacklist,
            "scenario_weights": scenario_weights or {},
        },
        parent=parent,
        artifacts=artifacts or {},
    )
