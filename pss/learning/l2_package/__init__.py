"""L2 prototype packages: signed, versioned, append-only, rollback-able.

Edge devices can read an activated package but cannot mutate it. The package
manager is the offline/backend side; the edge only sees ``active_package``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import shutil

import numpy as np

from ..signing import public_from_private, public_key_text, sign_bytes, verify_bytes


class PackageIntegrityError(Exception):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj))


@dataclass
class PrototypePackage:
    version: str
    categories: list[str]
    protos: dict[str, np.ndarray]
    stats: dict
    parent: str | None = None
    changelog: list[str] = field(default_factory=list)
    created: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    )
    signature: str = ""
    public_key: str = ""

    def save(self, path: str | Path, private_key: bytes | None = None) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        npz = path / "prototypes.npz"
        payload = {
            "categories_json": np.array(json.dumps(self.categories, ensure_ascii=False)),
            "stats_json": np.array(json.dumps(self.stats, ensure_ascii=False, default=_json_default)),
            "version": np.array(self.version),
            "parent": np.array(self.parent or ""),
        }
        for modality, arr in self.protos.items():
            payload[f"proto_{modality}"] = np.asarray(arr, dtype=np.float32)
        np.savez_compressed(npz, **payload)
        manifest = {
            "version": self.version,
            "parent": self.parent,
            "categories": self.categories,
            "files": ["prototypes.npz"],
            "sha256": _sha256(npz),
            "changelog": self.changelog,
            "created": self.created,
        }
        if private_key is not None:
            signed = json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode("utf-8")
            manifest["signature"] = sign_bytes(private_key, signed)
            manifest["public_key"] = public_key_text(public_from_private(private_key))
            self.signature = manifest["signature"]
            self.public_key = manifest["public_key"]
        (path / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        trusted_public_key: bytes | None = None,
    ) -> "PrototypePackage":
        path = Path(path)
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        npz = path / "prototypes.npz"
        if _sha256(npz) != manifest["sha256"]:
            raise PackageIntegrityError(f"hash mismatch for {manifest['version']}")
        signature = manifest.get("signature")
        public_key = manifest.get("public_key")
        if signature:
            if not public_key:
                raise PackageIntegrityError("signed package missing public key")
            if trusted_public_key is not None and public_key.encode("utf-8") != trusted_public_key:
                raise PackageIntegrityError("package public key does not match trusted key")
            base = {
                key: value
                for key, value in manifest.items()
                if key not in ("signature", "public_key")
            }
            payload = json.dumps(base, sort_keys=True, ensure_ascii=False).encode("utf-8")
            if not verify_bytes(public_key.encode("utf-8"), payload, signature):
                raise PackageIntegrityError("signature verification failed")
        elif trusted_public_key is not None:
            raise PackageIntegrityError("unsigned package rejected by trusted-key policy")
        with np.load(npz, allow_pickle=False) as f:
            categories = json.loads(str(f["categories_json"]))
            stats = json.loads(str(f["stats_json"]))
            protos = {
                k[len("proto_") :]: f[k].astype(np.float32)
                for k in f.files
                if k.startswith("proto_")
            }
        return cls(
            version=manifest["version"],
            categories=categories,
            protos=protos,
            stats=stats,
            parent=manifest.get("parent"),
            changelog=manifest.get("changelog", []),
            created=manifest.get("created", ""),
            signature=signature or "",
            public_key=public_key or "",
        )

    def verify(self, path: str | Path, trusted_public_key: bytes | None = None) -> bool:
        try:
            PrototypePackage.load(path, trusted_public_key=trusted_public_key)
            return True
        except Exception:
            return False

    def append_category(
        self,
        category: str,
        protos: dict[str, np.ndarray],
        stats: dict,
        version: str,
        changelog: str | None = None,
    ) -> "PrototypePackage":
        if category in self.categories:
            raise ValueError(f"category {category!r} already exists")
        new_protos = {
            m: np.vstack([self.protos[m], np.asarray(protos[m], dtype=np.float32)[None, :]])
            for m in self.protos
        }
        new_stats = {**self.stats, category: stats}
        return PrototypePackage(
            version=version,
            categories=self.categories + [category],
            protos=new_protos,
            stats=new_stats,
            parent=self.version,
            changelog=self.changelog + [changelog or f"add {category}"],
        )


class PackageManager:
    """Offline/backend package store with atomic activation and rollback."""

    def __init__(
        self,
        root: str | Path,
        trusted_public_key: bytes | None = None,
        private_key: bytes | None = None,
    ):
        self.root = Path(root)
        self.packages: dict[str, Path] = {}
        self.active: str | None = None
        self.history: list[str] = []
        self.trusted_public_key = trusted_public_key
        self.private_key = private_key

    def install(self, package: PrototypePackage) -> Path:
        path = self.root / "packages" / package.version
        package.save(path, private_key=self.private_key)
        self.packages[package.version] = path
        return path

    def verify(self, version: str) -> bool:
        path = self.packages.get(version)
        return bool(
            path
            and PrototypePackage.load(
                path, trusted_public_key=self.trusted_public_key
            )
        )

    def activate(self, version: str) -> None:
        if version not in self.packages:
            raise KeyError(version)
        PrototypePackage.load(
            self.packages[version], trusted_public_key=self.trusted_public_key
        )  # verifies integrity and signature
        self.root.mkdir(parents=True, exist_ok=True)
        ptr = self.root / "ACTIVE"
        tmp = self.root / "ACTIVE.tmp"
        tmp.write_text(version, encoding="utf-8")
        os.replace(tmp, ptr)
        if self.active is not None:
            self.history.append(self.active)
        self.active = version

    def rollback(self) -> str:
        if not self.history:
            raise RuntimeError("no previous version")
        previous = self.history.pop()
        ptr = self.root / "ACTIVE"
        tmp = self.root / "ACTIVE.tmp"
        tmp.write_text(previous, encoding="utf-8")
        os.replace(tmp, ptr)
        self.active = previous
        return previous

    def active_package(self) -> PrototypePackage | None:
        if self.active is None:
            return None
        return PrototypePackage.load(
            self.packages[self.active], trusted_public_key=self.trusted_public_key
        )

    def copy_package(self, version: str, dest: str | Path) -> Path:
        dest = Path(dest)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(self.packages[version], dest)
        return dest
