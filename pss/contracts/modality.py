"""Modality registry: the single source of truth for supported modalities.

The PSS core should not hard-code ``("vision", "tactile", "geometry")``.  A
modality is a row in ``contracts/modalities.yaml``; adding one means adding a
spec, an adapter, a projection head and a contract test — not editing the
prototype bank or the readout.

This module is intentionally dependency-light (PyYAML only) so it can be
imported by ``pss.contracts.schema`` without creating a cycle.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml

Normalize = Literal["l2", "none", "standardize"]
TimeUnit = Literal["frame", "ms", "s"]
MissingPolicy = Literal["skip", "impute", "raise"]
CalibrationKind = Literal["temperature", "none"]
ModalityStatus = Literal["active", "experimental", "retired"]

STATUSES = ("active", "experimental", "retired")
NORMALIZERS = ("l2", "none", "standardize")
TIME_UNITS = ("frame", "ms", "s")
MISSING_POLICIES = ("skip", "impute", "raise")
CALIBRATIONS = ("temperature", "none")

DEFAULT_FIELDS = {
    "normalize": "l2",
    "time_unit": "frame",
    "max_latency_ms": 50.0,
    "missing_policy": "skip",
    "calibration": "temperature",
    "status": "active",
}
_KNOWN_FIELDS = {
    "name",
    "kind",
    "input_schema",
    "native_dim",
    "preprocess",
    "normalize",
    "time_unit",
    "max_latency_ms",
    "reliability_prior",
    "missing_policy",
    "calibration",
    "capabilities",
    "status",
    "version",
}
_FALLBACK = ("vision", "tactile", "geometry")


class ModalityContractError(ValueError):
    """Raised when a modality spec is malformed or unknown."""


@dataclass(frozen=True)
class ModalitySpec:
    name: str
    kind: str
    input_schema: dict[str, Any]
    native_dim: int | None
    preprocess: str
    normalize: Normalize
    time_unit: TimeUnit
    max_latency_ms: float
    reliability_prior: float
    missing_policy: MissingPolicy
    calibration: CalibrationKind
    capabilities: tuple[str, ...]
    status: ModalityStatus
    version: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["capabilities"] = list(self.capabilities)
        return data


def default_registry_path() -> Path:
    env = os.environ.get("PSS_MODALITIES")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "contracts" / "modalities.yaml"


def _validate(name: str, spec: dict[str, Any]) -> ModalitySpec:
    unknown = set(spec) - _KNOWN_FIELDS
    if unknown:
        warnings.warn(
            f"modality {name!r} has unknown fields: {sorted(unknown)}",
            stacklevel=3,
        )
    required = {"kind", "preprocess", "reliability_prior", "version"}
    missing = sorted(required - set(spec))
    if missing:
        raise ModalityContractError(f"modality {name!r} missing fields: {missing}")

    normalize = spec["normalize"]
    time_unit = spec["time_unit"]
    missing_policy = spec["missing_policy"]
    calibration = spec["calibration"]
    status = spec["status"]
    for value, allowed, field in (
        (normalize, NORMALIZERS, "normalize"),
        (time_unit, TIME_UNITS, "time_unit"),
        (missing_policy, MISSING_POLICIES, "missing_policy"),
        (calibration, CALIBRATIONS, "calibration"),
        (status, STATUSES, "status"),
    ):
        if value not in allowed:
            raise ModalityContractError(
                f"modality {name!r} has invalid {field}={value!r}; expected {allowed}"
            )

    reliability = float(spec["reliability_prior"])
    if not 0.0 <= reliability <= 1.0:
        raise ModalityContractError(
            f"modality {name!r} reliability_prior must be in [0,1], got {reliability}"
        )
    latency = float(spec["max_latency_ms"])
    if latency < 0:
        raise ModalityContractError(
            f"modality {name!r} max_latency_ms must be >= 0, got {latency}"
        )
    native_dim = spec.get("native_dim")
    if native_dim is not None and int(native_dim) <= 0:
        raise ModalityContractError(
            f"modality {name!r} native_dim must be positive or null"
        )

    capabilities = spec.get("capabilities") or ()
    if not isinstance(capabilities, (list, tuple)):
        raise ModalityContractError(
            f"modality {name!r} capabilities must be a list"
        )
    input_schema = spec.get("input_schema") or {}
    if not isinstance(input_schema, dict):
        raise ModalityContractError(
            f"modality {name!r} input_schema must be a mapping"
        )
    version = str(spec["version"])
    if not version:
        raise ModalityContractError(f"modality {name!r} version must be non-empty")

    return ModalitySpec(
        name=name,
        kind=str(spec["kind"]),
        input_schema=dict(input_schema),
        native_dim=None if native_dim is None else int(native_dim),
        preprocess=str(spec["preprocess"]),
        normalize=normalize,
        time_unit=time_unit,
        max_latency_ms=latency,
        reliability_prior=reliability,
        missing_policy=missing_policy,
        calibration=calibration,
        capabilities=tuple(str(c) for c in capabilities),
        status=status,
        version=version,
    )


@lru_cache(maxsize=8)
def load_modalities(path: str | None = None) -> dict[str, ModalitySpec]:
    """Load and validate a modality registry; results are cached by path."""
    registry = Path(path) if path else default_registry_path()
    if not registry.exists():
        warnings.warn(
            f"modality registry {registry} not found; using fallback {_FALLBACK}",
            stacklevel=2,
        )
        return {
            name: ModalitySpec(
                name=name,
                kind="unknown",
                input_schema={},
                native_dim=None,
                preprocess="unknown",
                normalize="l2",
                time_unit="frame",
                max_latency_ms=50.0,
                reliability_prior=0.5,
                missing_policy="skip",
                calibration="none",
                capabilities=(),
                status="active",
                version="0.0.0",
            )
            for name in _FALLBACK
        }

    data = yaml.safe_load(registry.read_text(encoding="utf-8")) or {}
    defaults = {**DEFAULT_FIELDS, **(data.get("defaults") or {})}
    entries = data.get("modalities") or {}
    if not isinstance(entries, dict) or not entries:
        raise ModalityContractError(f"{registry}: 'modalities' must be a non-empty mapping")

    specs: dict[str, ModalitySpec] = {}
    for name, raw in entries.items():
        if not isinstance(raw, dict):
            raise ModalityContractError(f"modality {name!r} must be a mapping")
        merged = {**defaults, **raw}
        if "name" in raw and str(raw["name"]) != str(name):
            raise ModalityContractError(
                f"modality key {name!r} disagrees with name {raw['name']!r}"
            )
        merged["name"] = str(name)
        specs[str(name)] = _validate(str(name), merged)
    return specs


def active_modalities(path: str | None = None) -> tuple[str, ...]:
    """Registered modalities in registry order, active only."""
    return tuple(
        name for name, spec in load_modalities(path).items() if spec.status == "active"
    )


def get_modalities(
    path: str | None = None,
    *,
    include_experimental: bool = False,
    include_retired: bool = False,
) -> tuple[str, ...]:
    allowed = {"active"}
    if include_experimental:
        allowed.add("experimental")
    if include_retired:
        allowed.add("retired")
    return tuple(
        name for name, spec in load_modalities(path).items() if spec.status in allowed
    )


def get_spec(name: str, path: str | None = None) -> ModalitySpec:
    specs = load_modalities(path)
    if name not in specs:
        raise ModalityContractError(
            f"unknown modality {name!r}; registered: {sorted(specs)}"
        )
    return specs[name]


def resolve_modalities(names, path: str | None = None) -> tuple[str, ...]:
    """Validate names and return them in registry order (canonical)."""
    specs = load_modalities(path)
    requested = {str(name) for name in names}
    unknown = sorted(requested - set(specs))
    if unknown:
        raise ModalityContractError(
            f"unknown modalities {unknown}; registered: {sorted(specs)}"
        )
    return tuple(name for name in specs if name in requested)
