"""Load, lint and register world specs."""

from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path
from typing import Any

import yaml

from .schema import (
    LintResult,
    SpecValidationError,
    WorldSpec,
    hash_spec,
    json_schema,
    validate,
)

__all__ = [
    "LintResult",
    "SpecValidationError",
    "WorldSpec",
    "hash_spec",
    "json_schema",
    "lint_file",
    "load_spec",
    "load_yaml",
    "register",
    "validate",
]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = REPO_ROOT / "specs" / "registry.jsonl"


def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read a YAML mapping; empty documents are rejected by validation."""
    p = Path(path)
    try:
        with p.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except FileNotFoundError:
        raise
    except yaml.YAMLError as exc:
        raise SpecValidationError([f"{p}: YAML parse error: {exc}"]) from exc
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise SpecValidationError([f"{p}: top level must be a mapping"])
    return data


def lint_file(path: str | os.PathLike[str]) -> LintResult:
    """Validate a spec file and return errors/warnings plus its canonical hash."""
    p = Path(path)
    try:
        raw = load_yaml(p)
    except SpecValidationError as exc:
        return LintResult(p.stem, "", list(exc.errors))
    except OSError as exc:
        return LintResult(p.stem, "", [f"{p}: cannot read spec: {exc}"])
    return validate(raw)


def load_spec(path: str | os.PathLike[str], *, strict: bool = True) -> WorldSpec:
    """Load and validate a spec, returning the typed view with canonical hash."""
    p = Path(path)
    raw = load_yaml(p)
    result = validate(raw)
    if strict and not result.ok:
        raise SpecValidationError(result.errors, result.warnings)
    return WorldSpec(raw=raw, sha256=result.sha256, source=p.resolve())


def register(
    result: LintResult,
    path: str | os.PathLike[str],
    *,
    registry: str | os.PathLike[str] | None = None,
) -> bool:
    """Append a successful lint result to the registry (idempotent).

    Returns True when a new line was written, False when the same
    ``(spec_id, sha256)`` was already registered.
    """
    if not result.ok:
        raise SpecValidationError(result.errors, result.warnings)

    registry_path = Path(registry) if registry is not None else DEFAULT_REGISTRY
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path = Path(path)
    try:
        rel_path = spec_path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        rel_path = spec_path.as_posix()

    entry = {
        "spec_id": result.spec_id,
        "sha256": result.sha256,
        "path": rel_path,
        "spec_version": 1,
        "linted_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "warnings": len(result.warnings),
    }

    if registry_path.exists():
        with registry_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    old = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if old.get("spec_id") == entry["spec_id"] and old.get("sha256") == entry["sha256"]:
                    return False

    with registry_path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    return True
