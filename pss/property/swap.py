"""Hot-swap registry for expert modules.

The registry owns the active module per modality and the swap stack.  A swap is
validated against :mod:`pss.experts.contract` before it becomes active; an
invalid module is rejected and the old module stays in place.  Rollback restores
the previous module and a behavioural hash is used for exactness checks.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

import numpy as np

from pss.experts.contract import ExpertValidationError, validate_expert


@dataclass
class SwapResult:
    success: bool
    modality: str
    old_version: str | None
    new_version: str | None
    latency_ms: float
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "success": bool(self.success),
            "modality": self.modality,
            "old_version": self.old_version,
            "new_version": self.new_version,
            "latency_ms": float(self.latency_ms),
            "error": self.error,
        }


class ModuleRegistry:
    def __init__(self, public_dim: int = 128, probe=None):
        self.public_dim = int(public_dim)
        self.probe = probe
        self._modules: dict[str, object] = {}
        self._stack: list[tuple[str, object | None]] = []
        self._audit: list[dict] = []
        self._versions: dict[str, list[str]] = {}

    # ── registration / lookup ──────────────────────────────────────────────

    def register(self, expert, probe=None) -> None:
        validate_expert(
            expert,
            public_dim=self.public_dim,
            probe=probe if probe is not None else self.probe,
        )
        self._modules[expert.modality] = expert
        self._versions.setdefault(expert.modality, []).append(expert.version)
        self._audit.append(
            {"action": "register", "modality": expert.modality, "version": expert.version}
        )

    def get(self, modality: str):
        return self._modules.get(modality)

    def remove(self, modality: str) -> bool:
        """Remove a module (used by add/remove-module tests)."""
        if modality not in self._modules:
            return False
        self._stack.append((modality, self._modules.pop(modality)))
        self._audit.append({"action": "remove", "modality": modality})
        return True

    def active_versions(self) -> dict[str, str]:
        return {m: e.version for m, e in self._modules.items()}

    def history(self) -> list[dict]:
        return list(self._audit)

    def versions(self) -> dict[str, list[str]]:
        return {m: list(v) for m, v in self._versions.items()}

    # ── swap ───────────────────────────────────────────────────────────────

    def swap(self, expert, probe=None) -> SwapResult:
        start = time.perf_counter()
        modality = getattr(expert, "modality", "")
        old = self._modules.get(modality)
        old_version = getattr(old, "version", None) if old is not None else None
        try:
            validate_expert(
                expert,
                public_dim=self.public_dim,
                probe=probe if probe is not None else self.probe,
            )
        except (ExpertValidationError, Exception) as exc:
            latency = (time.perf_counter() - start) * 1000.0
            result = SwapResult(
                success=False,
                modality=str(modality),
                old_version=old_version,
                new_version=None,
                latency_ms=latency,
                error=str(exc),
            )
            self._audit.append({"action": "swap_rejected", **result.to_dict()})
            return result

        self._stack.append((modality, old))
        self._modules[modality] = expert
        self._versions.setdefault(modality, []).append(expert.version)
        latency = (time.perf_counter() - start) * 1000.0
        result = SwapResult(
            success=True,
            modality=str(modality),
            old_version=old_version,
            new_version=expert.version,
            latency_ms=latency,
        )
        self._audit.append({"action": "swap", **result.to_dict()})
        return result

    def rollback(self) -> SwapResult:
        start = time.perf_counter()
        if not self._stack:
            return SwapResult(False, "", None, None, 0.0, "empty swap stack")
        modality, old = self._stack.pop()
        current = self._modules.get(modality)
        current_version = getattr(current, "version", None) if current is not None else None
        if old is None:
            self._modules.pop(modality, None)
            old_version = None
        else:
            self._modules[modality] = old
            old_version = getattr(old, "version", None)
        latency = (time.perf_counter() - start) * 1000.0
        result = SwapResult(
            success=True,
            modality=modality,
            old_version=current_version,
            new_version=old_version,
            latency_ms=latency,
        )
        self._audit.append({"action": "rollback", **result.to_dict()})
        return result

    # ── behavioural hashes ─────────────────────────────────────────────────

    def behavior_hash(self, modality: str, probe=None) -> str:
        expert = self._modules.get(modality)
        if expert is None:
            return "missing"
        probe = probe if probe is not None else self.probe
        if probe is None:
            return f"{modality}:{expert.version}:no-probe"
        out = np.asarray(expert.encode(probe), dtype=np.float32)
        digest = hashlib.sha256(out.tobytes()).hexdigest()[:16]
        return f"{modality}:{expert.version}:{digest}"

    def snapshot(self, probe=None) -> dict:
        return {
            modality: {
                "version": expert.version,
                "hash": self.behavior_hash(modality, probe=probe),
            }
            for modality, expert in self._modules.items()
        }

    def restore(self, payload: dict) -> None:
        """Restore module versions listed in a snapshot (objects must exist)."""

        for modality, entry in payload.items():
            if modality not in self._modules:
                continue
            # The registry does not serialize experts; callers restore by
            # swapping the matching object back in.
            self._versions.setdefault(modality, []).append(str(entry.get("version")))
