"""Adapter registry for the PSS evaluation suite."""

from __future__ import annotations

from pathlib import Path

from .adapter import BenchmarkAdapter

ADAPTERS = {
    "bfcl": ("pss.eval.suite.adapters.bfcl", "BFCLAdapter"),
    "openood": ("pss.eval.suite.adapters.openood", "OpenOODAdapter"),
    "zsre": ("pss.eval.suite.adapters.zsre", "ZsREAdapter"),
    "clevr_cogent": ("pss.eval.suite.adapters.clevr_cogent", "CLEVRCoGenTAdapter"),
    "tao": ("pss.eval.suite.adapters.tao", "TAOAdapter"),
}


def get_adapter(name: str, root: str | Path | None = None) -> BenchmarkAdapter:
    if name not in ADAPTERS:
        raise KeyError(f"unknown benchmark {name!r}; available: {sorted(ADAPTERS)}")
    module_name, class_name = ADAPTERS[name]
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)(root=root)


def list_adapters() -> list[str]:
    return sorted(ADAPTERS)
