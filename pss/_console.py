"""Console helpers for Windows terminals with a legacy code page."""

from __future__ import annotations

import sys


def ensure_utf8() -> None:
    """Reconfigure stdout/stderr to UTF-8 when possible.

    Windows consoles default to the system code page (often GBK on zh-CN
    machines), which cannot encode characters such as ``✓``, ``✅`` or ``²``.
    Reconfiguring keeps the demos runnable without requiring ``chcp 65001``.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
