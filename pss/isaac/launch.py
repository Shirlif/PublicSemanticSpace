"""Isaac Sim launch configuration shared by build, smoke and episode runs.

On this machine (RTX 4070 Laptop, driver 616.64) the RTX scene DB crashes at
startup when Kit uses Vulkan; D3D12 boots cleanly.  The choice is a frozen part
of the stack: both evaluation machines must use the same graphics API, so it is
recorded in ``freeze.json``.

Override with ``PSS_ISAAC_EXTRA_ARGS`` (whitespace-separated Kit arguments)
when a machine is validated on a different backend.
"""

from __future__ import annotations

import os

DEFAULT_EXTRA_ARGS = ["--/app/vulkan=false"]  # D3D12


def extra_args() -> list[str]:
    override = os.environ.get("PSS_ISAAC_EXTRA_ARGS")
    if override is not None:
        return override.split()
    return list(DEFAULT_EXTRA_ARGS)


def launch_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {"headless": True, "extra_args": extra_args()}
    config.update(overrides)
    return config
