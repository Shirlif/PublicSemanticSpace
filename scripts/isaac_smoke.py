"""D1 smoke test: boot Isaac Sim headless, step physics, report startup/memory.

Run with the Isaac interpreter, from the repo root::

    D:\\isaac\\env\\Scripts\\python.exe scripts\\isaac_smoke.py

Writes JSON to stdout; append the result to ``notes/env.md``.
"""

from __future__ import annotations

import json
import os
import subprocess
import time

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")


def gpu_memory_for_pid(pid: int) -> str | None:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except Exception:
        return None
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0] == str(pid):
            return parts[1]
    return None


def main() -> int:
    t0 = time.time()
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "width": 640, "height": 480})
    startup_s = time.time() - t0
    boot_memory = gpu_memory_for_pid(os.getpid())

    from isaacsim.core.api import SimulationContext

    sim = SimulationContext(physics_dt=1 / 120.0, rendering_dt=1 / 120.0)
    sim.play()
    t1 = time.time()
    steps = 120
    for _ in range(steps):
        sim.step(render=False)
    step_s = time.time() - t1
    sim_time = float(sim.current_time)
    run_memory = gpu_memory_for_pid(os.getpid())

    sim.stop()
    app.close()
    print(
        json.dumps(
            {
                "startup_s": round(startup_s, 2),
                "physics_steps": steps,
                "physics_loop_s": round(step_s, 2),
                "per_step_ms": round(1000.0 * step_s / steps, 3),
                "sim_time_s": round(sim_time, 4),
                "gpu_memory_after_boot": boot_memory,
                "gpu_memory_after_steps": run_memory,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
