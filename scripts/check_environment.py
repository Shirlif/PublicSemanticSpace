"""A0/T40 environment alignment check against a freeze file."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", default="freeze/p7-w11.json")
    parser.add_argument("--device", choices=("dev", "official"), required=True)
    parser.add_argument("--isaac-python", default=r"D:\isaac\env\Scripts\python.exe")
    args = parser.parse_args()
    freeze_path = Path(args.freeze)
    if not freeze_path.is_absolute():
        freeze_path = REPO_ROOT / freeze_path
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))

    checks = []

    def check(name: str, actual, expected, ok: bool, detail: str = "") -> None:
        checks.append(
            {
                "check": name,
                "ok": bool(ok),
                "actual": actual,
                "expected": expected,
                "detail": detail,
            }
        )

    gpu_line = _run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]
    )
    gpu_name, gpu_mem, driver = (gpu_line.split(", ") if gpu_line else ["", "", ""])
    expected_gpu = freeze["gpu"][args.device]
    check("gpu", gpu_name, expected_gpu, expected_gpu.split()[1] in gpu_name if expected_gpu else False)
    check("driver", driver, freeze.get("driver"), driver == freeze.get("driver"))

    pss_python = platform.python_version()
    check("python.pss", pss_python, freeze.get("python", {}).get("pss"),
          pss_python == freeze.get("python", {}).get("pss"))

    isaac_version = _run(
        [args.isaac_python, "-c", "import importlib.metadata as m; print(m.version('isaacsim'))"]
    )
    check(
        "isaac_sim",
        isaac_version,
        freeze.get("sim", {}).get("isaac_sim"),
        isaac_version == freeze.get("sim", {}).get("isaac_sim"),
    )
    isaac_python = _run([args.isaac_python, "-c", "import platform; print(platform.python_version())"])
    check(
        "python.isaac",
        isaac_python,
        freeze.get("python", {}).get("isaac"),
        isaac_python == freeze.get("python", {}).get("isaac"),
    )

    if args.device == "official":
        check(
            "device_class_guard",
            gpu_name,
            "RTX 3090 Ti",
            "3090" in gpu_name,
            "official numbers may only come from the 3090 Ti",
        )

    ok = all(item["ok"] for item in checks)
    print(json.dumps({"device": args.device, "freeze": str(freeze_path), "ok": ok, "checks": checks}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
