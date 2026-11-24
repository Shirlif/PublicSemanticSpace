"""``pss sim ...`` -- Isaac Sim build / run / verify commands."""

from __future__ import annotations

import argparse
import json
import os

from pss.spec import load_spec


def _cmd_build(args: argparse.Namespace) -> int:
    # ``import isaacsim`` bootstraps the Kit kernel paths and makes ``pxr``
    # importable without starting the full application.
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    import isaacsim  # noqa: F401

    from .scene_builder import build_scene

    spec = load_spec(args.spec)
    manifest = build_scene(spec, cache_dir=args.cache_dir, force=args.force)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from .runner import run

    run(
        args.spec,
        episodes=args.episodes,
        seed=args.seed,
        device=args.device,
        out_root=args.out,
        suite=args.suite,
        variant=args.variant,
        cache_dir=args.cache_dir,
        force=args.force,
    )
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    from .verify import print_report, verify_bundle

    report = verify_bundle(
        args.bundle,
        expect_instances=args.expect_instances,
        expect_obs_frames=args.expect_frames,
        expect_physics_steps=args.expect_steps,
    )
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print_report(report)
    return 0 if report["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pss sim", description="Isaac Sim episode tooling")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="compile a spec into a cached USD scene (A2)")
    build.add_argument("spec")
    build.add_argument("--cache-dir", help="override ~/.cache/pss_usd")
    build.add_argument("--force", action="store_true", help="rebuild even if cached")
    build.set_defaults(func=_cmd_build)

    run = sub.add_parser("run", help="run episodes and write bundles (A3)")
    run.add_argument("spec")
    run.add_argument("--episodes", type=int, default=1)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--device", choices=("dev", "official"), default="dev")
    run.add_argument("--out", help="bundle root (default: <repo>/bundles)")
    run.add_argument("--suite", help="suite name (default: spec id without _vN)")
    run.add_argument("--variant", default="full")
    run.add_argument("--cache-dir", help="override ~/.cache/pss_usd")
    run.add_argument("--force", action="store_true")
    run.set_defaults(func=_cmd_run)

    verify = sub.add_parser("verify", help="read-back checks for one bundle (D5)")
    verify.add_argument("bundle")
    verify.add_argument("--expect-instances", type=int, default=4)
    verify.add_argument("--expect-frames", type=int, default=100)
    verify.add_argument("--expect-steps", type=int, default=1200)
    verify.add_argument("--json", action="store_true")
    verify.set_defaults(func=_cmd_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
