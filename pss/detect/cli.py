"""``pss detect`` and ``pss replay`` commands."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .oracle import detect_oracle, write_detections
from .replay import replay_bundle


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_meta(bundle: str | Path, meta: dict) -> None:
    root = Path(bundle)
    (root / "detect_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["detector_tier"] = meta.get("tier", manifest.get("detector_tier"))
        stack = manifest.setdefault("stack", {})
        stack["detector"] = {
            "name": meta.get("detector"),
            "sha256": meta.get("weights_sha256"),
            "embedding": meta.get("embedding"),
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )


def _cmd_detect(args: argparse.Namespace) -> int:
    if args.tier == "oracle":
        rows = detect_oracle(args.bundle, min_visibility=args.min_visibility)
        _write_meta(
            args.bundle,
            {
                "tier": "oracle",
                "detector": "gt_boxes",
                "min_visibility": args.min_visibility,
                "rows": len(rows),
            },
        )
    elif args.tier == "frozen_real":
        from .groundingdino import DEFAULT_MODEL_DIR, detect_bundle, write_groundingdino_detections

        rows = detect_bundle(
            args.bundle,
            prompt=args.prompt,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            max_frames=args.max_frames,
            plane_z=None if args.depth_pose else args.plane_z,
        )
        path = write_groundingdino_detections(args.bundle, rows)
        weights = DEFAULT_MODEL_DIR / "model.safetensors"
        _write_meta(
            args.bundle,
            {
                "tier": "frozen_real",
                "detector": "IDEA-Research/grounding-dino-tiny",
                "embedding": "facebook/dinov2-small",
                "weights_sha256": _sha256_file(weights) if weights.exists() else None,
                "prompt": args.prompt,
                "box_threshold": args.box_threshold,
                "text_threshold": args.text_threshold,
                "pose_mode": "depth" if args.depth_pose else "plane",
                "plane_z": None if args.depth_pose else args.plane_z,
                "rows": len(rows),
            },
        )
        print(f"[detect] frozen_real {len(rows)} rows -> {path}")
        return 0
    elif args.tier == "noisy":
        import json

        import pandas as pd

        from .noise import apply_noise

        bundle = Path(args.bundle)
        base = pd.read_parquet(bundle / "obs_detections.parquet")
        gt = pd.read_parquet(bundle / "gt.parquet")
        profile = json.loads(args.noise_profile) if args.noise_profile else None
        rows = apply_noise(
            base.to_dict(orient="records"), gt, profile=profile, seed=args.seed
        )
        path = write_detections(args.bundle, rows)
        _write_meta(
            args.bundle,
            {
                "tier": "noisy",
                "detector": "gt_boxes+noise",
                "profile": profile or "defaults",
                "seed": args.seed,
                "rows": len(rows),
            },
        )
        print(f"[detect] noisy {len(rows)} rows -> {path}")
        return 0
    else:
        raise NotImplementedError(f"tier {args.tier!r} not implemented")
    path = write_detections(args.bundle, rows)
    print(f"[detect] {len(rows)} rows -> {path}")
    return 0


def _cmd_replay(args: argparse.Namespace) -> int:
    trace, pss = replay_bundle(args.bundle, write=not args.no_write, variant=args.variant)
    print(f"[replay] {len(trace)} assignments, {len(pss)} pss_trace rows (variant={args.variant})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pss detect", description="Detection and replay")
    sub = parser.add_subparsers(dest="command", required=True)

    detect = sub.add_parser("detect", help="write obs_detections.parquet for a bundle")
    detect.add_argument("bundle")
    detect.add_argument(
        "--tier", default="oracle", choices=("oracle", "frozen_real", "noisy")
    )
    detect.add_argument("--min-visibility", type=float, default=0.001)
    detect.add_argument("--prompt", default=None)
    detect.add_argument("--box-threshold", type=float, default=0.3)
    detect.add_argument("--text-threshold", type=float, default=0.25)
    detect.add_argument("--max-frames", type=int, default=None)
    detect.add_argument(
        "--plane-z",
        type=float,
        default=0.79,
        help="tabletop plane for 3D backprojection (default 0.79)",
    )
    detect.add_argument(
        "--depth-pose",
        action="store_true",
        help="use depth at bbox center instead of tabletop plane",
    )
    detect.add_argument("--noise-profile", default=None, help="inline JSON noise profile")
    detect.add_argument("--seed", type=int, default=0)
    detect.set_defaults(func=_cmd_detect)

    replay = sub.add_parser("replay", help="run PSS over detections and write traces")
    replay.add_argument("bundle")
    replay.add_argument("--no-write", action="store_true")
    replay.add_argument(
        "--variant",
        default="full",
        choices=(
            "full",
            "no_entity",
            "no_relations",
            "no_memory",
            "no_active_set",
            "no_uncertainty",
            "no_audit",
            "flat_context",
            "no_tools",
        ),
    )
    replay.set_defaults(func=_cmd_replay)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
