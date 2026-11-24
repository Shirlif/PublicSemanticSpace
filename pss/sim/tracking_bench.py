"""Generate the PSS vs Transformer tracking benchmark (P1).

Balanced subsets:
- mixed: different categories
- samecat: three cups with appearance variation
- near_identical: three identical cups

Each sequence is 24 frames: 5 visible, 14 fully occluded, 5 visible, with
object motion during occlusion.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import dataset
from .generator import generate_occlusion_v3_sequence

SUBSETS = ("mixed", "samecat", "near_identical")
SPLIT_WEIGHTS = {"train": 0.5, "val": 0.25, "test": 0.25}
TOTAL_WEIGHTS = {"mixed": 0.5, "samecat": 0.25, "near_identical": 0.25}


def _sequence_kwargs(subset: str) -> dict:
    if subset == "mixed":
        return {"category": None, "variation_specs": False, "near_identical": False}
    if subset == "samecat":
        return {"category": "cup", "variation_specs": True, "near_identical": False}
    if subset == "near_identical":
        return {"category": "cup", "variation_specs": False, "near_identical": True}
    raise ValueError(f"unknown subset: {subset}")


def generate_tracking_bench(
    out_dir: str | Path,
    train_count: int = 200,
    val_count: int = 40,
    test_count: int = 60,
    seed: int = 0,
    width: int = 128,
    height: int = 128,
    frames: int = 24,
    visible_head: int = 5,
    visible_tail: int = 5,
    randomize_motion: bool = True,
    camera_jitter: bool = True,
    n_objects: int = 3,
    min_separation: float = 0.15,
    min_pixels: int = 20,
    data_scale: str = "tracking_bench_v1",
) -> dict:
    out = Path(out_dir)
    (out / "occlusion").mkdir(parents=True, exist_ok=True)
    entries = []
    split_counts = {"train": train_count, "val": val_count, "test": test_count}
    serial = 0
    for subset in SUBSETS:
        for split in ("train", "val", "test"):
            count = int(round(split_counts[split] * TOTAL_WEIGHTS[subset]))
            for i in range(count):
                sequence_seed = seed + serial * 1009
                serial += 1
                sample = None
                for attempt in range(20):
                    candidate = generate_occlusion_v3_sequence(
                        sequence_seed + attempt * 7919,
                        width=width,
                        height=height,
                        frames=frames,
                        visible_head=visible_head,
                        visible_tail=visible_tail,
                        randomize_motion=randomize_motion,
                        camera_jitter=camera_jitter,
                        n_objects=n_objects,
                        min_separation=min_separation,
                        **_sequence_kwargs(subset),
                    )
                    if not candidate["fully_occluded"]:
                        continue
                    valid = True
                    for phase in ("head", "tail"):
                        for counts in candidate["visible_object_pixel_counts"][phase]:
                            if min(counts.values()) < min_pixels:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        sample = candidate
                        break
                if sample is None:
                    raise RuntimeError(
                        f"{subset}/{split}/{i}: could not generate a valid sequence"
                    )
                sample["split"] = split
                sample["subset"] = subset
                rel = f"occlusion/{subset}_{split}_{i:04d}.npz"
                dataset.save_sample(out / rel, sample)
                entries.append(
                    {
                        "id": f"{subset}_{split}_{i:04d}",
                        "kind": "occlusion",
                        "path": rel,
                        "split": split,
                        "subset": subset,
                        "categories": sample["categories"],
                        "instance_ids": sample["instance_ids"],
                        "frames": int(sample["rgb"].shape[0]),
                        "occlusion_start": int(sample["occlusion_start"]),
                        "occlusion_end": int(sample["occlusion_end"]),
                        "fully_occluded": bool(sample["fully_occluded"]),
                    }
                )
    config = {
        "scale": data_scale,
        "seed": seed,
        "width": width,
        "height": height,
        "counts": {
            "train": train_count,
            "val": val_count,
            "test": test_count,
            "frames": frames,
            "visible_head": visible_head,
            "visible_tail": visible_tail,
            "n_objects": n_objects,
            "min_separation": min_separation,
        },
        "subsets": {name: TOTAL_WEIGHTS[name] for name in SUBSETS},
        "randomize_motion": bool(randomize_motion),
        "camera_jitter": bool(camera_jitter),
        "protocol": (
            "5 visible frames, 14 fully occluded frames with object motion, "
            "5 visible frames. Near-identical uses identical size/color/material."
        ),
    }
    dataset.write_index(out, entries, config)
    dataset.write_data_card(out, entries, config)
    return {
        "out_dir": str(out),
        "entries": len(entries),
        "train": sum(1 for e in entries if e["split"] == "train"),
        "val": sum(1 for e in entries if e["split"] == "val"),
        "test": sum(1 for e in entries if e["split"] == "test"),
    }
