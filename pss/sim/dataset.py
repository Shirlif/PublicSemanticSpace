"""Dataset storage helpers: one ``.npz`` per sample plus a JSON index.

Metadata is stored as a JSON string inside the ``.npz`` so loading never needs
``allow_pickle=True``.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def save_sample(path: Path, sample: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in sample.items() if isinstance(v, np.ndarray)}
    meta = {k: v for k, v in sample.items() if not isinstance(v, np.ndarray)}
    arrays["meta_json"] = np.array(json.dumps(meta, ensure_ascii=False, default=_json_default))
    np.savez_compressed(path, **arrays)


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(type(obj))


def load_sample(path: Path) -> dict:
    with np.load(Path(path), allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}
    meta = json.loads(str(data.pop("meta_json")))
    return {**data, **meta}


def write_index(out_dir: Path, entries: list[dict], config: dict) -> Path:
    out_dir = Path(out_dir)
    payload = {
        "version": "v1",
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "config": config,
        "count": len(entries),
        "entries": entries,
    }
    path = out_dir / "index.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return path


def write_data_card(out_dir: Path, entries: list[dict], config: dict, notes: str | None = None) -> Path:
    out_dir = Path(out_dir)
    kinds = Counter(e["kind"] for e in entries)
    splits = Counter(e["split"] for e in entries)
    categories = Counter(e.get("category") for e in entries if e.get("category"))
    lines = [
        "# Dataset v1 — Tabletop Proxy",
        "",
        f"- generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"- simulator: MuJoCo (primitive proxy assets, no external meshes)",
        f"- images: {config.get('width')}x{config.get('height')}",
        f"- seed: {config.get('seed')}",
        "",
        "## Counts",
        "",
        "| kind | count |",
        "|---|---:|",
    ]
    for k in sorted(kinds):
        lines.append(f"| {k} | {kinds[k]} |")
    lines += ["", "## Splits", "", "| split | count |", "|---|---:|"]
    for s in sorted(splits):
        lines.append(f"| {s} | {splits[s]} |")
    lines += ["", "## Categories", "", "| category | count |", "|---|---:|"]
    for c in sorted(categories):
        lines.append(f"| {c} | {categories[c]} |")

    lines += [
        "",
        "## Schema",
        "",
        "Category sample (`.npz`):",
        "",
        "| key | shape | dtype | meaning |",
        "|---|---|---|---|",
        "| rgb | (V, H, W, 3) | uint8 | RGB views |",
        "| depth | (V, H, W) | float32 | metric depth |",
        "| segmentation | (V, H, W, 2) | int32 | [geom id, object type] |",
        "| tactile | (V, 8, 8) | float32 | synthetic material-based proxy |",
        "| meta_json | scalar | unicode | category, instance, material, camera |",
        "",
        "Episode (`.npz`):",
        "",
        "| key | shape | dtype | meaning |",
        "|---|---|---|---|",
        "| rgb | (T, H, W, 3) | uint8 | frames |",
        "| depth | (T, H, W) | float32 | frames |",
        "| segmentation | (T, H, W, 2) | int32 | frames |",
        "| tactile | (T, N, 8, 8) | float32 | contact-derived proxy per object |",
        "| poses | (T, N, 7) | float32 | xyz + quat per object |",
        "| meta_json | scalar | unicode | categories, instance ids, actions, relations |",
        "",
        "## Splits",
        "",
        *(
            [f"- {s}" for s in config["split_policy"]]
            if config.get("split_policy")
            else [
                "- Held-out categories: `cloth`, `tool` -> `test_category`.",
                "- Other categories: instance 0 -> train, 1 -> val, 2 -> test.",
                "- Episodes and occlusion sequences: 60% train, 20% val, 20% test by index.",
            ]
        ),
        "",
        *(["## Notes", "", notes, ""] if notes else []),
        "## Known biases / caveats",
        "",
        "- **Primitive proxy assets**: no real textures, meshes, deformables, or articulated containers.",
        "- **Synthetic tactile**: category-level tactile is generated from material parameters, not measured contact.",
        "  Episode tactile is contact-derived but only coarse (8x8) and includes object-object contacts.",
        "- **Scripted actions**: lift/place are kinematic proxies, not learned grasping.",
        "- **Single fixed camera per episode**: occlusion sequences vary the occluder, not the camera.",
        "- **No real sensor noise model**: add noise/domain randomization before sim2real claims.",
        "",
        "## Generation",
        "",
        "```bash",
        config.get(
            "generation_command",
            f"python -m pss.sim.scripts.generate_dataset --out {out_dir} "
            f"--scale {config.get('scale', 'small')} --seed {config.get('seed')}",
        ),
        "```",
        "",
    ]
    path = out_dir / "DATA_CARD.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
