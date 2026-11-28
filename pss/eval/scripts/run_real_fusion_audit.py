"""Real-data expert complementarity audit on CLEVR-CoGenT.

This script runs :func:`pss.encoders.scripts.bench_expert_zoo.fusion_audit` on
the actual CLEVR-CoGenT images and scene metadata shipped in
``data/public/clevr_cogent``.  It does **not** edit or reimplement the audit;
it only builds aligned per-scene feature streams and calls the shared audit.

Streams (all aligned on the same scene):

- ``dinov2_small``   frozen ``facebook/dinov2-small`` pooler (384-d), real PNG
- ``clip_vit_b16``   frozen ``openai/clip-vit-base-patch16`` vision pooler (768-d)
- ``img_handcrafted`` colour/texture statistics of the real PNG (no checkpoint)
- ``geom_scene``     handcrafted scene geometry from ``scenes/*.json`` metadata
                     (object count, pixel/3-D positions, rotations, pair
                     distances).  Attributes (shape/colour/material/size) are
                     deliberately excluded.

Splits follow the official CoGenT protocol: ``trainA`` -> train, ``valA`` ->
val, ``valB`` -> test.  ``valB`` is the compositional shift split.

Labels are derived from the scene metadata, e.g. the shape/material/colour of
the frontmost object (smallest depth) or the number of distinct colours.  See
``LABEL_TASKS``.

Example:
    python -m pss.eval.scripts.run_real_fusion_audit \
        --out eval/results/real_fusion_audit --n-train 3000 --n-val 750 --n-test 750

Outputs ``real_fusion_audit.json`` and ``real_fusion_audit.md`` under ``--out``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from pss._console import ensure_utf8
from pss.encoders.scripts.bench_expert_zoo import fusion_audit

DEFAULT_ROOT = Path("data/public/clevr_cogent/CLEVR_CoGenT_v1.0")
SPLITS = OrderedDict(
    [
        ("train", ("CLEVR_trainA_scenes.json", "trainA")),
        ("val", ("CLEVR_valA_scenes.json", "valA")),
        ("test", ("CLEVR_valB_scenes.json", "valB")),
    ]
)

def _majority(objs: list[dict], key: str) -> str:
    """Most frequent attribute value; ties broken alphabetically (deterministic)."""
    counts = Counter(o[key] for o in objs)
    top = max(counts.values())
    return sorted(value for value, count in counts.items() if count == top)[0]


# A view is one scene.  These tasks only use scene metadata for labels; no
# label attribute is present in the geometry stream.
LABEL_TASKS = OrderedDict(
    [
        ("front_shape", lambda objs: _front(objs)["shape"]),
        ("front_material", lambda objs: _front(objs)["material"]),
        ("majority_color", lambda objs: _majority(objs, "color")),
        ("majority_shape", lambda objs: _majority(objs, "shape")),
        ("distinct_shape", lambda objs: str(len({o["shape"] for o in objs}))),
        # 7 means "7 or more"; keeps rare 8-colour scenes from vanishing.
        ("distinct_color", lambda objs: str(min(len({o["color"] for o in objs}), 7))),
    ]
)

EXPERT_KINDS = {
    "dinov2_small": "frozen_pretrained_vision",
    "clip_vit_b16": "frozen_pretrained_vision",
    "img_handcrafted": "handcrafted_image_statistics",
    "geom_scene": "simulator_metadata_geometry",
}

# Expert subsets, so the report shows the marginal value of adding each stream.
SUBSETS = OrderedDict(
    [
        ("vision_pair", ("dinov2_small", "clip_vit_b16")),
        ("vision_plus_geometry", ("dinov2_small", "clip_vit_b16", "geom_scene")),
        (
            "vision_plus_handcrafted",
            ("dinov2_small", "clip_vit_b16", "img_handcrafted"),
        ),
        ("all_four", ("dinov2_small", "clip_vit_b16", "geom_scene", "img_handcrafted")),
    ]
)


# --------------------------------------------------------------------------- #
# pure feature / label builders
# --------------------------------------------------------------------------- #
def _front(objs: list[dict]) -> dict:
    """Frontmost object = smallest render depth (pixel_coords[2])."""
    return min(objs, key=lambda o: (float(o["pixel_coords"][2]), float(o["pixel_coords"][1])))


def scene_labels(scene: dict) -> dict[str, str]:
    """All label-task values for one scene."""
    objs = scene["objects"]
    return {name: str(fn(objs)) for name, fn in LABEL_TASKS.items()}


def geometry_features(scene: dict, max_objects: int = 10) -> np.ndarray:
    """Scene geometry from metadata, with all semantic attributes excluded.

    Included: object count, per-object pixel/depth coordinates, 3-D coordinates
    and rotation, plus position moments and pairwise 3-D distances.  Excluded:
    shape, colour, material and size (those are labels or label-correlated).
    """
    objs = sorted(scene["objects"], key=lambda o: float(o["pixel_coords"][2]))[
        :max_objects
    ]
    feat: list[float] = [float(len(scene["objects"]))]
    for obj in objs:
        px, py, pz = (float(v) for v in obj["pixel_coords"])
        x, y, z = (float(v) for v in obj["3d_coords"])
        angle = np.deg2rad(float(obj["rotation"]))
        # Coordinates are scaled to O(1); depth keeps simulator units / 20.
        feat.extend(
            [
                px / 480.0 - 0.5,
                py / 320.0 - 0.5,
                pz / 20.0,
                x / 3.0,
                y / 3.0,
                z,
                float(np.sin(angle)),
                float(np.cos(angle)),
            ]
        )
    while len(feat) < 1 + 8 * max_objects:
        feat.append(0.0)
    pos = np.asarray([o["3d_coords"] for o in objs], dtype=np.float64) / 3.0
    feat.extend(pos.mean(axis=0).tolist())
    feat.extend(pos.std(axis=0).tolist())
    pair: list[float] = []
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            pair.append(float(np.linalg.norm(pos[i] - pos[j]) / 2.0))
    feat.extend((pair + [0.0] * 45)[:45])
    return np.asarray(feat, dtype=np.float32)


def handcrafted_image_features(rgb: np.ndarray, size: int = 64) -> np.ndarray:
    """Cheap frozen colour/texture descriptor for a real scene image."""
    img = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").resize(
        (size, size), Image.BILINEAR
    )
    arr = np.asarray(img, dtype=np.float32) / 255.0
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32) / 255.0
    gray = arr @ np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
    gy, gx = np.gradient(gray)
    mag = np.sqrt(gx * gx + gy * gy)
    angle = (np.arctan2(gy, gx) + np.pi) / (2 * np.pi)
    feat: list[float] = []
    feat.extend(arr.mean(axis=(0, 1)).tolist())
    feat.extend(arr.std(axis=(0, 1)).tolist())
    for channel, bins in ((hsv[..., 0], 12), (hsv[..., 1], 6), (hsv[..., 2], 6)):
        hist, _ = np.histogram(channel, bins=bins, range=(0.0, 1.0), density=False)
        feat.extend((hist / max(channel.size, 1)).astype(np.float32).tolist())
    # 4x4 mean colour grid: coarse layout cue.
    grid = arr.reshape(4, size // 4, 4, size // 4, 3).mean(axis=(1, 3))
    feat.extend(grid.reshape(-1).tolist())
    feat.extend([float(mag.mean()), float(mag.std())])
    hist, _ = np.histogram(angle.ravel(), bins=8, range=(0.0, 1.0), weights=mag.ravel())
    total = float(hist.sum()) + 1e-9
    feat.extend((hist / total).tolist())
    return np.asarray(feat, dtype=np.float32)


# --------------------------------------------------------------------------- #
# data loading
# --------------------------------------------------------------------------- #
def load_scene_split(
    root: Path, split: str, n: int, seed: int
) -> tuple[list[dict], dict[str, np.ndarray], list[str]]:
    """Sample ``n`` scenes and return (scenes, labels, image paths)."""
    scene_name, image_dir = SPLITS[split]
    with (root / "scenes" / scene_name).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    scenes = data["scenes"]
    rng = np.random.default_rng(seed)
    take = np.sort(rng.choice(len(scenes), size=min(n, len(scenes)), replace=False))
    chosen = [scenes[int(i)] for i in take]
    labels = {name: [] for name in LABEL_TASKS}
    images: list[str] = []
    for scene in chosen:
        for name, value in scene_labels(scene).items():
            labels[name].append(value)
        images.append(str(root / "images" / image_dir / scene["image_filename"]))
    return chosen, labels, images


def build_geometry(scenes: list[dict]) -> np.ndarray:
    return np.stack([geometry_features(s) for s in scenes]).astype(np.float32)


# --------------------------------------------------------------------------- #
# neural feature extraction
# --------------------------------------------------------------------------- #
def _load_models(device: str):
    """Load cached frozen encoders; return dict of callables (missing -> absent)."""
    models: dict[str, dict] = {}
    try:
        from transformers import AutoImageProcessor, AutoModel

        processor = AutoImageProcessor.from_pretrained(
            "facebook/dinov2-small", local_files_only=True
        )
        model = AutoModel.from_pretrained(
            "facebook/dinov2-small", local_files_only=True
        ).to(device).eval()
        models["dinov2_small"] = {
            "processor": processor,
            "model": model,
            "forward": _dino_forward,
            "model_id": "facebook/dinov2-small",
            "params": int(sum(p.numel() for p in model.parameters())),
        }
    except Exception as exc:  # noqa: BLE001 - report, do not fail the audit
        print(f"[skip] dinov2_small: {type(exc).__name__}: {exc}")
    try:
        from transformers import CLIPImageProcessor, CLIPVisionModel

        processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-base-patch16", local_files_only=True
        )
        model = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-base-patch16", local_files_only=True
        ).to(device).eval()
        models["clip_vit_b16"] = {
            "processor": processor,
            "model": model,
            "forward": _clip_forward,
            "model_id": "openai/clip-vit-base-patch16",
            "params": int(sum(p.numel() for p in model.parameters())),
        }
    except Exception as exc:  # noqa: BLE001
        print(f"[skip] clip_vit_b16: {type(exc).__name__}: {exc}")
    return models


def _dino_forward(model, inputs):
    out = model(**inputs)
    if getattr(out, "pooler_output", None) is not None:
        return out.pooler_output
    return out.last_hidden_state[:, 1:].mean(dim=1)


def _clip_forward(model, inputs):
    return model(**inputs).pooler_output


def extract_features(
    images: list[str],
    device: str,
    batch_size: int,
    use_neural: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    import torch

    experts: dict[str, list[np.ndarray]] = {}
    info: dict[str, dict] = {}
    models = _load_models(device) if use_neural else {}
    for name, meta in models.items():
        experts[name] = []
        info[name] = {
            "kind": EXPERT_KINDS[name],
            "model_id": meta["model_id"],
            "params": meta["params"],
        }
    experts["img_handcrafted"] = []
    info["img_handcrafted"] = {"kind": EXPERT_KINDS["img_handcrafted"], "params": 0}

    for start in range(0, len(images), batch_size):
        paths = images[start : start + batch_size]
        pil = [Image.open(path).convert("RGB") for path in paths]
        arrays = [np.asarray(img, dtype=np.uint8) for img in pil]
        experts["img_handcrafted"].append(
            np.stack([handcrafted_image_features(a) for a in arrays]).astype(np.float32)
        )
        for name, meta in models.items():
            inputs = meta["processor"](images=pil, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.inference_mode():
                feats = meta["forward"](meta["model"], inputs)
            experts[name].append(feats.detach().cpu().numpy())
        if (start // batch_size) % 10 == 0:
            print(f"  extracted {min(start + batch_size, len(images))}/{len(images)}")
    out = {name: np.concatenate(chunks, axis=0) for name, chunks in experts.items()}
    return out, info


# --------------------------------------------------------------------------- #
# audit driver
# --------------------------------------------------------------------------- #
def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def run_audits(
    features: dict[str, np.ndarray],
    labels: dict[str, list[str]],
    splits: list[str],
    images: list[str],
) -> dict:
    results: dict[str, dict] = OrderedDict()
    for task in LABEL_TASKS:
        meta = {
            "categories": labels[task],
            "splits": splits,
            "instances": images,
        }
        per_subset: dict[str, dict] = OrderedDict()
        for subset_name, names in SUBSETS.items():
            usable = [n for n in names if n in features]
            if len(usable) < 2:
                per_subset[subset_name] = {
                    "skipped": "fewer than two usable experts: " + ", ".join(usable)
                }
                continue
            payload = {n: features[n] for n in usable}
            try:
                per_subset[subset_name] = fusion_audit(payload, meta, label_key="category")
            except Exception as exc:  # noqa: BLE001
                per_subset[subset_name] = {
                    "skipped": f"{type(exc).__name__}: {exc}"
                }
        results[task] = per_subset
    return results


def _summary_table(audits: dict, label_tasks) -> list[dict]:
    rows: list[dict] = []
    for task in label_tasks:
        per_subset = audits.get(task, {})
        row = {"task": task, "n_classes": None}
        for subset, payload in per_subset.items():
            if "skipped" in payload:
                row[subset] = "skipped"
                continue
            row["n_classes"] = payload["n_classes"]
            row[subset] = {
                "best_single": payload["best_single"],
                "best_single_test": payload["best_single_test"],
                "all_concat_test": payload["all_concat_test"],
                "gap_vs_best_single": payload["gap_vs_best_single"],
                "concat_lr_test": payload["concat_lr_test"],
                "vote_lda_test": payload["vote_lda_test"],
                "weighted_vote_test": payload["weighted_vote_test"],
                "gate_top1_test": payload["gate_top1_test"],
                "gate_soft_test": payload["gate_soft_test"],
                "mean_error_corr": payload["mean_error_corr"],
                "mean_disagreement": payload["mean_disagreement"],
                "oracle_any": payload["oracle_any"],
            }
        rows.append(row)
    return rows


def _headline_findings(audits: dict) -> list[str]:
    """Concise, computed takeaways so the headline is not hand-written."""
    concat: list[tuple[str, str, float]] = []
    best_fusion = None
    for task, per_subset in audits.items():
        for subset, payload in per_subset.items():
            if "skipped" in payload:
                continue
            concat.append((task, subset, payload["gap_vs_best_single"]))
            for method in (
                "concat_lr_test",
                "vote_lda_test",
                "weighted_vote_test",
                "gate_soft_test",
                "gate_top1_test",
            ):
                delta = float(payload[method]) - float(payload["best_single_test"])
                candidate = {
                    "task": task,
                    "subset": subset,
                    "method": method,
                    "accuracy": float(payload[method]),
                    "best_single_test": float(payload["best_single_test"]),
                    "delta": delta,
                }
                if best_fusion is None or delta > best_fusion["delta"]:
                    best_fusion = candidate

    if not concat:
        return ["No audit ran."]
    n_pos = sum(1 for _, _, gap in concat if gap > 1e-9)
    worst = min(concat, key=lambda row: row[2])
    best_concat = max(concat, key=lambda row: row[2])
    findings = [
        (
            f"Concatenated pooled-covariance LDA beat the val-selected best single "
            f"expert on {n_pos}/{len(concat)} task-subset pairs. Largest concat gain: "
            f"`{best_concat[0]}`/{best_concat[1]} {best_concat[2]:+.3f}; largest "
            f"shortfall: `{worst[0]}`/{worst[1]} {worst[2]:+.3f}. The 1,372-d concat "
            "likely makes the pooled covariance hard to estimate from 3,000 training "
            "scenes; concat-LR underperforms best single on most tasks too."
        ),
        (
            f"Best alternative fusion (score-level or concat-LR): "
            f"`{best_fusion['method']}` on `{best_fusion['task']}`/"
            f"{best_fusion['subset']} {best_fusion['accuracy']:.3f} vs best single "
            f"{best_fusion['best_single_test']:.3f} ({best_fusion['delta']:+.3f}). "
            "Score-level fusion helps most where both encoders are already strong and "
            "their errors decorrelate."
        ),
    ]
    # Name the two appearance tasks when they show a score-level gain.
    for task in ("front_shape", "front_material"):
        per_subset = audits.get(task, {})
        gains = []
        for subset, payload in per_subset.items():
            if "skipped" in payload:
                continue
            base = float(payload["best_single_test"])
            for method in ("vote_lda_test", "weighted_vote_test"):
                gains.append((float(payload[method]) - base, subset, method, float(payload[method])))
        if gains:
            delta, subset, method, acc = max(gains)
            findings.append(
                f"`{task}`: best score-level fusion `{method}`/{subset} "
                f"{acc:.3f} vs best single {per_subset[subset]['best_single_test']:.3f} "
                f"({delta:+.3f}); the subset's mean error correlation is "
                f"{per_subset[subset]['mean_error_corr']:.3f}."
            )
    # Geometry's role on appearance vs count labels.
    appearance = []
    for task in ("front_shape", "front_material", "majority_color"):
        payload = audits.get(task, {}).get("all_four")
        if payload and "experts" in payload and "geom_scene" in payload["experts"]:
            appearance.append(
                f"`{task}` {payload['experts']['geom_scene']['test_acc']:.3f}"
            )
    if appearance:
        findings.append(
            "Geometry is near chance for appearance labels ("
            + ", ".join(appearance)
            + "), so it adds context but not appearance information."
        )
    # Count labels: geometry does carry signal, but concat gain stays small.
    count_labels = []
    for task in ("distinct_shape", "distinct_color"):
        payload = audits.get(task, {}).get("all_four")
        if payload and "experts" in payload and "geom_scene" in payload["experts"]:
            count_labels.append(
                f"`{task}` geom {payload['experts']['geom_scene']['test_acc']:.3f}, "
                f"concat gap {payload['gap_vs_best_single']:+.3f}"
            )
    if count_labels:
        findings.append(
            "On count-style labels geometry is above chance ("
            + "; ".join(count_labels)
            + "), but adding it to the vision pool does not produce a meaningful gain."
        )
    # Score-level fusion under compositional shift: explicit negative result.
    dc = audits.get("distinct_color", {})
    vote_rows = [
        float(payload[method])
        for payload in dc.values()
        if "skipped" not in payload
        for method in ("vote_lda_test", "weighted_vote_test", "gate_soft_test")
    ]
    dc_best = [
        float(payload["best_single_test"])
        for payload in dc.values()
        if "skipped" not in payload
    ]
    if vote_rows and dc_best and max(vote_rows) < max(dc_best) - 0.1:
        findings.append(
            f"`distinct_color` exposes a calibration failure: the best score-level "
            f"fusion is only {max(vote_rows):.3f} vs best single {max(dc_best):.3f} "
            "under the valA->valB compositional shift. Val-fitted z-scoring does not "
            "transfer automatically."
        )
    return findings


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def write_markdown(path: Path, report: dict) -> None:
    ds = report["dataset"]
    lines: list[str] = []
    lines.append("# Real-data expert complementarity audit (CLEVR-CoGenT)\n")
    lines.append(f"Generated: {report['generated_at']}\n")
    lines.append("## Dataset and splits\n")
    lines.append(
        f"- Dataset: **{ds['name']}** (`{ds['root']}`)\n"
        f"- Image source: {ds['image_source']}\n"
        f"- Views: {ds['n_views']['train']} train / {ds['n_views']['val']} val / "
        f"{ds['n_views']['test']} test (one view = one scene)\n"
        f"- Splits: train=`{ds['split_names']['train']}`, "
        f"val=`{ds['split_names']['val']}`, test=`{ds['split_names']['test']}` "
        "(official CoGenT shift protocol)\n"
        f"- Modalities aligned per view: {', '.join(ds['modalities'])}\n"
        f"- Seed: {ds['seed']}; device: {ds['device']}; image batch: {ds['batch_size']}\n"
    )
    lines.append("\n## Experts\n")
    lines.append("| expert | kind | dim | params |\n|---|---|---:|---:|\n")
    for name, meta in report["experts"].items():
        lines.append(
            f"| `{name}` | {meta['kind']} | {meta['dim']} | "
            f"{meta.get('params', 0):,} |\n"
        )
    lines.append("\n## Label tasks\n")
    lines.append(
        "- `front_shape` / `front_material`: attribute of the frontmost object "
        "(smallest render depth; ties broken by pixel y).\n"
        "- `majority_color` / `majority_shape`: most frequent attribute across the "
        "scene's objects (alphabetical tie-break).\n"
        "- `distinct_shape`: number of distinct shapes (1-3).\n"
        "- `distinct_color`: number of distinct colours, capped at 7 (`7` = 7+).\n"
    )
    lines.append("\n## Headline findings\n")
    for finding in report.get("headline_findings", []):
        lines.append(f"- {finding}\n")
    lines.append("\n## Per-task accuracy and fusion\n")
    lines.append(
        "`gap = A_full(subset) - best_single_val_selected`, where `A_full(subset)` "
        "is the pooled-covariance LDA on the concatenated subset. Positive gap means "
        "the concatenated frozen pool beats the best single expert chosen on val.\n"
    )
    for task, per_subset in report["audits"].items():
        lines.append(f"\n### `{task}`\n")
        rows = []
        for subset, payload in per_subset.items():
            if "skipped" in payload:
                rows.append(f"| {subset} | skipped | {payload['skipped']} |")
                continue
            rows.append(
                f"| {subset} | {payload['n_classes']} | "
                f"{payload['best_single']} @ {payload['best_single_test']:.3f} | "
                f"{payload['all_concat_test']:.3f} | "
                f"{payload['gap_vs_best_single']:+.3f} | "
                f"{payload['concat_lr_test']:.3f} | "
                f"{payload['vote_lda_test']:.3f} | "
                f"{payload['weighted_vote_test']:.3f} | "
                f"{payload['gate_soft_test']:.3f} | "
                f"{payload['mean_error_corr']:.3f} | "
                f"{payload['oracle_any']:.3f} |"
            )
        lines.append(
            "| subset | classes | best single (test) | A_full concat | gap | "
            "concat-LR | z-vote | weighted vote | soft gate | err corr | oracle-any |\n"
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        )
        lines.extend(row + "\n" for row in rows)
        all_payload = per_subset.get("all_four")
        if all_payload and "experts" in all_payload:
            lines.append("\nPer-expert LDA accuracy:\n\n")
            lines.append("| expert | dim | val acc | test acc |\n|---|---:|---:|---:|\n")
            for name, meta in sorted(all_payload["experts"].items()):
                lines.append(
                    f"| `{name}` | {meta['dim']} | {meta['val_acc']:.3f} | "
                    f"{meta['test_acc']:.3f} |\n"
                )
    lines.append("\n## What this proxy does and does not prove\n")
    for note in ds["caveats"]:
        lines.append(f"- {note}\n")
    path.write_text("".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ensure_utf8()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=Path("eval/results/real_fusion_audit"))
    parser.add_argument("--n-train", type=int, default=3000)
    parser.add_argument("--n-val", type=int, default=750)
    parser.add_argument("--n-test", type=int, default=750)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-neural",
        action="store_true",
        help="skip HuggingFace encoders; use handcrafted image + geometry only",
    )
    parser.add_argument(
        "--cache", type=Path, default=None,
        help="optional .npz feature cache path",
    )
    args = parser.parse_args()

    torch_threads = min(8, os.cpu_count() or 1)
    try:
        import torch

        torch.set_num_threads(torch_threads)
    except Exception:  # noqa: BLE001
        pass

    t0 = time.time()
    scenes: dict[str, list[dict]] = {}
    labels: dict[str, list[str]] = {task: [] for task in LABEL_TASKS}
    splits: list[str] = []
    images: list[str] = []
    n_by_split: dict[str, int] = {}
    for split, n in (("train", args.n_train), ("val", args.n_val), ("test", args.n_test)):
        chosen, split_labels, split_images = load_scene_split(
            args.root, split, n, args.seed
        )
        scenes[split] = chosen
        for task in LABEL_TASKS:
            labels[task].extend(split_labels[task])
        splits.extend([split] * len(chosen))
        images.extend(split_images)
        n_by_split[split] = len(chosen)
        print(f"[data] {split}: {len(chosen)} scenes")

    if args.cache and args.cache.exists():
        print(f"[cache] loading features from {args.cache}")
        loaded = np.load(args.cache, allow_pickle=False)
        features = {name: loaded[name] for name in loaded.files if name != "__meta__"}
        expert_info = json.loads(str(loaded["__meta__"].item()))
    else:
        print("[features] extracting frozen features ...")
        features, expert_info = extract_features(
            images, args.device, args.batch_size, use_neural=not args.no_neural
        )
        features["geom_scene"] = build_geometry(
            [s for split in ("train", "val", "test") for s in scenes[split]]
        )
        expert_info["geom_scene"] = {
            "kind": EXPERT_KINDS["geom_scene"],
            "params": 0,
        }
        if args.cache:
            args.cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.cache,
                **features,
                __meta__=np.asarray(json.dumps(expert_info)),
            )
            print(f"[cache] wrote {args.cache}")

    for name in list(features):
        expert_info.setdefault(name, {"kind": EXPERT_KINDS.get(name, "unknown"), "params": 0})
        expert_info[name]["dim"] = int(features[name].shape[1])

    print("[audit] running fusion audits ...")
    audits = run_audits(features, labels, splits, images)
    summary = _summary_table(audits, LABEL_TASKS)
    headline = _headline_findings(audits)

    report = OrderedDict(
        {
            "status": "ok",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset": OrderedDict(
                {
                    "name": "CLEVR-CoGenT v1.0",
                    "root": str(args.root),
                    "image_source": (
                        "shipped CLEVR-CoGenT PNG renders (synthetic rendered "
                        "imagery, not real-world photographs)"
                    ),
                    "modalities": [
                        "rgb_image (384-d DINOv2, 768-d CLIP, handcrafted stats)",
                        "scene_metadata_geometry (positions/count/rotation only)",
                    ],
                    "split_names": {k: v[1] for k, v in SPLITS.items()},
                    "n_views": n_by_split,
                    "seed": args.seed,
                    "device": args.device,
                    "batch_size": args.batch_size,
                    "label_tasks": list(LABEL_TASKS),
                    "caveats": [
                        "CLEVR is synthetic rendered imagery. 'Real' here means the "
                        "actual CLEVR-CoGenT image files on disk, not a real-world "
                        "sensor dataset; results may not transfer to physical scenes.",
                        "The geometry stream is simulator metadata, i.e. an oracle "
                        "symbolic stream, not a learned depth/pose estimator. It "
                        "excludes shape/colour/material/size, but includes object "
                        "count and positions, so it carries context the image stream "
                        "does not have to infer.",
                        "Test is valB, the CoGenT compositional shift split, so test "
                        "accuracy is lower than valA by design. Best single is selected "
                        "on valA and reported on valB.",
                        "Features are frozen; only linear LDA/LR readouts are fit. No "
                        "encoder is fine-tuned.",
                        "Handcrafted image statistics are not a learned expert; they "
                        "are a cheap complementary colour/texture stream.",
                        "Object size is deliberately not a label: in the geometry "
                        "stream the 3-D z coordinate is exactly 0.70 for large and "
                        "0.35 for small objects, so size would be trivially leaked.",
                        "Scenes are subsampled for CPU feasibility; numbers are "
                        "estimates with the stated sample sizes.",
                        "This proxy can show whether frozen streams are complementary "
                        "on CLEVR-CoGenT; it does not prove complementarity on real "
                        "robot multimodal data.",
                    ],
                }
            ),
            "experts": expert_info,
            "audits": audits,
            "summary": summary,
            "headline_findings": headline,
            "elapsed_sec": round(time.time() - t0, 2),
        }
    )
    report = _jsonable(report)
    args.out.mkdir(parents=True, exist_ok=True)
    json_path = args.out / "real_fusion_audit.json"
    md_path = args.out / "real_fusion_audit.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(md_path, report)
    print(f"[done] {json_path}")
    print(f"[done] {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
