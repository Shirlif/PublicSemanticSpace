"""Vision-expert bake-off on real image files: CLEVR-CoGenT.

The synthetic category benchmark feeds simulator ``.npz`` samples to every
expert.  This script asks the same question on *actual image files*: given the
same object crop, how do the task-trained SmallCNN, a frozen DINOv2 tower and a
zero-parameter handcrafted descriptor compare on accuracy, shift robustness and
cost?

Data and protocol
-----------------
CLEVR-CoGenT ships 480x320 PNG renders plus per-scene object metadata.  The
official compositional-shift protocol is used:

- train: ``CLEVR_trainA`` / ``trainA``
- in-distribution test: ``CLEVR_valA`` / ``valA``
- compositional shift: ``CLEVR_valB`` / ``valB`` (unseen shape/colour/material
  combinations)

Each view is the object crop around the frontmost object (smallest render
depth).  Labels come from the scene metadata: ``front_shape``,
``front_material`` and ``majority_color``.  Features are frozen; only the
linear probe is fit, so this is an expert comparison, not a training run.

Important: CLEVR is rendered imagery, not a real-world photograph dataset.
It exercises real PNG files and a distribution shift, but it is still a
synthetic renderer.  The report says so; do not present it as real-robot data.

Example:
    python -m pss.eval.scripts.run_vision_real_bench \
        --out eval/results/real_vision_bench --n-train 2000 --n-val 800 --n-test 800
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
import zipfile
from collections import Counter, OrderedDict, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pss._console import ensure_utf8
from pss.encoders.data import normalize_image
from pss.encoders.vision import load_vision_expert
from pss.encoders.vision.handcrafted import handcrafted_features

DEFAULT_ROOT = Path("data/public/clevr_cogent/CLEVR_CoGenT_v1.0")
SPLITS = OrderedDict(
    [
        ("train", ("CLEVR_trainA_scenes.json", "trainA")),
        ("val", ("CLEVR_valA_scenes.json", "valA")),
        ("shift", ("CLEVR_valB_scenes.json", "valB")),
    ]
)
LABEL_TASKS = OrderedDict(
    [
        ("front_shape", lambda objs: _front(objs)["shape"]),
        ("front_material", lambda objs: _front(objs)["material"]),
        ("majority_color", lambda objs: _majority(objs, "color")),
        ("front_size", lambda objs: _front(objs)["size"]),
    ]
)


def _front(objs: list[dict]) -> dict:
    """Frontmost object = smallest render depth; ties broken by pixel y."""
    return min(objs, key=lambda o: (float(o["pixel_coords"][2]), float(o["pixel_coords"][1])))


def _majority(objs: list[dict], key: str) -> str:
    counts = Counter(o[key] for o in objs)
    top = max(counts.values())
    return sorted(value for value, count in counts.items() if count == top)[0]


def _fit_probe(train_x, train_y):
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced"),
    )
    clf.fit(train_x, train_y)
    return clf


def _crop_front(scene: dict, image_path: Path, crop: int = 128) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    px, py, _ = (float(v) for v in _front(scene["objects"])["pixel_coords"])
    w, h = img.size
    half = crop // 2
    x0 = max(0, int(px) - half)
    y0 = max(0, int(py) - half)
    x1 = min(w, x0 + crop)
    y1 = min(h, y0 + crop)
    x0 = max(0, x1 - crop)
    y0 = max(0, y1 - crop)
    return np.asarray(img.crop((x0, y0, x1, y1)), dtype=np.uint8)


def load_split(root: Path, split: str, n: int, seed: int) -> list[tuple[Path, dict]]:
    scene_file, image_dir = SPLITS[split]
    data = json.loads((root / "scenes" / scene_file).read_text(encoding="utf-8"))
    scenes = data["scenes"]
    rng = np.random.default_rng(seed)
    take = np.sort(rng.choice(len(scenes), size=min(n, len(scenes)), replace=False))
    return [
        (root / "images" / image_dir / scenes[int(i)]["image_filename"], scenes[int(i)])
        for i in take
    ]


TAO_ROOT = Path("data/public/tao")
TAO_ANN = TAO_ROOT / "annotations/TAO_val/gt/gt.json"
TAO_ZIP = TAO_ROOT / "charades/Charades_v1_480.zip"
TAO_RAW = TAO_ROOT / "charades/raw"


def _tao_extract_video(zf: zipfile.ZipFile, name: str) -> Path:
    member = next((n for n in zf.namelist() if n.endswith(f"/{name}.mp4")), None)
    if member is None:
        raise FileNotFoundError(f"{name}.mp4 not in {TAO_ZIP}")
    TAO_RAW.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(suffix=".mp4", dir=TAO_RAW, delete=False)
    with zf.open(member) as src:
        handle.write(src.read())
    handle.close()
    return Path(handle.name)


def _tao_crop(frame: np.ndarray, bbox, margin: float = 0.1) -> np.ndarray | None:
    x, y, w, h = (float(v) for v in bbox)
    mx, my = w * margin, h * margin
    x0 = max(0, int(x - mx))
    y0 = max(0, int(y - my))
    x1 = min(frame.shape[1], int(x + w + mx))
    y1 = min(frame.shape[0], int(y + h + my))
    if x1 <= x0 or y1 <= y0:
        return None
    return np.asarray(frame[y0:y1, x0:x1], dtype=np.uint8)


def load_tao_items(
    min_samples: int = 80,
    min_per_split: int = 10,
    max_per_class: int = 300,
    seed: int = 0,
) -> tuple[list, list, list, dict]:
    """Real Charades photographs from TAO, split by video, cropped by GT box."""
    import cv2

    data = json.loads(TAO_ANN.read_text(encoding="utf-8"))
    images = {im["id"]: im for im in data["images"]}
    videos = {v["id"]: v for v in data["videos"]}
    categories = {c["id"]: str(c["name"]) for c in data["categories"]}
    charades = {
        v["id"] for v in data["videos"] if v.get("metadata", {}).get("dataset") == "Charades"
    }

    rows: list[dict] = []
    for ann in data["annotations"]:
        image = images.get(ann.get("image_id"))
        if image is None or image.get("video_id") not in charades:
            continue
        name = categories.get(ann.get("category_id"))
        if not name:
            continue
        x, y, w, h = (float(v) for v in ann.get("bbox", [0, 0, 0, 0]))
        if w * h < 32 * 32:
            continue
        video = videos.get(image["video_id"], {})
        rows.append(
            {
                "category": name,
                "video_id": image["video_id"],
                "video_name": str(video.get("name", "")).split("/")[-1],
                "frame_index": int(image.get("frame_index", 0)),
                "bbox": (x, y, w, h),
                # Annotations use the original video resolution; the local
                # Charades copy is 480p, so crops must be scaled to the frame.
                "image_size": (
                    float(image.get("width", 0) or 0),
                    float(image.get("height", 0) or 0),
                ),
            }
        )

    counts = Counter(row["category"] for row in rows)
    keep = {name for name, count in counts.items() if count >= min_samples}
    rows = [row for row in rows if row["category"] in keep]

    # Global video split: no video appears in two splits, so no scene leakage.
    video_ids = sorted({row["video_id"] for row in rows})
    rng = np.random.default_rng(seed)
    rng.shuffle(video_ids)
    n = len(video_ids)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    split_of: dict[int, str] = {}
    for i, video_id in enumerate(video_ids):
        split_of[video_id] = (
            "train" if i < n_train else ("val" if i < n_train + n_val else "shift")
        )

    per_split = {
        split: Counter(
            row["category"] for row in rows if split_of[row["video_id"]] == split
        )
        for split in ("train", "val", "shift")
    }
    common = {
        name
        for name in keep
        if per_split["train"][name] >= min_samples
        and per_split["val"][name] >= min_per_split
        and per_split["shift"][name] >= min_per_split
    }
    rows = [row for row in rows if row["category"] in common]

    # Cap per (split, class) before touching the videos.
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(split_of[row["video_id"]], row["category"])].append(row)
    selected: list[dict] = []
    for key, group in grouped.items():
        group.sort(key=lambda r: (r["video_id"], r["frame_index"]))
        selected.extend(group[:max_per_class])

    items: dict[str, list[tuple[np.ndarray, dict]]] = {
        "train": [],
        "val": [],
        "shift": [],
    }
    by_video: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in selected:
        by_video[(split_of[row["video_id"]], row["video_id"])].append(row)

    with zipfile.ZipFile(TAO_ZIP) as zf:
        for (split, video_id), video_rows in sorted(by_video.items()):
            name = video_rows[0]["video_name"]
            path = _tao_extract_video(zf, name)
            try:
                cap = cv2.VideoCapture(str(path))
                if not cap.isOpened():
                    continue
                by_frame: dict[int, list[dict]] = defaultdict(list)
                for row in video_rows:
                    by_frame[row["frame_index"]].append(row)
                for frame_index, frame_rows in sorted(by_frame.items()):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        continue
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    for row in frame_rows:
                        iw, ih = row["image_size"]
                        sx = frame.shape[1] / iw if iw > 0 else 1.0
                        sy = frame.shape[0] / ih if ih > 0 else 1.0
                        x, y, w, h = row["bbox"]
                        crop = _tao_crop(rgb, (x * sx, y * sy, w * sx, h * sy))
                        if crop is None or crop.size == 0:
                            continue
                        items[split].append((crop, {"category": row["category"]}))
                cap.release()
            finally:
                path.unlink(missing_ok=True)

    present = Counter(
        row["category"] for row in selected if split_of[row["video_id"]] == "train"
    )
    meta = {
        "dataset_label": "TAO / Charades real photographs",
        "input_note": "Real-world video frames from Charades, cropped by TAO GT boxes.",
        "splits_note": "Video-level split (70/15/15): held-out test, not a domain shift.",
        "shift_label": "test",
        "classes": dict(sorted(present.items())),
        "n_videos": len(video_ids),
    }
    return items["train"], items["val"], items["shift"], meta


# ── feature extractors ──────────────────────────────────────────────────────


class SmallCNNFeatures:
    name = "smallcnn_v0"
    kind = "task_trained_cnn"

    def __init__(
        self,
        checkpoint: str | Path = "checkpoints/category_pretrain_experts/vision.pt",
        device: str = "cpu",
    ) -> None:
        import torch

        self.torch = torch
        self.expert = load_vision_expert(checkpoint, device=device)
        self.image_size = int(self.expert.image_size)
        self.device = device
        self.params = int(sum(p.numel() for p in self.expert.encoder.parameters()))

    def encode(self, crops: list[np.ndarray], batch_size: int = 64) -> np.ndarray:
        torch = self.torch
        out = []
        self.expert.encoder.eval()
        with torch.inference_mode():
            for start in range(0, len(crops), batch_size):
                batch = crops[start : start + batch_size]
                xs = []
                for crop in batch:
                    img = Image.fromarray(crop).resize(
                        (self.image_size, self.image_size), Image.BILINEAR
                    )
                    xs.append(normalize_image(np.asarray(img, dtype=np.uint8)))
                xb = torch.as_tensor(np.stack(xs), device=self.device)
                out.append(self.expert.encoder(xb).detach().cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)

    def encode_one(self, crop: np.ndarray) -> np.ndarray:
        return self.encode([crop], batch_size=1)[0]


class Dinov2Features:
    name = "dinov2_small"
    kind = "frozen_pretrained_vision"

    def __init__(self, device: str = "cpu", model_id: str = "facebook/dinov2-small"):
        from transformers import AutoImageProcessor, AutoModel

        self.processor = AutoImageProcessor.from_pretrained(
            model_id, local_files_only=True
        )
        self.model = AutoModel.from_pretrained(model_id, local_files_only=True).to(device).eval()
        self.device = device
        self.model_id = model_id
        self.params = int(sum(p.numel() for p in self.model.parameters()))

    def encode(self, crops: list[np.ndarray], batch_size: int = 32) -> np.ndarray:
        import torch

        out = []
        with torch.inference_mode():
            for start in range(0, len(crops), batch_size):
                pil = [Image.fromarray(c) for c in crops[start : start + batch_size]]
                inputs = self.processor(images=pil, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                feat = self.model(**inputs)
                feat = (
                    feat.pooler_output
                    if getattr(feat, "pooler_output", None) is not None
                    else feat.last_hidden_state[:, 1:].mean(dim=1)
                )
                out.append(feat.detach().cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)

    def encode_one(self, crop: np.ndarray) -> np.ndarray:
        return self.encode([crop], batch_size=1)[0]


class HandcraftedFeatures:
    name = "handcrafted_v0"
    kind = "handcrafted_image_statistics"

    def __init__(self, size: int = 64) -> None:
        self.size = int(size)
        self.params = 0

    def encode(self, crops: list[np.ndarray], batch_size: int = 256) -> np.ndarray:
        return np.stack(
            [handcrafted_features(c, size=self.size) for c in crops]
        ).astype(np.float32)

    def encode_one(self, crop: np.ndarray) -> np.ndarray:
        return handcrafted_features(crop, size=self.size)


# ── evaluation ──────────────────────────────────────────────────────────────


def _task_report(
    train_labels: list[str],
    val_labels: list[str],
    shift_labels: list[str],
    features: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> dict:
    rows: dict[str, dict] = {}
    for name, (xtr, xva, xte) in features.items():
        clf = _fit_probe(xtr, train_labels)
        pred_val = clf.predict(xva)
        pred_shift = clf.predict(xte)
        rows[name] = {
            "val_top1": float(accuracy_score(val_labels, pred_val)),
            "shift_top1": float(accuracy_score(shift_labels, pred_shift)),
            "shift_macro_f1": float(
                f1_score(shift_labels, pred_shift, average="macro", zero_division=0)
            ),
        }
    classes = sorted(set(train_labels))
    majority = Counter(train_labels).most_common(1)[0][1] / len(train_labels)
    return {
        "n_train": len(train_labels),
        "n_val": len(val_labels),
        "n_shift": len(shift_labels),
        "classes": classes,
        "train_class_counts": dict(Counter(train_labels)),
        "majority_baseline": float(majority),
        "experts": rows,
    }


def _measure_latency(encoder, crops: list[np.ndarray], repeats: int = 10) -> dict:
    for _ in range(min(3, repeats)):
        encoder.encode_one(crops[0])
    times = []
    for i in range(max(1, repeats)):
        crop = crops[i % len(crops)]
        t0 = time.perf_counter()
        encoder.encode_one(crop)
        times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "latency_ms_p50": float(np.percentile(times, 50)),
        "latency_ms_p95": float(np.percentile(times, 95)),
        "params": int(encoder.params),
    }


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


def _fmt(value, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def render_markdown(payload: dict) -> str:
    shift_label = payload.get("shift_label", "shift")
    lines = [
        f"# Vision-expert bake-off: {payload['dataset_label']}",
        "",
        f"> Generated: {payload['generated']}",
        f"> Dataset: {payload['input_note']}",
        f"> Root: `{payload['root']}`",
        f"> Splits: {payload['splits_note']}",
        f"> Views: {payload['n_train']} / {payload['n_val']} / {payload['n_shift']}"
        f" (train / val / {shift_label}).",
        f"> Device: `{payload['device']}`; seed: {payload['seed']}.",
        "",
        "## Experts",
        "",
        "| expert | kind | dim | params | params source |",
        "|---|---|---:|---:|---|",
    ]
    for name, meta in payload["experts"].items():
        lines.append(
            f"| `{name}` | {meta['kind']} | {meta['dim']} | {meta['params']:,} "
            f"| {meta.get('model_id', 'checkpoint')} |"
        )
    lines += [
        "",
        "## Accuracy by label task",
        "",
        f"`Δ` = {shift_label} − val; a larger negative Δ means the frozen feature"
        " transfers worse to the held-out split.",
        "",
        f"| task | classes | majority | expert | val top-1 | {shift_label} top-1 "
        f"| {shift_label} macro-F1 | Δ |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for task, report in payload["tasks"].items():
        for name, row in report["experts"].items():
            lines.append(
                f"| `{task}` | {len(report['classes'])} | {_fmt(report['majority_baseline'])} "
                f"| `{name}` | {_fmt(row['val_top1'])} | {_fmt(row['shift_top1'])} "
                f"| {_fmt(row['shift_macro_f1'])} "
                f"| {_fmt(row['shift_top1'] - row['val_top1'], 3)} |"
            )
    lines += [
        "",
        "## Class distribution",
        "",
    ]
    for task, report in payload["tasks"].items():
        lines.append(f"- `{task}`: {report['train_class_counts']}")
    lines += [
        "",
        "## Cost (batch=1, object crop)",
        "",
        "| expert | p50 (ms) | p95 (ms) | params |",
        "|---|---:|---:|---:|",
    ]
    # Latency is measured once per expert, not per task.
    for name, cost in payload["cost"].items():
        lines.append(
            f"| `{name}` | {_fmt(cost['latency_ms_p50'])} | {_fmt(cost['latency_ms_p95'])} "
            f"| {cost['params']:,} |"
        )
    lines += [
        "",
        "## Automatic reading",
        "",
    ]
    for task, report in payload["tasks"].items():
        ranked = sorted(
            report["experts"].items(), key=lambda kv: kv[1]["shift_top1"], reverse=True
        )
        best = ranked[0]
        hand = report["experts"].get("handcrafted_v0")
        cnn = report["experts"].get("smallcnn_v0")
        dino = report["experts"].get("dinov2_small")
        parts = [
            f"- `{task}`: best on {shift_label} = `{best[0]}` "
            f"({_fmt(best[1]['shift_top1'])})."
        ]
        if hand and cnn and dino:
            parts.append(
                f" handcrafted {_fmt(hand['shift_top1'])}, SmallCNN "
                f"{_fmt(cnn['shift_top1'])}, DINOv2 {_fmt(dino['shift_top1'])};"
                f" val→{shift_label} drop "
                f"{_fmt(hand['val_top1'] - hand['shift_top1'])} / "
                f"{_fmt(cnn['val_top1'] - cnn['shift_top1'])} / "
                f"{_fmt(dino['val_top1'] - dino['shift_top1'])}."
            )
        lines.append("".join(parts))
    lines += [
        "",
        "## Boundaries",
        "",
    ]
    if payload["dataset"] == "clevr":
        lines += [
            "- CLEVR-CoGenT is rendered, not real-world photography. The real-robot /",
            "  real-sensor gap (RD2) is still open.",
            "- Labels are metadata-derived attributes of the frontmost object, not a",
            "  robot task label.",
            "- Crops are centered on the annotated object centre; occlusion/background",
            "  composition is not controlled.",
        ]
    else:
        lines += [
            "- TAO/Charades is real photography, but the labels are coarse object",
            "  categories from tracking annotations, not a robot task label.",
            "- Crops come from GT boxes with a 10% margin; detection errors are not",
            "  modelled.",
            "- Only Charades (the subset with local video files) is used; the wider",
            "  TAO split has more domains but no local frames.",
        ]
    lines += ["", ""]
    return "\n".join(lines)


def _clevr_items(root: Path, n_train: int, n_val: int, n_shift: int, crop: int, seed: int):
    splits = {
        "train": load_split(root, "train", n_train, seed),
        "val": load_split(root, "val", n_val, seed),
        "shift": load_split(root, "shift", n_shift, seed),
    }
    out: dict[str, list[tuple[np.ndarray, dict]]] = {}
    for split, items in splits.items():
        out[split] = [
            (
                _crop_front(scene, path, crop),
                {name: str(fn(scene["objects"])) for name, fn in LABEL_TASKS.items()},
            )
            for path, scene in items
        ]
    meta = {
        "dataset_label": "CLEVR-CoGenT (rendered PNGs)",
        "input_note": "Shipped CLEVR-CoGenT PNG renders; not real-world photographs.",
        "splits_note": "trainA -> train, valA -> in-distribution val, valB -> compositional shift.",
        "shift_label": "shift",
        "classes": {},
        "n_videos": None,
    }
    return out["train"], out["val"], out["shift"], meta


def run(args) -> dict:
    root = Path(args.root)
    if args.dataset == "clevr":
        train_items, val_items, shift_items, meta = _clevr_items(
            root, args.n_train, args.n_val, args.n_test, args.crop, args.seed
        )
    else:
        train_items, val_items, shift_items, meta = load_tao_items(
            min_samples=args.min_samples,
            min_per_split=args.min_per_split,
            max_per_class=args.max_per_class,
            seed=args.seed,
        )

    tasks_to_run = list(args.tasks) if args.tasks else list(train_items[0][1])
    for split, items in (("train", train_items), ("val", val_items), ("shift", shift_items)):
        if not items:
            raise SystemExit(
                f"empty {split} split after filtering; lower --min-samples/"
                "--min-per-split or increase the data pool"
            )
    for task in tasks_to_run:
        if task not in train_items[0][1]:
            raise SystemExit(
                f"unknown task {task!r} for dataset {args.dataset!r}; "
                f"choose from {list(train_items[0][1])}"
            )

    train_crops = [crop for crop, _ in train_items]
    val_crops = [crop for crop, _ in val_items]
    shift_crops = [crop for crop, _ in shift_items]

    encoders = [
        SmallCNNFeatures(device=args.device),
        Dinov2Features(device=args.device),
        HandcraftedFeatures(size=64),
    ]
    features: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    dims: dict[str, int] = {}
    for enc in encoders:
        print(f"[extract] {enc.name}")
        xtr = enc.encode(train_crops)
        xva = enc.encode(val_crops)
        xte = enc.encode(shift_crops)
        features[enc.name] = (xtr, xva, xte)
        dims[enc.name] = int(xtr.shape[1])
        print(f"  {enc.name}: {xtr.shape}")

    labels = {
        split: {task: [row[task] for _, row in items] for task in tasks_to_run}
        for split, items in (
            ("train", train_items),
            ("val", val_items),
            ("shift", shift_items),
        )
    }

    tasks: dict[str, dict] = OrderedDict()
    for task in tasks_to_run:
        tasks[task] = _task_report(
            labels["train"][task],
            labels["val"][task],
            labels["shift"][task],
            features,
        )

    cost = {}
    for enc in encoders:
        cost[enc.name] = _measure_latency(enc, val_crops[:8], repeats=args.repeats)

    payload = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "dataset": args.dataset,
        "dataset_label": meta["dataset_label"],
        "input_note": meta["input_note"],
        "splits_note": meta["splits_note"],
        "shift_label": meta.get("shift_label", "shift"),
        "root": str(root) if args.dataset == "clevr" else str(TAO_ROOT),
        "device": args.device,
        "seed": int(args.seed),
        "n_train": len(train_items),
        "n_val": len(val_items),
        "n_shift": len(shift_items),
        "crop": int(args.crop),
        "classes": meta.get("classes", {}),
        "experts": {
            enc.name: {
                "kind": enc.kind,
                "dim": dims[enc.name],
                "params": int(enc.params),
                "model_id": getattr(enc, "model_id", "checkpoint"),
            }
            for enc in encoders
        },
        "tasks": tasks,
        "cost": cost,
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "real_vision_bench.json").write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "REAL_VISION_BENCH.md").write_text(render_markdown(payload), encoding="utf-8")
    print(f"Wrote {out / 'REAL_VISION_BENCH.md'}")
    return payload


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="clevr", choices=["clevr", "tao"])
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--out", default="eval/results/real_vision_bench")
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-val", type=int, default=800)
    ap.add_argument("--n-test", type=int, default=800)
    ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--min-samples", type=int, default=80)
    ap.add_argument("--min-per-split", type=int, default=10)
    ap.add_argument("--max-per-class", type=int, default=300)
    ap.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="label tasks; defaults to all dataset tasks (TAO uses 'category')",
    )
    args = ap.parse_args()
    if args.device == "auto":
        import torch

        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    run(args)


if __name__ == "__main__":
    main()
