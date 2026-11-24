"""ObjectFolder 2.0 object-level cross-modal R@1.

ObjectFolder 2.0 stores each object as an implicit ObjectFile (VisionNet /
AudioNet / TouchNet). This module renders a few vision views and tactile
images per object, extracts DINOv2 features, and measures same-object
retrieval across modalities.

The task is object-level, not synchronized interaction: each object is an
instance, and a tactile image of object X should retrieve the vision image of
the same X. This is the public-data proxy for RD2's instance-level alignment.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def discover_objects(root: str | Path, limit: int | None = None) -> list[Path]:
    root = Path(root)
    objects = sorted({path.parent for path in root.rglob("ObjectFile.pth")})
    if limit is not None:
        objects = objects[: int(limit)]
    return objects


def obj_vertices(path: str | Path) -> np.ndarray:
    vertices = []
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.asarray(vertices, dtype=np.float32)


def write_touch_queries(
    object_dir: Path, out_dir: Path, touches: int = 4, seed: int = 0
) -> tuple[Path, Path] | None:
    mesh_path = None
    for name in ("model.obj", "mesh.obj", "object.obj"):
        candidate = object_dir / name
        if candidate.exists():
            mesh_path = candidate
            break
    if mesh_path is None:
        return None
    vertices = obj_vertices(mesh_path)
    if len(vertices) == 0:
        return None
    rng = np.random.default_rng(seed)
    take = rng.choice(len(vertices), size=min(touches, len(vertices)), replace=False)
    sampled = vertices[take]
    gelinfo = np.stack(
        [
            rng.uniform(0.0, np.radians(15.0), len(sampled)),
            rng.uniform(0.0, 2.0 * np.pi, len(sampled)),
            rng.uniform(0.0005, 0.0020, len(sampled)),
        ],
        axis=1,
    ).astype(np.float32)
    out_dir.mkdir(parents=True, exist_ok=True)
    vertices_path = out_dir / "touch_vertices.npy"
    gelinfo_path = out_dir / "touch_gelinfo.npy"
    np.save(vertices_path, sampled.astype(np.float32))
    np.save(gelinfo_path, gelinfo)
    return vertices_path, gelinfo_path


def render_object(
    object_dir: Path,
    renderer_dir: Path,
    vision_query: Path,
    out_dir: Path,
    touches: int = 4,
    seed: int = 0,
) -> dict:
    object_file = object_dir / "ObjectFile.pth"
    vision_dir = out_dir / "vision"
    touch_dir = out_dir / "touch"
    queries_dir = out_dir / "queries"
    if not object_file.exists():
        return {"ok": False, "error": "missing ObjectFile.pth"}
    vision_ok = vision_dir.exists() and any(vision_dir.glob("*.png"))
    existing_touch = list(touch_dir.glob("*.png")) if touch_dir.exists() else []
    touch_ok = len(existing_touch) >= touches
    if vision_ok and touch_ok:
        return {"ok": True, "skipped": True}

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    base = [
        sys.executable,
        str(renderer_dir / "OF_render.py"),
        "--object_file_path",
        str(object_file.resolve()),
    ]
    if not vision_ok:
        vision_cmd = base + [
            "--modality",
            "vision",
            "--vision_test_file_path",
            str(vision_query.resolve()),
            "--vision_results_dir",
            str(vision_dir.resolve()),
        ]
        result = subprocess.run(
            vision_cmd, cwd=renderer_dir, env=env, capture_output=True, text=True
        )
        if result.returncode != 0:
            return {"ok": False, "error": f"vision: {result.stderr[-800:]}"}

    if not touch_ok:
        for path in existing_touch:
            path.unlink()
        queries = write_touch_queries(
            object_dir, queries_dir, touches=touches, seed=seed
        )
        if queries is None:
            return {"ok": False, "error": "no mesh for touch vertices"}
        vertices_path, gelinfo_path = queries
        touch_cmd = base + [
            "--modality",
            "touch",
            "--touch_vertices_file_path",
            str(vertices_path.resolve()),
            "--touch_gelinfo_file_path",
            str(gelinfo_path.resolve()),
            "--touch_results_dir",
            str(touch_dir.resolve()),
        ]
        result = subprocess.run(
            touch_cmd, cwd=renderer_dir, env=env, capture_output=True, text=True
        )
        if result.returncode != 0:
            return {"ok": False, "error": f"touch: {result.stderr[-800:]}"}
    return {"ok": True, "skipped": False, "vision_rendered": not vision_ok, "touch_rendered": not touch_ok}


def _image_tensor(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((224, 224), Image.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(array).permute(2, 0, 1)


def _load_encoder():
    from transformers import AutoModel

    model = AutoModel.from_pretrained("facebook/dinov2-small")
    model.eval()
    return model


def _encode_paths(model, paths: list[Path], device: str, batch_size: int = 8) -> np.ndarray:
    out = []
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            batch = torch.stack([_image_tensor(p) for p in paths[start : start + batch_size]])
            outputs = model(pixel_values=batch.to(device))
            feat = outputs.pooler_output
            if feat is None:
                feat = outputs.last_hidden_state.mean(dim=1)
            out.append(feat.detach().cpu().numpy())
    return np.concatenate(out, axis=0)


def _l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def _retrieval(query: np.ndarray, gallery: np.ndarray) -> dict:
    query = _l2(query)
    gallery = _l2(gallery)
    sim = query @ gallery.T
    order = np.argsort(-sim, axis=1)
    n = len(query)
    r1 = float(np.mean(order[:, 0] == np.arange(n)))
    k = min(5, n)
    r5 = float(np.mean([k_ in order[i, :k] for i, k_ in enumerate(np.arange(n))]))
    return {"r1": r1, "r5": r5, "n": n, "chance": 1.0 / n}


def _ridge_retrieval(train_a, train_b, test_a, test_b) -> dict:
    from sklearn.linear_model import Ridge

    reg = Ridge(alpha=1.0)
    reg.fit(train_a, train_b)
    return _retrieval(reg.predict(test_a), test_b)


def run_objectfolder(
    root: str | Path,
    renderer_dir: str | Path,
    out_dir: str | Path,
    vision_query: str | Path | None = None,
    limit: int | None = None,
    touches: int = 4,
    seed: int = 0,
    render: bool = True,
    render_only: bool = False,
) -> dict:
    root = Path(root).resolve()
    renderer_dir = Path(renderer_dir).resolve()
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    vision_query = (
        Path(vision_query).resolve()
        if vision_query
        else renderer_dir / "demo" / "vision_demo.npy"
    )
    objects = discover_objects(root, limit=limit)
    if not objects:
        raise SystemExit(f"no ObjectFile.pth found under {root}")

    renders_root = out / "renders"
    render_reports = {}
    for i, object_dir in enumerate(objects):
        key = object_dir.name
        if render:
            report = render_object(
                object_dir,
                renderer_dir,
                vision_query,
                renders_root / key,
                touches=touches,
                seed=seed + i,
            )
            render_reports[key] = report
            print(f"[render] {i + 1}/{len(objects)} {key}: {report}")

    if render_only:
        return {
            "render_only": True,
            "root": str(root),
            "objects_requested": len(objects),
            "render_reports": render_reports,
        }

    names: list[str] = []
    vision_features: list[np.ndarray] = []
    touch_features: list[np.ndarray] = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_encoder().to(device)
    for object_dir in objects:
        key = object_dir.name
        vision_paths = sorted((renders_root / key / "vision").glob("*.png"))
        touch_paths = sorted((renders_root / key / "touch").glob("*.png"))
        if not vision_paths or not touch_paths:
            continue
        vision = _encode_paths(model, vision_paths, device).mean(0)
        touch = _encode_paths(model, touch_paths, device).mean(0)
        names.append(key)
        vision_features.append(vision)
        touch_features.append(touch)
    if len(names) < 2:
        raise SystemExit("need at least 2 rendered objects")
    V = np.stack(vision_features)
    T = np.stack(touch_features)

    order = np.random.default_rng(seed).permutation(len(names))
    cut = len(names) // 2
    train_idx, test_idx = order[:cut], order[cut:]

    results = {
        "root": str(root),
        "renderer": str(renderer_dir),
        "license": "CC-BY-4.0",
        "citation": "Gao et al., ObjectFolder 2.0, CVPR 2022",
        "objects_requested": len(objects),
        "objects_rendered": len(names),
        "vision_images_per_object": sorted(
            {len(list((renders_root / n / "vision").glob("*.png"))) for n in names}
        ),
        "touch_images_per_object": sorted(
            {len(list((renders_root / n / "touch").glob("*.png"))) for n in names}
        ),
        "zero_shot": {
            "vision_to_touch": _retrieval(V, T),
            "touch_to_vision": _retrieval(T, V),
        },
        "ridge_half_split": {
            "vision_to_touch": _ridge_retrieval(
                V[train_idx], T[train_idx], V[test_idx], T[test_idx]
            ),
            "touch_to_vision": _ridge_retrieval(
                T[train_idx], V[train_idx], T[test_idx], V[test_idx]
            ),
        },
        "render_reports": render_reports,
        "note": (
            "Object-level retrieval, not synchronized interaction. DINOv2 zero-shot "
            "features; Ridge is trained on half the objects and evaluated on the other half."
        ),
    }
    (out / "objectfolder_r1.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "OBJECTFOLDER_R1.md").write_text(_render_report(results), encoding="utf-8")
    return results


def _fmt(value, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _render_report(results: dict) -> str:
    lines = [
        "# ObjectFolder 2.0 object-level cross-modal R@1",
        "",
        f"- license: {results.get('license', 'CC-BY-4.0')}",
        f"- citation: {results.get('citation', 'Gao et al., ObjectFolder 2.0, CVPR 2022')}",
        f"- objects requested: {results['objects_requested']}",
        f"- objects rendered: {results['objects_rendered']}",
        f"- vision images per object: {results['vision_images_per_object']}",
        f"- touch images per object: {results['touch_images_per_object']}",
        "",
        "## Zero-shot DINOv2 features",
        "",
        "| direction | R@1 | R@5 | chance |",
        "|---|---:|---:|---:|",
    ]
    for direction, metric in results["zero_shot"].items():
        lines.append(
            f"| {direction} | {_fmt(metric['r1'])} | {_fmt(metric['r5'])} | {_fmt(metric['chance'])} |"
        )
    lines += [
        "",
        "## Ridge mapping (half objects train, half test)",
        "",
        "| direction | R@1 | R@5 | chance |",
        "|---|---:|---:|---:|",
    ]
    for direction, metric in results["ridge_half_split"].items():
        lines.append(
            f"| {direction} | {_fmt(metric['r1'])} | {_fmt(metric['r5'])} | {_fmt(metric['chance'])} |"
        )
    lines += [
        "",
        "## Caveats",
        "",
        "- Object-level pairing, not synchronized robot interaction.",
        "- DINOv2 zero-shot is a baseline; a trained tactile encoder may do better.",
        "- Rendered from implicit ObjectFiles, not raw sensor captures.",
        "",
    ]
    return "\n".join(lines)
