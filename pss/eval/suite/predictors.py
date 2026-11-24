"""Deterministic predictors for the suite smoke path.

These are baselines for plumbing, not capability claims. Replace a predictor
with a PSS-backed one without changing the runner or adapters.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from pss.eval.suite.schema import EvalEpisode, Prediction
from pss.knowledge import FactStore


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z_]+", text.lower()))


def predict_bfcl(episode: EvalEpisode) -> Prediction:
    query = episode.query or ""
    q_tokens = _tokens(query)
    best, best_score = None, -1.0
    for tool in episode.tools:
        text = " ".join(
            [
                str(tool.get("name", "")),
                str(tool.get("description", "")),
                " ".join(str(k) for k in (tool.get("parameters") or {})),
            ]
        )
        overlap = len(q_tokens & _tokens(text))
        if overlap > best_score:
            best, best_score = tool, overlap
    tool_name = best.get("name") if best else None
    params = best.get("parameters") if best else {}
    args = _extract_args(query, params or {})
    return Prediction(
        episode_id=episode.id,
        output={"tool": tool_name, "args": args},
        meta={"expected": episode.expected},
    )


def _extract_args(query: str, params: dict) -> dict:
    numbers = re.findall(r"-?\d+(?:\.\d+)?", query)
    codes = re.findall(r"\b[A-Z]{2,4}\b", query)
    paths = re.findall(r"/[^\s,;]+", query)
    quoted = re.findall(r"'([^']+)'|\"([^\"]+)\"", query)
    quoted = [a or b for a, b in quoted]
    capitals = re.findall(r"\b[A-Z][a-z]+\b", query)
    out = {}
    num_i, code_i, path_i, quote_i, cap_i = 0, 0, 0, 0, 0
    for name, spec in params.items():
        kind = str(spec).lower()
        if "int" in kind or "float" in kind or "num" in kind:
            value = float(numbers[num_i]) if num_i < len(numbers) else 0
            num_i += 1
            out[name] = int(value) if "int" in kind else value
        elif "path" in name.lower() or "dir" in name.lower() or "file" in name.lower():
            out[name] = paths[path_i] if path_i < len(paths) else ""
            path_i += 1
        elif "from" in name.lower() or "to" in name.lower() or "currency" in name.lower():
            out[name] = codes[code_i] if code_i < len(codes) else ""
            code_i += 1
        elif quote_i < len(quoted):
            out[name] = quoted[quote_i]
            quote_i += 1
        elif cap_i < len(capitals):
            out[name] = capitals[cap_i]
            cap_i += 1
        else:
            out[name] = ""
    return out


def predict_openood(episode: EvalEpisode) -> Prediction:
    features = np.asarray(episode.meta.get("features"), dtype=np.float64)
    prototypes = np.asarray(episode.meta.get("prototypes"), dtype=np.float64)
    f = features / (np.linalg.norm(features) + 1e-9)
    p = prototypes / (np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-9)
    score = float((p @ f).max())
    return Prediction(
        episode_id=episode.id,
        output={"score": score},
        meta={"expected": episode.expected},
    )


def predict_zsre(episode: EvalEpisode) -> Prediction:
    facts = {(f.entity, f.attribute): f.value for f in episode.facts}
    store = FactStore(facts)
    subject = episode.expected["subject"]
    prop = episode.expected["property"]
    old = episode.expected["old"]
    new = episode.expected["new"]
    version = store.version
    store.update(subject, prop, new, reason="suite edit")
    edited = store.get(subject, prop) == new
    locality = all(
        store.get(entity, attribute) == value
        for (entity, attribute), value in facts.items()
        if (entity, attribute) != (subject, prop)
    )
    store.rollback(version)
    rollback = store.get(subject, prop) == old
    return Prediction(
        episode_id=episode.id,
        output={"edited": edited, "locality": locality, "rollback": rollback},
        meta={"expected": episode.expected},
    )


def _clevr_relation(a, b, eps=5.0):
    """Image-space relation from GT boxes: pixel x + pixel depth."""
    ax, bx = float(a[0]), float(b[0])
    ad, bd = float(a[2]), float(b[2])
    if ax < bx - eps:
        return "left"
    if ax > bx + eps:
        return "right"
    if ad < bd - 1e-6:
        return "front"
    return "behind"


def predict_clevr(episode: EvalEpisode) -> Prediction:
    objects = episode.meta.get("objects", [])
    pair = episode.meta.get("pair", [0, 1])
    i, j = int(pair[0]), int(pair[1])
    relation = None
    if len(objects) > max(i, j):
        a = objects[i].get("pixel") or objects[i].get("pos")
        b = objects[j].get("pixel") or objects[j].get("pos")
        relation = _clevr_relation(a, b)
    expected = episode.expected.get("relation")
    answer = relation == expected
    return Prediction(
        episode_id=episode.id,
        output={"answer": answer, "relation": relation},
        meta={"expected": episode.expected, "split": episode.split},
    )


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def predict_tao(episode: EvalEpisode) -> Prediction:
    from scipy.optimize import linear_sum_assignment

    meta = episode.meta
    assignments = {}
    next_id = 1
    active: list[tuple[int, list]] = []
    frames = meta.get("frames")
    if not frames:
        # Sample episodes: boxes map image_id -> {ann_id: bbox}.
        boxes_by_image = meta.get("boxes", {})
        frame_ids = sorted(boxes_by_image)
    else:
        boxes_by_image = {}
        anns = meta.get("annotations", {})
        for im in frames:
            image_id = str(im["id"])
            rows = anns.get(image_id) or anns.get(im["id"]) or []
            boxes_by_image[im["id"]] = {str(a["id"]): a["bbox"] for a in rows}
        frame_ids = [im["id"] for im in sorted(frames, key=lambda x: x.get("frame_index", 0))]

    for image_id in frame_ids:
        boxes = boxes_by_image.get(image_id, {})
        ids = list(boxes)
        if not boxes:
            continue
        matrix = np.zeros((len(active), len(ids)))
        for i, (_, bbox) in enumerate(active):
            for j, ann_id in enumerate(ids):
                matrix[i, j] = _iou(bbox, boxes[ann_id])
        matched = {}
        if active and ids:
            rows, cols = linear_sum_assignment(-matrix)
            for r, c in zip(rows, cols):
                if matrix[r, c] > 0.1:
                    matched[int(c)] = active[int(r)][0]
        new_active = []
        for j, ann_id in enumerate(ids):
            if j in matched:
                pred_id = matched[j]
            else:
                pred_id = next_id
                next_id += 1
            assignments[image_id] = assignments.get(image_id, {})
            assignments[image_id][str(ann_id)] = pred_id
            new_active.append((pred_id, boxes[ann_id]))
        active = new_active
    return Prediction(
        episode_id=episode.id,
        output={"assignments": assignments},
        meta={"episode": meta},
    )


def predict_episode(episode: EvalEpisode) -> Prediction:
    if episode.benchmark == "clevr_cogent":
        return predict_clevr(episode)
    if episode.benchmark == "tao":
        return predict_tao(episode)
    if episode.benchmark == "bfcl":
        return predict_bfcl(episode)
    if episode.benchmark == "openood":
        return predict_openood(episode)
    if episode.benchmark == "zsre":
        return predict_zsre(episode)
    raise KeyError(f"no predictor for benchmark {episode.benchmark!r}")


def predict_tao_pss(episode: EvalEpisode, device: str | None = None, config=None) -> Prediction:
    """TAO association with the PSS tracker (appearance + pose).

    GT boxes are used as detections (association benchmark); crops are encoded
    with frozen DINOv2 to provide appearance. This tests the PSS entity layer,
    not a detector.
    """
    import torch
    from PIL import Image

    from pss.encoders.data import normalize_image
    from pss.encoders.vision.dinov2 import load_dinov2_encoder
    from pss.memory import SceneGraph, Tracker, TrackerConfig
    from pss.memory.perception import Observation

    meta = episode.meta
    frames_dir = meta.get("frames_dir")
    frames = meta.get("frames") or []
    anns = meta.get("annotations") or {}
    if not frames_dir or not frames:
        return Prediction(episode_id=episode.id, output={"assignments": {}}, meta={"episode": meta})
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    encoder = _get_dino_encoder(device)
    tracker = Tracker(
        config
        or TrackerConfig(
            gate_radius=90.0, w_pose=0.4, w_app=0.6, w_category=0.0, max_cost=1.0
        ),
        SceneGraph(),
    )
    assignments: dict[int, dict[str, int]] = {}
    for im in sorted(frames, key=lambda x: x.get("frame_index", 0)):
        rows = anns.get(str(im["id"])) or anns.get(im["id"]) or []
        if not rows:
            continue
        path = Path(frames_dir) / im["file_name"]
        if not path.exists():
            continue
        image = Image.open(path).convert("RGB")
        crops, obs = [], []
        sx = image.width / float(im.get("width") or image.width)
        sy = image.height / float(im.get("height") or image.height)
        for ann in rows:
            x, y, w, h = ann["bbox"]
            x, y, w, h = x * sx, y * sy, w * sx, h * sy
            box = (max(0, int(x)), max(0, int(y)), min(image.width, int(x + w)), min(image.height, int(y + h)))
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            crop = image.crop(box).resize((224, 224), Image.BILINEAR)
            crops.append(normalize_image(np.asarray(crop, dtype=np.uint8)))
            obs.append(
                {
                    "ann": str(ann["id"]),
                    "pos": np.array([x + w / 2.0, y + h / 2.0, 0.0]),
                }
            )
        if not obs:
            continue
        with torch.inference_mode():
            feats = encoder.encode(torch.as_tensor(np.stack(crops), dtype=torch.float32)).cpu().numpy()
        observations = [
            Observation(
                gt_id=item["ann"],
                pos=item["pos"],
                quat=np.array([1.0, 0.0, 0.0, 0.0]),
                appearance=feats[i],
                raw={},
                confidence=1.0,
                category=None,
                frame=int(im.get("frame_index", 0)),
            )
            for i, item in enumerate(obs)
        ]
        events = tracker.step(observations, int(im.get("frame_index", 0)))
        mapping = {}
        for key in ("matched", "new", "reid"):
            for tid, gid in events.get(key, []):
                # Tracker IDs are strings like obj_0001; metrics use integers.
                mapping[str(gid)] = int(str(tid).split("_")[-1])
        assignments[int(im["id"])] = {
            str(ann["id"]): mapping[str(ann["id"])]
            for ann in rows
            if str(ann["id"]) in mapping
        }
    return Prediction(
        episode_id=episode.id,
        output={"assignments": assignments},
        meta={"episode": meta},
    )


_DINO = {}


def _get_dino_encoder(device: str):
    if device not in _DINO:
        from pss.encoders.vision.dinov2 import load_dinov2_encoder

        _DINO[device] = load_dinov2_encoder(device)
    return _DINO[device]
