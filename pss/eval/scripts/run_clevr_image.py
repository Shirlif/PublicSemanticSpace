"""CLEVR-CoGenT image-based relation and property-composition evaluation.

Step 1: image crops -> frozen DINOv2 -> relation head; compare with the
pixel-geometry rule and a vision-only variant.
Step 2: object property composition trainA -> valB; compare a joint
combination classifier with a factorized multi-head classifier.

GT boxes (pixel_coords) are used to crop objects; this is a detection-oracle
image baseline, not a detector. Raw data stays under data/public/.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from pss._console import ensure_utf8
from pss.encoders.data import normalize_image
from pss.encoders.vision.dinov2 import load_dinov2_encoder

ROOT = Path("data/public/clevr_cogent/CLEVR_CoGenT_v1.0")
OUT = "eval/results/clevr_image"
CACHE = ROOT / "cache"
RELATIONS = ("left", "right", "front", "behind")
REL_INDEX = {r: i for i, r in enumerate(RELATIONS)}
ATTRS = ("shape", "color", "material", "size")
MAX_TRAIN = 8000
MAX_VAL = 3000
HALF = 32
SIZE = 224


def _fmt(v, digits=3):
    if v is None:
        return "n/a"
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


def _scenes(split: str):
    name = {"traina": "CLEVR_trainA_scenes.json", "vala": "CLEVR_valA_scenes.json", "valb": "CLEVR_valB_scenes.json"}[split]
    payload = json.loads((ROOT / "scenes" / name).read_text(encoding="utf-8"))
    return payload.get("scenes", [])


def _image_path(split: str, filename: str) -> Path:
    directory = {"traina": "trainA", "vala": "valA", "valb": "valB"}[split]
    return ROOT / "images" / directory / filename


def _crop(image: Image.Image, pixel, half: int = HALF) -> np.ndarray:
    x, y = float(pixel[0]), float(pixel[1])
    box = (max(0, int(x - half)), max(0, int(y - half)),
           min(image.width, int(x + half)), min(image.height, int(y + half)))
    if box[2] <= box[0] or box[3] <= box[1]:
        box = (0, 0, image.width, image.height)
    crop = image.crop(box).resize((SIZE, SIZE), Image.BILINEAR)
    return normalize_image(np.asarray(crop, dtype=np.uint8))


def color_feat(image: Image.Image, pixel, half: int = HALF) -> np.ndarray:
    """HSV/RGB statistics of the crop; robust to shape-color combinations."""
    x, y = float(pixel[0]), float(pixel[1])
    box = (max(0, int(x - half)), max(0, int(y - half)),
           min(image.width, int(x + half)), min(image.height, int(y + half)))
    if box[2] <= box[0] or box[3] <= box[1]:
        box = (0, 0, image.width, image.height)
    crop = np.asarray(image.crop(box).resize((48, 48), Image.BILINEAR), dtype=np.float32) / 255.0
    hsv = np.asarray(Image.fromarray((crop * 255).astype(np.uint8)).convert("HSV"), dtype=np.float32) / 255.0
    h, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = sat > 0.25
    if mask.sum() < 5:
        mask = np.ones_like(sat, dtype=bool)
    hist = np.histogram(h[mask], bins=12, range=(0, 1))[0].astype(np.float32)
    hist /= hist.sum() + 1e-9
    flat = crop.reshape(-1, 3)
    return np.concatenate(
        [hist, [sat.mean(), val.mean(), sat.std(), val.std(), float(mask.mean())], flat.mean(0), flat.std(0)]
    ).astype(np.float32)


def _enrich_color(payload: dict) -> dict:
    scenes = payload["scenes"]
    scene_ids = payload["scene"]
    obj_ids = payload["obj"]
    pixels = payload["pixel"]
    by: dict[int, list[tuple[int, int]]] = {}
    for idx, (si, oi) in enumerate(zip(scene_ids, obj_ids)):
        by.setdefault(int(si), []).append((idx, int(oi)))
    colors = np.zeros((len(scene_ids), 23), dtype=np.float32)
    for si, items in by.items():
        image = Image.open(_image_path(payload.get("split", "vala"), scenes[si]["image_filename"])).convert("RGB")
        for idx, _oi in items:
            colors[idx] = color_feat(image, pixels[idx])
    payload["color"] = colors
    return payload


def extract_split(split: str, max_scenes: int, device: str, force: bool = False):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{split}_{max_scenes}.pt"
    if path.exists() and not force:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if "color" not in payload:
            payload = _enrich_color(payload)
            torch.save(payload, path)
        return payload
    encoder = load_dinov2_encoder(device=device)
    scenes = _scenes(split)[:max_scenes]
    embs, scene_ids, obj_ids = [], [], []
    pixels, attrs, color_feats = [], [], []
    for si, scene in enumerate(scenes):
        image = Image.open(_image_path(split, scene["image_filename"])).convert("RGB")
        crops = [_crop(image, obj.get("pixel_coords", [0, 0, 0])) for obj in scene.get("objects", [])]
        colors = [color_feat(image, obj.get("pixel_coords", [0, 0, 0])) for obj in scene.get("objects", [])]
        if not crops:
            continue
        with torch.inference_mode():
            z = encoder.encode(torch.as_tensor(np.stack(crops), dtype=torch.float32).to(device))
        embs.append(z.cpu().numpy())
        for oi, obj in enumerate(scene.get("objects", [])):
            scene_ids.append(si)
            obj_ids.append(oi)
            pixels.append(list(map(float, obj.get("pixel_coords", [0, 0, 0]))))
            attrs.append([obj.get(a) for a in ATTRS])
            color_feats.append(colors[oi])
        if (si + 1) % 1000 == 0:
            print(f"  {split}: {si + 1}/{len(scenes)}", flush=True)
    payload = {
        "embeddings": np.concatenate(embs, axis=0).astype(np.float32),
        "scene": np.asarray(scene_ids, dtype=np.int32),
        "obj": np.asarray(obj_ids, dtype=np.int32),
        "pixel": np.asarray(pixels, dtype=np.float32),
        "attrs": attrs,
        "color": np.stack(color_feats).astype(np.float32),
        "scenes": scenes,
        "split": split,
    }
    torch.save(payload, path)
    return payload


def _opposite(rel: str) -> str:
    return {"left": "right", "right": "left", "front": "behind", "behind": "front"}[rel]


def _true_pair(scene):
    relationships = scene.get("relationships") or {}
    objects = scene.get("objects", [])
    for i in range(len(objects)):
        for pred in RELATIONS:
            rows = relationships.get(pred)
            if rows is None or i >= len(rows):
                continue
            for j in rows[i]:
                j = int(j)
                if j != i and j < len(objects):
                    return pred, (j, i)  # j is pred of i
    return None, None


def relation_examples(cache):
    by_scene: dict[int, dict[int, int]] = {}
    for idx, (s, o) in enumerate(zip(cache["scene"], cache["obj"])):
        by_scene.setdefault(int(s), {})[int(o)] = idx
    E = cache["embeddings"]
    P = cache["pixel"]
    Xv, Xp, y = [], [], []
    for si, scene in enumerate(cache["scenes"]):
        pred, pair = _true_pair(scene)
        if pred is None or si not in by_scene:
            continue
        i, j = pair
        objs = by_scene[si]
        if i not in objs or j not in objs:
            continue
        for subj, obj, label in ((i, j, pred), (j, i, _opposite(pred))):
            a, b = E[objs[subj]], E[objs[obj]]
            Xv.append(np.concatenate([a, b, a - b, a * b]))
            dx = (P[objs[subj], 0] - P[objs[obj], 0]) / 100.0
            dy = (P[objs[subj], 1] - P[objs[obj], 1]) / 100.0
            dd = (P[objs[subj], 2] - P[objs[obj], 2]) / 10.0
            Xp.append([dx, dy, dd])
            y.append(REL_INDEX[label])
    return torch.as_tensor(np.stack(Xv), dtype=torch.float32), torch.as_tensor(np.stack(Xp), dtype=torch.float32), torch.as_tensor(y, dtype=torch.long)


class RelationMLP(nn.Module):
    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, len(RELATIONS)),
        )

    def forward(self, x):
        return self.net(x)


def train_relation(X, y, epochs=30, seed=0):
    torch.manual_seed(seed)
    model = RelationMLP(X.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(X))
        for start in range(0, len(perm), 512):
            idx = perm[start:start + 512]
            loss = F.cross_entropy(model(X[idx]), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


def eval_relation(model, X, y):
    model.eval()
    with torch.no_grad():
        pred = model(X).argmax(1)
    return float((pred == y).float().mean()), pred


def geometry_relation(Xp):
    dx, dy, dd = Xp[:, 0], Xp[:, 1], Xp[:, 2]
    pred = torch.where(dx < 0, torch.zeros_like(dx, dtype=torch.long), torch.ones_like(dx, dtype=torch.long))
    pred = torch.where(dx.abs() > 0.05, pred, torch.where(dd < 0, torch.full_like(pred, 2), torch.full_like(pred, 3)))
    return pred


# ── property composition ────────────────────────────────────────────────────


def property_data(cache):
    E = cache["embeddings"]
    C = cache.get("color")
    labels = {a: [] for a in ATTRS}
    combos = []
    for row in cache["attrs"]:
        for k, a in enumerate(ATTRS):
            labels[a].append(row[k])
        combos.append(tuple(row))
    return E, C, {a: np.asarray(v) for a, v in labels.items()}, combos


def _fit_head(X, y, n_classes, epochs=30, seed=0):
    torch.manual_seed(seed)
    head = nn.Linear(X.shape[1], n_classes)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for start in range(0, len(perm), 512):
            idx = perm[start:start + 512]
            loss = F.cross_entropy(head(X[idx]), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    head.eval()
    return head


def _predict(head, X):
    with torch.no_grad():
        return head(X).argmax(1)


def run(device="cuda", max_train=MAX_TRAIN, max_val=MAX_VAL, force=False) -> dict:
    torch.manual_seed(0)
    caches = {
        "traina": extract_split("traina", max_train, device, force),
        "vala": extract_split("vala", max_val, device, force),
        "valb": extract_split("valb", max_val, device, force),
    }
    relation = {}
    data = {}
    for split, cache in caches.items():
        Xv, Xp, y = relation_examples(cache)
        data[split] = (Xv, Xp, y)
    train_v, train_p, train_y = data["traina"]
    for name, use_pos in (("vision", False), ("vision_pos", True)):
        X = torch.cat([train_v, train_p], 1) if use_pos else train_v
        model = train_relation(X, train_y)
        relation[name] = {}
        for split in ("vala", "valb"):
            Xv, Xp, y = data[split]
            Xte = torch.cat([Xv, Xp], 1) if use_pos else Xv
            acc, _ = eval_relation(model, Xte, y)
            relation[name][split] = {"accuracy": acc, "n": len(y)}
    # Geometry rule on val splits.
    geometry = {}
    for split in ("vala", "valb"):
        _, Xp, y = data[split]
        pred = geometry_relation(Xp)
        geometry[split] = {"accuracy": float((pred == y).float().mean()), "n": len(y)}

    # Property composition.
    prop = {
        "joint": {},
        "factorized_dino": {},
        "factorized_color": {},
        "per_attribute": {},
        "color_only": {},
    }
    trainE, trainC, trainL, train_combos = property_data(caches["traina"])
    seen = {c: i for i, c in enumerate(sorted(set(train_combos)))}
    y_joint = torch.as_tensor([seen[c] for c in train_combos], dtype=torch.long)
    joint_head = _fit_head(torch.as_tensor(trainE), y_joint, len(seen))
    heads_dino, heads_color = {}, {}
    for a in ATTRS:
        classes = sorted(set(trainL[a]))
        cmap = {v: i for i, v in enumerate(classes)}
        y = torch.as_tensor([cmap[v] for v in trainL[a]], dtype=torch.long)
        heads_dino[a] = (_fit_head(torch.as_tensor(trainE), y, len(classes)), classes)
        # Color is trained on the color histogram; other attributes stay on DINOv2.
        source = torch.as_tensor(trainC) if a == "color" and trainC is not None else torch.as_tensor(trainE)
        heads_color[a] = (_fit_head(source, y, len(classes)), classes)
    inv_joint = {i: c for c, i in seen.items()}
    for split in ("vala", "valb"):
        E, C, L, combos = property_data(caches[split])
        X = torch.as_tensor(E)
        Xc = torch.as_tensor(C) if C is not None else X
        pj = _predict(joint_head, X)
        joint_exact = float(np.mean([inv_joint[int(i)] == c for i, c in zip(pj, combos)]))
        prop["joint"][split] = {"exact": joint_exact, "n": len(combos)}
        for label, heads, features in (
            ("factorized_dino", heads_dino, X),
            ("factorized_color", heads_color, None),
        ):
            exact = []
            factor_pred = {}
            for a in ATTRS:
                head, classes = heads[a]
                source = Xc if (label == "factorized_color" and a == "color") else X
                pred = _predict(head, source)
                factor_pred[a] = [classes[int(i)] for i in pred]
                exact.append(np.array(factor_pred[a]) == np.array(L[a]))
            exact_all = np.stack(exact, 1).all(1)
            prop[label][split] = {"exact": float(exact_all.mean()), "n": len(combos)}
            if label == "factorized_color":
                prop["per_attribute"][split] = {
                    a: float(np.mean(np.array(factor_pred[a]) == np.array(L[a]))) for a in ATTRS
                }
        head_dino, classes_dino = heads_dino["color"]
        pred_dino = _predict(head_dino, X)
        head_hist, classes_hist = heads_color["color"]
        pred_hist = _predict(head_hist, Xc)
        prop["color_only"][split] = {
            "dino": float(np.mean(np.array([classes_dino[int(i)] for i in pred_dino]) == np.array(L["color"]))),
            "hist": float(np.mean(np.array([classes_hist[int(i)] for i in pred_hist]) == np.array(L["color"]))),
        }

    payload = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "counts": {k: {"scenes": len(c["scenes"]), "objects": len(c["embeddings"])} for k, c in caches.items()},
        "relation": {"geometry": geometry, **relation},
        "property": prop,
    }
    out = Path(OUT)
    out.mkdir(parents=True, exist_ok=True)
    (out / "clevr_image.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "clevr_image.md").write_text(render(payload), encoding="utf-8")
    return payload


def render(payload: dict) -> str:
    lines = [
        "# CLEVR-CoGenT image relation + property composition",
        "",
        f"> generated: {payload['generated']}",
        f"> counts: {payload['counts']}",
        "",
        "## Relation (valA / valB)",
        "",
        "| method | valA | valB |",
        "|---|---:|---:|",
    ]
    rel = payload["relation"]
    for name in ("geometry", "vision", "vision_pos"):
        row = rel[name]
        lines.append(
            f"| {name} | {_fmt(row['vala']['accuracy'])} | {_fmt(row['valb']['accuracy'])} |"
        )
    lines += [
        "",
        "## Property composition (trainA -> valA / valB)",
        "",
        "| method | valA exact | valB exact |",
        "|---|---:|---:|",
    ]
    for name in ("joint", "factorized_dino", "factorized_color"):
        row = payload["property"][name]
        lines.append(f"| {name} | {_fmt(row['vala']['exact'])} | {_fmt(row['valb']['exact'])} |")
    lines += [
        "",
        "### Color head: DINOv2 vs histogram",
        "",
        "| split | DINOv2 | histogram |",
        "|---|---:|---:|",
    ]
    for split in ("vala", "valb"):
        row = payload["property"]["color_only"][split]
        lines.append(f"| {split} | {_fmt(row['dino'])} | {_fmt(row['hist'])} |")
    lines += [
        "",
        "### Factorized per-attribute accuracy (color histogram head)",
        "",
        "| split | shape | color | material | size |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in ("vala", "valb"):
        row = payload["property"]["per_attribute"][split]
        lines.append(
            f"| {split} | {_fmt(row['shape'])} | {_fmt(row['color'])} | "
            f"{_fmt(row['material'])} | {_fmt(row['size'])} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- GT boxes (`pixel_coords`) are used to crop objects; this is a detection-oracle",
        "  image baseline, not a detector.",
        "- `vision` uses crop embeddings only; `vision_pos` also uses image-space",
        "  dx/dy/depth.",
        "- Property `joint` predicts a seen combination; `factorized_dino` uses",
        "  DINOv2 for every attribute; `factorized_color` uses the color histogram",
        "  for color and DINOv2 for shape/material/size. CoGenT valB contains",
        "  unseen combinations.",
        "",
    ]
    return "\n".join(lines)


def main():
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--train", type=int, default=MAX_TRAIN)
    ap.add_argument("--val", type=int, default=MAX_VAL)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    payload = run(args.device, args.train, args.val, args.force)
    for name in ("geometry", "vision", "vision_pos"):
        row = payload["relation"][name]
        print(f"{name}: valA {row['vala']['accuracy']:.3f} valB {row['valb']['accuracy']:.3f}")
    for name in ("joint", "factorized_dino", "factorized_color"):
        row = payload["property"][name]
        print(f"property {name}: valA {row['vala']['exact']:.3f} valB {row['valb']['exact']:.3f}")
    print(f"output: {OUT}")


if __name__ == "__main__":
    main()
