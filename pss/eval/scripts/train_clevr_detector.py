"""Train a tiny CLEVR center detector (CenterNet-lite) and evaluate on valA/valB.

The detector predicts object centers and a depth value per center from the
image. It is trained on trainA with center/depth supervision, then used to
replace GT `pixel_coords` in the relation/property pipeline.

No GT boxes are used at test time; GT centers are used only to score detection
recall and to attach labels for evaluation.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment

from pss._console import ensure_utf8
from pss.encoders.data import normalize_image
from pss.encoders.vision.dinov2 import load_dinov2_encoder
from pss.eval.scripts.run_clevr_image import (
    ATTRS,
    RELATIONS,
    _fit_head,
    _predict,
    _scenes,
    color_feat,
    property_data,
    relation_examples,
    train_relation,
)

ROOT = Path("data/public/clevr_cogent/CLEVR_CoGenT_v1.0")
CKPT = ROOT / "detector_centernet.pt"
OUT = Path("eval/results/clevr_detector")
SPLITS = {"traina": "trainA", "vala": "valA", "valb": "valB"}
INPUT = 256
STRIDE = 4
HM = INPUT // STRIDE  # 64
CROP = 64


class CenterNetLite(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, 3, 2, 1), nn.BatchNorm2d(96), nn.ReLU(inplace=True),
            nn.Conv2d(96, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(128, 96, 3, 1, 1), nn.BatchNorm2d(96), nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, 1, 1), nn.BatchNorm2d(96), nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(96, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.heat = nn.Conv2d(64, 1, 1)
        self.depth = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x = self.stem(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.up1(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.up2(x)
        return self.heat(x), self.depth(x)


def _load_images(split: str, max_scenes: int):
    scenes = _scenes(split)[:max_scenes]
    images = np.zeros((len(scenes), INPUT, INPUT, 3), dtype=np.uint8)
    centers, depths = [], []
    for i, scene in enumerate(scenes):
        image = Image.open(ROOT / "images" / SPLITS[split] / scene["image_filename"]).convert("RGB")
        images[i] = np.asarray(image.resize((INPUT, INPUT), Image.BILINEAR), dtype=np.uint8)
        cs, ds = [], []
        for obj in scene.get("objects", []):
            x, y, z = obj.get("pixel_coords", [0, 0, 0])
            cs.append([float(x) * INPUT / image.width, float(y) * INPUT / image.height])
            ds.append(float(z))
        centers.append(np.asarray(cs, dtype=np.float32))
        depths.append(np.asarray(ds, dtype=np.float32))
        if (i + 1) % 1000 == 0:
            print(f"  load {split}: {i + 1}/{len(scenes)}", flush=True)
    return images, centers, depths, scenes


def _targets(centers, depths):
    hm = torch.zeros(HM, HM)
    dep = torch.zeros(HM, HM)
    yy, xx = torch.meshgrid(torch.arange(HM), torch.arange(HM), indexing="ij")
    for (cx, cy), z in zip(centers, depths):
        gx, gy = cx / STRIDE, cy / STRIDE
        sigma = 1.2
        g = torch.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * sigma ** 2))
        hm = torch.maximum(hm, g)
        ix, iy = int(round(gx)), int(round(gy))
        if 0 <= ix < HM and 0 <= iy < HM:
            dep[iy, ix] = float(z) / 10.0
    return hm, dep


def train_detector(max_train: int, epochs: int, device: str, batch: int = 32):
    print(f"training detector: {max_train} scenes, {epochs} epochs", flush=True)
    images, centers, depths, _ = _load_images("traina", max_train)
    n = len(images)
    model = CenterNetLite().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        perm = np.random.permutation(n)
        total = 0.0
        for start in range(0, n, batch):
            idx = perm[start:start + batch]
            xb = torch.as_tensor(images[idx], dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255.0
            xb = (xb - 0.5) / 0.25
            hms, deps = zip(*[_targets(centers[i], depths[i]) for i in idx])
            hm_t = torch.stack(hms).to(device)
            dep_t = torch.stack(deps).to(device)
            with torch.cuda.amp.autocast(enabled=device == "cuda"):
                hm_logits, dep_pred = model(xb)
                pos_weight = torch.tensor(50.0, device=device)
                hm_loss = F.binary_cross_entropy_with_logits(hm_logits[:, 0], hm_t, pos_weight=pos_weight)
                center_mask = dep_t > 0
                dep_loss = F.l1_loss(dep_pred[:, 0][center_mask], dep_t[center_mask]) if center_mask.any() else hm_loss * 0
                loss = hm_loss + dep_loss
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(opt)
            scaler.update()
            total += float(loss.detach()) * len(idx)
        print(f"  epoch {epoch + 1}/{epochs} loss {total / n:.4f} ({time.time() - t0:.1f}s)", flush=True)
    torch.save(model.state_dict(), CKPT)
    print(f"saved {CKPT}", flush=True)
    return model


def load_detector(device: str):
    model = CenterNetLite().to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.eval()
    return model


@torch.inference_mode()
def detect_image(model, image: Image.Image, device: str, threshold: float = 0.3, topk: int = 20):
    arr = np.asarray(image.resize((INPUT, INPUT), Image.BILINEAR), dtype=np.uint8)
    x = torch.as_tensor(normalize_image(arr), dtype=torch.float32).unsqueeze(0).to(device)
    hm_logits, dep = model(x)
    hm = torch.sigmoid(hm_logits)[0, 0]
    dep = dep[0, 0]
    pool = F.max_pool2d(hm[None, None], 3, 1, 1)[0, 0]
    peaks = (hm == pool) & (hm > threshold)
    ys, xs = torch.where(peaks)
    if len(xs) == 0:
        return [], []
    scores = hm[ys, xs]
    order = torch.argsort(scores, descending=True)[:topk]
    ys, xs = ys[order], xs[order]
    centers, depths = [], []
    for y, x in zip(ys.tolist(), xs.tolist()):
        cx = (x + 0.5) * STRIDE * image.width / INPUT
        cy = (y + 0.5) * STRIDE * image.height / INPUT
        centers.append([cx, cy])
        depths.append(float(dep[y, x]) * 10.0)
    return centers, depths


def _match_centers(pred, gts, threshold=14.0):
    if not pred or not gts:
        return {}
    cost = np.linalg.norm(np.asarray(gts)[:, None, :] - np.asarray(pred)[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(cost)
    return {int(r): (int(c), float(cost[r, c])) for r, c in zip(rows, cols) if cost[r, c] <= threshold}


def extract_cache(model, split: str, max_scenes: int, device: str):
    scenes = _scenes(split)[:max_scenes]
    encoder = load_dinov2_encoder(device)
    emb, colors, pixels, attrs, scene_ids, obj_ids, errs = [], [], [], [], [], [], []
    recalls, precisions = [], []
    for si, scene in enumerate(scenes):
        image = Image.open(ROOT / "images" / SPLITS[split] / scene["image_filename"]).convert("RGB")
        centers, depths = detect_image(model, image, device)
        gts = [obj.get("pixel_coords", [0, 0, 0])[:2] for obj in scene.get("objects", [])]
        matched = _match_centers(centers, gts)
        recalls.append(len(matched) / max(1, len(gts)))
        precisions.append(len(matched) / max(1, len(centers)))
        if not matched:
            continue
        crops, meta = [], []
        for gi, (pi, err) in matched.items():
            cx, cy = centers[pi]
            half = CROP // 2
            box = (max(0, int(cx - half)), max(0, int(cy - half)),
                   min(image.width, int(cx + half)), min(image.height, int(cy + half)))
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            crop = image.crop(box).resize((224, 224), Image.BILINEAR)
            crops.append(np.asarray(crop, dtype=np.uint8))
            meta.append({"gi": gi, "pixel": [cx, cy, depths[pi]], "err": err})
        if not crops:
            continue
        xs = np.stack([normalize_image(c) for c in crops])
        z = encoder.encode(torch.as_tensor(xs, dtype=torch.float32).to(device)).cpu().numpy()
        for k, m in enumerate(meta):
            obj = scene["objects"][m["gi"]]
            emb.append(z[k])
            colors.append(color_feat(Image.fromarray(crops[k]), (112, 112, 0), half=112))
            pixels.append(m["pixel"])
            attrs.append([obj.get(a) for a in ATTRS])
            scene_ids.append(si)
            obj_ids.append(m["gi"])
            errs.append(m["err"])
        if (si + 1) % 1000 == 0:
            print(f"  detect {split}: {si + 1}/{len(scenes)}", flush=True)
    return {
        "embeddings": np.stack(emb).astype(np.float32),
        "color": np.stack(colors).astype(np.float32),
        "pixel": np.stack(pixels).astype(np.float32),
        "attrs": attrs,
        "scene": np.asarray(scene_ids),
        "obj": np.asarray(obj_ids),
        "err": np.asarray(errs),
        "scenes": scenes,
        "recall": float(np.mean(recalls)),
        "precision": float(np.mean(precisions)),
        "center_error": float(np.mean(errs)) if errs else float("nan"),
    }


def _fmt(v, digits=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{float(v):.{digits}f}"


def evaluate(model, device: str, max_train: int, max_val: int):
    train_cache = extract_cache(model, "traina", max_train, device)
    tv, tp, ty = relation_examples(train_cache)
    relation_model = train_relation(torch.cat([tv, tp], 1), ty)
    trainE, trainC, trainL, _ = property_data(train_cache)
    heads = {}
    for a in ATTRS:
        classes = sorted(set(trainL[a]))
        cmap = {v: i for i, v in enumerate(classes)}
        yy = torch.as_tensor([cmap[v] for v in trainL[a]], dtype=torch.long)
        source = torch.as_tensor(trainC) if a == "color" else torch.as_tensor(trainE)
        heads[a] = (_fit_head(source, yy, len(classes)), classes)

    results = {}
    for split in ("vala", "valb"):
        cache = extract_cache(model, split, max_val, device)
        Xv, Xp, y = relation_examples(cache)
        pred = relation_model(torch.cat([Xv, Xp], 1)).argmax(1)
        rel_acc = float((pred == y).float().mean())
        per_rel = {RELATIONS[k]: float((pred[y == k] == y[y == k]).float().mean()) for k in range(len(RELATIONS)) if (y == k).any()}
        E, C, L, _ = property_data(cache)
        X, Xc = torch.as_tensor(E), torch.as_tensor(C)
        exact, factor = [], {}
        for a in ATTRS:
            head, classes = heads[a]
            pr = _predict(head, Xc if a == "color" else X)
            factor[a] = np.array([classes[int(i)] for i in pr])
            exact.append(factor[a] == np.array(L[a]))
        exact_all = np.stack(exact, 1).all(1)
        results[split] = {
            "detection_recall": cache["recall"],
            "detection_precision": cache["precision"],
            "center_error_px": cache["center_error"],
            "n_objects": len(cache["embeddings"]),
            "relation_accuracy": rel_acc,
            "relation_per_class": per_rel,
            "property_exact": float(exact_all.mean()),
            "property_per_attribute": {a: float(np.mean(factor[a] == np.array(L[a]))) for a in ATTRS},
        }
    return results


def run(device="cuda", max_train=8000, max_val=3000, epochs=25, train_only=False):
    if CKPT.exists():
        model = load_detector(device)
        print(f"loaded {CKPT}", flush=True)
    else:
        model = train_detector(max_train, epochs, device)
    if train_only:
        return {}
    results = evaluate(model, device, max_train, max_val)
    payload = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "detector": "CenterNet-lite (center + depth), trained on trainA",
        "checkpoint": str(CKPT),
        "results": results,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "clevr_centernet.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# CLEVR with trained detector (no GT boxes)",
        "",
        f"> generated: {payload['generated']}",
        "",
        "| split | det recall | det precision | center err (px) | relation | property exact | shape | color | material | size |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, row in results.items():
        p = row["property_per_attribute"]
        lines.append(
            f"| {split} | {_fmt(row['detection_recall'])} | {_fmt(row['detection_precision'])} | "
            f"{_fmt(row['center_error_px'], 2)} | {_fmt(row['relation_accuracy'])} | {_fmt(row['property_exact'])} | "
            f"{_fmt(p['shape'])} | {_fmt(p['color'])} | {_fmt(p['material'])} | {_fmt(p['size'])} |"
        )
    lines += [
        "",
        "Detector: CenterNet-lite (center heatmap + depth head), trained on trainA.",
        "Test crops are fixed 64x64 around detected centers; GT centers are used",
        "only for detection scoring and label attachment.",
        "",
    ]
    (OUT / "clevr_centernet.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def main():
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--train", type=int, default=8000)
    ap.add_argument("--val", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--train-only", action="store_true")
    args = ap.parse_args()
    payload = run(args.device, args.train, args.val, args.epochs, args.train_only)
    if payload:
        for split, row in payload["results"].items():
            print(split, row)
        print(f"output: {OUT}")


if __name__ == "__main__":
    main()
