"""TAO Charades person tracking with a real detector instead of GT boxes.

Detector: torchvision Faster R-CNN MobileNetV3-320 pretrained on COCO, person
class only. GT boxes are used only for scoring detection recall/precision and
for attaching track labels in the IDF1/retention/switch metrics.

References on the same person-only subset:
  - GT boxes + PSS tracker (oracle boxes)
  - detected boxes + IoU tracker
  - detected boxes + PSS tracker
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

os.environ.setdefault("TORCH_HOME", "D:/hf_cache/torch")

from pss._console import ensure_utf8
from pss.eval.suite.adapters.tao import TAOAdapter, _idf1, _retention, _switches
from pss.memory import SceneGraph, Tracker, TrackerConfig
from pss.memory.perception import Observation

ROOT = Path("data/public/tao")
ANN = ROOT / "annotations/TAO_val/gt/gt.json"
FRAMES = ROOT / "frames"
OUT = Path("eval/results/tao_detector")
PERSON_CAT = 805  # TAO person.n.01 (name "baby", synonyms include person)
VIDEOS = {2162: "21WN7", 2142: "IONV9", 2134: "J7BOV", 2141: "OEIR9", 2160: "UR55B"}

_DETECTOR = None
_DINO = None


def _get_detector():
    global _DETECTOR
    if _DETECTOR is None:
        from torchvision.models.detection import (
            FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
            fasterrcnn_mobilenet_v3_large_320_fpn,
        )

        weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        _DETECTOR = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights).eval()
    return _DETECTOR


def _get_dino(device):
    global _DINO
    if _DINO is None:
        from pss.encoders.vision.dinov2 import load_dinov2_encoder

        _DINO = load_dinov2_encoder(device)
    return _DINO


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _match(dets, gts, threshold=0.5):
    if not dets or not gts:
        return {}
    sim = np.zeros((len(gts), len(dets)))
    for i, g in enumerate(gts):
        for j, d in enumerate(dets):
            sim[i, j] = _iou(g, d)
    from scipy.optimize import linear_sum_assignment

    rows, cols = linear_sum_assignment(-sim)
    return {int(r): (int(c), float(sim[r, c])) for r, c in zip(rows, cols) if sim[r, c] >= threshold}


def _detect_person(image: Image.Image, threshold: float = 0.3):
    model = _get_detector()
    arr = np.asarray(image, dtype=np.uint8).copy()
    x = torch.as_tensor(arr, dtype=torch.float32).permute(2, 0, 1) / 255.0
    with torch.inference_mode():
        out = model([x])[0]
    boxes, scores = [], []
    for lab, score, box in zip(out["labels"], out["scores"], out["boxes"]):
        if int(lab) != 1 or float(score) < threshold:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(float(score))
    return boxes, scores


def _person_rows(data, video_id):
    by_image = {}
    for ann in data["annotations"]:
        if ann["video_id"] == video_id and ann["category_id"] == PERSON_CAT:
            by_image.setdefault(ann["image_id"], []).append(ann)
    frames = [im for im in data["images"] if im["video_id"] == video_id]
    frames = sorted(frames, key=lambda x: x.get("frame_index", 0))
    return frames, by_image


def _gt_person(rows, im, image):
    sx = image.width / float(im.get("width") or image.width)
    sy = image.height / float(im.get("height") or image.height)
    out = {}
    for ann in rows:
        x, y, w, h = ann["bbox"]
        out[str(ann["id"])] = [x * sx, y * sy, w * sx, h * sy]
    return out


def _pss_track(records, device):
    encoder = _get_dino(device)
    tracker = Tracker(
        TrackerConfig(
            gate_radius=300, w_pose=0.2, w_app=0.8, w_category=0.0, max_cost=1.2,
            reid_pose_gate=300.0, reid_app_min=0.2, reid_max_age=1000,
            dormancy_missed=1, missing_missed=3, historical_missed=10,
        ),
        SceneGraph(),
    )
    assignments = {}
    for rec in records:
        image, boxes, scores, frame_index = rec["image"], rec["boxes"], rec["scores"], rec["frame_index"]
        if not boxes:
            continue
        crops = []
        for box in boxes:
            x, y, w, h = box
            crop_box = (max(0, int(x)), max(0, int(y)), min(image.width, int(x + w)), min(image.height, int(y + h)))
            if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
                crop_box = (0, 0, image.width, image.height)
            crops.append(image.crop(crop_box).resize((224, 224), Image.BILINEAR))
        from pss.encoders.data import normalize_image

        xs = np.stack([normalize_image(np.asarray(c, dtype=np.uint8)) for c in crops])
        with torch.inference_mode():
            feats = encoder.encode(torch.as_tensor(xs, dtype=torch.float32).to(device)).cpu().numpy()
        obs = [
            Observation(
                gt_id=f"det_{i}",
                pos=np.array([boxes[i][0] + boxes[i][2] / 2.0, boxes[i][1] + boxes[i][3] / 2.0, 0.0]),
                quat=np.array([1.0, 0.0, 0.0, 0.0]),
                appearance=feats[i],
                raw={"score": scores[i]},
                confidence=scores[i],
                category=None,
                frame=frame_index,
            )
            for i in range(len(boxes))
        ]
        events = tracker.step(obs, frame_index)
        mapping = {}
        for key in ("matched", "new", "reid"):
            for tid, gid in events.get(key, []):
                mapping[str(gid)] = int(str(tid).split("_")[-1])
        assignments[rec["image_id"]] = {f"det_{i}": mapping[f"det_{i}"] for i in range(len(boxes)) if f"det_{i}" in mapping}
    return assignments


def _iou_track(records):
    assignments = {}
    active = []
    next_id = 1
    for rec in records:
        boxes = rec["boxes"]
        if not boxes:
            continue
        matched = {}
        if active:
            cost = np.zeros((len(active), len(boxes)))
            for i, (_, abox) in enumerate(active):
                for j, box in enumerate(boxes):
                    cost[i, j] = _iou(abox, box)
            from scipy.optimize import linear_sum_assignment

            rows, cols = linear_sum_assignment(-cost)
            for r, c in zip(rows, cols):
                if cost[r, c] >= 0.3:
                    matched[int(c)] = active[int(r)][0]
        new_active = []
        row = {}
        for j, box in enumerate(boxes):
            tid = matched.get(j, next_id)
            if j not in matched:
                next_id += 1
            row[f"det_{j}"] = tid
            new_active.append((tid, box))
        assignments[rec["image_id"]] = row
        active = new_active
    return assignments


def _attach_gt(records, assignments):
    """Map GT person annotation ids to predicted detection track ids by IoU."""
    out = {}
    det_hits = det_total = gt_total = 0
    for rec in records:
        gts = rec["gt"]
        dets = rec["boxes"]
        gt_total += len(gts)
        det_total += len(dets)
        matched = _match(dets, list(gts.values()), 0.5)
        row = {}
        gt_ids = list(gts)
        pred = assignments.get(rec["image_id"], {})
        for gi, (di, _iou_val) in matched.items():
            ann_id = gt_ids[gi]
            tid = pred.get(f"det_{di}")
            det_hits += 1
            if tid is not None:
                row[ann_id] = tid
        out[rec["image_id"]] = row
    recall = det_hits / max(1, gt_total)
    precision = det_hits / max(1, det_total)
    return out, recall, precision


def _metrics(gt, assignments):
    return {
        "idf1": _idf1(gt, assignments),
        "id_retention": _retention(gt, assignments),
        "id_switches": _switches(gt, assignments),
    }


def run(device="cuda", threshold=0.3):
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    data = json.loads(ANN.read_text(encoding="utf-8"))
    results = {}
    det_stats = []
    for video_id, short in VIDEOS.items():
        frames, by_image = _person_rows(data, video_id)
        records = []
        for im in frames:
            rows = by_image.get(im["id"], [])
            path = FRAMES / im["file_name"]
            if not rows or not path.exists():
                continue
            image = Image.open(path).convert("RGB")
            boxes, scores = _detect_person(image, threshold)
            records.append(
                {
                    "image": image,
                    "image_id": str(im["id"]),
                    "frame_index": int(im.get("frame_index", 0)),
                    "boxes": boxes,
                    "scores": scores,
                    "gt": _gt_person(rows, im, image),
                    "gt_track": {str(ann["id"]): int(ann["track_id"]) for ann in rows},
                }
            )
        if not records:
            continue
        gt_boxes = {r["image_id"]: r["gt"] for r in records}
        gt = {r["image_id"]: r["gt_track"] for r in records}
        t0 = time.time()
        iou_assign = _iou_track(records)
        pss_assign = _pss_track(records, device)
        iou_gt, recall, precision = _attach_gt(records, iou_assign)
        pss_gt, _, _ = _attach_gt(records, pss_assign)
        # Oracle: GT person boxes + PSS tracker (same association code).
        oracle_records = [
            {**r, "boxes": list(r["gt"].values()), "scores": [1.0] * len(r["gt"])} for r in records
        ]
        oracle_assign = _pss_track(oracle_records, device)
        oracle_gt, _, _ = _attach_gt(oracle_records, oracle_assign)
        results[f"{video_id}_{short}"] = {
            "frames": len(records),
            "gt_person_instances": sum(len(v) for v in gt.values()),
            "detector_recall": recall,
            "detector_precision": precision,
            "iou_detector": _metrics(gt, iou_gt),
            "pss_detector": _metrics(gt, pss_gt),
            "pss_gt_boxes": _metrics(gt, oracle_gt),
            "seconds": time.time() - t0,
        }
        det_stats.append((recall, precision))
        print(f"{short}: {results[f'{video_id}_{short}']}", flush=True)
    summary = {}
    for key in ("iou_detector", "pss_detector", "pss_gt_boxes"):
        for metric in ("idf1", "id_retention", "id_switches"):
            vals = [r[key][metric] for r in results.values()]
            summary[f"{key}/{metric}"] = float(np.mean(vals)) if vals else None
    payload = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "detector": "Faster R-CNN MobileNetV3-320 FPN (COCO person), score>=%.2f" % threshold,
        "videos": len(results),
        "detection_recall": float(np.mean([x[0] for x in det_stats])) if det_stats else None,
        "detection_precision": float(np.mean([x[1] for x in det_stats])) if det_stats else None,
        "summary": summary,
        "per_video": results,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tao_detector.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# TAO Charades: detector vs GT boxes (person-only)",
        "",
        f"> generated: {payload['generated']}",
        f"> videos: {payload['videos']}; person detection recall {payload['detection_recall']:.3f}, precision {payload['detection_precision']:.3f}",
        "",
        "| method | IDF1 | ID retention | ID switches |",
        "|---|---:|---:|---:|",
    ]
    for key, label in (
        ("iou_detector", "detected boxes + IoU"),
        ("pss_detector", "detected boxes + PSS"),
        ("pss_gt_boxes", "GT boxes + PSS (oracle)"),
    ):
        row = summary
        lines.append(
            f"| {label} | {row[f'{key}/idf1']:.3f} | {row[f'{key}/id_retention']:.3f} | "
            f"{row[f'{key}/id_switches']:.3f} |"
        )
    lines += [
        "",
        "Detector: COCO Faster R-CNN MobileNetV3-320, person class only.",
        "GT boxes are used only to score detection and attach labels; tracker input",
        "is always detected boxes (except the oracle row).",
        "",
    ]
    (OUT / "tao_detector.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def main():
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threshold", type=float, default=0.3)
    args = ap.parse_args()
    payload = run(args.device, args.threshold)
    print(json.dumps(payload["summary"], indent=1))
    print(f"output: {OUT}")


if __name__ == "__main__":
    main()
