"""TAO tracking adapter (official annotations).

``prepare`` expects the official annotation file:

```
data/public/tao/annotations/TAO_val/gt/gt.json
data/public/tao/annotations/TAO_train/gt/gt.json
```

Frames are optional for the association benchmark: the predictor uses the GT
boxes as oracle detections and evaluates identity association (IDF1, ID
switches, retention). To run a perception-based tracker, place official frames
under ``frames/<split>/...`` following the ``file_name`` fields.

The official TAO evaluator (``tao.toolkit.tao.TaoEval``) can be used for a full
HOTA/IDF1 run once tracker results are exported; this adapter implements the
association metrics needed by the suite.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np

from pss.eval.suite.adapter import BenchmarkAdapter, BenchmarkUnavailable
from pss.eval.suite.metrics import auroc  # noqa: F401  (kept for future metrics)
from pss.eval.suite.schema import EvalEpisode, EvalResult, Prediction


class TAOAdapter(BenchmarkAdapter):
    name = "tao"
    layer = "entity"
    license = "TAO research"
    homepage = "https://taodataset.org"

    def _ann_path(self) -> Path:
        if self.root is None:
            raise BenchmarkUnavailable(
                "TAO annotations not found. Download TAOLabels.zip from "
                "motchallenge.net and extract under data/public/tao/annotations/."
            )
        root = Path(self.root)
        for candidate in (
            root / "annotations/TAO_val/gt/gt.json",
            root / "TAO_val/gt/gt.json",
            root / "annotations/TAO_train/gt/gt.json",
            root / "TAO_train/gt/gt.json",
        ):
            if candidate.exists():
                return candidate
        raise BenchmarkUnavailable(f"no TAO gt.json under {root}")

    def prepare(self) -> None:
        self._ann_path()

    def episodes(self, split: str = "test", max_videos: int = 50, dataset: str | None = None) -> Iterator[EvalEpisode]:
        data = json.loads(self._ann_path().read_text(encoding="utf-8"))
        images = {im["id"]: im for im in data.get("images", [])}
        anns_by_image: dict[int, list[dict]] = {}
        for ann in data.get("annotations", []):
            anns_by_image.setdefault(ann["image_id"], []).append(ann)
        by_video: dict[int, list[dict]] = {}
        for im in data.get("images", []):
            by_video.setdefault(im.get("video_id"), []).append(im)
        videos = {v["id"]: v for v in data.get("videos", [])}
        yielded = 0
        for video_id, frames in by_video.items():
            frames = sorted(frames, key=lambda x: x.get("frame_index", 0))
            if len(frames) < 3:
                continue
            gt = {}
            for im in frames:
                rows = anns_by_image.get(im["id"], [])
                if rows:
                    gt[im["id"]] = {str(a["id"]): int(a["track_id"]) for a in rows}
            if len(gt) < 3 or sum(len(v) for v in gt.values()) < 6:
                continue
            video = videos.get(video_id, {})
            if dataset is not None and video.get("metadata", {}).get("dataset") != dataset:
                continue
            yield EvalEpisode(
                id=f"tao_{video_id}",
                benchmark=self.name,
                layer=self.layer,
                split=split,
                capabilities={"entities", "frames", "tracks"},
                expected={"tracks": gt},
                meta={
                    "video_id": video_id,
                    "video": video.get("name", ""),
                    "frames": frames,
                    "annotations": {str(k): v for k, v in anns_by_image.items()},
                    "gt": gt,
                    "frames_dir": str(Path(self.root) / "frames"),
                },
            )
            yielded += 1
            if yielded >= max_videos:
                return

    def evaluate(self, predictions: list[Prediction]) -> list[EvalResult]:
        idf1s, retention, switches = [], [], []
        for pred in predictions:
            episode = pred.meta.get("episode", {})
            gt = episode.get("gt", {})
            assignments = pred.output.get("assignments", {})
            if not gt:
                continue
            idf1s.append(_idf1(gt, assignments))
            retention.append(_retention(gt, assignments))
            switches.append(_switches(gt, assignments))
        return [
            EvalResult(self.name, self.layer, "test", "idf1", float(np.nanmean(idf1s)) if idf1s else None, len(idf1s)),
            EvalResult(self.name, self.layer, "test", "id_retention", float(np.nanmean(retention)) if retention else None, len(retention)),
            EvalResult(self.name, self.layer, "test", "id_switches", float(np.nanmean(switches)) if switches else None, len(switches)),
        ]

    def sample_episodes(self) -> list[EvalEpisode]:
        episodes = []
        for v in range(2):
            images = [
                {"id": v * 10 + t, "frame_index": t, "video_id": v}
                for t in range(4)
            ]
            gt = {}
            for t, im in enumerate(images):
                gt[im["id"]] = {"a": 1, "b": 2}
            episodes.append(
                EvalEpisode(
                    id=f"tao_sample_{v}",
                    benchmark=self.name,
                    layer=self.layer,
                    capabilities={"entities", "tracks"},
                    expected={"tracks": gt},
                    meta={
                        "video_id": v,
                        "images": images,
                        "gt": gt,
                        "boxes": {
                            im["id"]: {
                                "a": [10 + t, 10, 30, 30],
                                "b": [60 + t, 20, 30, 30],
                            }
                            for t, im in enumerate(images)
                        },
                    },
                )
            )
        return episodes


# ── metrics ─────────────────────────────────────────────────────────────────


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _idf1(gt, assignments) -> float:
    from scipy.optimize import linear_sum_assignment

    pairs: dict[tuple, int] = {}
    total_gt = 0
    total_pred = 0
    for image_id, rows in gt.items():
        preds = assignments.get(image_id, {})
        for ann_id, gt_track in rows.items():
            total_gt += 1
            pred_track = preds.get(str(ann_id))
            if pred_track is None:
                continue
            total_pred += 1
            key = (int(gt_track), int(pred_track))
            pairs[key] = pairs.get(key, 0) + 1
    if not pairs:
        return 0.0
    gt_ids = sorted({k[0] for k in pairs})
    pred_ids = sorted({k[1] for k in pairs})
    cost = np.zeros((len(gt_ids), len(pred_ids)))
    for (g, p), count in pairs.items():
        cost[gt_ids.index(g), pred_ids.index(p)] = count
    rows, cols = linear_sum_assignment(-cost)
    idtp = float(cost[rows, cols].sum())
    idfp = total_pred - idtp
    idfn = total_gt - idtp
    denom = 2 * idtp + idfp + idfn
    return 2 * idtp / denom if denom > 0 else 0.0


def _retention(gt, assignments) -> float:
    per_track: dict[int, list[int]] = {}
    for image_id, rows in gt.items():
        preds = assignments.get(image_id, {})
        for ann_id, gt_track in rows.items():
            pred_track = preds.get(str(ann_id))
            if pred_track is not None:
                per_track.setdefault(int(gt_track), []).append(int(pred_track))
    if not per_track:
        return 0.0
    values = []
    for preds in per_track.values():
        counts = {}
        for p in preds:
            counts[p] = counts.get(p, 0) + 1
        values.append(max(counts.values()) / len(preds))
    return float(np.mean(values))


def _switches(gt, assignments) -> float:
    per_track: dict[int, list[int]] = {}
    for image_id, rows in gt.items():
        preds = assignments.get(image_id, {})
        for ann_id, gt_track in rows.items():
            pred_track = preds.get(str(ann_id))
            if pred_track is not None:
                per_track.setdefault(int(gt_track), []).append(int(pred_track))
    if not per_track:
        return 0.0
    values = []
    for preds in per_track.values():
        values.append(sum(1 for a, b in zip(preds, preds[1:]) if a != b))
    return float(np.mean(values))
