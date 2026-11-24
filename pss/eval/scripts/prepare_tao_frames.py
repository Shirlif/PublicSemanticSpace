"""Extract TAO official frames from Charades videos.

TAO annotations reference Charades frames as
``val/Charades/<video>/frameXXXXXX.jpg`` with a ``frame_index``. This script
extracts the matching frames from the official Charades 480p videos.

Example:
    python -m pss.eval.scripts.prepare_tao_frames --videos 5
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import cv2

from pss._console import ensure_utf8

TAO_ROOT = Path("data/public/tao")
ANN = TAO_ROOT / "annotations/TAO_val/gt/gt.json"
ZIP = TAO_ROOT / "charades/Charades_v1_480.zip"
RAW = TAO_ROOT / "charades/raw"
FRAMES = TAO_ROOT / "frames"


def _charades_videos(data: dict, limit: int):
    videos = [
        v for v in data.get("videos", [])
        if v.get("metadata", {}).get("dataset") == "Charades"
    ]
    by_video = {}
    for im in data.get("images", []):
        by_video.setdefault(im.get("video_id"), []).append(im)
    out = []
    for v in videos:
        if by_video.get(v["id"]):
            out.append((v, sorted(by_video[v["id"]], key=lambda x: x.get("frame_index", 0))))
        if len(out) >= limit:
            break
    return out


def _extract_video(zf: zipfile.ZipFile, video: str) -> Path | None:
    member = None
    for name in zf.namelist():
        if name.endswith(f"/{video}.mp4"):
            member = name
            break
    if member is None:
        return None
    RAW.mkdir(parents=True, exist_ok=True)
    target = RAW / f"{video}.mp4"
    if not target.exists():
        with zf.open(member) as src, open(target, "wb") as dst:
            dst.write(src.read())
    return target


def run(limit: int = 5, max_frames_per_video: int = 40) -> dict:
    data = json.loads(ANN.read_text(encoding="utf-8"))
    videos = _charades_videos(data, limit)
    FRAMES.mkdir(parents=True, exist_ok=True)
    extracted = {}
    with zipfile.ZipFile(ZIP) as zf:
        for video, images in videos:
            name = video["name"].split("/")[-1]
            path = _extract_video(zf, name)
            if path is None:
                extracted[name] = {"status": "video_not_in_zip"}
                continue
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                extracted[name] = {"status": "open_failed"}
                continue
            count = 0
            for im in images[:max_frames_per_video]:
                target = FRAMES / im["file_name"]
                if target.exists():
                    count += 1
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(im.get("frame_index", 0)))
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                cv2.imwrite(str(target), frame)
                count += 1
            cap.release()
            extracted[name] = {"status": "ok", "frames": count, "video": str(path)}
    return {"videos": extracted, "frames_root": str(FRAMES)}


def main():
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=int, default=5)
    ap.add_argument("--frames", type=int, default=40)
    args = ap.parse_args()
    payload = run(args.videos, args.frames)
    for name, row in payload["videos"].items():
        print(name, row)
    print("frames:", payload["frames_root"])


if __name__ == "__main__":
    main()
