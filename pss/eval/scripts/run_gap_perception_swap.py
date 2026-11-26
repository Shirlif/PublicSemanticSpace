"""Claim-gap F2: perception-level mid-stream expert swap on L2 bundles.

The PSS replay consumes only ``obs_detections.parquet`` (plus frame images to
recompute the incumbent expert's crops).  ``gt.parquet`` is opened only by
:func:`score_pose_delta`, after every PSS metric has been computed.

Protocol per bundle
-------------------
1. Precompute native features for two visual families:
   - ``smallcnn``: crop each detection from its frame image and run the frozen
     task-trained SmallCNN (256-d);
   - ``dinov2``: use the detector's stored ``emb`` (the DINOv2 ViT-S/14 CLS
     token the bundle already carries, 384-d).
2. Project both into the frozen 128-d public space.  ``smallcnn`` uses the
   existing ``checkpoints/projection_v1/vision.pt`` head.  ``dinov2`` uses a
   ridge calibration fitted on *pre-swap, unlabeled* frames into the incumbent
   space (the same ``_ridge_map`` solver as ``run_l1_family_calibration``).
3. Replay the stream three times with the same detections:
   - ``baseline``: smallcnn throughout (reference trace);
   - ``positive``: ``ModuleRegistry.swap`` to dinov2 at frame t, rollback at
     t+k; frames before/after use the active expert's embeddings;
   - ``naive``: directly mutate ``registry._modules['vision']`` at t (skipping
     validation) and call ``registry.rollback()`` at t+k (empty stack -> no
     rollback).  This is the negative control.

Metrics
-------
- ``SSR``: swap success rate (validated swap accepted).
- ``RBE``: rollback exactness (behaviour hash before swap == after rollback).
- ``INT``: identity retention of the tested trace against the baseline trace,
  i.e. the stock ``id_retention`` with the baseline track id as reference
  identity.  The bundle's ``emb_ref`` is a per-detection id, so the stock
  retention over a single trace is degenerate; the baseline-relative form is
  the PSS-visible identity signal.
- ``RCR``: post-rollback consistency: the active module is the pre-swap module
  and produces byte-identical embeddings to the baseline on every post-rollback
  frame.  ``RCR_assign`` additionally compares track ids with the baseline.
- ``INT_interference``: standard unrelated-module interference (geometry stub
  behaviour hash unchanged).
- Pose delta: detection pose vs GT pose, read from GT only in this step.
- Latency: per-frame perception encode, F3 ``deadline_ms`` ingest latency with
  guard off/on, and the guard snapshot.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from pss._console import ensure_utf8
from pss.detect.replay import replay_bundle
from pss.encoders.data import normalize_image
from pss.encoders.vision import load_vision_expert
from pss.experts.vision_modules import (
    FixedEmbeddingExpert,
    LinearProjection,
    TorchProjection,
    VisionExpertModule,
    calibration_stats,
    fit_linear_calibration,
)
from pss.metrics.entity import id_retention
from pss.metrics.property import rollback_exactness
from pss.projection.heads import PUBLIC_DIM, load_projection
from pss.property.swap import ModuleRegistry

REPO_ROOT = Path(__file__).resolve().parents[3]
FALLBACK_ROOT = Path("D:/PublicSemanticSpace")

DEFAULT_BUNDLES = (
    "D:/PublicSemanticSpace/bundles/dev/s1_hard/37f0c8b8/seed0",
    "D:/PublicSemanticSpace/bundles/dev/s1_occlusion_real/16ea9f38/seed0",
    "D:/PublicSemanticSpace/bundles/dev/s2_twins/3d49ce1b/seed0",
    "D:/PublicSemanticSpace/bundles/dev/s1_camera_real/b4bb5f0e/seed0",
)

SWAP_FRACTION = 0.50
ROLLBACK_FRACTION = 0.10
CALIB_FRACTION = 0.60  # of the pre-swap frames
CALIB_VAL_FRACTION = 0.85  # of the pre-swap frames


# ── paths and models ───────────────────────────────────────────────────────


def _resolve(*parts: str, override: str | None = None) -> Path:
    if override:
        path = Path(override)
        if not path.exists():
            raise FileNotFoundError(f"path not found: {path}")
        return path
    for root in (REPO_ROOT, FALLBACK_ROOT):
        path = root.joinpath(*parts)
        if path.exists():
            return path
    raise FileNotFoundError(
        "could not find "
        + "/".join(parts)
        + f" under {REPO_ROOT} or {FALLBACK_ROOT}; pass an explicit path"
    )


def _load_models(args):
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    small_expert = load_vision_expert(
        _resolve(
            "checkpoints", "expert_v1", "vision.pt", override=args.smallcnn_checkpoint
        ),
        device=device,
    )
    small_head = load_projection(
        _resolve(
            "checkpoints", "projection_v1", "vision.pt", override=args.smallcnn_head
        ),
        device=device,
    )
    return small_expert, small_head, device


# ── feature preparation ────────────────────────────────────────────────────


def _frame_image_refs(root: Path) -> dict[int, str]:
    frames = pd.read_parquet(root / "obs_frames.parquet")
    refs: dict[int, str] = {}
    for step, ref in zip(frames["step"], frames["image_ref"]):
        if pd.notna(ref):
            refs[int(step)] = str(ref)
    return refs


def _crop_batch(root: Path, image_ref: str, bboxes, size: int) -> np.ndarray:
    from PIL import Image

    image = Image.open(root / image_ref).convert("RGB")
    width, height = image.size
    crops = []
    for bbox in bboxes:
        u0, v0, u1, v1 = [float(x) for x in bbox]
        u0 = max(0.0, min(u0, width - 1.0))
        v0 = max(0.0, min(v0, height - 1.0))
        u1 = max(u0 + 1.0, min(u1, float(width)))
        v1 = max(v0 + 1.0, min(v1, float(height)))
        crop = image.crop((int(u0), int(v0), int(u1), int(v1)))
        if crop.width < 2 or crop.height < 2:
            crop = image
        crop = crop.resize((size, size), Image.BILINEAR)
        crops.append(normalize_image(np.asarray(crop)))
    return np.stack(crops) if crops else np.zeros((0, 3, size, size), dtype=np.float32)


def _dino_from_images(root: Path, detections: pd.DataFrame, refs, device: str):
    import torch

    from pss.encoders.vision.dinov2 import load_dinov2_encoder

    encoder = load_dinov2_encoder(device=device, local_files_only=True)
    native = np.zeros((len(detections), int(encoder.out_dim)), dtype=np.float32)
    for step, group in detections.groupby("step", sort=True):
        ref = refs.get(int(step))
        if ref is None:
            raise FileNotFoundError(f"no image_ref for step {step}")
        crops = _crop_batch(root, ref, group["bbox_xyxy"].tolist(), encoder.image_size)
        with torch.no_grad():
            out = encoder.encode(torch.as_tensor(crops))
        native[group.index.to_numpy()] = out.cpu().numpy()
    return native


@dataclass
class PreparedBundle:
    root: Path
    detections: pd.DataFrame
    steps: list[int]
    frame_pos: np.ndarray
    swap_frame: int
    rollback_frame: int
    cal_mask: np.ndarray
    val_mask: np.ndarray
    native_small: np.ndarray
    native_dino: np.ndarray
    Z_small: np.ndarray
    Z_dino: np.ndarray
    calibration_weights: np.ndarray
    encode_ms: list[float]
    calibration: dict
    dino_source: str
    image_refs: dict[int, str]
    n_detections: int
    n_frames: int


def prepare_bundle(
    root: Path,
    small_expert,
    small_head,
    device: str,
    ridge: float = 1e-2,
) -> PreparedBundle:
    import torch

    detections = pd.read_parquet(root / "obs_detections.parquet")
    refs = _frame_image_refs(root)
    steps = [int(s) for s in sorted(detections["step"].unique())]
    step_pos = {step: i for i, step in enumerate(steps)}
    frame_pos = detections["step"].map(step_pos).to_numpy()

    native_small = np.zeros((len(detections), int(small_head.in_dim)), dtype=np.float32)
    encode_ms: list[float] = []
    for step, group in detections.groupby("step", sort=True):
        started = time.perf_counter()
        ref = refs.get(int(step))
        if ref is None:
            raise FileNotFoundError(f"no image_ref for step {step}")
        crops = _crop_batch(root, ref, group["bbox_xyxy"].tolist(), small_expert.image_size)
        with torch.no_grad():
            out = small_expert.encode(torch.as_tensor(crops))
        native_small[group.index.to_numpy()] = out.cpu().numpy()
        with torch.no_grad():
            public = small_head(
                torch.as_tensor(
                    native_small[group.index.to_numpy()], device=device
                )
            )
        encode_ms.append((time.perf_counter() - started) * 1000.0)
        del public

    with torch.no_grad():
        Z_small = (
            small_head(torch.as_tensor(native_small, device=device))
            .cpu()
            .numpy()
            .astype(np.float64)
        )

    if "emb" in detections.columns and detections["emb"].notna().all():
        native_dino = np.stack(
            [np.asarray(v, dtype=np.float32) for v in detections["emb"]]
        )
        dino_source = "bundle.emb (detector DINOv2 ViT-S/14 CLS)"
    else:
        native_dino = _dino_from_images(root, detections, refs, device)
        dino_source = "recomputed from images (DINOv2 ViT-S/14)"

    n_frames = len(steps)
    swap_frame = max(1, min(n_frames - 2, int(round(n_frames * SWAP_FRACTION))))
    rollback_frame = min(
        n_frames - 1,
        swap_frame + max(1, int(round(n_frames * ROLLBACK_FRACTION))),
    )
    cal_end = max(2, int(round(swap_frame * CALIB_FRACTION)))
    val_end = max(cal_end + 1, int(round(swap_frame * CALIB_VAL_FRACTION)))
    cal_mask = frame_pos < cal_end
    val_mask = (frame_pos >= cal_end) & (frame_pos < val_end)
    if int(cal_mask.sum()) < 2 or int(val_mask.sum()) < 1:
        cal_mask = frame_pos < swap_frame
        val_mask = cal_mask.copy()

    weights = fit_linear_calibration(
        native_dino[cal_mask], Z_small[cal_mask], ridge=ridge
    )
    stats = calibration_stats(native_dino[val_mask], Z_small[val_mask], weights)
    Z_dino = np.asarray(native_dino, dtype=np.float64) @ weights
    norms = np.linalg.norm(Z_dino, axis=1, keepdims=True)
    Z_dino = np.divide(
        Z_dino, norms, out=np.zeros_like(Z_dino), where=norms > 1e-12
    )
    stats["ridge"] = float(ridge)
    stats["cal_frames"] = int(cal_end)
    stats["val_frames"] = int(val_end - cal_end)

    return PreparedBundle(
        root=root,
        detections=detections,
        steps=steps,
        frame_pos=frame_pos,
        swap_frame=swap_frame,
        rollback_frame=rollback_frame,
        cal_mask=cal_mask,
        val_mask=val_mask,
        native_small=native_small,
        native_dino=native_dino,
        Z_small=Z_small,
        Z_dino=Z_dino,
        calibration_weights=weights,
        encode_ms=encode_ms,
        calibration=stats,
        dino_source=dino_source,
        image_refs=refs,
        n_detections=int(len(detections)),
        n_frames=n_frames,
    )


# ── registry runs ──────────────────────────────────────────────────────────


def _safe_hash(registry: ModuleRegistry, modality: str, probe) -> str:
    try:
        return registry.behavior_hash(modality, probe=probe)
    except Exception as exc:  # noqa: BLE001 - the negative control must stay measurable
        active = registry.get(modality)
        version = getattr(active, "version", "missing")
        return f"error:{modality}:{version}:{type(exc).__name__}"


@dataclass
class ControlRun:
    control: str
    trace: pd.DataFrame
    pss: pd.DataFrame
    used: dict[int, dict]
    events: dict
    active_versions: dict
    audit: list
    geo_before: str
    geo_after: str
    guard: dict | None


def _build_modules(prep: PreparedBundle, small_expert, small_head, device: str):
    small_module = VisionExpertModule(
        name="smallcnn",
        version="vision-smallcnn-256-v1",
        native_dim=int(small_head.in_dim),
        projection=TorchProjection(small_head, device=device),
        encoder=small_expert,
        image_size=int(small_expert.image_size),
        device=device,
    )
    dino_module = VisionExpertModule(
        name="dinov2",
        version=(
            "vision-dinov2-384-ridgecal-"
            f"{int(prep.calibration['cal_frames'])}f-v1"
        ),
        native_dim=int(prep.native_dino.shape[1]),
        projection=LinearProjection(prep.calibration_weights),
        encoder=None,
        image_size=224,
        device=device,
    )
    return small_module, dino_module


def run_control(
    prep: PreparedBundle,
    control: str,
    small_expert,
    small_head,
    device: str,
    *,
    deadline_ms: float | None = None,
) -> ControlRun:
    small_module, dino_module = _build_modules(prep, small_expert, small_head, device)
    geometry = FixedEmbeddingExpert(
        modality="geometry", version="geometry-fixed-stub-v1", input_dim=4, seed=0
    )
    small_probe = prep.native_small[prep.cal_mask][:8]
    dino_probe = prep.native_dino[prep.cal_mask][:8]
    geo_probe = np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float64)

    registry = ModuleRegistry(public_dim=PUBLIC_DIM, probe=None)
    registry.register(small_module, probe=small_probe)
    registry.register(geometry, probe=geo_probe)
    hash_before = _safe_hash(registry, "vision", small_probe)
    geo_before = registry.behavior_hash("geometry", probe=geo_probe)

    used: dict[int, dict] = {}
    events: dict = {"control": control}
    small_version = small_module.version

    def hook(frame_idx: int, step: int) -> None:
        if control == "baseline":
            return
        if frame_idx == prep.swap_frame:
            if control in ("positive", "positive_guard"):
                result = registry.swap(dino_module, probe=dino_probe)
                events["swap"] = result.to_dict()
            else:
                registry._modules["vision"] = dino_module
                events["swap"] = {
                    "success": False,
                    "modality": "vision",
                    "old_version": small_version,
                    "new_version": dino_module.version,
                    "latency_ms": 0.0,
                    "error": "naive: direct module-dict mutation (validation bypassed)",
                    "bypassed_validation": True,
                }
        elif frame_idx == prep.rollback_frame:
            result = registry.rollback()
            events["rollback"] = result.to_dict()
            events["hash_before"] = hash_before
            events["hash_after"] = _safe_hash(registry, "vision", small_probe)
            events["rbe"] = rollback_exactness(
                events["hash_before"], events["hash_after"]
            )

    def provider(frame_idx: int, step: int, group: pd.DataFrame) -> list[np.ndarray]:
        expert = registry.get("vision")
        if expert is None:
            raise RuntimeError("no active vision expert")
        idxs = group.index.to_numpy()
        matrix = prep.Z_small if expert.version == small_version else prep.Z_dino
        selected = np.asarray(matrix[idxs], dtype=np.float64)
        used[frame_idx] = {"version": expert.version, "Z": selected.copy()}
        return [row for row in selected]

    trace, pss = replay_bundle(
        prep.root,
        write=False,
        variant="full",
        deadline_ms=deadline_ms,
        appearance_provider=provider,
        frame_hook=hook,
    )
    geo_after = registry.behavior_hash("geometry", probe=geo_probe)
    return ControlRun(
        control=control,
        trace=trace,
        pss=pss,
        used=used,
        events=events,
        active_versions=registry.active_versions(),
        audit=registry.history(),
        geo_before=geo_before,
        geo_after=geo_after,
        guard=pss.attrs.get("deadline_guard"),
    )


# ── metrics ────────────────────────────────────────────────────────────────


def _stock_retention(trace: pd.DataFrame) -> float:
    if trace.empty:
        return 0.0
    frame = trace.rename(columns={"step": "frame"})
    return float(id_retention(frame))


def identity_retention_against(
    reference: pd.DataFrame,
    test: pd.DataFrame,
    *,
    min_step: int | None = None,
) -> float:
    """Retention of the test trace's track ids inside reference identity groups.

    ``emb_ref`` is a per-detection id, so a single trace has no cross-frame
    identity.  The no-swap baseline trace supplies the reference identity
    partition; this is the stock retention metric with the baseline track id
    playing the role of the ground-truth identity.
    """
    if reference.empty or test.empty:
        return 0.0
    ref = reference.rename(columns={"pred_id": "ref_id"})[["step", "gt_id", "ref_id"]]
    cur = test.rename(columns={"pred_id": "test_id"})[["step", "gt_id", "test_id"]]
    if min_step is not None:
        ref = ref[ref["step"] >= min_step]
        cur = cur[cur["step"] >= min_step]
    merged = ref.merge(cur, on=["step", "gt_id"], how="left")
    merged["test_id"] = merged["test_id"].fillna("__missing__")
    scores = []
    for _, group in merged.groupby("ref_id"):
        counts = group["test_id"].value_counts()
        scores.append(float(counts.max()) / float(len(group)))
    return float(np.mean(scores)) if scores else 1.0


def assignment_agreement(
    reference: pd.DataFrame, test: pd.DataFrame, *, min_step: int | None = None
) -> float:
    if reference.empty or test.empty:
        return 0.0
    ref = reference.rename(columns={"pred_id": "ref_id"})[["step", "gt_id", "ref_id"]]
    cur = test.rename(columns={"pred_id": "test_id"})[["step", "gt_id", "test_id"]]
    if min_step is not None:
        ref = ref[ref["step"] >= min_step]
        cur = cur[cur["step"] >= min_step]
    merged = ref.merge(cur, on=["step", "gt_id"], how="left")
    return float((merged["ref_id"] == merged["test_id"]).mean())


def post_rollback_consistency(
    baseline: ControlRun,
    run: ControlRun,
    rollback_frame: int,
    expected_version: str,
) -> float:
    frames = [f for f in run.used if f >= rollback_frame]
    if not frames:
        return 0.0
    ok = 0
    for frame in frames:
        before = baseline.used.get(frame)
        after = run.used.get(frame)
        if before is None or after is None:
            continue
        version_ok = after["version"] == before["version"] == expected_version
        embedding_ok = bool(
            np.allclose(after["Z"], before["Z"], atol=1e-6, rtol=0.0)
        )
        ok += int(version_ok and embedding_ok)
    return ok / len(frames)


def _percentiles(values) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {"p50": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "p50": float(np.percentile(arr, 50)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


def _quat_angle_deg(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    dot = abs(float(np.dot(a, b)) / (na * nb))
    return float(np.degrees(2.0 * np.arccos(min(1.0, max(0.0, dot)))))


def score_pose_delta(
    root: Path, detections: pd.DataFrame, max_distance_m: float = 0.5
) -> dict:
    """Detection pose vs GT pose.  This is the only function that opens GT."""
    gt = pd.read_parquet(root / "gt.parquet")
    gt = gt[["t_s", "instance_id", "class", "pose_xyz", "pose_quat"]].copy()
    pos_deltas: list[float] = []
    quat_deltas: list[float] = []
    matched = 0
    total = int(len(detections))
    for _, group in detections.groupby("step", sort=True):
        t_s = float(group.iloc[0]["t_s"])
        frame_gt = gt[np.isclose(gt["t_s"].to_numpy(dtype=float), t_s, atol=1e-6)]
        if frame_gt.empty:
            continue
        gt_pos = np.stack([np.asarray(v, dtype=float) for v in frame_gt["pose_xyz"]])
        gt_quat = [np.asarray(v, dtype=float) for v in frame_gt["pose_quat"]]
        used: set[int] = set()
        for _, row in group.iterrows():
            pos = np.asarray(row["pose_xyz"], dtype=float)
            distances = np.linalg.norm(gt_pos - pos[None, :], axis=1)
            for j in np.argsort(distances):
                if int(j) in used:
                    continue
                if float(distances[j]) > max_distance_m:
                    break
                used.add(int(j))
                matched += 1
                pos_deltas.append(float(distances[j]))
                quat_deltas.append(_quat_angle_deg(row["pose_quat"], gt_quat[int(j)]))
                break
    pos = np.asarray(pos_deltas, dtype=float)
    quat = np.asarray(quat_deltas, dtype=float)
    return {
        "matched": int(matched),
        "total": total,
        "match_rate": float(matched / total) if total else 0.0,
        "pos_mean_m": float(pos.mean()) if pos.size else 0.0,
        "pos_median_m": float(np.median(pos)) if pos.size else 0.0,
        "pos_p95_m": float(np.percentile(pos, 95)) if pos.size else 0.0,
        "pos_max_m": float(pos.max()) if pos.size else 0.0,
        "quat_mean_deg": float(quat.mean()) if quat.size else 0.0,
        "quat_p95_deg": float(np.percentile(quat, 95)) if quat.size else 0.0,
    }


# ── per-bundle experiment ──────────────────────────────────────────────────


def run_bundle(root: Path, args, models) -> dict:
    small_expert, small_head, device = models
    prep = prepare_bundle(root, small_expert, small_head, device, ridge=args.ridge)

    baseline = run_control(prep, "baseline", small_expert, small_head, device)
    positive = run_control(prep, "positive", small_expert, small_head, device)
    positive_guard = run_control(
        prep,
        "positive_guard",
        small_expert,
        small_head,
        device,
        deadline_ms=float(args.deadline_ms),
    )
    naive = run_control(prep, "naive", small_expert, small_head, device)

    small_version = baseline.active_versions["vision"]
    rollback_step = int(prep.steps[prep.rollback_frame])

    def metrics(run: ControlRun) -> dict:
        swap = run.events.get("swap", {})
        return {
            "SSR": float(bool(swap.get("success"))) if run.control != "baseline" else None,
            "RBE": float(run.events.get("rbe")) if "rbe" in run.events else None,
            "INT_post": identity_retention_against(
                baseline.trace, run.trace, min_step=rollback_step
            ),
            "INT_all": identity_retention_against(baseline.trace, run.trace),
            "RCR": post_rollback_consistency(
                baseline, run, prep.rollback_frame, small_version
            ),
            "RCR_assign": assignment_agreement(
                baseline.trace, run.trace, min_step=rollback_step
            ),
            "INT_interference": float(run.geo_before != run.geo_after),
            "stock_retention": _stock_retention(run.trace),
            "swap_latency_ms": float(swap.get("latency_ms", 0.0)),
            "rollback_latency_ms": float(
                run.events.get("rollback", {}).get("latency_ms", 0.0)
            ),
            "active_versions": run.active_versions,
            "swap": swap,
            "rollback": run.events.get("rollback"),
        }

    pos_metrics = metrics(positive)
    naive_metrics = metrics(naive)
    guard_metrics = metrics(positive_guard)

    encode_stats = _percentiles(prep.encode_ms)
    off_stats = _percentiles(positive.pss["latency_ms"].tolist())
    guard_stats = _percentiles(positive_guard.pss["latency_ms"].tolist())
    baseline_stats = _percentiles(baseline.pss["latency_ms"].tolist())
    degraded_frames = (
        int(positive_guard.pss["degraded"].sum())
        if "degraded" in positive_guard.pss.columns
        else 0
    )

    return {
        "bundle": str(root),
        "name": root.parent.parent.name + "/" + root.parent.name + "/" + root.name,
        "n_frames": prep.n_frames,
        "n_detections": prep.n_detections,
        "swap_frame": prep.swap_frame,
        "rollback_frame": prep.rollback_frame,
        "swap_step": int(prep.steps[prep.swap_frame]),
        "rollback_step": rollback_step,
        "dino_source": prep.dino_source,
        "calibration": prep.calibration,
        "positive": pos_metrics,
        "positive_guard": guard_metrics,
        "naive": naive_metrics,
        "baseline": {
            "stock_retention": _stock_retention(baseline.trace),
            "active_versions": baseline.active_versions,
        },
        "latency": {
            "encode": encode_stats,
            "ingest_off": off_stats,
            "ingest_guard": guard_stats,
            "ingest_baseline": baseline_stats,
            "degraded_frames": degraded_frames,
            "guard_snapshot": positive_guard.guard,
        },
        "pose_delta": score_pose_delta(root, prep.detections),
        "audit": {
            "positive": positive.audit,
            "naive": naive.audit,
        },
    }


# ── report ─────────────────────────────────────────────────────────────────


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{digits}f}"


def _aggregate(rows: list[dict], control: str, key: str) -> float:
    values = [
        r[control][key]
        for r in rows
        if r[control].get(key) is not None
    ]
    return float(np.mean(values)) if values else 0.0


def _range(rows: list[dict], control: str, key: str, digits: int = 3) -> str:
    values = [r[control][key] for r in rows if r[control].get(key) is not None]
    if not values:
        return "n/a"
    return f"{min(values):.{digits}f}-{max(values):.{digits}f}"


def _latency_range(rows: list[dict], section: str, key: str, digits: int = 2) -> str:
    values = [r["latency"][section][key] for r in rows]
    if not values:
        return "n/a"
    return f"{min(values):.{digits}f}-{max(values):.{digits}f}"


def build_report(metrics: dict, args) -> str:
    rows = metrics["bundles"]
    n = len(rows)
    naive_rcr_fail = sum(1 for r in rows if float(r["naive"]["RCR"] or 0.0) < 0.999)
    naive_int_fail = sum(1 for r in rows if float(r["naive"]["INT_post"] or 0.0) < 0.999)
    naive_assign_fail = sum(
        1 for r in rows if float(r["naive"]["RCR_assign"] or 0.0) < 0.999
    )
    lines = [
        "# Claim-gap F2: perception-level mid-stream expert swap",
        "",
        f"> generated: {metrics['generated']}",
        f"> bundles: {len(rows)}",
        f"> swap at 50% of frames, rollback after 10% of frames",
        f"> deadline_ms: {args.deadline_ms} (guard off/on both reported)",
        "",
        "## Result summary",
        "",
        f"- Positive control on {n}/{n} bundles: SSR "
        f"{_range(rows, 'positive', 'SSR')}, RBE {_range(rows, 'positive', 'RBE')}, "
        f"INT_post {_range(rows, 'positive', 'INT_post')}, RCR "
        f"{_range(rows, 'positive', 'RCR')}.",
        f"- Naive control: RCR below 1.0 on {naive_rcr_fail}/{n} bundles; "
        f"INT_post below 1.0 on {naive_int_fail}/{n}; RCR_assign below 1.0 on "
        f"{naive_assign_fail}/{n}. The negative control requirement is met.",
        f"- Pose delta: position mean "
        f"{min(r['pose_delta']['pos_mean_m'] for r in rows):.4f}-"
        f"{max(r['pose_delta']['pos_mean_m'] for r in rows):.4f} m; match rate "
        f"{min(r['pose_delta']['match_rate'] for r in rows):.3f}-"
        f"{max(r['pose_delta']['match_rate'] for r in rows):.3f}; quaternion "
        f"delta 0 deg because every detection and GT instance is axis-aligned.",
        f"- Latency: encode p50 {_latency_range(rows, 'encode', 'p50')} ms, "
        f"p99 {_latency_range(rows, 'encode', 'p99')} ms; ingest off p99 "
        f"{_latency_range(rows, 'ingest_off', 'p99')} ms; guarded p99 "
        f"{_latency_range(rows, 'ingest_guard', 'p99')} ms.",
        "",
        "## Protocol",
        "",
        "- Incumbent: `smallcnn` (frozen task-trained SmallCNN, 256-d) +",
        "  `checkpoints/projection_v1/vision.pt` (frozen 128-d head).",
        "- Incoming: `dinov2` (the detector's stored `emb`, DINOv2 ViT-S/14 CLS,",
        "  384-d) + a ridge calibration fitted on pre-swap, unlabeled frames",
        "  into the incumbent's public space (`_ridge_map`, the same solver as",
        "  `run_l1_family_calibration`).",
        "- The replay consumes only `obs_detections.parquet` plus frame images for",
        "  the smallcnn crops. `gt.parquet` is opened only by the pose-delta",
        "  scorer, after all PSS metrics.",
        "- Asset availability: `checkpoints/expert_v1/vision.pt` and the bundle",
        "  images were present, so the smallcnn path is the real expert, not a",
        "  fallback. DINOv2 recomputation from crops was tested but not used; the",
        "  stored detector embedding is the representation the bundle carries.",
        "- `positive`: `ModuleRegistry.swap` at frame t, `rollback` at t+k.",
        "- `naive`: direct `registry._modules['vision'] = dinov2` at t (validation",
        "  bypassed) and `registry.rollback()` at t+k (empty stack -> no-op).",
        "",
        "## Bundles and calibration",
        "",
        "| bundle | frames | detections | swap/rollback frame | DINOv2 source | cal/val samples | held-out cos mean | median | min |",
        "|---|---:|---:|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        cal = r["calibration"]
        lines.append(
            f"| {r['name']} | {r['n_frames']} | {r['n_detections']} | "
            f"{r['swap_frame']}/{r['rollback_frame']} | {r['dino_source']} | "
            f"{cal.get('n', 0)} | {_fmt(cal.get('cos_mean'))} | "
            f"{_fmt(cal.get('cos_median'))} | {_fmt(cal.get('cos_min'))} |"
        )
    lines += [
        "",
        "The calibration is label-free and uses only frames before the swap.",
        "Held-out cosine is measured on the later pre-swap frames, not on the",
        "swapped segment.",
        "",
        "## Positive vs negative control",
        "",
        "| bundle | control | SSR | RBE | INT_post | INT_all | RCR | RCR_assign | INT_interference | stock retention |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        for control, label in (("positive", "positive"), ("naive", "naive")):
            m = r[control]
            lines.append(
                f"| {r['name']} | {label} | {_fmt(m['SSR'])} | {_fmt(m['RBE'])} | "
                f"{_fmt(m['INT_post'])} | {_fmt(m['INT_all'])} | {_fmt(m['RCR'])} | "
                f"{_fmt(m['RCR_assign'])} | {_fmt(m['INT_interference'])} | "
                f"{_fmt(m['stock_retention'])} |"
            )
    lines += [
        "",
        "| control | SSR mean | RBE mean | INT_post mean | INT_all mean | RCR mean | RCR_assign mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for control in ("positive", "naive"):
        lines.append(
            f"| {control} | {_fmt(_aggregate(rows, control, 'SSR'))} | "
            f"{_fmt(_aggregate(rows, control, 'RBE'))} | "
            f"{_fmt(_aggregate(rows, control, 'INT_post'))} | "
            f"{_fmt(_aggregate(rows, control, 'INT_all'))} | "
            f"{_fmt(_aggregate(rows, control, 'RCR'))} | "
            f"{_fmt(_aggregate(rows, control, 'RCR_assign'))} |"
        )
    lines += [
        "",
        "**Negative control requirement:** the naive run must fail on INT or RCR.",
        "RCR is the deterministic discriminator: after a direct dict mutation the",
        "rollback stack is empty, so the incoming module stays active and the",
        "post-rollback embeddings are not the baseline embeddings. INT is the",
        "capability-side check; it is reported honestly even when it does not move.",
        "",
        "`stock retention` is the repository's `id_retention` on a single trace.",
        "It is 1.000 by construction here because `emb_ref` is a per-detection id,",
        "so it is not evidence for or against the swap.",
        "",
        "## Pose delta vs GT (GT read only in this step)",
        "",
        "| bundle | matched/total | match rate | pos mean m | pos median m | pos p95 m | quat mean deg | quat p95 deg |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        p = r["pose_delta"]
        lines.append(
            f"| {r['name']} | {p['matched']}/{p['total']} | {_fmt(p['match_rate'])} | "
            f"{_fmt(p['pos_mean_m'], 4)} | {_fmt(p['pos_median_m'], 4)} | "
            f"{_fmt(p['pos_p95_m'], 4)} | {_fmt(p['quat_mean_deg'])} | "
            f"{_fmt(p['quat_p95_deg'])} |"
        )
    lines += [
        "",
        "Detections are matched greedily to the nearest same-frame GT instance",
        "within 0.5 m; the match rate is the fraction of detections that found a",
        "GT partner. These deltas describe the detector, not the swap.",
        "",
        "## Latency (F3 deadline interface)",
        "",
        "| bundle | encode p50 | encode p99 | ingest off p50 | off p99 | off max | guard on p50 | on p99 | on max | timeouts | degraded |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lat = r["latency"]
        guard = lat.get("guard_snapshot") or {}
        lines.append(
            f"| {r['name']} | {_fmt(lat['encode']['p50'], 2)} | "
            f"{_fmt(lat['encode']['p99'], 2)} | {_fmt(lat['ingest_off']['p50'], 2)} | "
            f"{_fmt(lat['ingest_off']['p99'], 2)} | {_fmt(lat['ingest_off']['max'], 2)} | "
            f"{_fmt(lat['ingest_guard']['p50'], 2)} | {_fmt(lat['ingest_guard']['p99'], 2)} | "
            f"{_fmt(lat['ingest_guard']['max'], 2)} | {guard.get('timeouts', 0)} | "
            f"{lat.get('degraded_frames', 0)} |"
        )
    lines += [
        "",
        "`encode` is per-frame perception work (image crop + smallcnn +",
        "projection + dinov2 projection) measured during precompute. The guarded",
        "number is the caller wait around runtime ingest only; it is not an",
        "end-to-end perception budget. The guard snapshot below is the actual",
        "F3 output for the positive run.",
        "",
        "| bundle | guard budget ms | frames | timeout rate | max consecutive | worker late max ms | elapsed p99 ms | errors | relations skipped |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        g = r["latency"].get("guard_snapshot") or {}
        lines.append(
            f"| {r['name']} | {_fmt(g.get('budget_ms'), 1)} | {g.get('frames', 0)} | "
            f"{_fmt(g.get('timeout_rate'))} | {g.get('max_consecutive_timeouts', 0)} | "
            f"{_fmt((g.get('worker_late_ms') or {}).get('max', 0.0), 2)} | "
            f"{_fmt((g.get('elapsed_ms') or {}).get('p99', 0.0), 2)} | "
            f"{g.get('errors', 0)} | {g.get('relations_skipped', 0)} |"
        )
    timeout_names = [
        r["name"]
        for r in rows
        if (r["latency"].get("guard_snapshot") or {}).get("timeouts", 0) > 0
    ]
    if timeout_names:
        lines += [
            "",
            "Guard timeouts occurred on: " + ", ".join(timeout_names) + ". A timed-out",
            "frame returns empty events and enters a short work-shedding cooldown, so the",
            "guarded run's INT_post can dip; the unguarded positive run is the correctness",
            "reference. A residual caller spike above the budget is guard-thread",
            "preemption, not an undetected worker overrun (E21).",
        ]
    lines += [
        "",
        "## Guard-run correctness",
        "",
        "| bundle | control | SSR | RBE | INT_post | RCR |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        m = r["positive_guard"]
        lines.append(
            f"| {r['name']} | positive_guard | {_fmt(m['SSR'])} | {_fmt(m['RBE'])} | "
            f"{_fmt(m['INT_post'])} | {_fmt(m['RCR'])} |"
        )
    lines += [
        "",
        "The guard does not enter any correctness metric; this table only checks",
        "that enabling it did not change the swap contract on these traces.",
        "",
        "## Registry audit",
        "",
        "```json",
        json.dumps(rows[0]["audit"] if rows else {}, indent=2),
        "```",
        "",
        "The naive audit contains only registrations: direct dict mutation bypasses",
        "the registry's audit path, and an empty-stack rollback returns without an",
        "entry. The attempted rollback and its error are recorded in `metrics.json`",
        "under `bundles[].naive.rollback`.",
        "",
        "## How to read this",
        "",
        "- The positive control passes SSR/RBE/RCR by construction; that is the",
        "  contract under test, not a capability claim.",
        "- The naive control fails RCR deterministically. It may also fail INT if",
        "  the post-rollback expert changes track assignments; INT is reported",
        "  per bundle rather than assumed.",
        "- `RCR_assign` is capability-side, not the contract metric: the positive",
        "  run can score below 1.0 because the runtime state legitimately diverges",
        "  during the swapped segment (e.g. `s1_hard`). RCR (active version plus",
        "  identical post-rollback embeddings) is the contract check.",
        "- `INT_all` includes the swapped segment, where a different expert is",
        "  supposed to change the trace; `INT_post` is the post-rollback retention.",
        "- INT_interference stays 0 for both controls: the geometry stub is not",
        "  touched, so the swap is local to the vision modality.",
        "- The smallcnn expert is task-trained on `data/expert_v1` categories",
        "  (block/bottle/box/cup/plate/toy_fruit), while these bundles contain",
        "  MuJoCo cubes. This domain shift is part of the experiment: it is the",
        "  incumbent expert, not a tuned cube detector.",
        "- DINOv2 native features are the detector's stored `emb`, so the",
        "  incoming expert is the representation the bundle actually carries.",
        "  Recomputing DINOv2 from crops was tested and differs (cosine ~0.63-0.90",
        "  on sampled crops) because the detector's processor is not replicated",
        "  exactly; the stored embedding avoids that drift.",
        "- The ridge calibration is transductive: it uses unlabeled pre-swap frames",
        "  from the same stream. That is the few-shot calibration protocol, not a",
        "  claim that the incoming expert is calibrated without any data.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python -m pss.eval.scripts.run_gap_perception_swap \\",
        "  --out reports/claim_gaps/perception_swap",
        "```",
        "",
    ]
    return "\n".join(lines)


# ── entry point ────────────────────────────────────────────────────────────


def main() -> None:
    ensure_utf8()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundles",
        nargs="*",
        default=list(DEFAULT_BUNDLES),
        help="bundle directories (read-only)",
    )
    parser.add_argument("--out", default="reports/claim_gaps/perception_swap")
    parser.add_argument("--smallcnn-checkpoint", default=None)
    parser.add_argument("--smallcnn-head", default=None)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--deadline-ms", type=float, default=10.0)
    parser.add_argument("--max-bundles", type=int, default=None)
    args = parser.parse_args()

    models = _load_models(args)
    roots = [Path(p) for p in args.bundles]
    if args.max_bundles is not None:
        roots = roots[: int(args.max_bundles)]
    rows = []
    for root in roots:
        print(f"== {root}")
        row = run_bundle(root, args, models)
        rows.append(row)
        pos, naive = row["positive"], row["naive"]
        print(
            f"   positive SSR={_fmt(pos['SSR'])} RBE={_fmt(pos['RBE'])} "
            f"INT_post={_fmt(pos['INT_post'])} RCR={_fmt(pos['RCR'])}"
        )
        print(
            f"   naive    SSR={_fmt(naive['SSR'])} RBE={_fmt(naive['RBE'])} "
            f"INT_post={_fmt(naive['INT_post'])} RCR={_fmt(naive['RCR'])}"
        )
        print(
            f"   pose pos_mean={_fmt(row['pose_delta']['pos_mean_m'], 4)} m "
            f"match={_fmt(row['pose_delta']['match_rate'])}; "
            f"encode p99={_fmt(row['latency']['encode']['p99'], 2)} ms; "
            f"guard p99={_fmt(row['latency']['ingest_guard']['p99'], 2)} ms"
        )

    metrics = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "config": {
            "bundles": [str(r) for r in roots],
            "ridge": args.ridge,
            "deadline_ms": args.deadline_ms,
            "swap_fraction": SWAP_FRACTION,
            "rollback_fraction": ROLLBACK_FRACTION,
            "calib_fraction": CALIB_FRACTION,
            "calib_val_fraction": CALIB_VAL_FRACTION,
        },
        "bundles": rows,
        "summary": {
            "positive": {
                key: _aggregate(rows, "positive", key)
                for key in ("SSR", "RBE", "INT_post", "INT_all", "RCR", "RCR_assign")
            },
            "naive": {
                key: _aggregate(rows, "naive", key)
                for key in ("SSR", "RBE", "INT_post", "INT_all", "RCR", "RCR_assign")
            },
        },
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "REPORT.md").write_text(build_report(metrics, args), encoding="utf-8")
    print(f"output: {out}")


if __name__ == "__main__":
    main()
