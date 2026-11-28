"""U6: explicit geometry skeleton under sensor noise (sim2real *readiness*).

This is a **sensor-noise stress test**, not a sim2real result.  The repository
ships no real RGB-D / robot data; this script asks a narrower question: when a
clean-trained symbolic relation head is handed noisy geometry, how much of the
U3d accuracy (symbolic 0.996 / residual 0.999, absence 0.043-0.045) survives?

Two noise families, both evaluated on the **same** CLEVR object pairs, dead
zone and pair types as U3/U3d:

1. ``coord``: iid Gaussian noise on the normalised pixel coordinates of each
   object, sigma in {0.005, 0.01, 0.02, 0.04}.  This is the direct
   "geometry-precision sensitivity" axis.
2. ``canonical`` (3D -> 2D): CLEVR scene ``3d_coords`` are projected into a
   table-top (world XY) plane with a pinhole camera calibrated by DLT on the
   training cache.  The camera pose is perturbed by Gaussian yaw/pitch
   (1/3/5 deg, sampled per scene) and per-object depth by Gaussian
   multiplicative noise (0.5/1/2 %).  Pose and depth errors are applied to the
   3D points before the XY projection, so this is a real 3D->2D projection,
   not a pixel-space shortcut.  Combined levels are (1 deg, 0.5 %),
   (3 deg, 1 %), (5 deg, 2 %); pose-only and depth-only variants are reported
   for attribution.

The clean head is trained once per seed on clean geometry.  Test-time labels
and pair types come from clean geometry; only the head input is noisy.  This
matches the deployment question: the sensor is noisy, the physical relation is
not.

Why not ``data/category_pretrain_instance8``?
    That dataset has per-sample ``depth`` and a single object ``pose`` but no
    object-object pairs, no camera intrinsics and no shared scene geometry, so
    it cannot form the relation pairs this task needs.  CLEVR scenes already
    carry ``3d_coords`` for every object, which gives a paired 3D->2D
    projection on exactly the U3 pairs.  Using it is the stronger option; the
    category tabletop proxy is only mentioned as a boundary.

Pre-registered reading rule (fixed before the run)
--------------------------------------------------
``u_geom_noise`` is read at the **highest noise level of each family**, and
the arm is scored on the **worse** of the two readings:

    worst_cell_acc = min(coord sigma=0.04, canonical pose 5 deg + depth 2%)
    delta_vs_full  = min(coord sigma=0.04 - clean pixel,
                         canonical P3 - clean canonical)

``delta_vs_full`` is measured against the matched clean head (same seed, same
test pairs), because "full" here means full sensor fidelity.  Bars:
``worst_cell_acc_min=0.85`` and ``delta_vs_full_min=-0.10``.  A failure is
kept in the report; the bar is never relaxed after seeing the numbers.

Example
-------
python -u -m pss.eval.scripts.run_upper_bound_u6 \
  --out eval/results/upper_bound/u6 --device cuda
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import rq

from pss._console import ensure_utf8
from pss.eval.readout_reasoning_preregistration import (
    check_readout_reasoning_preregistration,
)
from pss.eval.scripts.run_attribution_grid import _write_json
from pss.eval.scripts.run_geometry_validation import _train
from pss.eval.scripts.run_upper_bound_u3 import (
    CACHE_FILES,
    _evaluate,
    _load_cache,
    _normalise,
    _symbolic,
)
from pss.eval.upper_bound import PAIR_TYPES, pair_labels, pair_types

COORD_SIGMAS = (0.005, 0.01, 0.02, 0.04)
# (label, yaw/pitch sigma in degrees, depth sigma as a fraction)
POSE_DEPTH_LEVELS = (
    ("pose1_depth0.5", 1.0, 0.005),
    ("pose3_depth1", 3.0, 0.01),
    ("pose5_depth2", 5.0, 0.02),
)


def _pair_indices(cache: dict, max_pairs: int, seed: int):
    """Mirror ``run_upper_bound_u3._pairs`` but keep the object indices.

    The sampling must stay byte-for-byte identical to U3 so the pairs are the
    same task; only the returned values differ.
    """

    scene = np.asarray(cache["scene"])
    groups: dict[int, list[int]] = defaultdict(list)
    for i, s in enumerate(scene):
        groups[int(s)].append(i)

    pairs = []
    for idxs in groups.values():
        for i in idxs:
            for j in idxs:
                if i != j:
                    pairs.append((i, j))

    rng = np.random.default_rng(seed)
    if len(pairs) > max_pairs:
        chosen = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[int(k)] for k in chosen]

    ia = np.asarray([p[0] for p in pairs], dtype=np.int64)
    ib = np.asarray([p[1] for p in pairs], dtype=np.int64)
    return ia, ib


def _xyz(cache: dict) -> np.ndarray:
    """World 3D centre of every cached object (CLEVR ``3d_coords``)."""

    scenes = cache["scenes"]
    out = np.empty((len(cache["scene"]), 3), dtype=np.float64)
    for i, (si, oi) in enumerate(zip(cache["scene"], cache["obj"])):
        out[i] = scenes[int(si)]["objects"][int(oi)]["3d_coords"]
    return out


def _fit_camera(cache: dict, max_points: int, seed: int) -> dict:
    """DLT pinhole fit ``pixel ~ P [X, 1]`` plus a K/R/t decomposition.

    Only ``R`` and the camera centre ``C`` are used downstream: a camera-pose
    error rotates the 3D points about ``C``.  The fit residual is reported as a
    diagnostic, not as a claim of sensor calibration.
    """

    X = _xyz(cache)
    px = np.asarray(cache["pixel"], dtype=np.float64)[:, :2]
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    if len(idx) > max_points:
        idx = rng.choice(len(idx), size=max_points, replace=False)

    rows = []
    for point, (u, v) in zip(X[idx], px[idx]):
        h = np.array([point[0], point[1], point[2], 1.0])
        rows.append(np.concatenate([h, np.zeros(4), -u * h]))
        rows.append(np.concatenate([np.zeros(4), h, -v * h]))
    _, _, vt = np.linalg.svd(np.asarray(rows))
    P = vt[-1].reshape(3, 4)

    def _decompose(P):
        K, R = rq(P[:, :3])
        d = np.sign(np.diag(K))
        d[d == 0] = 1.0
        D = np.diag(d)
        K, R = K @ D, D @ R
        return K, R

    K, R = _decompose(P)
    if np.linalg.det(R) < 0:  # overall sign ambiguity of the DLT solution
        P = -P
        K, R = _decompose(P)
    if np.linalg.det(R) < 0:
        raise RuntimeError("camera decomposition produced a reflection")
    t = np.linalg.solve(K, P[:, 3])
    C = -R.T @ t

    Xh = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    proj = Xh @ P.T
    proj = proj[:, :2] / proj[:, 2:3]
    rms = float(np.sqrt(np.mean(np.sum((proj - px) ** 2, axis=1))))
    xc = (R @ X.T).T + t
    return {
        "P": P.tolist(),
        "R": R.tolist(),
        "C": C.tolist(),
        "rms_px": rms,
        "depth_z_min": float(xc[:, 2].min()),
        "depth_z_max": float(xc[:, 2].max()),
        "fit_points": int(len(idx)),
        "det_R": float(np.linalg.det(R)),
    }


def _canonical_xy(
    xyz: np.ndarray,
    scene_ids: np.ndarray,
    cam: dict,
    *,
    yaw_deg: float,
    pitch_deg: float,
    depth_sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Perturbed 3D -> table-top XY projection.

    A camera pose error ``R' = R @ Ry(yaw) @ Rx(pitch)`` moves a world point
    about the camera centre by ``M = R'^T R = Rx(-pitch) @ Ry(-yaw)``.
    A depth error scales the ray ``X' = C + (1 + eps)(X - C)``.  Both are
    applied in 3D, then the world XY plane is the 2D observation.
    """

    C = np.asarray(cam["C"], dtype=np.float64)
    pts = np.asarray(xyz, dtype=np.float64).copy()
    n = len(pts)

    unique_scenes, inverse = np.unique(scene_ids, return_inverse=True)
    yaw = (
        np.deg2rad(rng.normal(0.0, yaw_deg, len(unique_scenes)))[inverse]
        if yaw_deg > 0
        else np.zeros(n)
    )
    pitch = (
        np.deg2rad(rng.normal(0.0, pitch_deg, len(unique_scenes)))[inverse]
        if pitch_deg > 0
        else np.zeros(n)
    )

    # Camera pose error R' = R Ry(yaw) Rx(pitch) moves a world point about the
    # camera centre by M = R'^T R = Rx(-pitch) Ry(-yaw).  Written out, the
    # per-object world displacement is:
    #   dx' = cy*dx - sy*dz
    #   dy' = sp*sy*dx + cp*dy + sp*cy*dz
    #   dz' = cp*sy*dx - sp*dy + cp*cy*dz
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    dx, dy, dz = (pts - C).T
    pts[:, 0] = C[0] + cy * dx - sy * dz
    pts[:, 1] = C[1] + sp * sy * dx + cp * dy + sp * cy * dz
    pts[:, 2] = C[2] + cp * sy * dx - sp * dy + cp * cy * dz

    if depth_sigma > 0:
        eps = rng.normal(0.0, depth_sigma, n)
        pts = C + (1.0 + eps)[:, None] * (pts - C)
    return pts[:, :2]


def _predict(head, X: np.ndarray, device: str) -> np.ndarray:
    with torch.no_grad():
        logits = head.net(torch.as_tensor(X, dtype=torch.float32, device=device))
        return (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(bool)


def _summarise(rows: list[dict]) -> dict:
    def stat(key):
        values = [float(r[key]) for r in rows]
        return {"mean": float(np.mean(values)), "std": float(np.std(values))}

    return {
        "overall": stat("overall"),
        "worst_cell_acc": stat("worst_cell_acc"),
        "absence_horizontal": stat("absence_horizontal"),
        "absence_vertical": stat("absence_vertical"),
        "cells": {
            name: {
                "mean": float(np.mean([r["cells"].get(name, 0.0) for r in rows])),
                "std": float(np.std([r["cells"].get(name, 0.0) for r in rows])),
            }
            for name in PAIR_TYPES
        },
        "n": len(rows),
    }


def run(args) -> dict:
    root = Path(args.clevr_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train = _load_cache(root, args.train_split)
    test = _load_cache(root, args.test_split)

    ia_tr, ib_tr = _pair_indices(train, args.train_pairs, args.seed)
    ia_te, ib_te = _pair_indices(test, args.test_pairs, args.seed + 1)

    pix_tr = np.asarray(train["pixel"], dtype=np.float32)[:, :2]
    pix_te = np.asarray(test["pixel"], dtype=np.float32)[:, :2]
    xyz_tr = _xyz(train)
    xyz_te = _xyz(test)
    scene_te = np.asarray(test["scene"])

    # ── primary family: normalised pixel coordinates (exactly the U3 task) ──
    pa_tr, pb_tr = pix_tr[ia_tr], pix_tr[ib_tr]
    pa_te, pb_te = pix_te[ia_te], pix_te[ib_te]
    lo = np.minimum(pa_tr.min(0), pb_tr.min(0))
    hi = np.maximum(pa_tr.max(0), pb_tr.max(0))
    pa_tr, pb_tr = _normalise(pa_tr, lo, hi), _normalise(pb_tr, lo, hi)
    pa_te, pb_te = _normalise(pa_te, lo, hi), _normalise(pb_te, lo, hi)

    Yp_tr = pair_labels(pa_tr, pb_tr, eps=args.type_eps)
    Yp_te = pair_labels(pa_te, pb_te, eps=args.type_eps)
    Tp_te = pair_types(pa_te, pb_te, eps=args.type_eps)
    Xp_tr = _symbolic(pa_tr, pb_tr)
    Xp_te_clean = _symbolic(pa_te, pb_te)

    # ── secondary family: 3D -> table-top XY with pose/depth noise ──────────
    cam = _fit_camera(train, args.camera_points, args.seed)
    ka_tr, kb_tr = xyz_tr[ia_tr][:, :2], xyz_tr[ib_tr][:, :2]
    ka_te, kb_te = xyz_te[ia_te][:, :2], xyz_te[ib_te][:, :2]
    lo_k = np.minimum(ka_tr.min(0), kb_tr.min(0))
    hi_k = np.maximum(ka_tr.max(0), kb_tr.max(0))
    ka_tr, kb_tr = _normalise(ka_tr, lo_k, hi_k), _normalise(kb_tr, lo_k, hi_k)
    ka_te, kb_te = _normalise(ka_te, lo_k, hi_k), _normalise(kb_te, lo_k, hi_k)

    Yk_tr = pair_labels(ka_tr, kb_tr, eps=args.type_eps)
    Yk_te = pair_labels(ka_te, kb_te, eps=args.type_eps)
    Tk_te = pair_types(ka_te, kb_te, eps=args.type_eps)
    Xk_tr = _symbolic(ka_tr, kb_tr)
    Xk_te_clean = _symbolic(ka_te, kb_te)

    print(
        f"[data] train pairs {len(ia_tr)} test pairs {len(ia_te)} "
        f"pixel cells {dict(zip(*np.unique(Tp_te, return_counts=True)))} "
        f"canonical cells {dict(zip(*np.unique(Tk_te, return_counts=True)))}",
        flush=True,
    )
    print(
        f"[camera] rms {cam['rms_px']:.2f} px det(R) {cam['det_R']:+.3f} "
        f"C {np.round(cam['C'], 2).tolist()}",
        flush=True,
    )

    clean_pixel: list[dict] = []
    clean_canonical: list[dict] = []
    coord_rows: dict[float, list[dict]] = {s: [] for s in COORD_SIGMAS}
    canon_rows: dict[str, list[dict]] = {name: [] for name, _, _ in POSE_DEPTH_LEVELS}
    pose_rows: dict[str, list[dict]] = {name: [] for name, _, _ in POSE_DEPTH_LEVELS}
    depth_rows: dict[str, list[dict]] = {name: [] for name, _, _ in POSE_DEPTH_LEVELS}

    for seed in args.seeds:
        head_p = _train(Xp_tr, Yp_tr, "relative", seed, args.epochs, args.device)
        head_k = _train(Xk_tr, Yk_tr, "relative", seed, args.epochs, args.device)

        clean_pixel.append(_evaluate(_predict(head_p, Xp_te_clean, args.device), Yp_te, Tp_te))
        clean_canonical.append(
            _evaluate(_predict(head_k, Xk_te_clean, args.device), Yk_te, Tk_te)
        )

        for i, sigma in enumerate(COORD_SIGMAS):
            rng = np.random.default_rng(seed * 1000 + i)
            pa_n = pa_te + rng.normal(0.0, sigma, pa_te.shape).astype(np.float32)
            pb_n = pb_te + rng.normal(0.0, sigma, pb_te.shape).astype(np.float32)
            pred = _predict(head_p, _symbolic(pa_n, pb_n), args.device)
            coord_rows[sigma].append(_evaluate(pred, Yp_te, Tp_te))

        for j, (name, yaw, depth_sigma) in enumerate(POSE_DEPTH_LEVELS):
            # combined: pose + depth
            rng = np.random.default_rng(seed * 100000 + j + 17)
            xy = _canonical_xy(
                xyz_te, scene_te, cam,
                yaw_deg=yaw, pitch_deg=yaw, depth_sigma=depth_sigma, rng=rng,
            )
            aa, bb = _normalise(xy[ia_te], lo_k, hi_k), _normalise(xy[ib_te], lo_k, hi_k)
            canon_rows[name].append(
                _evaluate(_predict(head_k, _symbolic(aa, bb), args.device), Yk_te, Tk_te)
            )
            # attribution: pose only
            rng = np.random.default_rng(seed * 100000 + j + 401)
            xy = _canonical_xy(
                xyz_te, scene_te, cam,
                yaw_deg=yaw, pitch_deg=yaw, depth_sigma=0.0, rng=rng,
            )
            aa, bb = _normalise(xy[ia_te], lo_k, hi_k), _normalise(xy[ib_te], lo_k, hi_k)
            pose_rows[name].append(
                _evaluate(_predict(head_k, _symbolic(aa, bb), args.device), Yk_te, Tk_te)
            )
            # attribution: depth only
            rng = np.random.default_rng(seed * 100000 + j + 801)
            xy = _canonical_xy(
                xyz_te, scene_te, cam,
                yaw_deg=0.0, pitch_deg=0.0, depth_sigma=depth_sigma, rng=rng,
            )
            aa, bb = _normalise(xy[ia_te], lo_k, hi_k), _normalise(xy[ib_te], lo_k, hi_k)
            depth_rows[name].append(
                _evaluate(_predict(head_k, _symbolic(aa, bb), args.device), Yk_te, Tk_te)
            )

        print(
            f"[seed {seed}] clean pixel {clean_pixel[-1]['overall']:.3f} "
            f"clean canonical {clean_canonical[-1]['overall']:.3f} "
            f"coord0.04 {coord_rows[COORD_SIGMAS[-1]][-1]['overall']:.3f} "
            f"canon(L3) {canon_rows[POSE_DEPTH_LEVELS[-1][0]][-1]['overall']:.3f}",
            flush=True,
        )

    summary = {
        "clean_pixel": _summarise(clean_pixel),
        "clean_canonical": _summarise(clean_canonical),
        "coord": {str(s): _summarise(coord_rows[s]) for s in COORD_SIGMAS},
        "canonical": {name: _summarise(canon_rows[name]) for name, _, _ in POSE_DEPTH_LEVELS},
        "canonical_pose_only": {
            name: _summarise(pose_rows[name]) for name, _, _ in POSE_DEPTH_LEVELS
        },
        "canonical_depth_only": {
            name: _summarise(depth_rows[name]) for name, _, _ in POSE_DEPTH_LEVELS
        },
    }

    # Pre-registered read: worse of the two highest single-family levels.
    coord_high = summary["coord"][str(COORD_SIGMAS[-1])]
    canon_high = summary["canonical"][POSE_DEPTH_LEVELS[-1][0]]
    clean_p = summary["clean_pixel"]
    clean_k = summary["clean_canonical"]
    delta_coord = coord_high["worst_cell_acc"]["mean"] - clean_p["worst_cell_acc"]["mean"]
    delta_canon = canon_high["worst_cell_acc"]["mean"] - clean_k["worst_cell_acc"]["mean"]
    observed = {
        "u_geom_noise": {
            "worst_cell_acc": float(
                min(coord_high["worst_cell_acc"]["mean"], canon_high["worst_cell_acc"]["mean"])
            ),
            "delta_vs_full": float(min(delta_coord, delta_canon)),
            # diagnostics (not asserted by the arm definition)
            "coord_high_sigma": float(COORD_SIGMAS[-1]),
            "coord_high_worst_cell_acc": float(coord_high["worst_cell_acc"]["mean"]),
            "coord_high_overall": float(coord_high["overall"]["mean"]),
            "coord_high_delta_vs_clean": float(delta_coord),
            "canonical_high_level": POSE_DEPTH_LEVELS[-1][0],
            "canonical_high_worst_cell_acc": float(canon_high["worst_cell_acc"]["mean"]),
            "canonical_high_overall": float(canon_high["overall"]["mean"]),
            "canonical_high_delta_vs_clean": float(delta_canon),
            "clean_pixel_worst_cell_acc": float(clean_p["worst_cell_acc"]["mean"]),
            "clean_canonical_worst_cell_acc": float(clean_k["worst_cell_acc"]["mean"]),
        }
    }
    check = check_readout_reasoning_preregistration(observed)

    payload = {
        "stage": "u6",
        "clevr_root": str(root),
        "train_split": args.train_split,
        "test_split": args.test_split,
        "train_pairs": int(len(ia_tr)),
        "test_pairs": int(len(ia_te)),
        "type_eps": float(args.type_eps),
        "epochs": int(args.epochs),
        "seeds": list(args.seeds),
        "coord_sigmas": list(COORD_SIGMAS),
        "pose_depth_levels": [
            {"name": n, "yaw_pitch_deg": y, "depth_sigma": d}
            for n, y, d in POSE_DEPTH_LEVELS
        ],
        "reading_rule": (
            "u_geom_noise is read at the highest level of each family; the arm "
            "takes the worse of coord sigma=0.04 and canonical pose5+depth2, "
            "with delta_vs_full measured against the matched clean head."
        ),
        "camera": cam,
        "summary": summary,
        "observed": observed,
        "preregistration": check,
        "note": (
            "sensor-noise stress test, not sim2real: no real RGB-D, no real "
            "sensor calibration, CLEVR is rendered. Clean heads are trained "
            "on clean geometry; test labels and pair types stay clean while "
            "only the head input is noisy."
        ),
    }
    _write_json(out / "u6.json", payload)
    (out / "U6_RESULTS.md").write_text(_render(payload), encoding="utf-8")
    return payload


def _fmt(stat: dict) -> str:
    return f"{stat['mean']:.3f} ± {stat['std']:.3f}"


def _table(summary: dict, levels: list[tuple[str, dict]]) -> list[str]:
    lines = [
        "| level | horizontal | vertical | diagonal | overall | worst cell | absence err |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in levels:
        cells = row["cells"]
        absence = max(
            row["absence_horizontal"]["mean"], row["absence_vertical"]["mean"]
        )
        lines.append(
            f"| {name} | {_fmt(cells['horizontal'])} | {_fmt(cells['vertical'])} | "
            f"{_fmt(cells['diagonal'])} | {_fmt(row['overall'])} | "
            f"{_fmt(row['worst_cell_acc'])} | {absence:.3f} |"
        )
    return lines


def _render(p: dict) -> str:
    s = p["summary"]
    coord_levels = [("clean", s["clean_pixel"])] + [
        (f"sigma {sigma}", s["coord"][str(sigma)]) for sigma in p["coord_sigmas"]
    ]
    canon_levels = [("clean", s["clean_canonical"])] + [
        (name, s["canonical"][name]) for name, _, _ in POSE_DEPTH_LEVELS
    ]
    pose_levels = [(name, s["canonical_pose_only"][name]) for name, _, _ in POSE_DEPTH_LEVELS]
    depth_levels = [(name, s["canonical_depth_only"][name]) for name, _, _ in POSE_DEPTH_LEVELS]

    lines = [
        "# U6：显式几何骨架的传感器噪声压力测试（sim2real readiness）",
        "",
        f"- CLEVR: `{p['clevr_root']}` · train `{p['train_split']}` "
        f"({p['train_pairs']} pairs) · test `{p['test_split']}` ({p['test_pairs']} pairs)",
        f"- epochs {p['epochs']} · seeds {p['seeds']} · pair-type eps {p['type_eps']}",
        f"- camera DLT: {p['camera']['fit_points']} points, "
        f"rms {p['camera']['rms_px']:.2f} px, det(R) {p['camera']['det_R']:+.3f}, "
        f"C {np.round(p['camera']['C'], 2).tolist()}",
        "",
        "> 这是传感器噪声压力测试，不是 sim2real。仓库没有真实 RGB-D/真机数据；"
        "CLEVR 是渲染几何，3D->2D 投影由 DLT 在训练 cache 上标定。",
        "",
        "## 1. 归一化像素坐标噪声",
        "",
        *_table(s, coord_levels),
        "",
        f"- `u_geom_noise` 在主族最高档 sigma={p['coord_sigmas'][-1]} 的 worst cell: "
        f"{_fmt(s['coord'][str(p['coord_sigmas'][-1])]['worst_cell_acc'])}",
        f"- 相对 clean 的 delta: "
        f"{p['observed']['u_geom_noise']['coord_high_delta_vs_clean']:+.3f}",
        "",
        "## 2. 3D -> 2D 投影：位姿 + 深度噪声",
        "",
        *_table(s, canon_levels),
        "",
        "### 2.1 归因：仅位姿 / 仅深度",
        "",
        "| level | pose-only worst cell | pose-only overall | depth-only worst cell | depth-only overall |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, _, _ in POSE_DEPTH_LEVELS:
        po = s["canonical_pose_only"][name]
        do = s["canonical_depth_only"][name]
        lines.append(
            f"| {name} | {_fmt(po['worst_cell_acc'])} | {_fmt(po['overall'])} | "
            f"{_fmt(do['worst_cell_acc'])} | {_fmt(do['overall'])} |"
        )
    lines += [
        "",
        "## 3. 预注册 `u_geom_noise`",
        "",
        f"- 读数规则: {p['reading_rule']}",
        f"- 检查通过: **{p['preregistration']['ok']}** · checked "
        f"{p['preregistration']['checked']}",
    ]
    if p["preregistration"]["mismatches"]:
        for mismatch in p["preregistration"]["mismatches"]:
            lines.append(f"- `{mismatch}`")
    lines += [
        "",
        "## 4. 边界",
        "",
        "- 不是 sim2real：无真实 RGB-D、无真实传感器标定、CLEVR 是渲染几何。",
        "- 位姿/深度噪声是各向同性高斯代理，不是从真实传感器误差分布拟合的模型。",
        "- 位姿误差按场景共享、深度误差按物体独立；真实标定误差可能相关。",
        "- 3D->2D 使用世界 XY 桌面平面；像素族与 U3 完全一致，canonical 族是同一批 pair 的重投影诊断。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--clevr-root",
        default="data/public/clevr_cogent/CLEVR_CoGenT_v1.0",
    )
    ap.add_argument("--train-split", choices=sorted(CACHE_FILES), default="traina")
    ap.add_argument("--test-split", choices=sorted(CACHE_FILES), default="vala")
    ap.add_argument("--train-pairs", type=int, default=30000)
    ap.add_argument("--test-pairs", type=int, default=10000)
    ap.add_argument("--type-eps", type=float, default=0.02)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--camera-points", type=int, default=4000)
    ap.add_argument("--out", default="eval/results/upper_bound/u6")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    payload = run(args)
    print(json.dumps(payload["observed"], indent=2))
    check = payload["preregistration"]
    print(f"preregistration ok={check['ok']} checked={check['checked']}")
    for mismatch in check["mismatches"]:
        print(f"  - {mismatch}")


if __name__ == "__main__":
    main()
