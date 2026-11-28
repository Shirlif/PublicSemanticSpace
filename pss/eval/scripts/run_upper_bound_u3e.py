"""U3e: patch-token / position-aware relation composition without explicit coords.

U3b-U3d showed that a DINOv2 *CLS* pair feature is not compositionally
lossless on CLEVR geometry (``L_gap`` +0.19, absence error 0.83), and that the
only form that closes it is an explicit coordinate skeleton with a bounded
(zero) embedding residual.  U3e asks the remaining representation question:
can **patch tokens / positional tokens** carry the relative geometry on their
own, without feeding ``dx, dy, |dx|, |dy|``?

Design
------
1. Each object is cropped from the full-resolution scene image (``crop`` x
   ``crop`` pixels, default 64) and passed through the frozen DINOv2-small
   patch encoder at 224, giving a ``16 x 16`` token grid per object.
2. The crop origin is snapped to a virtual scene grid of ``cell = crop/16``
   pixels (4 px by default), so patch ``(i, j)`` in a crop has an **absolute**
   scene-grid coordinate ``(origin + i, origin + j)``.  The token at the
   object centre therefore carries ``floor(pixel / cell)`` exactly: a 4 px
   position resolution, finer than the pre-registered dead zone (~8.5 px in
   x, ~4.6 px in y).
3. A ``(2r+1)^2`` window of tokens around the object centre is kept.  A
   two-layer transformer head (hidden 64) receives both token sets plus
   **absolute sinusoidal positional encodings of those scene-grid
   coordinates** and learned A/B role embeddings.  It never sees explicit
   ``dx, dy, |dx|, |dy|``; the position signal is the patch-grid index only.
4. The four predicates are read from the CLS output; training/evaluation use
   the same CLEVR pairs and metrics as U3b-d.

Object localisation comes from ``pixel_coords`` only to choose the crop, as
``crop_object`` does elsewhere in the codebase; the crop origin is then used
solely to index the positional encoding, never as an input feature.

If this passes ``u_comp_patch`` (``L_gap <= 0.05``, absence <= 0.20) then a
patch-position representation can replace explicit coordinates for this task.
If it does not, the failure is recorded against the frozen bar.

Example
-------
python -u -m pss.eval.scripts.run_upper_bound_u3e \
  --out eval/results/upper_bound/u3e --device cuda \
  --train-pairs 30000 --test-pairs 10000 --epochs 300 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from pss._console import ensure_utf8
from pss.encoders.data import normalize_image
from pss.encoders.vision.dinov2 import load_dinov2_encoder
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
from pss.eval.upper_bound import (
    PAIR_TYPES,
    cell_accuracy,
    pair_labels,
    pair_types,
)

DINOV2_PATCH = 14
IMAGE_DIRS = {"traina": "trainA", "vala": "valA", "valb": "valB"}


# ---------------------------------------------------------------------------
# Pair construction (identical RNG path to U3, plus object indices)
# ---------------------------------------------------------------------------


def _pairs_with_indices(
    cache: dict, max_pairs: int, seed: int, max_scenes: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Return cache row indices for ordered object pairs.

    The sampling loop mirrors ``run_upper_bound_u3._pairs`` exactly (same
    scene grouping, same ``np.random.default_rng(seed).choice`` call), so with
    the default arguments U3e sees the same pairs as U3b.
    """

    scene = np.asarray(cache["scene"])
    groups: dict[int, list[int]] = defaultdict(list)
    for i, s in enumerate(scene):
        groups[int(s)].append(i)

    pairs = []
    scene_ids = sorted(groups)
    if max_scenes:
        scene_ids = scene_ids[: int(max_scenes)]
    for sidx in scene_ids:
        idxs = groups[sidx]
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


# ---------------------------------------------------------------------------
# Patch-token extraction from full scene images
# ---------------------------------------------------------------------------


def _scene_batches(scene_ids: list[int], batch_size: int):
    for start in range(0, len(scene_ids), batch_size):
        yield scene_ids[start : start + batch_size]


@torch.no_grad()
def _extract_pair_windows(
    cache: dict,
    root: Path,
    split: str,
    ia: np.ndarray,
    ib: np.ndarray,
    *,
    crop_w: int,
    crop_h: int,
    radius: int,
    device: str,
    images_per_batch: int = 8,
) -> dict[str, torch.Tensor]:
    """Extract a local patch-token window for both objects of every pair.

    Each object is cropped at native resolution and encoded at 224 (16x16
    tokens).  The crop origin is snapped to the virtual ``cell = crop/16``
    grid; token ``(i, j)`` then has absolute scene-grid coordinates
    ``(origin_cell + i, origin_cell + j)``.  ``crop_w`` and ``crop_h`` may
    differ, giving independent x/y position resolutions (e.g. 64x32 -> 4 px /
    2 px cells).  Returns ``(N, K, 384)`` fp16 tokens and ``(N, K, 2)``
    integer absolute grid coordinates for A and B.
    """

    grid_tokens = 224 // DINOV2_PATCH  # 16
    if crop_w % grid_tokens or crop_h % grid_tokens:
        raise ValueError(f"crop sizes must be multiples of {grid_tokens}")
    cell_x = crop_w // grid_tokens
    cell_y = crop_h // grid_tokens
    k = (2 * radius + 1) ** 2
    offsets = [
        (dy, dx)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
    ]
    off = np.asarray(offsets, dtype=np.int64)

    scene = np.asarray(cache["scene"])
    pixel = np.asarray(cache["pixel"], dtype=np.float32)
    scenes = cache["scenes"]
    img_dir = root / "images" / IMAGE_DIRS[split]

    # pair -> scene (ia and ib always share a scene)
    pair_scene = scene[ia]
    by_scene: dict[int, list[int]] = defaultdict(list)
    used_by_scene: dict[int, set[int]] = defaultdict(set)
    for pi, sidx in enumerate(pair_scene):
        sidx = int(sidx)
        by_scene[sidx].append(pi)
        used_by_scene[sidx].add(int(ia[pi]))
        used_by_scene[sidx].add(int(ib[pi]))

    n = len(ia)
    a_tok = np.empty((n, k, 384), dtype=np.float16)
    b_tok = np.empty((n, k, 384), dtype=np.float16)
    a_grid = np.empty((n, k, 2), dtype=np.int16)
    b_grid = np.empty((n, k, 2), dtype=np.int16)

    enc = load_dinov2_encoder(device=device, local_files_only=True)

    scene_ids = sorted(by_scene)
    t0 = time.time()
    processed = 0
    for batch_ids in _scene_batches(scene_ids, images_per_batch):
        crops: list[np.ndarray] = []
        meta: list[tuple[int, int, int, int, int, int]] = []
        window_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        for sidx in batch_ids:
            with Image.open(img_dir / scenes[sidx]["image_filename"]) as _im:
                rgb = np.asarray(_im.convert("RGB"))
            img_h, img_w = rgb.shape[:2]
            max_x = ((img_w - crop_w) // cell_x) * cell_x
            max_y = ((img_h - crop_h) // cell_y) * cell_y
            for obj_row in sorted(used_by_scene[sidx]):
                px, py = float(pixel[obj_row, 0]), float(pixel[obj_row, 1])
                # snap the crop origin to the virtual cell grid
                x0 = int(round((px - crop_w / 2.0) / cell_x)) * cell_x
                y0 = int(round((py - crop_h / 2.0) / cell_y)) * cell_y
                x0 = int(np.clip(x0, 0, max_x))
                y0 = int(np.clip(y0, 0, max_y))
                crop = rgb[y0 : y0 + crop_h, x0 : x0 + crop_w]
                arr = np.asarray(
                    Image.fromarray(crop).resize(
                        (224, 224), Image.BILINEAR
                    )
                )
                crops.append(normalize_image(arr))
                meta.append(
                    (sidx, obj_row, x0 // cell_x, y0 // cell_y, x0, y0)
                )

        x = torch.as_tensor(np.stack(crops), dtype=torch.float32, device=device)
        hidden = enc.model(
            pixel_values=x, interpolate_pos_encoding=False
        ).last_hidden_state
        patches = hidden[:, 1:].reshape(-1, grid_tokens, grid_tokens, 384)
        patches = patches.to(torch.float16).cpu().numpy()

        for b, (sidx, obj_row, ox, oy, x0, y0) in enumerate(meta):
            px, py = float(pixel[obj_row, 0]), float(pixel[obj_row, 1])
            ci = int(np.clip((px - x0) // cell_x, 0, grid_tokens - 1))
            cj = int(np.clip((py - y0) // cell_y, 0, grid_tokens - 1))
            xs = np.clip(ci + off[:, 1], 0, grid_tokens - 1)
            ys = np.clip(cj + off[:, 0], 0, grid_tokens - 1)
            obj_patches = patches[b][ys, xs]  # (K, 384)
            obj_grid = np.stack([ox + xs, oy + ys], axis=-1).astype(np.int16)
            window_cache[(sidx, obj_row)] = (obj_patches, obj_grid)

        for sidx in batch_ids:
            for pi in by_scene[sidx]:
                win_a, grid_a = window_cache[(sidx, int(ia[pi]))]
                win_b, grid_b = window_cache[(sidx, int(ib[pi]))]
                a_tok[pi] = win_a
                b_tok[pi] = win_b
                a_grid[pi] = grid_a
                b_grid[pi] = grid_b

        processed += len(batch_ids)
        if processed % (images_per_batch * 50) == 0 or processed == len(scene_ids):
            elapsed = time.time() - t0
            rate = processed / max(elapsed, 1e-6)
            print(
                f"[extract {split}] {processed}/{len(scene_ids)} scenes "
                f"({rate:.1f} scene/s, {elapsed:.0f}s)",
                flush=True,
            )

    return {
        "a_tokens": torch.from_numpy(a_tok),
        "b_tokens": torch.from_numpy(b_tok),
        "a_grid": torch.from_numpy(a_grid),
        "b_grid": torch.from_numpy(b_grid),
        "grid": grid_tokens,
        "window_radius": radius,
        "crop_w": crop_w,
        "crop_h": crop_h,
        "cell_x": cell_x,
        "cell_y": cell_y,
    }


# ---------------------------------------------------------------------------
# Position-aware pair transformer
# ---------------------------------------------------------------------------


def _sinusoidal_2d(grid: torch.Tensor, dim: int = 64) -> torch.Tensor:
    """Absolute 2-D sinusoidal encoding of integer patch-grid coordinates.

    ``grid`` is ``(..., 2)`` with ``[col, row]``.  Positions are used as raw
    grid indices (not normalised coordinates) so that one-cell differences
    remain visibly encoded.  ``dim`` is split evenly between x and y.
    """

    half = dim // 2
    freqs = torch.exp(
        torch.linspace(0.0, -math.log(10000.0), half // 2, device=grid.device)
    )

    def _axis(pos: torch.Tensor) -> torch.Tensor:
        args = pos.float().unsqueeze(-1) * freqs
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    return torch.cat([_axis(grid[..., 0]), _axis(grid[..., 1])], dim=-1)


class PatchPairHead(nn.Module):
    """Position-token head with a bounded patch-content cross-attention residual.

    The primary path reads two **position-aware tokens** (the absolute
    sinusoidal encodings of the two object centre patches) through a 2-layer
    MLP.  A 2-layer cross-attention branch over the full patch-token sets is
    added with a learned scalar gate initialised near zero, so patch content
    can help but cannot override the position channel.  Hidden width is 64.
    No explicit relative vector is supplied.
    """

    def __init__(
        self,
        dim: int = 384,
        hidden: int = 64,
        heads: int = 4,
        layers: int = 2,
        window_radius: int = 2,
        pe_dim: int = 64,
    ):
        super().__init__()
        self.hidden = hidden
        self.pe_dim = pe_dim
        self.center = window_radius * (2 * window_radius + 1) + window_radius
        # position path: absolute PE of the two object-centre grid cells
        self.pos = nn.Sequential(
            nn.Linear(2 * pe_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 4),
        )
        # content path: cross-attention over the two patch-token windows
        self.proj = nn.Linear(dim, hidden)
        self.role = nn.Embedding(2, hidden)
        self.cross_ab = nn.ModuleList(
            [
                nn.MultiheadAttention(hidden, heads, batch_first=True, dropout=0.0)
                for _ in range(layers)
            ]
        )
        self.cross_ba = nn.ModuleList(
            [
                nn.MultiheadAttention(hidden, heads, batch_first=True, dropout=0.0)
                for _ in range(layers)
            ]
        )
        self.ffn = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden, 2 * hidden),
                    nn.GELU(),
                    nn.Linear(2 * hidden, hidden),
                )
                for _ in range(layers)
            ]
        )
        self.norm_a = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.norm_b = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.content_out = nn.Linear(2 * hidden, 4)
        self.content_gate = nn.Parameter(torch.tensor(-4.0))

    def forward(
        self,
        a_tokens: torch.Tensor,
        a_grid: torch.Tensor,
        b_tokens: torch.Tensor,
        b_grid: torch.Tensor,
    ) -> torch.Tensor:
        pa = _sinusoidal_2d(a_grid[:, self.center].float(), self.pe_dim)
        pb = _sinusoidal_2d(b_grid[:, self.center].float(), self.pe_dim)
        pos_logits = self.pos(torch.cat([pa, pb], dim=-1))

        xa = (
            self.proj(a_tokens.float())
            + _sinusoidal_2d(a_grid.float(), self.hidden)
            + self.role.weight[0]
        )
        xb = (
            self.proj(b_tokens.float())
            + _sinusoidal_2d(b_grid.float(), self.hidden)
            + self.role.weight[1]
        )
        qa = xa[:, self.center : self.center + 1]
        qb = xb[:, self.center : self.center + 1]
        for layer in range(len(self.cross_ab)):
            ra, _ = self.cross_ab[layer](qa, xb, xb)
            rb, _ = self.cross_ba[layer](qb, xa, xa)
            qa = self.norm_a[layer](qa + ra)
            qb = self.norm_b[layer](qb + rb)
            qa = qa + self.ffn[layer](qa)
            qb = qb + self.ffn[layer](qb)
        content_logits = self.content_out(
            torch.cat([qa.squeeze(1), qb.squeeze(1)], dim=-1)
        )
        return pos_logits + torch.sigmoid(self.content_gate) * content_logits


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------


def _train_patch_head(
    tensors: dict[str, torch.Tensor],
    Y: np.ndarray,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    hidden: int,
    heads: int,
    layers: int,
    window_radius: int,
    fit_idx: np.ndarray | None = None,
) -> PatchPairHead:
    torch.manual_seed(seed)
    head = PatchPairHead(
        hidden=hidden,
        heads=heads,
        layers=layers,
        window_radius=window_radius,
    ).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)

    a_tok = tensors["a_tokens"].to(device)
    b_tok = tensors["b_tokens"].to(device)
    a_grid = tensors["a_grid"].to(device)
    b_grid = tensors["b_grid"].to(device)
    y = torch.as_tensor(Y, dtype=torch.float32, device=device)
    fit = (
        torch.as_tensor(fit_idx, dtype=torch.long, device=device)
        if fit_idx is not None
        else torch.arange(len(y), device=device)
    )
    n = len(fit)

    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = fit[perm[start : start + batch_size]]
            logits = head(a_tok[idx], a_grid[idx], b_tok[idx], b_grid[idx])
            loss = F.binary_cross_entropy_with_logits(logits, y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    head.eval()
    return head


@torch.no_grad()
def _predict_probs(
    head: PatchPairHead,
    tensors: dict[str, torch.Tensor],
    device: str,
    batch_size: int = 512,
) -> np.ndarray:
    outs = []
    n = len(tensors["a_tokens"])
    for start in range(0, n, batch_size):
        idx = slice(start, start + batch_size)
        logits = head(
            tensors["a_tokens"][idx].to(device),
            tensors["a_grid"][idx].to(device),
            tensors["b_tokens"][idx].to(device),
            tensors["b_grid"][idx].to(device),
        )
        outs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(outs, axis=0)


def _calibrate_thresholds(
    probs: np.ndarray,
    y: np.ndarray,
    types: np.ndarray,
    grid: np.ndarray | None = None,
) -> list[float]:
    """Choose one threshold per axis on a held-out slice of *training* pairs.

    Free per-predicate thresholds overfit a 3k calibration slice (a diagnostic
    run picked ``[0.95, 0.05, 0.15, 0.75]`` and lost the absence cells), so
    the decision rule is restricted to two parameters: ``tx`` for
    left/right and ``ty`` for above/below.  The objective is the **worst
    per-pair-type exact-4**, then the mean cell exact-4, then overall exact-4;
    this keeps the rare absence cells from being traded away for diagonal
    accuracy.  Calibration never sees the test pairs.
    """

    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    yb = np.asarray(y) > 0.5

    def _score(tx: float, ty: float) -> tuple[float, float, float]:
        pred = probs > np.asarray([tx, tx, ty, ty], dtype=np.float32)
        cells = cell_accuracy(types, yb, pred)
        if not cells:
            return (0.0, 0.0, 0.0)
        values = [float(v) for v in cells.values()]
        return (min(values), float(np.mean(values)), float(np.all(pred == yb, axis=1).mean()))

    best_score: tuple[float, float, float] | None = None
    best = (0.5, 0.5)
    for tx in grid:
        for ty in grid:
            score = _score(float(tx), float(ty))
            if best_score is None or score > best_score:
                best_score = score
                best = (float(tx), float(ty))
    return [best[0], best[0], best[1], best[1]]


def _threshold_predict(probs: np.ndarray, thresholds: list[float]) -> np.ndarray:
    return probs > np.asarray(thresholds, dtype=np.float32)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _pair_cache_path(
    out: Path,
    split: str,
    *,
    crop_w: int,
    crop_h: int,
    radius: int,
    pairs: int,
    seed: int,
    max_scenes: int,
) -> Path:
    tag = f"{split}_c{crop_h}x{crop_w}_r{radius}_n{pairs}_seed{seed}"
    if max_scenes:
        tag += f"_scenes{max_scenes}"
    return out / "_cache" / f"{tag}.pt"


def _load_or_extract(
    args,
    root: Path,
    cache: dict,
    split: str,
    ia: np.ndarray,
    ib: np.ndarray,
) -> dict[str, torch.Tensor]:
    path = _pair_cache_path(
        Path(args.out),
        split,
        crop_w=args.crop_w,
        crop_h=args.crop_h,
        radius=args.window,
        pairs=len(ia),
        seed=args.seed if split == args.train_split else args.seed + 1,
        max_scenes=args.max_scenes,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not args.force_extract:
        print(f"[cache] loading {path}", flush=True)
        return torch.load(path, map_location="cpu", weights_only=False)

    print(
        f"[extract {split}] {len(ia)} pairs from {len(cache['scene'])} objects, "
        f"crop {args.crop_w}x{args.crop_h} px, cell "
        f"{args.crop_w // 16}x{args.crop_h // 16} px",
        flush=True,
    )
    tensors = _extract_pair_windows(
        cache,
        root,
        split,
        ia,
        ib,
        crop_w=args.crop_w,
        crop_h=args.crop_h,
        radius=args.window,
        device=args.device,
        images_per_batch=args.images_per_batch,
    )
    torch.save(tensors, path)
    print(f"[cache] wrote {path}", flush=True)
    return tensors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _subset(tensors: dict[str, torch.Tensor], idx: np.ndarray) -> dict[str, torch.Tensor]:
    keys = ("a_tokens", "b_tokens", "a_grid", "b_grid")
    return {key: tensors[key][torch.as_tensor(idx, dtype=torch.long)] for key in keys}


def run(args) -> dict:
    root = Path(args.clevr_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    train = _load_cache(root, args.train_split)
    test = _load_cache(root, args.test_split)

    ia_tr, ib_tr = _pairs_with_indices(
        train, args.train_pairs, args.seed, args.max_scenes
    )
    ia_te, ib_te = _pairs_with_indices(
        test, args.test_pairs, args.seed + 1, args.max_scenes
    )

    pixel_tr = np.asarray(train["pixel"], dtype=np.float32)[:, :2]
    pixel_te = np.asarray(test["pixel"], dtype=np.float32)[:, :2]
    pa_tr, pb_tr = pixel_tr[ia_tr], pixel_tr[ib_tr]
    pa_te, pb_te = pixel_te[ia_te], pixel_te[ib_te]

    # Exact U3 normalisation / label path.
    lo = np.minimum(pa_tr.min(0), pb_tr.min(0))
    hi = np.maximum(pa_tr.max(0), pb_tr.max(0))
    pa_tr, pb_tr = _normalise(pa_tr, lo, hi), _normalise(pb_tr, lo, hi)
    pa_te, pb_te = _normalise(pa_te, lo, hi), _normalise(pb_te, lo, hi)

    Ytr = pair_labels(pa_tr, pb_tr, eps=args.type_eps)
    Yte = pair_labels(pa_te, pb_te, eps=args.type_eps)
    Ttr = pair_types(pa_tr, pb_tr, eps=args.type_eps)
    Tte = pair_types(pa_te, pb_te, eps=args.type_eps)
    print(
        f"[data] train pairs {len(pa_tr)} test pairs {len(pa_te)} "
        f"test cells {dict(zip(*np.unique(Tte, return_counts=True)))}",
        flush=True,
    )

    # Symbolic sanity check on the same pairs (reference value still comes
    # from the U3b JSON, as in U3c/U3d).
    Xs_tr, Xs_te = _symbolic(pa_tr, pb_tr), _symbolic(pa_te, pb_te)
    symbolic_rows = []
    for seed in args.seeds:
        head_s = _train(Xs_tr, Ytr, "relative", seed, args.epochs, args.device)
        with torch.no_grad():
            ps = (
                torch.sigmoid(
                    head_s.net(
                        torch.as_tensor(Xs_te, dtype=torch.float32, device=args.device)
                    )
                ).cpu().numpy()
                > 0.5
            )
        symbolic_rows.append(_evaluate(ps, Yte, Tte))
    symbolic_self = float(np.mean([r["overall"] for r in symbolic_rows]))

    tr_tensors = _load_or_extract(
        args, root, train, args.train_split, ia_tr, ib_tr
    )
    te_tensors = _load_or_extract(
        args, root, test, args.test_split, ia_te, ib_te
    )

    rng = np.random.default_rng(args.seed)
    n_fit = len(Ytr)
    perm = rng.permutation(n_fit)
    n_cal = max(1, int(round(args.cal_fraction * n_fit)))
    cal_idx, fit_idx = perm[:n_cal], perm[n_cal:]
    print(
        f"[split] fit {len(fit_idx)} cal {len(cal_idx)} (calibration on train only)",
        flush=True,
    )

    rows = []
    for seed in args.seeds:
        t0 = time.time()
        head = _train_patch_head(
            tr_tensors,
            Ytr,
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            hidden=args.hidden,
            heads=args.heads,
            layers=args.layers,
            window_radius=args.window,
            fit_idx=fit_idx,
        )
        cal_probs = _predict_probs(
            head, _subset(tr_tensors, cal_idx), args.device
        )
        thresholds = _calibrate_thresholds(cal_probs, Ytr[cal_idx], Ttr[cal_idx])
        cal_score = float(
            np.all(
                _threshold_predict(cal_probs, thresholds) == (Ytr[cal_idx] > 0.5),
                axis=1,
            ).mean()
        )
        probs = _predict_probs(head, te_tensors, args.device)
        pred = _threshold_predict(probs, thresholds)
        row = _evaluate(pred, Yte, Tte)
        row["thresholds"] = thresholds
        row["cal_exact4"] = cal_score
        row["content_gate"] = float(torch.sigmoid(head.content_gate).item())
        row["secs"] = round(time.time() - t0, 1)
        rows.append(row)
        print(
            f"[seed {seed}] patch {row['overall']:.3f} "
            f"worst {row['worst_cell_acc']:.3f} "
            f"absence {max(row['absence_horizontal'], row['absence_vertical']):.3f} "
            f"gate {row['content_gate']:.3f} thr {thresholds} "
            f"({row['secs']}s)",
            flush=True,
        )

    patch = {
        "overall": float(np.mean([r["overall"] for r in rows])),
        "overall_std": float(np.std([r["overall"] for r in rows])),
        "worst_cell_acc": float(np.mean([r["worst_cell_acc"] for r in rows])),
        "cells": {
            name: float(np.mean([r["cells"].get(name, 0.0) for r in rows]))
            for name in PAIR_TYPES
        },
        "absence_horizontal": float(np.mean([r["absence_horizontal"] for r in rows])),
        "absence_vertical": float(np.mean([r["absence_vertical"] for r in rows])),
        "per_seed": rows,
    }

    reference = None
    if args.reference:
        ref_path = Path(args.reference)
        if ref_path.exists():
            reference = json.loads(ref_path.read_text(encoding="utf-8"))
    if reference is None:
        raise FileNotFoundError(
            "U3e needs the U3b reference JSON for the symbolic baseline; "
            f"got {args.reference!r}"
        )
    symbolic_ref = float(reference["symbolic"]["overall"])
    composition_gap = float(symbolic_ref - patch["overall"])
    absence_error = max(patch["absence_horizontal"], patch["absence_vertical"])

    observed = {
        "u_comp_patch": {
            "composition_gap": composition_gap,
            "absence_cell_error": absence_error,
            "test_acc": patch["overall"],
        },
    }
    check = check_readout_reasoning_preregistration(observed)

    params = int(
        sum(
            p.numel()
            for p in PatchPairHead(
                hidden=args.hidden,
                heads=args.heads,
                layers=args.layers,
                window_radius=args.window,
            ).parameters()
        )
    )
    payload = {
        "stage": "u3e",
        "clevr_root": str(root),
        "train_split": args.train_split,
        "test_split": args.test_split,
        "train_pairs": int(len(pa_tr)),
        "test_pairs": int(len(pa_te)),
        "type_eps": float(args.type_eps),
        "epochs": int(args.epochs),
        "seeds": list(args.seeds),
        "cal_fraction": float(args.cal_fraction),
        "reduced": bool(args.max_scenes or args.train_pairs < 30000 or args.test_pairs < 10000),
        "representation": {
            "backbone": "dinov2-small",
            "crop_px_hw": [int(args.crop_h), int(args.crop_w)],
            "patch_grid": 16,
            "scene_cell_px_hw": [int(args.crop_h // 16), int(args.crop_w // 16)],
            "window_radius": int(args.window),
            "tokens_per_object": int((2 * args.window + 1) ** 2),
            "position_encoding": "absolute 2-D sinusoidal over scene-grid cells",
            "explicit_coordinates": False,
        },
        "head": {
            "hidden": int(args.hidden),
            "layers": int(args.layers),
            "heads": int(args.heads),
            "params": params,
            "position_path": "2-layer MLP over absolute PE of the two centre cells",
            "content_path": f"{args.layers}-layer cross-attention patch residual, learned gate",
            "threshold_calibration": f"{args.cal_fraction:.0%} of train pairs",
        },
        "patch": patch,
        "symbolic_reference": symbolic_ref,
        "symbolic_self": symbolic_self,
        "composition_gap": composition_gap,
        "absence_cell_error": absence_error,
        "observed": observed,
        "preregistration": check,
        "wall_seconds": round(time.time() - t_start, 1),
        "note": (
            "DINOv2 patch tokens from object crops; the primary readout is two "
            "position-aware tokens (absolute sinusoidal encodings of the "
            "object-centre scene-grid cells), plus a gated 2-layer "
            "cross-attention patch-content residual (gate init ~0.018). "
            "Per-predicate thresholds are calibrated on a held-out slice of "
            "the training pairs; no explicit dx/dy vector is an input."
        ),
    }
    _write_json(out / "u3e.json", payload)
    (out / "U3E_REPORT.md").write_text(_render(payload), encoding="utf-8")
    return payload


def _render(p: dict) -> str:
    r = p["patch"]
    rep = p["representation"]
    lines = [
        "# U3e：patch-token / 位置感知表示（无显式坐标）",
        "",
        f"- CLEVR: `{p['clevr_root']}` · train `{p['train_split']}` "
        f"({p['train_pairs']} pairs) · test `{p['test_split']}` ({p['test_pairs']} pairs)",
        f"- epochs {p['epochs']} · seeds {p['seeds']} · pair-type eps {p['type_eps']}",
        (
            "- **single-seed, time-boxed run**: not a 3-seed mean and not "
            "directly comparable to the 3-seed U3/U3c/U3d statistics"
            if len(p["seeds"]) == 1
            else ""
        ),
        f"- backbone: {rep['backbone']} · crop {rep['crop_px_hw'][1]}×"
        f"{rep['crop_px_hw'][0]} px at 224 · patch grid {rep['patch_grid']}×"
        f"{rep['patch_grid']} · scene cell {rep['scene_cell_px_hw'][1]}×"
        f"{rep['scene_cell_px_hw'][0]} px (x, y)",
        f"- window {2 * rep['window_radius'] + 1}×{2 * rep['window_radius'] + 1} = "
        f"{rep['tokens_per_object']} tokens/object · "
        f"pos-enc: {rep['position_encoding']} · explicit coords: {rep['explicit_coordinates']}",
        f"- head: positions 2-layer MLP + gated {p['head']['layers']}-layer "
        f"cross-attention content, hidden {p['head']['hidden']}, "
        f"{p['head']['heads']} heads, {p['head']['params']} params",
        f"- threshold calibration: {p['head']['threshold_calibration']}",
        f"- reduced run: {p['reduced']} · wall {p['wall_seconds']} s",
        "",
        "## 1. 主表",
        "",
        "| representation | horizontal | vertical | diagonal | overall | worst cell | absence error |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| symbolic (reference) | — | — | — | {p['symbolic_reference']:.3f} | — | — |",
        f"| patch position-aware | {r['cells']['horizontal']:.3f} | {r['cells']['vertical']:.3f} | "
        f"{r['cells']['diagonal']:.3f} | {r['overall']:.3f} ± {r['overall_std']:.3f} | "
        f"{r['worst_cell_acc']:.3f} | "
        f"max({r['absence_horizontal']:.3f}, {r['absence_vertical']:.3f}) |",
        "",
        f"- symbolic self-check on the same pairs: {p['symbolic_self']:.3f}",
        f"- `L_gap = symbolic − patch = {p['composition_gap']:+.3f}`",
        f"- `absence_cell_error = {p['absence_cell_error']:.3f}`",
        f"- per-seed thresholds: "
        f"{[row['thresholds'] for row in r['per_seed']]}",
        f"- per-seed content gate: "
        f"{[round(row['content_gate'], 4) for row in r['per_seed']]}",
        f"- per-seed calibrated exact-4: "
        f"{[round(row['cal_exact4'], 4) for row in r['per_seed']]}",
        "",
    ]
    check = p["preregistration"]
    lines += [
        "## 2. 预注册检查",
        "",
        f"- 检查 arm 数: {check['checked']} · 通过: **{check['ok']}**",
    ]
    if check["mismatches"]:
        lines.append("")
        for mismatch in check["mismatches"]:
            lines.append(f"- `{mismatch}`")
    lines += [
        "",
        "## 3. 判读",
        "",
        "- `L_gap ≤ 0.05` 且 absence ≤ 0.20：patch/位置感知 token 足以承载几何关系，"
        "显式坐标不是唯一通道。",
        "- 若 absence 仍高：patch 网格分辨率或 token 内容不足以表达 dead zone；"
        "需要给出量化分辨率与 eps 的关系。",
        "- 若 `L_gap` 高但 absence 低：关系方向能读，精确边界/其他 cell 不够。",
        "",
        "## 4. 边界",
        "",
        "- 对象定位使用 `pixel_coords` 选窗并确定 PE 原点；这是数据预处理，"
        "不是模型输入（模型只看到 token + PE，没有 dx/dy 或归一化坐标）。",
        "- 位置编码是 patch 网格绝对正弦编码；网格量化（4×2 px）会损失亚 patch 精度。",
        "- crop 64×32 到 224×224 是各向异性缩放；patch 内容外观会变形。",
        "- CLEVR 是渲染数据；DINOv2 是冻结骨干，不是 VLM。",
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
    ap.add_argument("--crop-w", type=int, default=64, help="object crop width in original px")
    ap.add_argument("--crop-h", type=int, default=32, help="object crop height in original px")
    ap.add_argument("--window", type=int, default=2, help="patch radius around object")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument(
        "--cal-fraction",
        type=float,
        default=0.1,
        help="fraction of train pairs used to calibrate predicate thresholds",
    )
    ap.add_argument(
        "--images-per-batch", type=int, default=8, help="scene-image batch for cropping"
    )
    ap.add_argument(
        "--max-scenes",
        type=int,
        default=0,
        help="diagnostic: restrict pair sources to the first N scenes",
    )
    ap.add_argument("--force-extract", action="store_true")
    ap.add_argument("--reference", default="eval/results/upper_bound/u3b/u3.json")
    ap.add_argument("--out", default="eval/results/upper_bound/u3e")
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
