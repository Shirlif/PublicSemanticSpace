"""U3c: two-branch explicit-geometry + embedding head.

U3b found that *concatenating* 4 coordinates to a 1,536-d embedding input does
not recover the symbolic head's absence cell (seed 0: hybrid 0.892 vs symbolic
0.998).  U3c tests the architectural fix: two separate branches, one on the
explicit relative coordinates and one on the DINOv2 pair feature, summed at the
logit level.  This is the PSS "explicit geometry skeleton + learned relation"
form rather than a single wide MLP.

Example
-------
python -u -m pss.eval.scripts.run_upper_bound_u3c \
  --out eval/results/upper_bound/u3c --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pss._console import ensure_utf8
from pss.eval.readout_reasoning_preregistration import (
    check_readout_reasoning_preregistration,
)
from pss.eval.scripts.run_attribution_grid import _write_json
from pss.eval.scripts.run_upper_bound_u3 import (
    CACHE_FILES,
    _evaluate,
    _learned,
    _load_cache,
    _normalise,
    _pairs,
    _symbolic,
)
from pss.eval.upper_bound import pair_labels, pair_types


class BranchHybridHead(nn.Module):
    """Coordinate branch + embedding branch, summed at the logit level."""

    def __init__(self, dim: int = 384, hidden: int = 64):
        super().__init__()
        self.coord = nn.Sequential(
            nn.Linear(4, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 4),
        )
        self.emb = nn.Sequential(
            nn.Linear(4 * dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 4),
        )

    def forward(self, x_emb: torch.Tensor, x_coord: torch.Tensor) -> torch.Tensor:
        return self.emb(x_emb) + self.coord(x_coord)


def _train_branch(
    Xe: np.ndarray,
    Xc: np.ndarray,
    Y: np.ndarray,
    *,
    dim: int,
    seed: int,
    epochs: int,
    device: str,
) -> BranchHybridHead:
    torch.manual_seed(seed)
    head = BranchHybridHead(dim=dim).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    Xe_t = torch.as_tensor(Xe, dtype=torch.float32, device=device)
    Xc_t = torch.as_tensor(Xc, dtype=torch.float32, device=device)
    Y_t = torch.as_tensor(Y, dtype=torch.float32, device=device)
    for _ in range(epochs):
        perm = torch.randperm(len(Xe_t), device=device)
        for start in range(0, len(perm), 128):
            idx = perm[start : start + 128]
            logits = head.emb(Xe_t[idx]) + head.coord(Xc_t[idx])
            loss = F.binary_cross_entropy_with_logits(logits, Y_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    head.eval()
    return head


def _predict_branch(
    head: BranchHybridHead, Xe: np.ndarray, Xc: np.ndarray, device: str
) -> np.ndarray:
    with torch.no_grad():
        logits = head.emb(torch.as_tensor(Xe, dtype=torch.float32, device=device)) + head.coord(
            torch.as_tensor(Xc, dtype=torch.float32, device=device)
        )
        return (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(bool)


def run(args) -> dict:
    root = Path(args.clevr_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train = _load_cache(root, args.train_split)
    test = _load_cache(root, args.test_split)
    pa_tr, pb_tr, za_tr, zb_tr = _pairs(train, args.train_pairs, args.seed)
    pa_te, pb_te, za_te, zb_te = _pairs(test, args.test_pairs, args.seed + 1)

    lo = np.minimum(pa_tr.min(0), pb_tr.min(0))
    hi = np.maximum(pa_tr.max(0), pb_tr.max(0))
    pa_tr, pb_tr = _normalise(pa_tr, lo, hi), _normalise(pb_tr, lo, hi)
    pa_te, pb_te = _normalise(pa_te, lo, hi), _normalise(pb_te, lo, hi)

    Ytr = pair_labels(pa_tr, pb_tr, eps=args.type_eps)
    Yte = pair_labels(pa_te, pb_te, eps=args.type_eps)
    Tte = pair_types(pa_te, pb_te, eps=args.type_eps)

    Xc_tr, Xc_te = _symbolic(pa_tr, pb_tr), _symbolic(pa_te, pb_te)
    Xe_tr_raw, Xe_te_raw = _learned(za_tr, zb_tr), _learned(za_te, zb_te)
    mu = Xe_tr_raw.mean(0)
    sd = Xe_tr_raw.std(0) + 1e-6
    Xe_tr = (Xe_tr_raw - mu) / sd
    Xe_te = (Xe_te_raw - mu) / sd

    rows = []
    for seed in args.seeds:
        head = _train_branch(
            Xe_tr, Xc_tr, Ytr,
            dim=za_tr.shape[1], seed=seed, epochs=args.epochs, device=args.device,
        )
        pred = _predict_branch(head, Xe_te, Xc_te, args.device)
        row = _evaluate(pred, Yte, Tte)
        rows.append(row)
        print(f"[seed {seed}] branch {row['overall']:.3f}", flush=True)

    branch = {
        "overall": float(np.mean([r["overall"] for r in rows])),
        "worst_cell_acc": float(np.mean([r["worst_cell_acc"] for r in rows])),
        "cells": {
            k: float(np.mean([r["cells"].get(k, 0.0) for r in rows]))
            for k in ("horizontal", "vertical", "diagonal")
        },
        "absence_horizontal": float(np.mean([r["absence_horizontal"] for r in rows])),
        "absence_vertical": float(np.mean([r["absence_vertical"] for r in rows])),
    }

    reference = None
    if args.reference:
        ref_path = Path(args.reference)
        if ref_path.exists():
            reference = json.loads(ref_path.read_text(encoding="utf-8"))
    if reference is None:
        raise FileNotFoundError(
            "U3c needs the U3b reference JSON for the symbolic baseline; "
            f"got {args.reference!r}"
        )
    symbolic_overall = float(reference["symbolic"]["overall"])
    composition_gap = float(symbolic_overall - branch["overall"])
    absence_error = max(branch["absence_horizontal"], branch["absence_vertical"])

    observed = {
        "u_comp_branch": {
            "composition_gap": composition_gap,
            "absence_cell_error": absence_error,
            "test_acc": branch["overall"],
        },
    }
    check = check_readout_reasoning_preregistration(observed)

    payload = {
        "stage": "u3c",
        "clevr_root": str(root),
        "train_split": args.train_split,
        "test_split": args.test_split,
        "train_pairs": int(len(pa_tr)),
        "test_pairs": int(len(pa_te)),
        "type_eps": float(args.type_eps),
        "epochs": int(args.epochs),
        "seeds": list(args.seeds),
        "branch": branch,
        "symbolic_reference": symbolic_overall,
        "composition_gap": composition_gap,
        "absence_cell_error": absence_error,
        "observed": observed,
        "preregistration": check,
        "note": (
            "two branches: explicit relative coords -> 4 logits; DINOv2 pair "
            "feature -> 4 logits; summed.  This is the separate-channel form, "
            "not the naive concat tested in U3b."
        ),
    }
    _write_json(out / "u3c.json", payload)
    (out / "U3C_REPORT.md").write_text(_render(payload), encoding="utf-8")
    return payload


def _render(p: dict) -> str:
    b = p["branch"]
    lines = [
        "# U3c：双分支（显式坐标 + DINOv2）",
        "",
        f"- CLEVR: `{p['clevr_root']}` · train `{p['train_split']}` "
        f"({p['train_pairs']} pairs) · test `{p['test_split']}` ({p['test_pairs']} pairs)",
        f"- epochs {p['epochs']} · seeds {p['seeds']} · pair-type eps {p['type_eps']}",
        "",
        "## 1. 主表",
        "",
        "| representation | horizontal | vertical | diagonal | overall | worst cell | absence error |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| symbolic (reference) | — | — | — | {p['symbolic_reference']:.3f} | — | — |",
        f"| branch hybrid | {b['cells']['horizontal']:.3f} | {b['cells']['vertical']:.3f} | "
        f"{b['cells']['diagonal']:.3f} | {b['overall']:.3f} | {b['worst_cell_acc']:.3f} | "
        f"max({b['absence_horizontal']:.3f}, {b['absence_vertical']:.3f}) |",
        "",
        f"- `L_gap (branch) = symbolic − branch = {p['composition_gap']:+.3f}`",
        f"- `absence_cell_error (branch) = {p['absence_cell_error']:.3f}`",
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
        "- 若 branch hybrid 的 `L_gap ≤ 0.05`，说明缺的是**显式几何通道**，"
        "不是 embedding 容量；U3b 的 naive concat 失败是架构问题（4 维被 1536 维淹没），"
        "不是坐标没用。",
        "- 若 branch 仍高 gap，说明即使分开通道，embedding 也在干扰坐标分支；"
        "应回到纯符号骨架，embedding 只做校准。",
        "",
        "## 4. 边界",
        "",
        "- 参考 symbolic 数字来自 U3b 的 JSON；U3c 只训练 branch 臂。",
        "- 分支在 logit 层相加，没有 learned gate；更复杂的门控未测。",
        "- CLEVR 是渲染数据；DINOv2 是冻结视觉骨干，不是 VLM。",
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
    ap.add_argument("--reference", default="eval/results/upper_bound/u3b/u3.json")
    ap.add_argument("--out", default="eval/results/upper_bound/u3c")
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
