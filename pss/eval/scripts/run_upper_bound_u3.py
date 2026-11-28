"""U3: composition fidelity of the learned public space (symbolic vs DINOv2).

Executable form of ``docs/plans/PSS-架构上界复核与下一阶段.md`` §6 (U3).

The B line measures relation composition on symbolic coordinates; the A line
measures recognition on learned embeddings.  U3 puts both on the *same* CLEVR
object pairs and asks whether the learned space preserves what the relations
need:

  symbolic   [dx, dy, |dx|, |dy|]                     (4-d, the B-line feature)
  learned    [z_a, z_b, z_b - z_a, |z_b - z_a|]       (4 x 384-d DINOv2)

Both heads have the same depth and hidden width; only the input dimension
differs, which is reported with parameter counts.  ``L_gap`` is the symbolic
minus learned exact-4.  If it is <= 0.05 the public space is compositionally
lossless enough for this task; if it is >= 0.10, losslessness is a real
bottleneck and belongs in f, not g.

Example
-------
python -u -m pss.eval.scripts.run_upper_bound_u3 \
  --out eval/results/upper_bound/u3 --device cuda
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
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
from pss.eval.scripts.run_geometry_validation import _train
from pss.eval.scripts.run_typed_relations import _pair_feature
from pss.eval.upper_bound import (
    PAIR_TYPES,
    absence_cell_error,
    cell_accuracy,
    exact4,
    pair_labels,
    pair_types,
    worst_cell_accuracy,
)

CACHE_FILES = {
    "traina": "traina_8000.pt",
    "vala": "vala_3000.pt",
    "valb": "valb_3000.pt",
}


class EmbeddingRelationHead(nn.Module):
    """Same depth/width as ``SpatialRelationHead``, wider input."""

    def __init__(self, dim: int = 384, hidden: int = 64):
        super().__init__()
        self.dim = int(dim)
        self.net = nn.Sequential(
            nn.Linear(4 * self.dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 4),
        )


class HybridRelationHead(nn.Module):
    """Embedding pair feature plus the explicit relative coordinates."""

    def __init__(self, dim: int = 384, hidden: int = 64):
        super().__init__()
        self.dim = int(dim)
        self.net = nn.Sequential(
            nn.Linear(4 * self.dim + 4, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 4),
        )


def _load_cache(root: Path, split: str) -> dict:
    return torch.load(root / "cache" / CACHE_FILES[split], map_location="cpu", weights_only=False)


def _pairs(cache: dict, max_pairs: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return pixel coords and DINOv2 embeddings for ordered object pairs."""

    scene = np.asarray(cache["scene"])
    pixel = np.asarray(cache["pixel"], dtype=np.float32)[:, :2]
    emb = np.asarray(cache["embeddings"], dtype=np.float32)
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
    return pixel[ia], pixel[ib], emb[ia], emb[ib]


def _normalise(points: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    span = np.where(hi > lo, hi - lo, 1.0)
    return (points - lo) / span


def _symbolic(pa: np.ndarray, pb: np.ndarray) -> np.ndarray:
    return np.asarray(
        [_pair_feature(pa[i], pb[i], "relative") for i in range(len(pa))],
        dtype=np.float32,
    )


def _learned(za: np.ndarray, zb: np.ndarray) -> np.ndarray:
    diff = zb - za
    return np.concatenate([za, zb, diff, np.abs(diff)], axis=1).astype(np.float32)


def _train_learned(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    dim: int,
    seed: int,
    epochs: int,
    device: str,
    hybrid: bool = False,
) -> nn.Module:
    torch.manual_seed(seed)
    head = (
        HybridRelationHead(dim=dim) if hybrid else EmbeddingRelationHead(dim=dim)
    ).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
    Y_t = torch.as_tensor(Y, dtype=torch.float32, device=device)
    for _ in range(epochs):
        perm = torch.randperm(len(X_t), device=device)
        for start in range(0, len(perm), 128):
            idx = perm[start : start + 128]
            loss = F.binary_cross_entropy_with_logits(head.net(X_t[idx]), Y_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    head.eval()
    return head


def _predict_learned(head: nn.Module, X: np.ndarray, device: str) -> np.ndarray:
    with torch.no_grad():
        logits = head.net(torch.as_tensor(X, dtype=torch.float32, device=device))
        return (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(bool)


def _evaluate(pred: np.ndarray, Y: np.ndarray, T: np.ndarray) -> dict:
    cells = cell_accuracy(T, Y, pred)
    return {
        "overall": exact4(Y, pred),
        "worst_cell_acc": worst_cell_accuracy(T, Y, pred),
        "cells": cells,
        "absence_horizontal": absence_cell_error(T, Y, pred, "horizontal"),
        "absence_vertical": absence_cell_error(T, Y, pred, "vertical"),
    }


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

    # The dead zone is the geometric absence case: pairs within ``type_eps``
    # of an axis have neither of that axis's predicates true.  The symbolic
    # head can read it from the coordinates; the learned head must recover it
    # from DINOv2 embeddings alone.
    Ytr = pair_labels(pa_tr, pb_tr, eps=args.type_eps)
    Yte = pair_labels(pa_te, pb_te, eps=args.type_eps)
    Tte = pair_types(pa_te, pb_te, eps=args.type_eps)
    Ttr = pair_types(pa_tr, pb_tr, eps=args.type_eps)
    print(
        f"[data] train pairs {len(pa_tr)} test pairs {len(pa_te)} "
        f"test cells {dict(zip(*np.unique(Tte, return_counts=True)))}",
        flush=True,
    )

    Xs_tr, Xs_te = _symbolic(pa_tr, pb_tr), _symbolic(pa_te, pb_te)
    Xl_tr_raw, Xl_te_raw = _learned(za_tr, zb_tr), _learned(za_te, zb_te)
    mu = Xl_tr_raw.mean(0)
    sd = Xl_tr_raw.std(0) + 1e-6
    Xl_tr = (Xl_tr_raw - mu) / sd
    Xl_te = (Xl_te_raw - mu) / sd

    # Hybrid: the learned embedding plus the explicit relative coordinates.
    # This is the PSS "explicit geometry + learned relation" form, not a pure
    # embedding readout.
    Xh_tr_raw = np.concatenate([Xl_tr_raw, Xs_tr], axis=1)
    Xh_te_raw = np.concatenate([Xl_te_raw, Xs_te], axis=1)
    mu_h = Xh_tr_raw.mean(0)
    sd_h = Xh_tr_raw.std(0) + 1e-6
    Xh_tr = (Xh_tr_raw - mu_h) / sd_h
    Xh_te = (Xh_te_raw - mu_h) / sd_h

    symbolic_rows, learned_rows, hybrid_rows = [], [], []
    for seed in args.seeds:
        head_s = _train(Xs_tr, Ytr, "relative", seed, args.epochs, args.device)
        with torch.no_grad():
            ps = (torch.sigmoid(head_s.net(torch.as_tensor(Xs_te, dtype=torch.float32, device=args.device))) > 0.5).cpu().numpy().astype(bool)
        symbolic_rows.append(_evaluate(ps, Yte, Tte))

        head_l = _train_learned(
            Xl_tr, Ytr, dim=za_tr.shape[1], seed=seed, epochs=args.epochs, device=args.device
        )
        pl = _predict_learned(head_l, Xl_te, args.device)
        learned_rows.append(_evaluate(pl, Yte, Tte))

        head_h = _train_learned(
            Xh_tr, Ytr, dim=za_tr.shape[1], seed=seed, epochs=args.epochs,
            device=args.device, hybrid=True,
        )
        ph = _predict_learned(head_h, Xh_te, args.device)
        hybrid_rows.append(_evaluate(ph, Yte, Tte))
        print(
            f"[seed {seed}] symbolic {symbolic_rows[-1]['overall']:.3f} "
            f"learned {learned_rows[-1]['overall']:.3f} "
            f"hybrid {hybrid_rows[-1]['overall']:.3f}",
            flush=True,
        )

    def _mean(rows, key):
        return float(np.mean([r[key] for r in rows]))

    symbolic = {
        "overall": _mean(symbolic_rows, "overall"),
        "worst_cell_acc": _mean(symbolic_rows, "worst_cell_acc"),
        "cells": {k: float(np.mean([r["cells"].get(k, 0.0) for r in symbolic_rows])) for k in PAIR_TYPES},
        "absence_horizontal": _mean(symbolic_rows, "absence_horizontal"),
        "absence_vertical": _mean(symbolic_rows, "absence_vertical"),
    }
    learned = {
        "overall": _mean(learned_rows, "overall"),
        "worst_cell_acc": _mean(learned_rows, "worst_cell_acc"),
        "cells": {k: float(np.mean([r["cells"].get(k, 0.0) for r in learned_rows])) for k in PAIR_TYPES},
        "absence_horizontal": _mean(learned_rows, "absence_horizontal"),
        "absence_vertical": _mean(learned_rows, "absence_vertical"),
    }
    hybrid = {
        "overall": _mean(hybrid_rows, "overall"),
        "worst_cell_acc": _mean(hybrid_rows, "worst_cell_acc"),
        "cells": {k: float(np.mean([r["cells"].get(k, 0.0) for r in hybrid_rows])) for k in PAIR_TYPES},
        "absence_horizontal": _mean(hybrid_rows, "absence_horizontal"),
        "absence_vertical": _mean(hybrid_rows, "absence_vertical"),
    }
    composition_gap = float(symbolic["overall"] - learned["overall"])
    absence_error = max(learned["absence_horizontal"], learned["absence_vertical"])
    hybrid_gap = float(symbolic["overall"] - hybrid["overall"])
    hybrid_absence = max(hybrid["absence_horizontal"], hybrid["absence_vertical"])

    observed = {
        "u_comp_symbolic": {"test_acc": symbolic["overall"]},
        "u_comp_learned": {
            "composition_gap": composition_gap,
            "absence_cell_error": absence_error,
            "test_acc": learned["overall"],
        },
        "u_comp_hybrid": {
            "composition_gap": hybrid_gap,
            "absence_cell_error": hybrid_absence,
            "test_acc": hybrid["overall"],
        },
    }
    check = check_readout_reasoning_preregistration(observed)

    payload = {
        "stage": "u3",
        "clevr_root": str(root),
        "train_split": args.train_split,
        "test_split": args.test_split,
        "train_pairs": int(len(pa_tr)),
        "test_pairs": int(len(pa_te)),
        "type_eps": float(args.type_eps),
        "epochs": int(args.epochs),
        "seeds": list(args.seeds),
        "symbolic": symbolic,
        "learned": learned,
        "hybrid": hybrid,
        "composition_gap": composition_gap,
        "absence_cell_error": absence_error,
        "hybrid_gap": hybrid_gap,
        "hybrid_absence_cell_error": hybrid_absence,
        "params": {
            "symbolic": int(sum(p.numel() for p in _train(Xs_tr[:1], Ytr[:1], "relative", 0, 0, "cpu").parameters())),
            "learned": int(sum(p.numel() for p in EmbeddingRelationHead(dim=za_tr.shape[1]).parameters())),
            "hybrid": int(sum(p.numel() for p in HybridRelationHead(dim=za_tr.shape[1]).parameters())),
        },
        "observed": observed,
        "preregistration": check,
        "note": (
            "same CLEVR pairs, same head depth/width, same training budget; "
            "symbolic sees coordinates, learned sees DINOv2 embeddings only, "
            "hybrid sees embeddings + explicit relative coordinates. "
            "L_gap <= 0.05 means compositionally lossless enough; >= 0.10 means "
            "losslessness is a real bottleneck."
        ),
    }
    _write_json(out / "u3.json", payload)
    (out / "U3_REPORT.md").write_text(_render(payload), encoding="utf-8")
    return payload


def _render(p: dict) -> str:
    s, l, h = p["symbolic"], p["learned"], p["hybrid"]
    lines = [
        "# U3：公共空间的组合保真度（符号坐标 vs DINOv2 vs 混合）",
        "",
        f"- CLEVR: `{p['clevr_root']}` · train `{p['train_split']}` "
        f"({p['train_pairs']} pairs) · test `{p['test_split']}` ({p['test_pairs']} pairs)",
        f"- epochs {p['epochs']} · seeds {p['seeds']} · pair-type eps {p['type_eps']}",
        "",
        "## 1. 主表",
        "",
        "| representation | horizontal | vertical | diagonal | overall | worst cell | absence error |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| symbolic (4-d) | {s['cells']['horizontal']:.3f} | {s['cells']['vertical']:.3f} | "
        f"{s['cells']['diagonal']:.3f} | {s['overall']:.3f} | {s['worst_cell_acc']:.3f} | "
        f"max({s['absence_horizontal']:.3f}, {s['absence_vertical']:.3f}) |",
        f"| learned (DINOv2) | {l['cells']['horizontal']:.3f} | {l['cells']['vertical']:.3f} | "
        f"{l['cells']['diagonal']:.3f} | {l['overall']:.3f} | {l['worst_cell_acc']:.3f} | "
        f"max({l['absence_horizontal']:.3f}, {l['absence_vertical']:.3f}) |",
        f"| hybrid (DINOv2 + coords) | {h['cells']['horizontal']:.3f} | {h['cells']['vertical']:.3f} | "
        f"{h['cells']['diagonal']:.3f} | {h['overall']:.3f} | {h['worst_cell_acc']:.3f} | "
        f"max({h['absence_horizontal']:.3f}, {h['absence_vertical']:.3f}) |",
        "",
        f"- `L_gap (learned) = symbolic − learned = {p['composition_gap']:+.3f}`",
        f"- `L_gap (hybrid)  = symbolic − hybrid  = {p['hybrid_gap']:+.3f}`",
        f"- `absence_cell_error (learned) = {p['absence_cell_error']:.3f}`",
        f"- `absence_cell_error (hybrid)  = {p['hybrid_absence_cell_error']:.3f}`",
        f"- params: symbolic {p['params']['symbolic']}, learned {p['params']['learned']}, "
        f"hybrid {p['params']['hybrid']}",
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
        "- `L_gap ≤ 0.05`：公共空间对该任务“组合上够无损”，B3 可判；无损度不进入 f 瓶颈。",
        "- `L_gap ≥ 0.10`：学习表示丢掉了关系所需的信息；无损度是 f 的真实因子，"
        "不能声称公共空间支持组合。",
        "- `hybrid` 是 PSS 的“显式几何 + 学习关系”形态：若它把 `L_gap` 压到 ≤0.05，"
        "说明缺的不是容量，而是表示里的几何通道；若它也修不好，问题在标签/任务。",
        "- 若 symbolic 本身不高，`L_gap` 小也不能证明无损；要看两列的绝对值。",
        "",
        "## 4. 边界",
        "",
        "- CLEVR 是渲染数据；DINOv2 是冻结视觉骨干，不是 VLM。",
        "- 学习头的输入维度（4×384）远大于符号头（4），参数量已报告；"
        "这是表示比较，不是严格等容量比较。",
        "- pair type 用像素容差 `eps` 定义；标签用同一 dead zone，因此 near-aligned "
        "是真正的 absence case（该轴谓词全为假），不是近似。",
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
    ap.add_argument("--out", default="eval/results/upper_bound/u3")
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
