"""U3d: frozen geometry skeleton + bounded embedding residual.

U3c showed that jointly training a coordinate branch and an embedding branch
recovers the overall gap (L_gap +0.046) but still loses the absence cell
(0.411): the 1,536-d branch can out-vote the 4-d branch during joint training.

U3d tests the PSS form directly: train the two branches **separately**, then
combine logits as ``coord + alpha * emb`` with ``alpha`` chosen on a held-out
slice of the training pairs.  At ``alpha=0`` this is exactly the symbolic
skeleton; the embedding can only calibrate, not override.

Example
-------
python -u -m pss.eval.scripts.run_upper_bound_u3d \
  --out eval/results/upper_bound/u3d --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pss._console import ensure_utf8
from pss.eval.readout_reasoning_preregistration import (
    check_readout_reasoning_preregistration,
)
from pss.eval.scripts.run_attribution_grid import _write_json
from pss.eval.scripts.run_geometry_validation import _train
from pss.eval.scripts.run_upper_bound_u3 import (
    CACHE_FILES,
    _evaluate,
    _learned,
    _load_cache,
    _normalise,
    _pairs,
    _symbolic,
    _train_learned,
)
from pss.eval.upper_bound import exact4, pair_labels, pair_types


def _logits(head, X: np.ndarray, device: str) -> np.ndarray:
    with torch.no_grad():
        return head.net(torch.as_tensor(X, dtype=torch.float32, device=device)).cpu().numpy()


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

    rng = np.random.default_rng(args.seed)
    n = len(Xc_tr)
    perm = rng.permutation(n)
    n_cal = max(1, int(round(args.cal_fraction * n)))
    cal_idx, fit_idx = perm[:n_cal], perm[n_cal:]

    rows = []
    for seed in args.seeds:
        head_c = _train(Xc_tr[fit_idx], Ytr[fit_idx], "relative", seed, args.epochs, args.device)
        head_e = _train_learned(
            Xe_tr[fit_idx], Ytr[fit_idx], dim=za_tr.shape[1],
            seed=seed, epochs=args.epochs, device=args.device,
        )
        cal_c = _logits(head_c, Xc_tr[cal_idx], args.device)
        cal_e = _logits(head_e, Xe_tr[cal_idx], args.device)
        cal_y = Ytr[cal_idx].astype(bool)

        best = None
        for alpha in args.alpha_grid:
            bits = torch.sigmoid(
                torch.as_tensor(cal_c + float(alpha) * cal_e)
            ).numpy() > 0.5
            score = exact4(cal_y, bits)
            if best is None or score > best["score"]:
                best = {"alpha": float(alpha), "score": float(score)}

        te_c = _logits(head_c, Xc_te, args.device)
        te_e = _logits(head_e, Xe_te, args.device)
        pred = torch.sigmoid(
            torch.as_tensor(te_c + best["alpha"] * te_e)
        ).numpy() > 0.5
        row = _evaluate(pred.astype(bool), Yte, Tte)
        row["alpha"] = best["alpha"]
        row["cal_score"] = best["score"]
        rows.append(row)
        print(
            f"[seed {seed}] alpha {best['alpha']} cal {best['score']:.3f} "
            f"test {row['overall']:.3f}",
            flush=True,
        )

    residual = {
        "overall": float(np.mean([r["overall"] for r in rows])),
        "worst_cell_acc": float(np.mean([r["worst_cell_acc"] for r in rows])),
        "cells": {
            k: float(np.mean([r["cells"].get(k, 0.0) for r in rows]))
            for k in ("horizontal", "vertical", "diagonal")
        },
        "absence_horizontal": float(np.mean([r["absence_horizontal"] for r in rows])),
        "absence_vertical": float(np.mean([r["absence_vertical"] for r in rows])),
        "alpha": float(np.mean([r["alpha"] for r in rows])),
        "alpha_values": [r["alpha"] for r in rows],
    }

    reference = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    symbolic_overall = float(reference["symbolic"]["overall"])
    composition_gap = float(symbolic_overall - residual["overall"])
    absence_error = max(residual["absence_horizontal"], residual["absence_vertical"])

    observed = {
        "u_comp_residual": {
            "composition_gap": composition_gap,
            "absence_cell_error": absence_error,
            "test_acc": residual["overall"],
        },
    }
    check = check_readout_reasoning_preregistration(observed)

    payload = {
        "stage": "u3d",
        "clevr_root": str(root),
        "train_split": args.train_split,
        "test_split": args.test_split,
        "train_pairs": int(len(pa_tr)),
        "test_pairs": int(len(pa_te)),
        "type_eps": float(args.type_eps),
        "epochs": int(args.epochs),
        "seeds": list(args.seeds),
        "alpha_grid": list(args.alpha_grid),
        "cal_fraction": float(args.cal_fraction),
        "residual": residual,
        "symbolic_reference": symbolic_overall,
        "composition_gap": composition_gap,
        "absence_cell_error": absence_error,
        "observed": observed,
        "preregistration": check,
        "note": (
            "coordinate and embedding branches trained separately; logits "
            "combined as coord + alpha*emb; alpha chosen on a held-out slice "
            "of the training pairs, test reported once."
        ),
    }
    _write_json(out / "u3d.json", payload)
    (out / "U3D_REPORT.md").write_text(_render(payload), encoding="utf-8")
    return payload


def _render(p: dict) -> str:
    r = p["residual"]
    lines = [
        "# U3d：几何骨架 + 有界 embedding 残差",
        "",
        f"- CLEVR: `{p['clevr_root']}` · train `{p['train_split']}` "
        f"({p['train_pairs']} pairs) · test `{p['test_split']}` ({p['test_pairs']} pairs)",
        f"- epochs {p['epochs']} · seeds {p['seeds']} · alpha grid {p['alpha_grid']}",
        f"- calibration slice: {p['cal_fraction']:.0%} of train pairs",
        "",
        "## 1. 主表",
        "",
        "| representation | horizontal | vertical | diagonal | overall | worst cell | absence error |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| symbolic (reference) | — | — | — | {p['symbolic_reference']:.3f} | — | — |",
        f"| coord + α·emb | {r['cells']['horizontal']:.3f} | {r['cells']['vertical']:.3f} | "
        f"{r['cells']['diagonal']:.3f} | {r['overall']:.3f} | {r['worst_cell_acc']:.3f} | "
        f"max({r['absence_horizontal']:.3f}, {r['absence_vertical']:.3f}) |",
        "",
        f"- chosen alpha: {r['alpha_values']} (mean {r['alpha']:.3f})",
        f"- `L_gap = symbolic − residual = {p['composition_gap']:+.3f}`",
        f"- `absence_cell_error = {p['absence_cell_error']:.3f}`",
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
        "- 若 alpha 选到 0 或很小，且 absence error 回到 symbolic 水平："
        "说明显式几何必须是主导通道，embedding 只能做有界校准；U3c 的 joint 干扰是训练方式问题。",
        "- 若 alpha 较大且 absence 仍高：说明 embedding 的信息与几何冲突，"
        "应完全分离，embedding 只用于非几何谓词。",
        "",
        "## 4. 边界",
        "",
        "- alpha 在训练对的 held-out 切片上选择，test 只报一次。",
        "- 两个分支分开训练，不是端到端联合；这正是“显式骨架 + 学习校准”的形态。",
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
    ap.add_argument("--cal-fraction", type=float, default=0.1)
    ap.add_argument("--alpha-grid", nargs="+", type=float, default=[0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0])
    ap.add_argument("--reference", default="eval/results/upper_bound/u3/u3.json")
    ap.add_argument("--out", default="eval/results/upper_bound/u3d")
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
