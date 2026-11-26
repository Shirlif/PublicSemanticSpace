"""Claim-gap F4: heterogeneous-width expert adapter.

E11 swapped three *same-width* expert families through one frozen projection
head.  F4 asks whether the property suite survives experts whose native widths
differ (DINOv2 384, CLIP 768, WT-MM 64).  Each expert gets:

- its own projection head, trained on its own head-fit split;
- its own calibration set, disjoint from the head-fit split;
- the same admission contract, evaluated in that expert's projected space.

The script reports k=0/1/5/20 calibration curves and the three admission probes
(near-duplicate, novel class, insufficient samples) for every width.

Runtime code is untouched: this only reads frozen features and fits heads.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from pss._console import ensure_utf8
from pss.encoders.hetero import (
    REQUIRED_WIDTHS,
    build_hetero_expert,
    get_expert_spec,
)
from pss.encoders.tactile import load_tactile_expert
from pss.encoders.vision import load_vision_expert
from pss.projection.features import load_paired_features
from pss.projection.hetero import (
    CONTRACT_CASES,
    EXPECTED_CONTRACT,
    run_width_family,
)
from pss.prototypes.bank import PrototypeBank


def _resolve_experts(experts, include_controls: bool) -> list[str]:
    names = [str(x).strip() for x in experts if str(x).strip()]
    if include_controls:
        for control in ("vision_smallcnn_256", "tactile_mlp_64"):
            if control not in names:
                names.append(control)
    if not names:
        raise ValueError("no experts selected")
    for name in names:
        get_expert_spec(name)  # fail fast on typos
    return names


def run(
    data: str = "data/category_pretrain",
    base_experts: str = "checkpoints/expert_v1",
    wtmm: str = "checkpoints/tactile_wtmm",
    prototypes: str = "checkpoints/prototypes_v1",
    experts=REQUIRED_WIDTHS,
    ks=(0, 1, 5, 20),
    seed: int = 0,
    device: str | None = None,
    epochs: int = 80,
    hidden: int = 256,
    max_per_class: int = 40,
    far_class: str = "plate",
    allow_download: bool = False,
    head_dir: str | None = "checkpoints/hetero_width",
    save_heads: bool = True,
) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    names = _resolve_experts(list(experts), include_controls=False)
    prototype_path = Path(prototypes) / "prototypes.npz"
    base_bank = PrototypeBank.load(prototype_path)

    base_vision = load_vision_expert(Path(base_experts) / "vision.pt", device=device)
    base_tactile = load_tactile_expert(Path(base_experts) / "tactile.pt", device=device)

    rows = []
    for name in names:
        spec = get_expert_spec(name)
        expert = build_hetero_expert(
            spec,
            base_expert_dir=base_experts,
            wtmm_dir=wtmm,
            device=device,
            allow_download=allow_download,
        )
        if spec.modality == "vision":
            vision, tactile = expert, base_tactile
        else:
            vision, tactile = base_vision, expert
        feats = load_paired_features(data, vision, tactile)
        head_path = (
            Path(head_dir) / f"{name}.pt"
            if save_heads and head_dir
            else None
        )
        row = run_width_family(
            feats,
            spec,
            base_bank,
            ks=tuple(int(k) for k in ks),
            seed=seed,
            device=device,
            epochs=epochs,
            hidden=hidden,
            max_per_class=max_per_class,
            far_class=far_class,
            save_head_path=head_path,
        )
        rows.append(row)
        curve = row["curves"]["ridge_to_frozen"]
        print(
            f"{name:24s} in_dim={row['in_dim']:3d} "
            f"k1={curve.get('1', float('nan')):.3f} "
            f"contract={'ok' if row['admission']['consistent_with_expected'] else 'FAIL'}"
        )

    required = [r for r in rows if r["name"] in REQUIRED_WIDTHS]
    vision_k1 = {
        r["name"]: r["curves"]["ridge_to_frozen"].get("1")
        for r in required
        if r["modality"] == "vision"
    }
    vision_k1_ok = (
        len(vision_k1) == 2
        and all(v is not None and float(v) >= 0.90 for v in vision_k1.values())
    )
    contract_ok = all(bool(r["admission"]["consistent_with_expected"]) for r in required)
    contract_decisions = {
        r["name"]: {
            case: r["admission"][case]["decision"] for case in CONTRACT_CASES
        }
        for r in required
    }

    return {
        "base": {
            "data": str(data),
            "experts": str(base_experts),
            "wtmm": str(wtmm),
            "prototypes": str(prototypes),
            "device": device,
        },
        "ks": [int(k) for k in ks],
        "seed": int(seed),
        "epochs": int(epochs),
        "rows": rows,
        "summary": {
            "required_experts": list(REQUIRED_WIDTHS),
            "widths": sorted({int(r["in_dim"]) for r in required}),
            "vision_k1": vision_k1,
            "vision_k1_ge_0_90": bool(vision_k1_ok),
            "contract_expected": dict(EXPECTED_CONTRACT),
            "contract_decisions": contract_decisions,
            "contract_consistent": bool(contract_ok),
        },
    }


def _fmt(value, digits: int = 3) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "nan"
    if np.isnan(v):
        return "nan"
    return f"{v:.{digits}f}"


def _report(metrics: dict) -> str:
    s = metrics["summary"]
    base = metrics["base"]
    lines = [
        "# Claim-gap F4: heterogeneous-width expert adapter",
        "",
        f"> data: `{base['data']}`",
        f"> base experts: `{base['experts']}`; WT-MM: `{base['wtmm']}`",
        f"> prototypes: `{base['prototypes']}`; device: `{base['device']}`",
        f"> ks: {metrics['ks']}; seed: {metrics['seed']}; head epochs: {metrics['epochs']}",
        "",
        "## Widths",
        "",
        "| expert | modality | native width | head-fit | calib/class | test |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in metrics["rows"]:
        calib = r["counts"]["calib_per_class"]
        per_class = min(calib.values()) if calib else 0
        lines.append(
            f"| {r['name']} | {r['modality']} | {r['in_dim']} | "
            f"{r['counts']['head_fit']} | {per_class} | {r['counts']['test']} |"
        )

    lines += [
        "",
        "## Calibration curves: ridge into the frozen public space",
        "",
        "Same protocol as E11: fit a linear map from the expert's projected features",
        "to the frozen prototype vectors with k examples per class.",
        "",
        "| expert | width | k=0 | k=1 | k=5 | k=20 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in metrics["rows"]:
        c = r["curves"]["ridge_to_frozen"]
        lines.append(
            f"| {r['name']} | {r['in_dim']} | {_fmt(c.get('0'))} | "
            f"{_fmt(c.get('1'))} | {_fmt(c.get('5'))} | {_fmt(c.get('20'))} |"
        )

    lines += [
        "",
        "## Few-shot prototype curves: independent head + calibration set",
        "",
        "k>=1 builds prototypes from k calibration examples per class in the",
        "expert's own projected space; k=0 is the frozen-bank zero-shot reference.",
        "",
        "| expert | width | k=0 | k=1 | k=5 | k=20 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in metrics["rows"]:
        c = r["curves"]["few_shot_prototype"]
        lines.append(
            f"| {r['name']} | {r['in_dim']} | {_fmt(c.get('0'))} | "
            f"{_fmt(c.get('1'))} | {_fmt(c.get('5'))} | {_fmt(c.get('20'))} |"
        )

    lines += [
        "",
        "## Admission contract across widths",
        "",
        "All three probes use the expert's held-out samples and the contract's own",
        "data-driven threshold (`calibrated_thresholds`). Expected: near-duplicate",
        "= reject, novel class = accept, insufficient = review.",
        "",
        "| expert | width | near-duplicate | novel class | insufficient "
        "| near dist | novel dist | threshold |",
        "|---|---:|---|---|---|---:|---:|---:|",
    ]
    for r in metrics["rows"]:
        a = r["admission"]
        lines.append(
            f"| {r['name']} | {r['in_dim']} | {a['near_dup']['decision']} | "
            f"{a['novel_class']['decision']} | {a['insufficient']['decision']} | "
            f"{_fmt(a['near_dup']['min_inter_distance'])} | "
            f"{_fmt(a['novel_class']['min_inter_distance'])} | "
            f"{_fmt(a['near_threshold'])} |"
        )

    lines += [
        "",
        "## Verdict",
        "",
        f"- Required widths: {s['widths']}",
        f"- Vision k=1 >= 0.90: **{'yes' if s['vision_k1_ge_0_90'] else 'no'}** "
        f"({', '.join(f'{k}={_fmt(v)}' for k, v in s['vision_k1'].items())})",
        f"- Admission contract consistent across required widths: "
        f"**{'yes' if s['contract_consistent'] else 'no'}**",
        "",
        "| expert | near-duplicate | novel class | insufficient |",
        "|---|---|---|---|",
    ]
    for name, decisions in s["contract_decisions"].items():
        lines.append(
            f"| {name} | {decisions['near_dup']} | {decisions['novel_class']} | "
            f"{decisions['insufficient']} |"
        )

    lines += [
        "",
        "## How to read this",
        "",
        "- The projection head is trained on one disjoint subset of each expert's",
        "  train split; the calibration set is the other subset.  No test sample",
        "  enters either.",
        "- `k=0` uses the frozen base codebook with the new head, so a low value is",
        "  expected: an independently trained head does not land in the shared",
        "  public space without calibration.  `k>=1` is the calibrated condition.",
        "- The admission bank is built in the expert's own projected space from the",
        "  full train split; the probes use val+test samples of the near class",
        "  (`block`, in the bank) and a far class (`plate`, removed from the bank).",
        "- WT-MM is a domain bridge: a real-GelSight CNN applied to upsampled 8x8",
        "  synthetic tactile.  Its curves are expected to stay well below vision.",
        "- `vision_dinov2_384` uses `facebook/dinov2-small`; `vision_clip_768` uses",
        "  `openai/clip-vit-base-patch16` (768-d), not the registry default ViT-L/14.",
        "  Both must be present in the local HF cache; pass `--allow-download` to",
        "  fetch a missing snapshot.",
        "- Runtime code is untouched; only frozen features and offline heads are",
        "  used.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "python -m pss.eval.scripts.run_gap_families_width \\",
        f"  --data {base['data']} \\",
        f"  --base-experts {base['experts']} \\",
        f"  --wtmm {base['wtmm']} \\",
        f"  --prototypes {base['prototypes']} \\",
        f"  --ks {','.join(str(k) for k in metrics['ks'])} \\",
        f"  --out reports/claim_gaps/families_width",
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/category_pretrain")
    ap.add_argument("--base-experts", default="checkpoints/expert_v1")
    ap.add_argument("--wtmm", default="checkpoints/tactile_wtmm")
    ap.add_argument("--prototypes", default="checkpoints/prototypes_v1")
    ap.add_argument(
        "--experts",
        default=",".join(REQUIRED_WIDTHS),
        help="comma-separated heterogeneous expert names",
    )
    ap.add_argument("--include-controls", action="store_true")
    ap.add_argument("--ks", default="0,1,5,20")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--max-per-class", type=int, default=40)
    ap.add_argument("--far-class", default="plate")
    ap.add_argument("--allow-download", action="store_true")
    ap.add_argument("--head-dir", default="checkpoints/hetero_width")
    ap.add_argument("--no-save-heads", action="store_true")
    ap.add_argument("--out", default="reports/claim_gaps/families_width")
    args = ap.parse_args()

    experts = [x for x in args.experts.split(",") if x.strip()]
    if args.include_controls:
        for control in ("vision_smallcnn_256", "tactile_mlp_64"):
            if control not in experts:
                experts.append(control)
    metrics = run(
        data=args.data,
        base_experts=args.base_experts,
        wtmm=args.wtmm,
        prototypes=args.prototypes,
        experts=experts,
        ks=tuple(int(x) for x in args.ks.split(",") if x.strip()),
        seed=args.seed,
        device=args.device,
        epochs=args.epochs,
        hidden=args.hidden,
        max_per_class=args.max_per_class,
        far_class=args.far_class,
        allow_download=args.allow_download,
        head_dir=args.head_dir,
        save_heads=not args.no_save_heads,
    )
    metrics["generated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "REPORT.md").write_text(_report(metrics), encoding="utf-8")

    s = metrics["summary"]
    print(
        f"vision k=1 >= 0.90: {s['vision_k1_ge_0_90']} ({s['vision_k1']}); "
        f"contract consistent: {s['contract_consistent']}"
    )
    print(f"output: {out}")


if __name__ == "__main__":
    main()
