"""Summarize attribution_v2 runs into the three-segment loss table.

Reads ``grid.json`` (required), ``encoder.json`` and ``bilevel.json`` (optional).
Selects configs on val where possible, reports test once, and decomposes

    total gap = encoder loss + projection loss + reasoning loss

with signed contributions (projection can be a gain) plus positive-only shares.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

READOUTS = ("lr", "mlp", "mlp_cap", "proto")


def load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def parse_tag(tag: str):
    raw = tag[5:] if tag.startswith("prod_") else tag
    m = re.match(r"^(raw|pca|rand|linear|mlp|res)(\d+)?(?:_(category|instance))?(?:_s(\d+))?$", raw)
    if not m:
        return {"transform": tag, "dim": None, "level": None, "seed": None, "prod": tag.startswith("prod_")}
    return {
        "transform": m.group(1),
        "dim": int(m.group(2)) if m.group(2) else None,
        "level": m.group(3),
        "seed": int(m.group(4)) if m.group(4) else None,
        "prod": tag.startswith("prod_"),
    }


def rows_from_arms(arms: dict):
    rows = []
    for tag, arm in arms.items():
        if not isinstance(arm, dict):
            continue
        meta = parse_tag(tag)
        for ro in READOUTS:
            if ro not in arm or not isinstance(arm[ro], dict):
                continue
            r = arm[ro]
            rows.append({
                **meta, "tag": tag, "readout": ro,
                "val": r.get("val_acc"), "test": r.get("test_acc"),
                "std": r.get("test_acc_std", 0.0),
                "ece": r.get("ece"), "nll": r.get("nll"),
                "auroc": r.get("auroc"),
                "dims": arm.get("_meta", {}).get("dims"),
            })
    return rows


def agg(rows, pred):
    sel = [r for r in rows if pred(r)]
    return sel


def mean_test(rows, pred, key="test"):
    sel = agg(rows, pred)
    vals = [r[key] for r in sel if r[key] is not None]
    return float(np.mean(vals)) if vals else None


def best_by_val(rows, pred):
    sel = [r for r in agg(rows, pred) if r["val"] is not None]
    if not sel:
        return None
    return max(sel, key=lambda r: r["val"])


def boot_delta(pred_a, pred_b, y, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    a = (np.asarray(pred_a) == np.asarray(y)).astype(float)
    b = (np.asarray(pred_b) == np.asarray(y)).astype(float)
    if len(a) != len(b):
        return None
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, len(a), len(a))
        diffs.append(a[idx].mean() - b[idx].mean())
    d = np.asarray(diffs)
    return {
        "mean": float(a.mean() - b.mean()),
        "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
    }


def fmt(v, d=3):
    return "n/a" if v is None else f"{float(v):.{d}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/results/attribution_v2")
    args = ap.parse_args()
    out = Path(args.out)
    grid = load(out / "grid_slim.json") or load(out / "grid.json")
    if grid is None:
        raise SystemExit("grid.json not found")
    grid_prod = load(out / "grid_prod.json")
    encoder = load(out / "encoder.json")
    bilevel = load(out / "bilevel.json")

    rows = rows_from_arms(grid["arms"])
    if grid_prod:
        prod_rows = rows_from_arms(grid_prod["arms"])
        rows += prod_rows
    lines = ["# 归因消融 v2：三段损失汇总", ""]
    lines += [f"- data: `{grid['data']}`", f"- regime: `{grid['regime']}`",
              f"- classes: {grid['classes']}", f"- counts: {grid['counts']}", ""]

    # ---------------------------------------------------------------- table
    lines += ["## 1. 全部臂（test）", "",
              "| tag | readout | val | test | std | ECE | AUROC |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for r in sorted(rows, key=lambda x: (x["tag"], x["readout"])):
        lines.append(
            f"| `{r['tag']}` | {r['readout']} | {fmt(r['val'])} | {fmt(r['test'])} "
            f"| {fmt(r['std'])} | {fmt(r['ece'])} | {fmt(r['auroc'])} |"
        )

    # ------------------------------------------------------ matched deltas
    raw_mlp = next((r for r in rows if r["tag"] == "raw" and r["readout"] == "mlp"), None)
    raw_lr = next((r for r in rows if r["tag"] == "raw" and r["readout"] == "lr"), None)
    lines += ["", "## 2. 匹配读出：投影效应", "",
              "| config | LR raw→proj | MLP raw→proj | MLP delta |",
              "|---|---:|---:|---:|"]
    for tag in sorted({r["tag"] for r in rows if r["tag"] != "raw"}):
        lr = next((r for r in rows if r["tag"] == tag and r["readout"] == "lr"), None)
        mlp = next((r for r in rows if r["tag"] == tag and r["readout"] == "mlp"), None)
        if lr is None or mlp is None:
            continue
        d_mlp = None if not (raw_mlp and mlp["test"] is not None) else mlp["test"] - raw_mlp["test"]
        lines.append(
            f"| `{tag}` | {fmt(lr['test'])} (Δ {fmt(None if not (raw_lr and lr['test'] is not None) else lr['test']-raw_lr['test'])}) "
            f"| {fmt(mlp['test'])} | {fmt(d_mlp)} |"
        )

    # ------------------------------------------------------ dimension / target
    lines += ["", "## 3. 维度与对齐目标", ""]
    for level in ("category", "instance"):
        for transform in ("linear", "mlp", "res"):
            a = next((r for r in rows if r["tag"] == f"{transform}128_{level}_s0" and r["readout"] == "mlp"), None)
            b = next((r for r in rows if r["tag"] == f"{transform}512_{level}_s0" and r["readout"] == "mlp"), None)
            if a and b:
                lines.append(f"- `{transform}` / `{level}`: MLP 128={fmt(a['test'])} → 512={fmt(b['test'])} "
                             f"(Δ {fmt(b['test']-a['test'])})")
    for transform in ("linear", "mlp", "res"):
        for dim in (128, 512):
            a = next((r for r in rows if r["tag"] == f"{transform}{dim}_category_s0" and r["readout"] == "mlp"), None)
            b = next((r for r in rows if r["tag"] == f"{transform}{dim}_instance_s0" and r["readout"] == "mlp"), None)
            if a and b:
                lines.append(f"- `{transform}{dim}`: category={fmt(a['test'])} → instance={fmt(b['test'])} "
                             f"(目标匹配 Δ {fmt(b['test']-a['test'])})")
    if grid_prod:
        lines.append("")
        lines.append("生产训练器（mlp / res）：")
        for transform in ("mlp", "res"):
            for level in ("category", "instance"):
                a = next((r for r in rows if r["tag"] == f"prod_{transform}128_{level}_s0" and r["readout"] == "mlp"), None)
                b = next((r for r in rows if r["tag"] == f"prod_{transform}512_{level}_s0" and r["readout"] == "mlp"), None)
                if a and b:
                    lines.append(f"- `prod_{transform}` / `{level}`: 128={fmt(a['test'])} → 512={fmt(b['test'])} "
                                 f"(Δ {fmt(b['test']-a['test'])})")
        for transform in ("mlp", "res"):
            for dim in (128, 512):
                a = next((r for r in rows if r["tag"] == f"prod_{transform}{dim}_category_s0" and r["readout"] == "mlp"), None)
                b = next((r for r in rows if r["tag"] == f"prod_{transform}{dim}_instance_s0" and r["readout"] == "mlp"), None)
                if a and b:
                    lines.append(f"- `prod_{transform}{dim}`: category={fmt(a['test'])} → instance={fmt(b['test'])} "
                                 f"(目标匹配 Δ {fmt(b['test']-a['test'])})")

    # ------------------------------------------------------ decomposition
    lines += ["", "## 4. 三段分解", ""]
    deployed = [r for r in rows if r["tag"].startswith("prod_mlp128_category_") and r["readout"] in ("mlp", "proto")]
    if not deployed:
        deployed = [r for r in rows if r["tag"].startswith("mlp128_category_") and r["readout"] in ("mlp", "proto")]
    fresh = [r for r in rows if r["tag"].startswith("mlp128_category_") and not r["tag"].startswith("prod_")]
    A_raw = raw_mlp["test"] if raw_mlp else mean_test(rows, lambda r: r["tag"] == "raw" and r["readout"] == "mlp")
    A_proj = mean_test(deployed, lambda r: r["readout"] == "mlp")
    A_pss = mean_test(deployed, lambda r: r["readout"] == "proto")
    A_proj_fresh = mean_test(fresh, lambda r: r["readout"] == "mlp")
    A_pss_fresh = mean_test(fresh, lambda r: r["readout"] == "proto")
    best_cfg = best_by_val([r for r in rows if r["tag"] != "raw" and r["readout"] in ("lr", "mlp", "mlp_cap")], lambda r: True)
    A_proj_best = best_cfg["test"] if best_cfg else A_proj

    A_ft = A_dino = None
    if encoder:
        arms = encoder.get("arms", {})
        ft = [v["test_acc"] for k, v in arms.items() if k.startswith("finetune_") and isinstance(v, dict) and "test_acc" in v]
        dn = []
        for k, v in arms.items():
            if k == "dino_raw" and isinstance(v, dict):
                if "mlp" in v and "test_acc" in v["mlp"]:
                    dn.append(v["mlp"]["test_acc"])
            elif k.startswith("dino_mlp128_") and isinstance(v, dict):
                if "mlp" in v and "test_acc" in v["mlp"]:
                    dn.append(v["mlp"]["test_acc"])
        A_ft = float(np.mean(ft)) if ft else None
        # encoder-ladder reference is the *best* DINOv2 arm, not the mean of
        # raw and projected variants (those are different pipelines).
        A_dino = float(np.max(dn)) if dn else None
    enc_refs = [x for x in (A_ft, A_dino) if x is not None]
    # reference = best encoder arm, but never below the frozen E0 readout
    A_ref = max(enc_refs + [A_raw]) if enc_refs else max(x for x in (A_proj_best, A_raw) if x is not None)

    def share(x, total):
        return None if (x is None or total in (None, 0)) else x / total

    enc_loss = None if (A_ref is None or A_raw is None) else A_ref - A_raw
    proj_loss = None if (A_raw is None or A_proj is None) else A_raw - A_proj
    reason_loss = None if (A_proj is None or A_pss is None) else A_proj - A_pss
    total = None if (A_ref is None or A_pss is None) else A_ref - A_pss
    pos = [x for x in (enc_loss, proj_loss, reason_loss) if x is not None and x > 0]
    pos_sum = sum(pos) if pos else None

    lines += [
        "| 量 | 值 | 说明 |", "|---|---:|---|",
        f"| A_ref（编码器/最强参照） | {fmt(A_ref)} | fine-tune={fmt(A_ft)} / DINOv2={fmt(A_dino)} / 最好投影臂={fmt(A_proj_best)} |",
        f"| A_raw（E0 原始 + MLP） | {fmt(A_raw)} | 冻结特征可读 |",
        f"| A_proj（PSS 投影 + MLP） | {fmt(A_proj)} | 生产训练器 / mlp128/category，均值 |",
        f"| A_pss（PSS 投影 + 原型） | {fmt(A_pss)} | 部署读出 |",
        f"| （参考）自写训练器 mlp128 | {fmt(A_proj_fresh)} / {fmt(A_pss_fresh)} | MLP / proto；仅作复现对照 |",
        f"| **encoder loss** | **{fmt(enc_loss)}** | A_ref − A_raw |",
        f"| **projection loss** | **{fmt(proj_loss)}** | A_raw − A_proj；负数 = 投影是增益 |",
        f"| **reasoning loss** | **{fmt(reason_loss)}** | A_proj − A_pss |",
        f"| **total gap** | **{fmt(total)}** | A_ref − A_pss（恒等于三段之和） |",
        "",
        "### 4.1 份额",
        "",
        "| 段 | 点数 | 占净缺口 | 占正损失 |",
        "|---|---:|---:|---:|",
        f"| encoder | {fmt(enc_loss)} | {fmt(share(enc_loss, total))} | {fmt(share(enc_loss, pos_sum))} |",
        f"| projection | {fmt(proj_loss)} | {fmt(share(proj_loss, total))} | {fmt(share(proj_loss, pos_sum))} |",
        f"| reasoning | {fmt(reason_loss)} | {fmt(share(reason_loss, total))} | {fmt(share(reason_loss, pos_sum))} |",
        "",
        "> projection 为负时，占净缺口会超过 100% 或为负：它表示投影把缺口**补掉**了一部分。"
        "看“占正损失”更直观：只把真正造成缺口的段归一化。",
        "",
        f"> 训练器口径：生产 `train_projections` 的 mlp128/category 为 {fmt(A_proj)}/"
        f"{fmt(A_pss)}（MLP/proto）；本文件自写训练器只有 {fmt(A_proj_fresh)}/"
        f"{fmt(A_pss_fresh)}。归因主表用生产口径；自写口径仅用于同训练器内部的维度/目标对比。",
        "",
        "### 4.2 seed 敏感性（部署配置 mlp128/category）",
        "",
        "| seed | MLP | proto | reasoning loss |",
        "|---|---:|---:|---:|",
    ]
    for s in sorted({r["seed"] for r in deployed if r["seed"] is not None}):
        m = next((r for r in deployed if r["seed"] == s and r["readout"] == "mlp"), None)
        q = next((r for r in deployed if r["seed"] == s and r["readout"] == "proto"), None)
        if m and q:
            lines.append(f"| {s} | {fmt(m['test'])} | {fmt(q['test'])} | {fmt(m['test']-q['test'])} |")
    lines.append(f"| mean | {fmt(A_proj)} | {fmt(A_pss)} | {fmt(None if (A_proj is None or A_pss is None) else A_proj-A_pss)} |")

    # ------------------------------------------------------ bootstrap
    preds = {}
    y_test = None
    for name in ("grid_preds.npz", "grid_prod_preds.npz"):
        p = out / name
        if p.exists():
            z = np.load(p, allow_pickle=True)
            preds.update({k: z[k] for k in z.files if k != "y_test"})
            if "y_test" in z.files:
                y_test = z["y_test"]
    if not preds and grid.get("raw_arrays"):
        # older runs serialized raw arms (including pred) into grid.json
        for tag, arm in grid["raw_arrays"].items():
            for ro in READOUTS:
                if isinstance(arm, dict) and ro in arm and isinstance(arm[ro], dict) and "pred" in arm[ro]:
                    preds[f"{tag}__{ro}"] = np.asarray(arm[ro]["pred"])
    if y_test is None:
        # reconstruct labels from the cached features
        try:
            zf = np.load(out / "paired_features.npz", allow_pickle=True)
            from pss.eval.scripts.run_attribution_grid import task_split
            task = task_split({k: zf[k] for k in zf.files}, grid["regime"])
            y_test = task["yte"]
        except Exception:
            y_test = None

    if preds:
        pairs = [
            ("raw__mlp", "prod_mlp128_category_s0__mlp"),
            ("prod_mlp128_category_s0__mlp", "prod_mlp128_category_s0__proto"),
            ("raw__mlp", "prod_mlp128_instance_s0__mlp"),
            ("prod_mlp128_category_s0__mlp", "prod_mlp128_instance_s0__mlp"),
            ("prod_mlp128_category_s0__mlp", "prod_mlp512_category_s0__mlp"),
            ("raw__mlp", "mlp128_category_s0__mlp"),
            ("mlp128_category_s0__mlp", "mlp128_category_s0__proto"),
        ]
        lines += ["", "## 5. 关键 delta 的 bootstrap", ""]
        if y_test is not None:
            lines += ["| delta | mean | 95% CI |", "|---|---:|---|"]
            for a, b in pairs:
                if a in preds and b in preds:
                    r = boot_delta(preds[a], preds[b], y_test)
                    if r:
                        lines.append(f"| `{a} − {b}` | {fmt(r['mean'])} | [{fmt(r['ci95'][0])}, {fmt(r['ci95'][1])}] |")
        else:
            lines += ["（无 y_test，跳过 CI）"]

    # ------------------------------------------------------ bilevel
    if bilevel:
        lines += ["", "## 6. 最小 bilevel（协同适应）", "",
                  "| arm | val | test |", "|---|---:|---:|"]
        for k, v in sorted(bilevel.get("arms", {}).items()):
            lines.append(f"| `{k}` | {fmt(v.get('val_acc'))} | {fmt(v.get('test_acc'))} |")

    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
