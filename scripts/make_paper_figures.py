"""Generate the paper figures from the claim-gap and property reports.

Usage:
    python scripts/make_paper_figures.py

Writes PDF (for LaTeX) and PNG (for preview) into ``paper/figures/``.
Palette: the validated default categorical slots (blue / orange / aqua),
with gray for baselines and direct labels on the marks.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
GRAY = "#7a7974"
LIGHT = "#b7b6ae"
RED = "#e34948"
INK = "#0b0b0b"
MUTED = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#e5e4df"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "legend.fontsize": 7.2,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.6,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
    }
)


def load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"wrote {name}.pdf / .png")


def style(ax, horizontal_grid: bool = True) -> None:
    ax.tick_params(colors=MUTED, width=0.6, length=3)
    if horizontal_grid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.5)
        ax.xaxis.grid(False)
    else:
        ax.xaxis.grid(True, color=GRID, linewidth=0.5)
        ax.yaxis.grid(False)
    ax.set_axisbelow(True)


def fig_binding() -> None:
    d = load("reports/claim_gaps/tactile_binding/metrics.json")
    deltas = d["deltas"]
    order = [
        ("graph_tactile", "Graph + tactile", BLUE, ""),
        ("graph_vision_zero", "Vision zeroed", AQUA, f" (−{deltas['vision_contribution']:.3f})"),
        ("graph_tactile_zero", "Tactile zeroed", ORANGE, f" (−{deltas['tactile_contribution']:.3f})"),
        ("simple_tracker", "Simple tracker", GRAY, f" (−{deltas['graph_contribution']:.3f})"),
        ("text_state_proxy", "Text-state proxy", LIGHT, f" (−{deltas['state_contribution']:.3f})"),
    ]
    vals = [d["results"][k]["overall"]["id_retention"] for k, _, _, _ in order]
    labels = [f"{label}{suffix}" for _, label, _, suffix in order]
    colors = [c for _, _, c, _ in order]

    fig, ax = plt.subplots(figsize=(3.35, 2.05), constrained_layout=True)
    y = np.arange(len(vals))[::-1]
    ax.barh(y, vals, color=colors, edgecolor=SURFACE, linewidth=0.8, height=0.62)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1.03)
    ax.set_xlabel("ID retention")
    style(ax, horizontal_grid=False)
    for yi, v in zip(y, vals):
        ax.text(v + 0.012, yi, f"{v:.3f}", va="center", fontsize=7.3, color=INK)
    ax.set_title("Binding under 70-frame occlusion", loc="left")
    save(fig, "fig_binding_signal")


def fig_fair() -> None:
    d = load("eval/results/pss_vs_transformer_v2/fair_attribution.json")
    cells = d["cells"]
    groups = ["Base", "+ tactile"]
    pss = [cells["base"]["pss"]["overall_mean"], cells["tactile"]["pss"]["overall_mean"]]
    tr = [cells["base"]["transformer"]["overall_mean"], cells["tactile"]["transformer"]["overall_mean"]]

    fig, ax = plt.subplots(figsize=(3.35, 2.05), constrained_layout=True)
    x = np.arange(2)
    w = 0.34
    ax.bar(x - w / 2, pss, w, label="PSS", color=BLUE, edgecolor=SURFACE, linewidth=0.8)
    ax.bar(x + w / 2, tr, w, label="Transformer", color=ORANGE, edgecolor=SURFACE, linewidth=0.8)
    ax.set_xticks(x, groups)
    ax.set_ylim(0, 1.22)
    ax.set_ylabel("re-ID accuracy")
    style(ax)
    for i, (p, t) in enumerate(zip(pss, tr)):
        ax.text(i - w / 2, p + 0.02, f"{p:.3f}", ha="center", fontsize=7.3, color=INK)
        ax.text(i + w / 2, t + 0.02, f"{t:.3f}", ha="center", fontsize=7.3, color=INK)
        ax.text(i, 1.14, f"Δ {p - t:+.3f}", ha="center", fontsize=8, color=INK)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    ax.set_title("Fair attribution of re-ID", loc="left")
    save(fig, "fig_fair_attribution")


def fig_relations() -> None:
    d = load("reports/claim_gaps/hybrid_relations/metrics.json")
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.15), constrained_layout=True)

    ax = axes[0]
    labels = ["spatial", "semantic", "tactile", "tactile\n(removed)"]
    vals = [0.957, 0.996, 0.970, 0.483]
    colors = [BLUE, BLUE, BLUE, ORANGE]
    x = np.arange(len(vals))
    ax.bar(x, vals, color=colors, edgecolor=SURFACE, linewidth=0.8, width=0.62)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("F1")
    style(ax)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.015, f"{v:.3f}", ha="center", fontsize=7.2, color=INK)
    ax.set_title("(a) Typed relation F1", loc="left")

    ax = axes[1]
    noise = d["noise_levels"]
    methods = [
        ("geometry_eps0.25", "Geometry (dead zone)", BLUE, "o"),
        ("learned", "Learned", ORANGE, "s"),
        ("hybrid", "Naive hybrid", GRAY, "^"),
        ("hybrid_diag", "Distribution-aware hybrid", AQUA, "D"),
    ]
    for key, label, color, marker in methods:
        vals = [d["summary"][str(n)][key]["mean"] for n in noise]
        ax.plot(noise, vals, marker=marker, color=color, label=label, linewidth=1.4, markersize=3.4)
    ax.set_xlabel("coordinate noise")
    ax.set_ylabel("exact-4 accuracy")
    ax.set_ylim(0.4, 1.04)
    style(ax)
    ax.legend(frameon=False, fontsize=6.4, loc="center", ncol=2)
    ax.set_title("(b) Hybrid under noise", loc="left")
    save(fig, "fig_relations")


def fig_memory() -> None:
    p4 = load("eval/results/pss_vs_transformer/p4_scaling.json")
    fact = load("reports/claim_gaps/fact_active/metrics.json")
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.15), constrained_layout=True)

    ax = axes[0]
    n = [r["n_entities"] for r in p4["pss"]]
    pss = [r["p50_ms"] for r in p4["pss"]]
    tr = [r["p50_ms"] for r in p4["transformer"]]
    ax.plot(n, pss, marker="o", color=BLUE, label="PSS active set", linewidth=1.4, markersize=3.5)
    ax.plot(n, tr, marker="s", color=ORANGE, label="Transformer", linewidth=1.4, markersize=3.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("entities / tokens")
    ax.set_ylabel("p50 latency (ms)")
    style(ax)
    ax.legend(frameon=False, fontsize=6.8, loc="upper left")
    ax.annotate(
        "23.6x",
        xy=(1000, 6.68),
        xytext=(260, 15),
        fontsize=7,
        color=MUTED,
        arrowprops=dict(arrowstyle="-", lw=0.5, color=MUTED),
    )
    ax.set_title("(a) Active-set compute", loc="left")

    ax = axes[1]
    ks = [4, 8, 16]
    methods = [("hot", "Hot", GRAY), ("learned_old", "Old learned", ORANGE), ("query_recency", "Query-conditioned", AQUA)]
    x = np.arange(len(ks))
    w = 0.26
    for i, (key, label, color) in enumerate(methods):
        vals = [fact["summary"][str(k)][key]["mean"] for k in ks]
        ax.bar(x + (i - 1) * w, vals, w, label=label, color=color, edgecolor=SURFACE, linewidth=0.8)
    ax.set_xticks(x, [f"k={k}" for k in ks])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("fact accuracy")
    style(ax)
    ax.legend(frameon=False, fontsize=6.6, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3)
    ax.set_title("(b) Query-conditioned facts", loc="left")
    save(fig, "fig_memory_compute")


def fig_detection_l2() -> None:
    d = load("reports/claim_gaps/detect_noise/metrics.json")
    l2 = load("reports/claim_gaps/l2_memory_all/metrics.json")
    valid_bundles = {
        r["bundle"]
        for r in l2["rows"]
        if r["variant"] == "full" and r["id_retention"] > 0.5
    }
    valid_rows = [r for r in l2["rows"] if r["bundle"] in valid_bundles]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.15), constrained_layout=True)

    ax = axes[0]
    conds = d["conditions"]
    miss = [c["miss_rate"] for c in conds]
    vals = [d["results"][c["name"]]["overall"]["id_retention"] for c in conds]
    ax.plot(miss, vals, marker="o", color=BLUE, linewidth=1.4, markersize=4)
    for m, v in zip(miss, vals):
        ax.annotate(f"{v:.3f}", (m, v), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=7)
    ax.set_xlabel("detection miss rate")
    ax.set_ylabel("ID retention")
    ax.set_ylim(0.6, 1.04)
    style(ax)
    ax.set_title("(a) Detection noise", loc="left")

    ax = axes[1]
    variants = ["full", "no_memory", "no_entity"]
    labels = ["Full", "No memory", "No entity"]
    vals = [
        float(np.mean([r["id_retention"] for r in valid_rows if r["variant"] == v]))
        for v in variants
    ]
    colors = [BLUE, ORANGE, AQUA]
    x = np.arange(3)
    ax.bar(x, vals, color=colors, edgecolor=SURFACE, linewidth=0.8, width=0.6)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("ID retention")
    style(ax)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.015, f"{v:.3f}", ha="center", fontsize=7.2, color=INK)
    ax.set_title("(b) L2 bundle replay", loc="left")
    save(fig, "fig_detection_l2")


def fig_families_retrain() -> None:
    fam = load("reports/claim_gaps/families/metrics.json")
    r0 = load("reports/claim_gaps/retrain_all/metrics.json")
    r5 = load("reports/claim_gaps/retrain_all_experts/metrics.json")
    r30 = load("reports/claim_gaps/retrain_all_experts_30ep/metrics.json")
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.15), constrained_layout=True)

    ax = axes[0]
    names = ["base", "+instance", "+instance8"]
    vz = [r["ZCR_vision"] for r in fam["rows"]]
    tz = [r["ZCR_tactile"] for r in fam["rows"]]
    vk = [r["calibration"]["vision"]["1"] for r in fam["rows"]]
    x = np.arange(3)
    w = 0.26
    ax.bar(x - w, vz, w, label="vision zero-shot", color=BLUE, edgecolor=SURFACE, linewidth=0.8)
    ax.bar(x, tz, w, label="tactile zero-shot", color=ORANGE, edgecolor=SURFACE, linewidth=0.8)
    ax.bar(x + w, vk, w, label="vision k=1", color=AQUA, edgecolor=SURFACE, linewidth=0.8)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("ZCR")
    style(ax)
    ax.legend(frameon=False, fontsize=6.6, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3)
    ax.set_title("(a) Cross-family calibration", loc="left")

    ax = axes[1]
    points = [
        ("prototype\nrebuild", r0["cost"]["prototype_build_ms"], r0["methods"]["frozen_heads_alt_prototypes"]["vision"], BLUE, (7, 4), "left"),
        ("projection\nretrain", r0["cost"]["retrain_projection_s"] * 1000, r0["methods"]["retrained_projection_prototypes"]["vision"], ORANGE, (-8, 10), "right"),
        ("5-epoch\nexpert", r5["cost"]["retrain_total_s"] * 1000, r5["methods"]["retrained_projection_prototypes"]["vision"], GRAY, (-8, -14), "right"),
        ("30-epoch\nexpert", r30["cost"]["retrain_total_s"] * 1000, r30["methods"]["retrained_projection_prototypes"]["vision"], AQUA, (-8, -14), "right"),
    ]
    for label, xv, yv, color, offset, ha in points:
        ax.scatter(xv, yv, s=30, color=color, zorder=3)
        ax.annotate(label, (xv, yv), textcoords="offset points", xytext=offset, fontsize=6.6, color=MUTED, ha=ha)
    ax.axhline(r0["methods"]["frozen_heads_alt_prototypes"]["vision"], color=LIGHT, linewidth=0.8, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("cost (ms, log)")
    ax.set_ylabel("vision ZCR")
    ax.set_ylim(0.93, 0.99)
    style(ax)
    ax.set_title("(b) Retraining cost", loc="left")
    save(fig, "fig_families_retrain")


def fig_conflict_openset() -> None:
    c = load("reports/claim_gaps/modality_conflict/modality_conflict.json")
    o = load("reports/claim_gaps/openset/p2_openset.json")
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.15), constrained_layout=True)

    ax = axes[0]
    conds = ["clean", "tactile_zero", "vision_shuffle", "geometry_shuffle"]
    labels = ["clean", "tactile 0", "vision shuf.", "geometry shuf."]
    fixed = [c["conditions"][k]["pss_fixed"]["accuracy"] for k in conds]
    abstain = [c["conditions"][k]["pss_abstain"]["accepted_accuracy"] for k in conds]
    tr = [c["conditions"][k]["transformer"]["accuracy_mean"] for k in conds]
    x = np.arange(4)
    w = 0.26
    ax.bar(x - w, fixed, w, label="PSS fixed", color=BLUE, edgecolor=SURFACE, linewidth=0.8)
    ax.bar(x, abstain, w, label="PSS abstain", color=ORANGE, edgecolor=SURFACE, linewidth=0.8)
    ax.bar(x + w, tr, w, label="Transformer", color=GRAY, edgecolor=SURFACE, linewidth=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("accuracy")
    style(ax)
    ax.legend(frameon=False, fontsize=6.4, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3)
    ax.set_title("(a) Modality conflict", loc="left")

    ax = axes[1]
    methods = ["PSS", "Softmax", "Energy"]
    auroc = [o["pss"]["auroc"], o["transformer"]["softmax"]["auroc_mean"], o["transformer"]["energy"]["auroc_mean"]]
    reject = [
        o["pss"]["unseen_reject"],
        o["transformer"]["softmax"]["unseen_reject_mean"],
        o["transformer"]["energy"]["unseen_reject_mean"],
    ]
    x = np.arange(3)
    w = 0.32
    ax.bar(x - w / 2, auroc, w, label="AUROC", color=BLUE, edgecolor=SURFACE, linewidth=0.8)
    ax.bar(x + w / 2, reject, w, label="Unseen reject", color=ORANGE, edgecolor=SURFACE, linewidth=0.8)
    ax.set_xticks(x, methods)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("score")
    style(ax)
    ax.legend(frameon=False, fontsize=6.6, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2)
    ax.set_title("(b) Open-set rejection", loc="left")
    save(fig, "fig_conflict_openset")


def fig_space_rag() -> None:
    s = load("reports/claim_gaps/space_ablation/metrics.json")["ablations"]
    r = load("reports/claim_gaps/rag_exact/metrics.json")
    base = load("reports/claim_gaps/rag_baseline/rag_baseline.json")
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.15), constrained_layout=True)

    ax = axes[0]
    labels = ["full", "sum", "random", "trained\nR@1", "random\nR@1"]
    vals = [
        s["full_accuracy"],
        s["cross_modal_sum_accuracy"],
        s["random_projection_accuracy"],
        s["trained_vision_tactile_r1"],
        s["random_projection_vision_tactile_r1"],
    ]
    colors = [BLUE, BLUE, BLUE, ORANGE, ORANGE]
    x = np.arange(len(vals))
    ax.bar(x, vals, color=colors, edgecolor=SURFACE, linewidth=0.8, width=0.62)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("score")
    style(ax)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.015, f"{v:.3f}", ha="center", fontsize=7.0, color=INK)
    ax.set_title("(a) Space ablation", loc="left")

    ax = axes[1]
    row = r["rows"][-1]
    learned_row = base["rows"][-1]
    pts = [
        ("PSS", r["pss"]["p50_ms"], r["pss"]["accuracy"], BLUE, (7, 4), "left"),
        ("Exact RAG", row["exact_rag_p50_ms"], row["exact_rag_accuracy"], ORANGE, (7, 4), "left"),
        ("Learned RAG", learned_row["rag_p50_ms"], learned_row["rag_accuracy"], GRAY, (-8, 10), "right"),
    ]
    for label, xv, yv, color, offset, ha in pts:
        ax.scatter(xv, yv, s=30, color=color, zorder=3)
        ax.annotate(label, (xv, yv), textcoords="offset points", xytext=offset, fontsize=6.8, color=MUTED, ha=ha)
    ax.set_xscale("log")
    ax.set_xlim(0.00015, 5.0)
    ax.set_xlabel("p50 latency (ms, log)")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.3, 1.05)
    style(ax)
    ax.set_title("(b) RAG vs active set", loc="left")
    save(fig, "fig_space_rag")


def main() -> None:
    fig_binding()
    fig_fair()
    fig_relations()
    fig_memory()
    fig_detection_l2()
    fig_families_retrain()
    fig_conflict_openset()
    fig_space_rag()
    print(f"figures written to {OUT}")


if __name__ == "__main__":
    main()
