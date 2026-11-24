"""P2-10: rollout-aware set world model.

Compares one-step MSE training with an additional multi-step rollout loss on
the same per-entity set model.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pss._console import ensure_utf8
from pss.world_model.set_model import train_set_world_model


def _mean_std(values):
    return float(np.mean(values)), float(np.std(values))


def run(
    data_dir="data/phase7_v2_more",
    out_dir="eval/results/pss_vs_transformer",
    seeds=3,
    epochs=300,
    hidden=256,
    device=None,
):
    rows = {}
    for name, weight in (("one_step", 0.0), ("rollout_aware", 2.0)):
        runs = []
        for seed in range(seeds):
            metrics = train_set_world_model(
                data_dir,
                Path(out_dir) / "p210_world_model" / f"{name}_seed{seed}",
                epochs=epochs,
                hidden=hidden,
                seed=seed,
                rollout_weight=weight,
            )
            runs.append(metrics)
        rows[name] = {
            "runs": runs,
            "one_step": _mean_std([m["one_step_pos_error"]["learned"] for m in runs]),
            "rollout": _mean_std([m["rollout_pos_error"]["learned"] for m in runs]),
            "rollout_target": _mean_std(
                [m["rollout_target_pos_error"]["learned"] for m in runs]
            ),
        }
    metrics = {
        "data_dir": str(data_dir),
        "seeds": seeds,
        "epochs": epochs,
        "variants": rows,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "p210_world_model.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _report(out / "p210_world_model.md", metrics)
    return metrics


def _report(path, metrics):
    lines = [
        "# P2-10: rollout-aware world model",
        "",
        f"- data: `{metrics['data_dir']}`, seeds: {metrics['seeds']}, epochs: {metrics['epochs']}",
        "",
        "| variant | 1-step pos error | rollout all | rollout target |",
        "|---|---:|---:|---:|",
    ]
    for name, r in metrics["variants"].items():
        a, sa = r["one_step"]
        b, sb = r["rollout"]
        c, sc = r["rollout_target"]
        lines.append(
            f"| {name} | {a:.4f} ± {sa:.4f} | **{b:.4f} ± {sb:.4f}** | "
            f"{c:.4f} ± {sc:.4f} |"
        )
    lines += [
        "",
        "Target: rollout all < 0.05 m.",
        "Rollout loss trains each predicted delta on states reached by the model's own rollout.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/phase7_v2_more")
    ap.add_argument("--out", default="eval/results/pss_vs_transformer")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args()
    m = run(args.data, args.out, args.seeds, args.epochs)
    for name, r in m["variants"].items():
        print(f"{name}: one-step {r['one_step'][0]:.4f}, rollout {r['rollout'][0]:.4f}")
    print(f"output: {args.out}")


if __name__ == "__main__":
    main()
