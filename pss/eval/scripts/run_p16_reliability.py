"""P1-6: learned modality reliability for conflict-aware fusion.

A small gate sees the three modality score vectors and learns when to trust
each one. Training augments clean modality scores with shuffle/noise/zero
corruptions; test uses the same corruption protocol as the conflict experiment.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pss._console import ensure_utf8
from pss.eval.scripts.run_p2_openset import MODALITIES
from pss.prototypes import PrototypeBank, recognize
from pss.prototypes.bank import l2
from pss.prototypes.build import extract_projected_features
from pss.prototypes.conflict import conflict_aware_recognize


def _scores_for(Z, bank, idx):
    return {m: np.asarray(bank.protos[m] @ l2(Z[m][idx]).T).T for m in MODALITIES}


def _corrupt_scores(scores, rng, p_clean=0.5):
    """Return a corrupted copy of a batch of score vectors."""
    out = {m: scores[m].copy() for m in MODALITIES}
    b = next(iter(out.values())).shape[0]
    for m in MODALITIES:
        u = rng.random(b)
        # clean / shuffle / noise / zero per sample
        perm = rng.permutation(b)
        noise = rng.normal(scale=0.5, size=out[m].shape).astype(np.float32)
        out[m][u < p_clean] = scores[m][u < p_clean]
        mask = (u >= p_clean) & (u < p_clean + 0.2)
        out[m][mask] = scores[m][perm][mask]
        mask = (u >= p_clean + 0.2) & (u < p_clean + 0.4)
        out[m][mask] = scores[m][mask] + noise[mask]
        mask = u >= p_clean + 0.4
        out[m][mask] = 0.0
    return out


class ReliabilityGate(nn.Module):
    def __init__(self, n_classes: int, n_mod: int = 3, hidden: int = 64):
        super().__init__()
        self.n_mod = n_mod
        self.n_classes = n_classes
        self.net = nn.Sequential(
            nn.Linear(n_classes * n_mod + 3 * n_mod, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_mod),
        )

    def forward(self, scores):
        feats = []
        for m in MODALITIES:
            s = scores[m]
            top = s.max(dim=1).values
            top2 = s.topk(2, dim=1).values
            margin = top2[:, 0] - top2[:, 1]
            present = (s.abs().sum(dim=1) > 0).float()
            feats.append(torch.stack([top, margin, present], dim=1))
        stats = torch.cat(feats, dim=1)
        raw = torch.cat([scores[m] for m in MODALITIES], dim=1)
        w = torch.softmax(self.net(torch.cat([raw, stats], dim=1)), dim=1)
        fused = sum(w[:, i : i + 1] * scores[m] for i, m in enumerate(MODALITIES))
        return fused, w


def _batch_tensors(scores, device):
    return {
        m: torch.as_tensor(scores[m], dtype=torch.float32, device=device)
        for m in MODALITIES
    }


def _train_gate(train_scores, train_labels, val_scores, val_labels, seeds, epochs, device):
    n_classes = train_scores[MODALITIES[0]].shape[1]
    runs = []
    for seed in range(seeds):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = ReliabilityGate(n_classes).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        y = torch.as_tensor(train_labels, dtype=torch.long, device=device)
        yv = torch.as_tensor(val_labels, dtype=torch.long, device=device)
        best_state, best_val = None, -1.0
        for _epoch in range(epochs):
            model.train()
            perm = rng.permutation(len(y))
            for start in range(0, len(perm), 64):
                idx = perm[start : start + 64]
                batch = {m: train_scores[m][idx] for m in MODALITIES}
                batch = _corrupt_scores(batch, rng)
                logits, _w = model(_batch_tensors(batch, device))
                loss = F.cross_entropy(logits, y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                vbatch = _corrupt_scores(
                    {m: val_scores[m] for m in MODALITIES}, rng, p_clean=0.5
                )
                logits, _ = model(_batch_tensors(vbatch, device))
                acc = float((logits.argmax(dim=1) == yv).float().mean().cpu())
            if acc > best_val:
                best_val = acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        model.eval()
        runs.append((model, best_val))
    return runs


def _gate_predict(model, scores, device):
    with torch.no_grad():
        logits, w = model(_batch_tensors(scores, device))
        probs = torch.softmax(logits, dim=1)
        return (
            logits.argmax(dim=1).cpu().numpy(),
            probs.max(dim=1).values.cpu().numpy(),
            w.cpu().numpy(),
        )


def _corrupt_Z(Z, idx, modality, op, rng, scale=1.0):
    out = {m: np.array(Z[m], copy=True) for m in MODALITIES}
    if modality is None or op == "clean":
        return out
    if op == "shuffle":
        perm = rng.permutation(idx)
        out[modality][idx] = Z[modality][perm]
    elif op == "noise":
        std = float(np.std(Z[modality]))
        out[modality][idx] += rng.normal(
            scale=scale * max(std, 1e-6), size=out[modality][idx].shape
        ).astype(np.float32)
    elif op == "zero":
        out[modality][idx] = 0.0
    return out


def run(
    data_dir="data/category_pretrain",
    expert_dir="checkpoints/category_pretrain_experts",
    projection_dir="checkpoints/phase45_projection",
    prototype_dir="checkpoints/phase45_prototypes",
    out_dir="eval/results/pss_vs_transformer",
    seeds=5,
    epochs=200,
    device="cpu",
):
    feats, Z = extract_projected_features(data_dir, expert_dir, projection_dir, device)
    bank = PrototypeBank.load(Path(prototype_dir) / "prototypes.npz")
    weights = {"vision": 1.0, "tactile": 0.5, "geometry": 1.0}
    mp = Path(prototype_dir) / "metrics.json"
    if mp.exists():
        weights = json.loads(mp.read_text(encoding="utf-8")).get("weights", weights)

    def indices(split):
        return [
            i
            for i, (s, c) in enumerate(zip(feats.splits, feats.categories))
            if s == split and c in bank.categories
        ]

    train_idx, val_idx, test_idx = indices("train"), indices("val"), indices("test")
    classes = list(bank.categories)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    train_scores = _scores_for(Z, bank, train_idx)
    val_scores = _scores_for(Z, bank, val_idx)
    train_y = np.asarray([class_to_idx[feats.categories[i]] for i in train_idx])
    val_y = np.asarray([class_to_idx[feats.categories[i]] for i in val_idx])

    models = _train_gate(train_scores, train_y, val_scores, val_y, seeds, epochs, device)

    # Calibrate an abstention threshold on clean val at 95% coverage.
    val_logits = []
    for model, _ in models:
        pred, prob, _w = _gate_predict(
            model, _scores_for(Z, bank, val_idx), device
        )
        val_logits.append(prob)
    threshold = float(np.percentile(np.concatenate(val_logits), 5.0))

    conditions = [
        ("clean", None, "clean", 1.0),
        ("tactile_shuffle", "tactile", "shuffle", 1.0),
        ("tactile_noise_1.0", "tactile", "noise", 1.0),
        ("tactile_zero", "tactile", "zero", 1.0),
        ("vision_shuffle", "vision", "shuffle", 1.0),
        ("geometry_shuffle", "geometry", "shuffle", 1.0),
        ("vision_zero", "vision", "zero", 1.0),
        ("geometry_zero", "geometry", "zero", 1.0),
    ]

    results = {}
    rng = np.random.default_rng(0)
    for name, modality, op, scale in conditions:
        Zc = _corrupt_Z(Z, test_idx, modality, op, rng, scale)
        scores = _scores_for(Zc, bank, test_idx)
        gt = [feats.categories[i] for i in test_idx]

        fixed, reliable = [], []
        for i, idx in enumerate(test_idx):
            q = {m: Zc[m][idx] for m in MODALITIES}
            r = recognize(q, bank, weights=weights, use_tree=False, threshold=None)
            fixed.append(classes.index(r.ranked[0][0]) if r.ranked else -1)
            d = conflict_aware_recognize(q, bank, weights)
            reliable.append(
                (classes.index(d.category), d.category is not None)
                if d.category is not None
                else (-1, False)
            )

        gate_acc, gate_cov, gate_acc_acc = [], [], []
        for model, _ in models:
            pred, prob, _w = _gate_predict(model, scores, device)
            ok = pred == np.asarray([class_to_idx[c] for c in gt])
            gate_acc.append(float(np.mean(ok)))
            accepted = prob >= threshold
            gate_cov.append(float(np.mean(accepted)))
            gate_acc_acc.append(float(np.mean(ok[accepted])) if accepted.any() else 0.0)

        rel_correct = [1 if c == class_to_idx[gt[i]] else 0 for i, (c, ok) in enumerate(reliable) if ok]
        results[name] = {
            "pss_fixed": float(np.mean([c == class_to_idx[g] for c, g in zip(fixed, gt)])),
            "pss_reliable": {
                "coverage": float(np.mean([ok for _c, ok in reliable])),
                "accepted_accuracy": float(np.mean(rel_correct)) if rel_correct else 0.0,
            },
            "learned": {
                "accuracy_mean": float(np.mean(gate_acc)),
                "accuracy_std": float(np.std(gate_acc)),
                "coverage_mean": float(np.mean(gate_cov)),
                "accepted_accuracy_mean": float(np.mean(gate_acc_acc)),
            },
        }

    metrics = {
        "data_dir": str(data_dir),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "threshold": threshold,
        "conditions": results,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "p16_reliability.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _report(out / "p16_reliability.md", metrics)
    return metrics


def _report(path, metrics):
    lines = [
        "# P1-6: learned modality reliability",
        "",
        f"- data: `{metrics['data_dir']}`, test samples: {metrics['n_test']}",
        f"- abstention threshold: {metrics['threshold']:.3f}",
        "",
        "| condition | PSS fixed | PSS reliable (cov/acc) | learned (cov/acc) |",
        "|---|---:|---:|---:|",
    ]
    for name, r in metrics["conditions"].items():
        rel = r["pss_reliable"]
        learn = r["learned"]
        lines.append(
            f"| {name} | {r['pss_fixed']:.3f} | "
            f"{rel['coverage']:.2f}/{rel['accepted_accuracy']:.3f} | "
            f"{learn['coverage_mean']:.2f}/{learn['accepted_accuracy_mean']:.3f} |"
        )
    lines += [
        "",
        "Learned gate is trained with clean/shuffle/noise/zero score corruptions.",
        "Coverage uses a 95%-coverage threshold calibrated on clean validation.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/category_pretrain")
    ap.add_argument("--experts", default="checkpoints/category_pretrain_experts")
    ap.add_argument("--projections", default="checkpoints/phase45_projection")
    ap.add_argument("--prototypes", default="checkpoints/phase45_prototypes")
    ap.add_argument("--out", default="eval/results/pss_vs_transformer")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    m = run(args.data, args.experts, args.projections, args.prototypes,
            args.out, args.seeds, args.epochs, args.device)
    for name, r in m["conditions"].items():
        print(
            f"{name}: fixed {r['pss_fixed']:.3f}, "
            f"learned {r['learned']['coverage_mean']:.2f}/{r['learned']['accepted_accuracy_mean']:.3f}"
        )
    print(f"output: {args.out}")


if __name__ == "__main__":
    main()
