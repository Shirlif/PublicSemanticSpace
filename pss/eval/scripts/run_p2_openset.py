"""P2: open-set unknown rejection, PSS prototypes vs transformer classifier.

Same frozen projected features, same seen/unseen split. PSS uses prototype
similarity and a val-calibrated threshold; the transformer uses softmax/energy
OOD scores and the same 95% seen-acceptance calibration.
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
from sklearn.metrics import roc_auc_score

from pss._console import ensure_utf8
from pss.prototypes import PrototypeBank, recognize
from pss.prototypes.build import extract_projected_features

MODALITIES = ("vision", "tactile", "geometry")


def _split_indices(feats, bank, split: str, seen: bool):
    if seen:
        return [
            i
            for i, (s, c) in enumerate(zip(feats.splits, feats.categories))
            if s == split and c in bank.categories
        ]
    return [
        i
        for i, (s, c) in enumerate(zip(feats.splits, feats.categories))
        if s == split and c not in bank.categories
    ]


def _query(Z, i):
    return {m: Z[m][i] for m in MODALITIES}


def _pss_scores(feats, Z, bank, indices, weights):
    scores, correct = [], 0
    for i in indices:
        result = recognize(
            _query(Z, int(i)), bank, weights=weights, use_tree=False, threshold=None
        )
        score = float(result.ranked[0][1]) if result.ranked else -1.0
        scores.append(score)
        if result.ranked and not result.unknown and result.ranked[0][0] == feats.categories[i]:
            correct += 1
    return np.asarray(scores, dtype=float), correct / max(1, len(indices))


def _threshold_at_acceptance(scores, acceptance=0.95):
    return float(np.percentile(scores, (1.0 - acceptance) * 100.0))


def _summary_with_correct(scores_seen, scores_unseen, correct_flags, threshold):
    accepted = scores_seen >= threshold
    correct = np.asarray(correct_flags, dtype=bool)
    return {
        "threshold": float(threshold),
        "auroc": float(
            roc_auc_score(
                [1] * len(scores_seen) + [0] * len(scores_unseen),
                list(scores_seen) + list(scores_unseen),
            )
        ),
        "seen_accept": float(np.mean(accepted)),
        "seen_top1": float(np.mean(correct)),
        "accepted_accuracy": float(np.mean(correct[accepted])) if accepted.any() else 0.0,
        "unseen_reject": float(np.mean(scores_unseen < threshold)),
        "n_seen": len(scores_seen),
        "n_unseen": len(scores_unseen),
    }


class ModalityTransformer(nn.Module):
    def __init__(self, dims, n_classes: int, d_model: int = 128, layers: int = 2, nhead: int = 4):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(d, d_model) for d in dims])
        self.modality = nn.Parameter(torch.randn(len(dims), d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, tokens):
        # tokens: [B, M, D_m] per modality, already projected to d_model? No:
        # this method takes a list of [B, D_m] tensors and projects each.
        h = torch.stack([proj(x) for proj, x in zip(self.proj, tokens)], dim=1)
        h = h + self.modality.unsqueeze(0)
        h = self.encoder(h)
        return self.head(h.mean(dim=1))


def _tensor_tokens(Z, indices, device):
    return [
        torch.as_tensor(Z[m][indices], dtype=torch.float32, device=device)
        for m in MODALITIES
    ]


def _train_transformer(Z, feats, bank, seed: int, epochs: int, device: str):
    torch.manual_seed(seed)
    classes = sorted(bank.categories)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    train_idx = _split_indices(feats, bank, "train", True)
    val_idx = _split_indices(feats, bank, "val", True)
    dims = [Z[m].shape[1] for m in MODALITIES]
    model = ModalityTransformer(dims, len(classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    x_train = _tensor_tokens(Z, train_idx, device)
    y_train = torch.as_tensor(
        [class_to_idx[feats.categories[i]] for i in train_idx],
        dtype=torch.long,
        device=device,
    )
    x_val = _tensor_tokens(Z, val_idx, device)
    y_val = torch.as_tensor(
        [class_to_idx[feats.categories[i]] for i in val_idx],
        dtype=torch.long,
        device=device,
    )
    best_state, best_val = None, -1.0
    for _epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(y_train), device=device)
        for start in range(0, len(perm), 64):
            idx = perm[start : start + 64]
            logits = model([x[idx] for x in x_train])
            loss = F.cross_entropy(logits, y_train[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(x_val).argmax(dim=1)
            val_acc = float((pred == y_val).float().mean().cpu())
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {"best_val_acc": best_val}


def _transformer_scores(model, Z, indices, device):
    x = _tensor_tokens(Z, indices, device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        max_prob = probs.max(dim=1).values.cpu().numpy()
        energy = torch.logsumexp(logits, dim=1).cpu().numpy()
        pred = logits.argmax(dim=1).cpu().numpy()
    return max_prob, energy, pred


def run_p2(
    data_dir,
    expert_dir,
    projection_dir,
    prototype_dir,
    out_dir,
    seeds: int = 5,
    epochs: int = 200,
    device: str = "cpu",
) -> dict:
    feats, Z = extract_projected_features(data_dir, expert_dir, projection_dir, device)
    bank = PrototypeBank.load(Path(prototype_dir) / "prototypes.npz")
    metrics_path = Path(prototype_dir) / "metrics.json"
    weights = {"vision": 1.0, "tactile": 0.5, "geometry": 1.0}
    if metrics_path.exists():
        weights = json.loads(metrics_path.read_text(encoding="utf-8")).get("weights", weights)

    val_seen = _split_indices(feats, bank, "val", True)
    test_seen = _split_indices(feats, bank, "test", True)
    test_unseen = _split_indices(feats, bank, "test_category", False)

    pss_val_scores, _ = _pss_scores(feats, Z, bank, val_seen, weights)
    pss_seen_scores, _ = _pss_scores(feats, Z, bank, test_seen, weights)
    pss_unseen_scores, _ = _pss_scores(feats, Z, bank, test_unseen, weights)
    pss_threshold = _threshold_at_acceptance(pss_val_scores)

    # Per-sample correctness for accepted accuracy.
    pss_correct = []
    for i in test_seen:
        result = recognize(_query(Z, int(i)), bank, weights=weights, use_tree=False)
        pss_correct.append(bool(result.ranked and result.ranked[0][0] == feats.categories[i]))
    pss_metrics = _summary_with_correct(
        pss_seen_scores, pss_unseen_scores, pss_correct, pss_threshold
    )

    tr_runs = []
    training = []
    for seed in range(seeds):
        model, info = _train_transformer(Z, feats, bank, seed, epochs, device)
        val_prob, val_energy, _ = _transformer_scores(model, Z, val_seen, device)
        seen_prob, seen_energy, seen_pred = _transformer_scores(model, Z, test_seen, device)
        unseen_prob, unseen_energy, _ = _transformer_scores(model, Z, test_unseen, device)
        class_to_idx = {c: i for i, c in enumerate(sorted(bank.categories))}
        seen_correct = [
            int(seen_pred[k] == class_to_idx[feats.categories[i]])
            for k, i in enumerate(test_seen)
        ]
        # Softmax threshold.
        prob_thr = _threshold_at_acceptance(val_prob)
        prob_metrics = _summary_with_correct(
            seen_prob, unseen_prob, seen_correct, prob_thr
        )
        # Energy threshold.
        energy_thr = _threshold_at_acceptance(val_energy)
        energy_metrics = _summary_with_correct(
            seen_energy, unseen_energy, seen_correct, energy_thr
        )
        tr_runs.append(
            {
                "seed": seed,
                "best_val_acc": info["best_val_acc"],
                "softmax": prob_metrics,
                "energy": energy_metrics,
            }
        )
        training.append({"seed": seed, "best_val_acc": info["best_val_acc"]})

    def aggregate(key):
        vals = [r[key]["auroc"] for r in tr_runs]
        rejects = [r[key]["unseen_reject"] for r in tr_runs]
        top1 = [r[key]["seen_top1"] for r in tr_runs]
        return {
            "auroc_mean": float(np.mean(vals)),
            "auroc_std": float(np.std(vals)),
            "unseen_reject_mean": float(np.mean(rejects)),
            "unseen_reject_std": float(np.std(rejects)),
            "seen_top1_mean": float(np.mean(top1)),
            "seen_top1_std": float(np.std(top1)),
        }

    metrics = {
        "data_dir": str(data_dir),
        "n_seen_train": len(_split_indices(feats, bank, "train", True)),
        "n_seen_val": len(val_seen),
        "n_seen_test": len(test_seen),
        "n_unseen_test": len(test_unseen),
        "pss": pss_metrics,
        "transformer": {
            "seeds": seeds,
            "softmax": aggregate("softmax"),
            "energy": aggregate("energy"),
            "runs": tr_runs,
        },
        "h2": {
            "pss_auroc": pss_metrics["auroc"],
            "transformer_softmax_auroc": aggregate("softmax")["auroc_mean"],
            "transformer_energy_auroc": aggregate("energy")["auroc_mean"],
            "pss_unseen_reject": pss_metrics["unseen_reject"],
            "transformer_softmax_reject": aggregate("softmax")["unseen_reject_mean"],
            "transformer_energy_reject": aggregate("energy")["unseen_reject_mean"],
        },
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "p2_openset.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_report(out / "p2_openset.md", metrics)
    return metrics


def _write_report(path: Path, metrics: dict) -> None:
    pss = metrics["pss"]
    tr = metrics["transformer"]
    lines = [
        "# P2: open-set rejection",
        "",
        f"- data: `{metrics['data_dir']}`",
        f"- seen train / val / test: {metrics['n_seen_train']} / {metrics['n_seen_val']} / {metrics['n_seen_test']}",
        f"- unseen test: {metrics['n_unseen_test']}",
        "",
        "| method | AUROC | unseen reject @ 95% seen accept | seen top-1 |",
        "|---|---:|---:|---:|",
        f"| PSS prototypes | {pss['auroc']:.3f} | {pss['unseen_reject']:.3f} | {pss['seen_top1']:.3f} |",
        f"| Transformer softmax | {tr['softmax']['auroc_mean']:.3f} ± {tr['softmax']['auroc_std']:.3f} | "
        f"{tr['softmax']['unseen_reject_mean']:.3f} | {tr['softmax']['seen_top1_mean']:.3f} |",
        f"| Transformer energy | {tr['energy']['auroc_mean']:.3f} ± {tr['energy']['auroc_std']:.3f} | "
        f"{tr['energy']['unseen_reject_mean']:.3f} | {tr['energy']['seen_top1_mean']:.3f} |",
        "",
        "Both methods calibrate their threshold on seen val to 95% acceptance.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
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
    metrics = run_p2(
        args.data,
        args.experts,
        args.projections,
        args.prototypes,
        args.out,
        seeds=args.seeds,
        epochs=args.epochs,
        device=args.device,
    )
    print("P2 complete:")
    print(f"  PSS AUROC={metrics['pss']['auroc']:.3f} "
          f"unseen_reject={metrics['pss']['unseen_reject']:.3f}")
    print(f"  Transformer softmax AUROC={metrics['transformer']['softmax']['auroc_mean']:.3f} "
          f"energy AUROC={metrics['transformer']['energy']['auroc_mean']:.3f}")
    print(f"  output: {args.out}")


if __name__ == "__main__":
    main()
