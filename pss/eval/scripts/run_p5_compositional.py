"""P5: compositional attribute OOD, PSS prototypes vs transformer.

Train on low-hardness samples of seen categories; test on:
- ID: low-hardness test samples;
- OOD: high-hardness samples of the same categories (unseen material regime).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pss._console import ensure_utf8
from pss.eval.scripts.run_p2_openset import MODALITIES, ModalityTransformer
from pss.prototypes import PrototypeBank, recognize
from pss.prototypes.build import extract_projected_features


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _bank_from_low_train(feats, Z, bank_classes, low_train):
    protos, stats = {}, {}
    for category in bank_classes:
        idx = [i for i in low_train if feats.categories[i] == category]
        rows = []
        for m in MODALITIES:
            if not idx:
                rows.append(np.zeros(Z[m].shape[1]))
            else:
                rows.append(_unit(Z[m][idx].mean(axis=0)))
        stats[category] = {}
        for m, row in zip(MODALITIES, rows):
            stats[category][m] = {
                "sample_count": len(idx),
                "variance": float(
                    np.mean(1.0 - np.asarray(Z[m][idx], dtype=float) @ row)
                )
                if idx
                else 0.0,
            }
    protos = {
        m: np.stack(
            [
                _unit(Z[m][[i for i in low_train if feats.categories[i] == c]].mean(axis=0))
                for c in bank_classes
            ]
        ).astype(np.float32)
        for m in MODALITIES
    }
    return PrototypeBank(list(bank_classes), protos, stats, version="attr_low")


def _accuracy(bank, Z, indices, weights, feats):
    correct = 0
    for i in indices:
        q = {m: Z[m][i] for m in MODALITIES}
        r = recognize(q, bank, weights=weights, use_tree=False, threshold=None)
        if r.ranked and r.ranked[0][0] == feats.categories[i]:
            correct += 1
    return correct / max(1, len(indices))


def _train_transformer(Z, feats, classes, train_idx, val_idx, seed, epochs, device):
    torch.manual_seed(seed)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    dims = [Z[m].shape[1] for m in MODALITIES]
    model = ModalityTransformer(dims, len(classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    x_train = [torch.as_tensor(Z[m][train_idx], dtype=torch.float32, device=device) for m in MODALITIES]
    y_train = torch.as_tensor([class_to_idx[feats.categories[i]] for i in train_idx], dtype=torch.long, device=device)
    x_val = [torch.as_tensor(Z[m][val_idx], dtype=torch.float32, device=device) for m in MODALITIES]
    y_val = torch.as_tensor([class_to_idx[feats.categories[i]] for i in val_idx], dtype=torch.long, device=device)
    best_state, best_val = None, -1.0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(y_train), device=device)
        for s in range(0, len(perm), 64):
            idx = perm[s : s + 64]
            logits = model([x[idx] for x in x_train])
            loss = F.cross_entropy(logits, y_train[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(x_val).argmax(dim=1)
            acc = float((pred == y_val).float().mean().cpu())
        if acc > best_val:
            best_val = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_val


def _model_accuracy(model, Z, indices, feats, classes, device):
    x = [torch.as_tensor(Z[m][indices], dtype=torch.float32, device=device) for m in MODALITIES]
    with torch.no_grad():
        pred = model(x).argmax(dim=1).cpu().numpy()
    class_to_idx = {c: i for i, c in enumerate(classes)}
    correct = sum(int(pred[k] == class_to_idx[feats.categories[i]]) for k, i in enumerate(indices))
    return correct / max(1, len(indices))


def run_p5(data_dir, expert_dir, projection_dir, prototype_dir, out_dir, seeds=5, epochs=200, device="cpu"):
    feats, Z = extract_projected_features(data_dir, expert_dir, projection_dir, device)
    base_bank = PrototypeBank.load(Path(prototype_dir) / "prototypes.npz")
    classes = sorted(base_bank.categories)
    weights = {"vision": 1.0, "tactile": 0.5, "geometry": 1.0}
    mp = Path(prototype_dir) / "metrics.json"
    if mp.exists():
        weights = json.loads(mp.read_text(encoding="utf-8")).get("weights", weights)

    hardness = np.asarray(feats.attrs[:, 0], dtype=float)
    low, ood = [], []
    for category in classes:
        cat_idx = [i for i, c in enumerate(feats.categories) if c == category]
        if not cat_idx:
            continue
        med = float(np.median(hardness[cat_idx]))
        for i in cat_idx:
            (low if hardness[i] <= med else ood).append(i)

    def by_split(split):
        return [i for i in low if feats.splits[i] == split]

    low_train, low_val, low_test = by_split("train"), by_split("val"), by_split("test")
    bank = _bank_from_low_train(feats, Z, classes, low_train)
    pss_id = _accuracy(bank, Z, low_test, weights, feats)
    pss_ood = _accuracy(bank, Z, ood, weights, feats)

    runs = []
    for seed in range(seeds):
        model, best_val = _train_transformer(
            Z, feats, classes, low_train, low_val, seed, epochs, device
        )
        runs.append(
            {
                "seed": seed,
                "best_val": best_val,
                "id": _model_accuracy(model, Z, low_test, feats, classes, device),
                "ood": _model_accuracy(model, Z, ood, feats, classes, device),
            }
        )
    tr_id = float(np.mean([r["id"] for r in runs]))
    tr_ood = float(np.mean([r["ood"] for r in runs]))
    tr_ood_std = float(np.std([r["ood"] for r in runs]))
    h5 = {
        "pss_ood": pss_ood,
        "transformer_ood": tr_ood,
        "delta": pss_ood - tr_ood,
        "threshold": 0.10,
        "pass": bool(pss_ood - tr_ood >= 0.10),
    }
    metrics = {
        "data_dir": str(data_dir),
        "n_low_train": len(low_train),
        "n_low_val": len(low_val),
        "n_low_test": len(low_test),
        "n_ood": len(ood),
        "pss": {"id": pss_id, "ood": pss_ood},
        "transformer": {"id": tr_id, "ood": tr_ood, "ood_std": tr_ood_std, "runs": runs},
        "h5": h5,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "p5_compositional.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    _report(out / "p5_compositional.md", metrics)
    return metrics


def _report(path, metrics):
    p, t, h = metrics["pss"], metrics["transformer"], metrics["h5"]
    lines = [
        "# P5: compositional attribute OOD",
        "",
        f"- low train/val/test: {metrics['n_low_train']} / {metrics['n_low_val']} / {metrics['n_low_test']}",
        f"- high-hardness OOD: {metrics['n_ood']}",
        "",
        "| method | ID accuracy | OOD accuracy |",
        "|---|---:|---:|",
        f"| PSS prototypes | {p['id']:.3f} | **{p['ood']:.3f}** |",
        f"| Transformer | {t['id']:.3f} | {t['ood']:.3f} ± {t['ood_std']:.3f} |",
        "",
        f"**H5**: OOD delta {h['delta']:+.3f} (threshold {h['threshold']:.2f}) -> "
        f"**{'PASS' if h['pass'] else 'FAIL'}**",
        "",
        "Train on low-hardness samples; OOD test is high-hardness samples of the same seen categories.",
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
    m = run_p5(args.data, args.experts, args.projections, args.prototypes, args.out,
               seeds=args.seeds, epochs=args.epochs, device=args.device)
    print(f"P5: PSS ID/OOD {m['pss']['id']:.3f}/{m['pss']['ood']:.3f}; "
          f"TR ID/OOD {m['transformer']['id']:.3f}/{m['transformer']['ood']:.3f}; "
          f"H5 delta {m['h5']['delta']:+.3f}")
    print(f"output: {args.out}")


if __name__ == "__main__":
    main()
