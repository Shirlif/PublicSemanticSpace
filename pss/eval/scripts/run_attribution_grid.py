"""Attribution grid v2: run the §5 ablation protocol end-to-end.

Stages
------
grid     frozen-encoder readouts over projection transforms / dimensions /
         alignment targets (raw, pca, rand, linear, mlp, res), plus PSS proto.
bilevel  minimal co-adaptation arm: projection + readout trained jointly vs the
         same projection frozen and the same readout trained alone.
encoder  encoder-ladder arm: frozen vs fine-tuned small vision expert (matched
         head), and a DINOv2 vision channel with the same projection/readout.

Everything is evaluated on the same task split. Config selection uses val only;
test is reported once per arm.

Example
-------
python -m pss.eval.scripts.run_attribution_grid --stage grid \
  --data data/category_pretrain_instance8 \
  --experts checkpoints/category_pretrain_instance8_experts \
  --regime view-instance --out eval/results/attribution_v2
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from pss._console import ensure_utf8
from pss.encoders.data import crop_object, load_index, normalize_image
from pss.encoders.tactile import load_tactile_expert
from pss.encoders.vision import load_vision_expert
from pss.projection.features import load_paired_features
from pss.projection.heads import ProjectionHead, load_projection
from pss.projection.losses import material_within_category, supcon_cross
from pss.projection.train import _material_bins
from pss.prototypes.bank import PrototypeBank, l2
from pss.sim.dataset import load_sample

MODALITIES = ("vision", "tactile", "geometry")
DEFAULT_WEIGHTS = {"vision": 1.0, "tactile": 0.5, "geometry": 1.0}
SEEDS = (0, 1, 2)
READOUTS = ("lr", "mlp", "mlp_cap", "proto")


# --------------------------------------------------------------------------- #
# features / task split
# --------------------------------------------------------------------------- #
def _cache_path(out: Path) -> Path:
    return out / "paired_features.npz"


def load_features(data_dir: str, experts: str, out: Path, *, refresh: bool = False) -> dict:
    cache = _cache_path(out)
    if cache.exists() and not refresh:
        z = np.load(cache, allow_pickle=True)
        return {k: z[k] for k in z.files}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vision = load_vision_expert(Path(experts) / "vision.pt", device=device)
    tactile = load_tactile_expert(Path(experts) / "tactile.pt", device=device)
    feats = load_paired_features(Path(data_dir), vision, tactile)
    payload = {
        "vision": np.asarray(feats.vision, dtype=np.float32),
        "tactile": np.asarray(feats.tactile, dtype=np.float32),
        "geometry": np.asarray(feats.geometry, dtype=np.float32),
        "attrs": np.asarray(feats.attrs, dtype=np.float32),
        "categories": np.asarray([str(x) for x in feats.categories]),
        "instances": np.asarray([str(x) for x in feats.instances]),
        "splits": np.asarray([str(x) for x in feats.splits]),
        "views": np.asarray(feats.views),
        "sample_ids": np.asarray([str(x) for x in feats.sample_ids]),
    }
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **payload)
    return payload


def task_split(feats: dict, regime: str) -> dict:
    splits = np.asarray(feats["splits"])
    views = np.asarray(feats["views"])
    if regime == "view-instance":
        labels = np.asarray([str(x) for x in feats["instances"]])
        seen = np.isin(splits, ["train", "val", "test"])
        vmax = int(views.max()) + 1
        if vmax < 3:
            raise ValueError("view-instance needs >= 3 views")
        train_idx = np.where(seen & (views <= vmax - 3))[0]
        val_idx = np.where(seen & (views == vmax - 2))[0]
        test_idx = np.where(seen & (views == vmax - 1))[0]
    else:
        labels = np.asarray([str(x) for x in feats["categories"]])
        train_idx = np.where(splits == "train")[0]
        val_idx = np.where(splits == "val")[0]
        test_idx = np.where(splits == "test")[0]
    held_idx = np.where(splits == "test_category")[0]
    classes = sorted({str(labels[i]) for i in train_idx})
    cls_to_i = {c: i for i, c in enumerate(classes)}
    return {
        "labels": labels,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "held_idx": held_idx,
        "classes": classes,
        "cls_to_i": cls_to_i,
        "ytr": np.array([cls_to_i[str(labels[i])] for i in train_idx]),
        "yva": np.array([cls_to_i[str(labels[i])] for i in val_idx]),
        "yte": np.array([cls_to_i[str(labels[i])] for i in test_idx]),
    }


# --------------------------------------------------------------------------- #
# small models / readouts
# --------------------------------------------------------------------------- #
class LinearHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: int = 256, cap: int | None = None):
        super().__init__()
        if cap is None:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_classes)
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, cap), nn.ReLU(),
                nn.Linear(cap, hidden), nn.ReLU(),
                nn.Linear(hidden, n_classes),
            )

    def forward(self, x):
        return self.net(x)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _acc(pred, y) -> float:
    return float(np.mean(np.asarray(pred) == np.asarray(y)))


def _ece(probs: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    conf, pred = probs.max(1), probs.argmax(1)
    correct = (pred == y).astype(np.float64)
    total = len(y)
    out = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        out += abs(correct[mask].mean() - conf[mask].mean()) * mask.sum() / total
    return float(out)


def _nll(probs: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)).mean())


def _softmax(x: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = (x - x.max(axis=1, keepdims=True)) / T
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _fit_temperature(scores: np.ndarray, y: np.ndarray) -> float:
    best_T, best_nll = 1.0, float("inf")
    for T in np.linspace(0.2, 10.0, 50):
        p = _softmax(scores, T)
        nll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0)).mean()
        if nll < best_nll:
            best_T, best_nll = float(T), float(nll)
    return best_T


def fit_mlp(Xtr, ytr, Xva, yva, Xte, n_classes, *, seed, epochs=300, lr=1e-3,
            wd=1e-4, hidden=256, cap=None, device="cuda"):
    set_seed(seed)
    model = MLPClassifier(Xtr.shape[1], n_classes, hidden=hidden, cap=cap).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss()
    Xtr_t = torch.as_tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.as_tensor(ytr, dtype=torch.long, device=device)
    Xva_t = torch.as_tensor(Xva, dtype=torch.float32, device=device)
    yva_t = torch.as_tensor(yva, dtype=torch.long, device=device)
    best, best_state = -1.0, {k: v.detach().clone() for k, v in model.state_dict().items()}
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(Xtr_t), ytr_t)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(Xva_t).argmax(1) == yva_t).float().mean().item()
        if acc > best:
            best = acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(torch.as_tensor(Xte, dtype=torch.float32, device=device)), 1).cpu().numpy()
    return probs, best


def fit_lr(Xtr, ytr, Xte):
    clf = LogisticRegression(max_iter=4000, C=1.0).fit(Xtr, ytr)
    return clf.predict_proba(Xte)


# --------------------------------------------------------------------------- #
# PSS prototype readout
# --------------------------------------------------------------------------- #
def build_bank(Z: dict[str, np.ndarray], train_rows, labels, classes) -> PrototypeBank:
    protos = {}
    for m in MODALITIES:
        rows = [l2(Z[m][[i for i in train_rows if labels[i] == c]].mean(0)) for c in classes]
        protos[m] = np.stack(rows).astype(np.float32)
    return PrototypeBank(categories=list(classes), protos=protos, stats={}, version="labels")


def fused_scores(Z, bank, rows, weights=DEFAULT_WEIGHTS) -> np.ndarray:
    total = sum(weights[m] for m in MODALITIES)
    scores = np.zeros((len(rows), len(bank.categories)), dtype=np.float64)
    for m in MODALITIES:
        z = np.asarray([l2(Z[m][i]) for i in rows], dtype=np.float64)
        scores += weights[m] * (z @ bank.protos[m].T.astype(np.float64))
    return scores / total


# --------------------------------------------------------------------------- #
# projection transforms
# --------------------------------------------------------------------------- #
def _label_map(labels, rows):
    values = sorted({str(labels[i]) for i in rows})
    return values, {v: i for i, v in enumerate(values)}


def train_projection(raw, attrs, labels, train_idx, *, transform, dim, seed,
                     epochs=120, batch_size=256, device="cuda", material_weight=1.0,
                     return_heads=False):
    """Train one projection configuration; return projected features for all rows."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_values, lmap = _label_map(labels, train_idx)
    y_all = np.array([lmap.get(str(v), -1) for v in labels], dtype=np.int64)
    tr = train_idx[y_all[train_idx] >= 0]
    mat_all = _material_bins(attrs, tr)
    y = torch.as_tensor(y_all, dtype=torch.long)
    mat = torch.as_tensor(mat_all, dtype=torch.long)
    X = {m: torch.as_tensor(raw[m], dtype=torch.float32) for m in MODALITIES}
    A = torch.as_tensor(attrs, dtype=torch.float32)

    heads, attr_heads = {}, {}
    for m in MODALITIES:
        d_in = X[m].shape[1]
        if transform == "linear":
            head = LinearHead(d_in, dim)
        else:
            head = ProjectionHead(d_in, dim, hidden=256, residual=(transform == "res"))
        heads[m] = head.to(device)
    for m in MODALITIES:
        attr_heads[m] = nn.Linear(dim, A.shape[1]).to(device)
    params = [p for h in heads.values() for p in h.parameters()]
    params += [p for h in attr_heads.values() for p in h.parameters()]
    opt = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-4)

    for _ in range(epochs):
        for h in heads.values():
            h.train()
        for h in attr_heads.values():
            h.train()
        perm = torch.randperm(len(tr))
        for start in range(0, len(tr), batch_size):
            idx = torch.as_tensor(tr[perm[start:start + batch_size]], dtype=torch.long)
            yb = y[idx].to(device)
            myb = mat[idx].to(device)
            z = {m: heads[m](X[m][idx].to(device)) for m in MODALITIES}
            cat_loss = (
                supcon_cross(z["vision"], z["tactile"], yb)
                + supcon_cross(z["tactile"], z["vision"], yb)
                + supcon_cross(z["vision"], z["geometry"], yb)
                + supcon_cross(z["geometry"], z["vision"], yb)
                + supcon_cross(z["tactile"], z["geometry"], yb)
                + supcon_cross(z["geometry"], z["tactile"], yb)
            ) / 6.0
            mat_loss = sum(
                material_within_category(z[m], yb, myb)
                for m in ("vision", "tactile")
            ) / 2.0
            attr_loss = sum(
                F.mse_loss(attr_heads[m](z[m]), A[idx].to(device)) for m in MODALITIES
            ) / 3.0
            loss = cat_loss + material_weight * mat_loss + 0.1 * attr_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

    out = {}
    with torch.no_grad():
        for m in MODALITIES:
            heads[m].eval()
            zs = []
            for start in range(0, len(X[m]), 512):
                zs.append(heads[m](X[m][start:start + 512].to(device)).cpu())
            out[m] = torch.cat(zs).numpy().astype(np.float64)
    if return_heads:
        return out, heads
    return out


def pca_transform(raw, train_idx, dim):
    out = {}
    for m in MODALITIES:
        d_in = raw[m].shape[1]
        k = int(min(dim, d_in, len(train_idx) - 1))
        pca = PCA(n_components=k, whiten=True, random_state=0).fit(raw[m][train_idx])
        z = pca.transform(raw[m])
        out[m] = z / np.clip(np.linalg.norm(z, axis=1, keepdims=True), 1e-12, None)
    return out


def rand_transform(raw, dim, seed):
    rng = np.random.default_rng(seed)
    out = {}
    for m in MODALITIES:
        d_in = raw[m].shape[1]
        W = rng.normal(0.0, 1.0 / np.sqrt(d_in), size=(d_in, dim))
        z = raw[m] @ W
        out[m] = z / np.clip(np.linalg.norm(z, axis=1, keepdims=True), 1e-12, None)
    return out


# --------------------------------------------------------------------------- #
# representation evaluation
# --------------------------------------------------------------------------- #
def concat(rep, rows):
    return np.concatenate([rep[m][rows] for m in MODALITIES], axis=1)


def eval_rep(rep, task, tag, *, seeds=SEEDS, device="cuda", open_set=False):
    tr, va, te = task["train_idx"], task["val_idx"], task["test_idx"]
    Xtr, Xva, Xte = concat(rep, tr), concat(rep, va), concat(rep, te)
    n_classes = len(task["classes"])
    out = {}
    t0 = time.time()

    # linear probe (val used only for reporting; no hyperparameter search)
    probs = fit_lr(Xtr, task["ytr"], Xte)
    pred = probs.argmax(1)
    out["lr"] = {
        "val_acc": _acc(fit_lr(Xtr, task["ytr"], Xva).argmax(1), task["yva"]),
        "test_acc": _acc(pred, task["yte"]),
        "ece": _ece(probs, task["yte"]),
        "nll": _nll(probs, task["yte"]),
        "f1": float(f1_score(task["yte"], pred, average="macro", zero_division=0)),
        "n_params": int(Xtr.shape[1] * n_classes),
        "probs": probs, "pred": pred,
    }

    # natural MLP
    prob_list, val_list = [], []
    for seed in seeds:
        pr, v = fit_mlp(Xtr, task["ytr"], Xva, task["yva"], Xte, n_classes, seed=seed, device=device)
        prob_list.append(pr)
        val_list.append(v)
    ens = np.mean(prob_list, axis=0)
    out["mlp"] = {
        "val_acc": float(np.mean(val_list)),
        "test_acc": _acc(ens.argmax(1), task["yte"]),
        "test_acc_std": float(np.std([_acc(p.argmax(1), task["yte"]) for p in prob_list])),
        "ece": _ece(ens, task["yte"]),
        "nll": _nll(ens, task["yte"]),
        "f1": float(f1_score(task["yte"], ens.argmax(1), average="macro", zero_division=0)),
        "n_params": int(Xtr.shape[1] * 256 + 256 * n_classes),
        "probs": ens, "pred": ens.argmax(1),
    }

    # capacity-capped MLP: fixed random 128-d projection, shared head
    rng = np.random.default_rng(12345)
    W = rng.normal(0.0, 1.0 / np.sqrt(Xtr.shape[1]), size=(Xtr.shape[1], 128))
    Xtr_c, Xva_c, Xte_c = Xtr @ W, Xva @ W, Xte @ W
    prob_list, val_list = [], []
    for seed in seeds:
        pr, v = fit_mlp(Xtr_c, task["ytr"], Xva_c, task["yva"], Xte_c, n_classes,
                        seed=seed, hidden=256, cap=None, device=device)
        prob_list.append(pr)
        val_list.append(v)
    ens_c = np.mean(prob_list, axis=0)
    out["mlp_cap"] = {
        "val_acc": float(np.mean(val_list)),
        "test_acc": _acc(ens_c.argmax(1), task["yte"]),
        "test_acc_std": float(np.std([_acc(p.argmax(1), task["yte"]) for p in prob_list])),
        "ece": _ece(ens_c, task["yte"]),
        "nll": _nll(ens_c, task["yte"]),
        "f1": float(f1_score(task["yte"], ens_c.argmax(1), average="macro", zero_division=0)),
        "n_params": int(Xtr.shape[1] * 128 + 128 * 256 + 256 * n_classes),
        "probs": ens_c, "pred": ens_c.argmax(1),
    }

    # PSS prototype readout
    bank = build_bank(rep, tr, task["labels"], task["classes"])
    val_scores = fused_scores(rep, bank, va)
    T = _fit_temperature(val_scores, task["yva"])
    te_scores = fused_scores(rep, bank, te)
    probs = _softmax(te_scores, T)
    out["proto"] = {
        "val_acc": _acc(val_scores.argmax(1), task["yva"]),
        "test_acc": _acc(te_scores.argmax(1), task["yte"]),
        "ece": _ece(probs, task["yte"]),
        "nll": _nll(probs, task["yte"]),
        "f1": float(f1_score(task["yte"], te_scores.argmax(1), average="macro", zero_division=0)),
        "n_params": 0,
        "temperature": T,
        "probs": probs, "pred": te_scores.argmax(1),
    }

    if open_set and len(task["held_idx"]) > 0:
        held = task["held_idx"]
        he_scores = fused_scores(rep, bank, held)
        out["proto"]["auroc"] = _auroc(te_scores.max(1), he_scores.max(1))
        Xhe = concat(rep, held)
        he_probs = np.mean(
            [fit_mlp(Xtr, task["ytr"], Xva, task["yva"], Xhe, n_classes, seed=s, device=device)[0]
             for s in seeds], axis=0
        )
        out["mlp"]["auroc"] = _auroc(ens.max(1), he_probs.max(1))
        out["lr"]["auroc"] = _auroc(
            fit_lr(Xtr, task["ytr"], Xhe).max(1),
            fit_lr(Xtr, task["ytr"], Xte).max(1),
        )

    out["_meta"] = {"tag": tag, "seconds": round(time.time() - t0, 2),
                    "dims": {m: int(rep[m].shape[1]) for m in MODALITIES}}
    return out


def _auroc(score_seen, score_unseen):
    y = np.concatenate([np.zeros(len(score_seen)), np.ones(len(score_unseen))])
    s = np.concatenate([-np.asarray(score_seen), -np.asarray(score_unseen)])
    return float(roc_auc_score(y, s))


# --------------------------------------------------------------------------- #
# grid stage
# --------------------------------------------------------------------------- #
def run_grid(args) -> dict:
    out = Path(args.out)
    feats = load_features(args.data, args.experts, out)
    task = task_split(feats, args.regime)
    raw = {m: np.asarray(feats[m], dtype=np.float32) for m in MODALITIES}
    attrs = np.asarray(feats["attrs"], dtype=np.float32)
    labels = task["labels"]

    # standardized raw arm
    sc = {m: StandardScaler().fit(raw[m][task["train_idx"]]) for m in MODALITIES}
    raw_std = {m: sc[m].transform(raw[m]).astype(np.float64) for m in MODALITIES}

    arms = {}
    def add(tag, rep, **kw):
        print(f"[eval] {tag}", flush=True)
        arms[tag] = eval_rep(rep, task, tag, seeds=tuple(args.seeds),
                             device=args.device, open_set=args.open_set, **kw)
        a = arms[tag]
        print(f"  lr={a['lr']['test_acc']:.3f} mlp={a['mlp']['test_acc']:.3f} "
              f"cap={a['mlp_cap']['test_acc']:.3f} proto={a['proto']['test_acc']:.3f}", flush=True)

    add("raw", raw_std)
    if "pca" in args.transforms:
        for dim in args.dims:
            add(f"pca{dim}", pca_transform(raw_std, task["train_idx"], dim))
    if "rand" in args.transforms:
        for dim in args.dims:
            add(f"rand{dim}", rand_transform(raw_std, dim, seed=0))

    trainable = [t for t in args.transforms if t in ("linear", "mlp", "res")]
    for transform in trainable:
        for dim in args.dims:
            for level in args.levels:
                level_labels = (
                    np.asarray([str(x) for x in feats["categories"]])
                    if level == "category"
                    else np.asarray([str(x) for x in feats["instances"]])
                )
                for seed in args.seeds:
                    tag = f"{transform}{dim}_{level}_s{seed}"
                    print(f"[train] {tag}", flush=True)
                    rep = train_projection(
                        raw, attrs, level_labels, task["train_idx"],
                        transform=transform, dim=dim, seed=seed, device=args.device,
                    )
                    add(tag, rep)

    payload = {
        "data": args.data, "experts": args.experts, "regime": args.regime,
        "transforms": args.transforms, "dims": args.dims, "levels": args.levels,
        "seeds": args.seeds,
        "counts": {k: int(len(task[k])) for k in ("train_idx", "val_idx", "test_idx", "held_idx")},
        "classes": len(task["classes"]),
        "arms": _strip_arrays(arms),
    }
    # keep predictions compactly for bootstrap CIs
    preds = {f"{tag}__{readout}": arms[tag][readout]["pred"]
             for tag in arms for readout in ("lr", "mlp", "mlp_cap", "proto")}
    preds["y_test"] = task["yte"]
    np.savez_compressed(out / "grid_preds.npz", **preds)
    _write_json(out / "grid.json", payload)
    return payload


# --------------------------------------------------------------------------- #
# production-recipe grid stage
# --------------------------------------------------------------------------- #
def run_grid_prod(args) -> dict:
    """Re-run the trainable projection grid with the production trainer.

    The hand-written trainer in this file is only an approximation; the
    production ``train_projections`` is the ground truth for the deployed space.
    This stage monkeypatches its feature loader to the cached features, so it
    runs without re-encoding images.
    """
    from pss.projection.features import PairedFeatures
    from pss.projection.train import train_projections
    from pss.projection.heads import load_projection

    out = Path(args.out)
    feats = load_features(args.data, args.experts, out)
    task = task_split(feats, args.regime)
    raw = {m: np.asarray(feats[m], dtype=np.float32) for m in MODALITIES}
    pf = PairedFeatures(
        vision=raw["vision"], tactile=raw["tactile"], geometry=raw["geometry"],
        attrs=np.asarray(feats["attrs"], dtype=np.float32),
        categories=[str(x) for x in feats["categories"]],
        instances=[str(x) for x in feats["instances"]],
        splits=[str(x) for x in feats["splits"]],
        sample_ids=[str(x) for x in feats["sample_ids"]],
        views=[int(x) for x in feats["views"]],
    )
    import pss.projection.train as T
    T.load_paired_features = lambda *a, **k: pf

    arms = {}
    ckpt_root = out / "prod_ckpt"
    for transform in ("mlp", "res"):
        for dim in args.dims:
            for level in args.levels:
                for seed in args.seeds:
                    tag = f"prod_{transform}{dim}_{level}_s{seed}"
                    ckpt = ckpt_root / f"{transform}{dim}_{level}_s{seed}"
                    print(f"[prod-train] {tag}", flush=True)
                    train_projections(
                        args.data, args.experts, ckpt, epochs=args.epochs,
                        batch_size=256, level=level, seed=seed, out_dim=dim,
                        material_weight=1.0, residual=(transform == "res"),
                    )
                    heads = {m: load_projection(ckpt / f"{m}.pt", device=args.device)
                             for m in MODALITIES}
                    proj = {}
                    with torch.no_grad():
                        for m in MODALITIES:
                            proj[m] = heads[m](
                                torch.as_tensor(raw[m]).to(args.device)
                            ).cpu().numpy().astype(np.float64)
                    print(f"[eval] {tag}", flush=True)
                    arms[tag] = eval_rep(proj, task, tag, seeds=tuple(args.seeds),
                                         device=args.device, open_set=args.open_set)
                    a = arms[tag]
                    print(f"  lr={a['lr']['test_acc']:.3f} mlp={a['mlp']['test_acc']:.3f} "
                          f"proto={a['proto']['test_acc']:.3f}", flush=True)

    payload = {
        "data": args.data, "experts": args.experts, "regime": args.regime,
        "recipe": "production train_projections (cached features)",
        "transforms": ["mlp", "res"], "dims": args.dims, "levels": args.levels,
        "seeds": args.seeds, "classes": len(task["classes"]),
        "counts": {k: int(len(task[k])) for k in ("train_idx", "val_idx", "test_idx", "held_idx")},
        "arms": _strip_arrays(arms),
    }
    preds = {f"{tag}__{ro}": arms[tag][ro]["pred"] for tag in arms for ro in READOUTS}
    preds["y_test"] = task["yte"]
    np.savez_compressed(out / "grid_prod_preds.npz", **preds)
    _write_json(out / "grid_prod.json", payload)
    return payload


# --------------------------------------------------------------------------- #
# bilevel stage
# --------------------------------------------------------------------------- #
def run_bilevel(args) -> dict:
    out = Path(args.out)
    feats = load_features(args.data, args.experts, out)
    task = task_split(feats, args.regime)
    raw = {m: np.asarray(feats[m], dtype=np.float32) for m in MODALITIES}
    attrs = np.asarray(feats["attrs"], dtype=np.float32)
    labels = task["labels"]
    device = args.device
    n_classes = len(task["classes"])

    # same trained projection initializes both arms; only the training scope differs
    init_dir = Path(getattr(args, "init_projections", "") or "")
    if init_dir and init_dir.exists():
        init_heads = {m: load_projection(init_dir / f"{m}.pt", device=device)
                      for m in MODALITIES}
        print(f"[bilevel] init from {init_dir}", flush=True)
    else:
        _, init_heads = train_projection(
            raw, attrs, labels, task["train_idx"], transform="mlp", dim=128, seed=0,
            device=device, return_heads=True,
        )
    init_state = {m: {k: v.detach().clone() for k, v in init_heads[m].state_dict().items()}
                  for m in MODALITIES}

    X = {m: torch.as_tensor(raw[m], dtype=torch.float32) for m in MODALITIES}
    tr_t = torch.as_tensor(task["train_idx"], dtype=torch.long)
    va_t = torch.as_tensor(task["val_idx"], dtype=torch.long)
    te_t = torch.as_tensor(task["test_idx"], dtype=torch.long)
    ytr_t = torch.as_tensor(task["ytr"], dtype=torch.long)
    yva_t = torch.as_tensor(task["yva"], dtype=torch.long)
    yte_t = torch.as_tensor(task["yte"], dtype=torch.long)

    def full_batch(rows, device):
        return {m: X[m][rows].to(device) for m in MODALITIES}

    results = {}
    for arm in ("frozen", "joint"):
        for seed in args.seeds:
            set_seed(seed)
            heads = {}
            for m in MODALITIES:
                h = ProjectionHead(X[m].shape[1], 128, hidden=256).to(device)
                h.load_state_dict(init_state[m])
                for p in h.parameters():
                    p.requires_grad_(arm == "joint")
                heads[m] = h
            head = MLPClassifier(128 * len(MODALITIES), n_classes, hidden=256).to(device)
            params = list(head.parameters())
            if arm == "joint":
                params += [p for h in heads.values() for p in h.parameters()]
            opt = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-4)

            ztr = full_batch(tr_t, device)
            zva = full_batch(va_t, device)
            zte = full_batch(te_t, device)
            ytr_d, yva_d, yte_d = ytr_t.to(device), yva_t.to(device), yte_t.to(device)

            best, best_state = -1.0, None
            for _ in range(args.epochs):
                for h in heads.values():
                    h.train(arm == "joint")
                head.train()
                if arm == "joint":
                    z = {m: heads[m](ztr[m]) for m in MODALITIES}
                else:
                    with torch.no_grad():
                        z = {m: heads[m](ztr[m]) for m in MODALITIES}
                loss = F.cross_entropy(head(torch.cat([z[m] for m in MODALITIES], 1)), ytr_d)
                opt.zero_grad()
                loss.backward()
                opt.step()

                for h in heads.values():
                    h.eval()
                head.eval()
                with torch.no_grad():
                    zva_b = {m: heads[m](zva[m]) for m in MODALITIES}
                    acc = (head(torch.cat([zva_b[m] for m in MODALITIES], 1)).argmax(1) == yva_d).float().mean().item()
                if acc > best:
                    best = acc
                    best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
                    best_heads = {m: {k: v.detach().clone() for k, v in heads[m].state_dict().items()}
                                  for m in MODALITIES}
            head.load_state_dict(best_state)
            for m in MODALITIES:
                heads[m].load_state_dict(best_heads[m])
            with torch.no_grad():
                for h in heads.values():
                    h.eval()
                zte_b = {m: heads[m](zte[m]) for m in MODALITIES}
                pred = head(torch.cat([zte_b[m] for m in MODALITIES], 1)).argmax(1).cpu().numpy()
            results[f"{arm}_s{seed}"] = {"val_acc": best, "test_acc": _acc(pred, task["yte"])}
    payload = {"data": args.data, "regime": args.regime, "arms": results}
    _write_json(out / "bilevel.json", payload)
    return payload


# --------------------------------------------------------------------------- #
# encoder stage
# --------------------------------------------------------------------------- #
def load_images(data_dir, image_size, preserve_aspect):
    index = load_index(Path(data_dir))
    imgs = []
    for entry in index["entries"]:
        if entry["kind"] != "category":
            continue
        sample = load_sample(Path(data_dir) / entry["path"])
        geom_id = int(sample["geom_id"])
        for v in range(sample["rgb"].shape[0]):
            crop = crop_object(sample["rgb"][v], sample["segmentation"][v], geom_id,
                               size=image_size, preserve_aspect=preserve_aspect)
            imgs.append(normalize_image(crop))
    return np.stack(imgs).astype(np.float32)


def encode_dino_stream(data_dir, dino, preserve, batch_size=32):
    index = load_index(Path(data_dir))
    out, buf = [], []
    for entry in index["entries"]:
        if entry["kind"] != "category":
            continue
        sample = load_sample(Path(data_dir) / entry["path"])
        geom_id = int(sample["geom_id"])
        for v in range(sample["rgb"].shape[0]):
            crop = crop_object(sample["rgb"][v], sample["segmentation"][v], geom_id,
                               size=dino.image_size, preserve_aspect=preserve)
            buf.append(normalize_image(crop))
            if len(buf) >= batch_size:
                out.append(dino.encode(torch.as_tensor(np.stack(buf)), batch_size=batch_size).cpu().numpy())
                buf = []
    if buf:
        out.append(dino.encode(torch.as_tensor(np.stack(buf)), batch_size=batch_size).cpu().numpy())
    return np.concatenate(out, 0).astype(np.float64)


def run_encoder(args) -> dict:
    out = Path(args.out)
    feats = load_features(args.data, args.experts, out)
    task = task_split(feats, args.regime)
    raw = {m: np.asarray(feats[m], dtype=np.float32) for m in MODALITIES}
    device = args.device
    index = load_index(Path(args.data))
    preserve = bool(index.get("config", {}).get("crop_preserve_aspect", False))

    vision = load_vision_expert(Path(args.experts) / "vision.pt", device=device)
    vision_dim = int(raw["vision"].shape[1])
    images = torch.as_tensor(load_images(args.data, vision.image_size, preserve))
    sc = {m: StandardScaler().fit(raw[m][task["train_idx"]]) for m in ("tactile", "geometry")}
    aux = {m: torch.as_tensor(sc[m].transform(raw[m]), dtype=torch.float32) for m in ("tactile", "geometry")}

    results = {}
    init_enc = {k: v.detach().clone() for k, v in vision.encoder.state_dict().items()}
    for arm in ("frozen", "finetune"):
        for seed in args.seeds:
            set_seed(seed)
            enc = vision.encoder.to(device)
            enc.load_state_dict(init_enc)
            for p in enc.parameters():
                p.requires_grad_(arm == "finetune")
            enc.train(arm == "finetune")
            head = MLPClassifier(vision_dim + sum(aux[m].shape[1] for m in aux),
                                 len(task["classes"]), hidden=256).to(device)
            params = list(head.parameters()) + ([p for p in enc.parameters() if p.requires_grad])
            opt = torch.optim.Adam([
                {"params": [p for p in enc.parameters() if p.requires_grad], "lr": 1e-4},
                {"params": head.parameters(), "lr": 1e-3},
            ], weight_decay=1e-4) if arm == "finetune" else torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
            ytr = torch.as_tensor(task["ytr"], dtype=torch.long, device=device)
            yva = torch.as_tensor(task["yva"], dtype=torch.long, device=device)
            yte = torch.as_tensor(task["yte"], dtype=torch.long, device=device)
            tr = torch.as_tensor(task["train_idx"], dtype=torch.long)
            va = torch.as_tensor(task["val_idx"], dtype=torch.long)
            te = torch.as_tensor(task["test_idx"], dtype=torch.long)
            best, best_state = -1.0, None
            for _ in range(args.epochs):
                perm = torch.randperm(len(tr))
                head.train()
                if arm == "finetune":
                    enc.train()
                for start in range(0, len(tr), 64):
                    idx = tr[perm[start:start + 64]]
                    imgs = images[idx].to(device)
                    if arm == "finetune":
                        vf = enc(imgs)
                    else:
                        enc.eval()
                        with torch.no_grad():
                            vf = enc(imgs)
                    x = torch.cat([vf] + [aux[m][idx].to(device) for m in aux], dim=1)
                    loss = F.cross_entropy(head(x), ytr[perm[start:start + 64]].to(device))
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                head.eval()
                enc.eval()
                with torch.no_grad():
                    vf = enc(images[va].to(device))
                    x = torch.cat([vf] + [aux[m][va].to(device) for m in aux], 1)
                    acc = (head(x).argmax(1) == yva.to(device)).float().mean().item()
                if acc > best:
                    best = acc
                    best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
                    best_enc = {k: v.detach().clone() for k, v in enc.state_dict().items()} if arm == "finetune" else None
            head.load_state_dict(best_state)
            if best_enc is not None:
                enc.load_state_dict(best_enc)
            with torch.no_grad():
                vf = enc(images[te].to(device))
                x = torch.cat([vf] + [aux[m][te].to(device) for m in aux], 1)
                pred = head(x).argmax(1).cpu().numpy()
            results[f"{arm}_s{seed}"] = {"val_acc": best, "test_acc": _acc(pred, task["yte"])}

    # DINOv2 channel
    try:
        from pss.encoders.vision.dinov2 import load_dinov2_encoder
        dino = load_dinov2_encoder(device=device, local_files_only=True)
        dfeat = encode_dino_stream(args.data, dino, preserve, batch_size=32)
        drep = {
            "vision": dfeat,
            "tactile": sc["tactile"].transform(raw["tactile"]).astype(np.float64),
            "geometry": sc["geometry"].transform(raw["geometry"]).astype(np.float64),
        }
        dino_eval = eval_rep(drep, task, "dino_raw", seeds=tuple(args.seeds), device=device)
        results["dino_raw"] = _strip_one(dino_eval)
        # DINOv2 + learned projection (instance level)
        for seed in args.seeds:
            proj = train_projection(
                {m: np.asarray(drep[m], dtype=np.float32) for m in MODALITIES},
                np.asarray(feats["attrs"], dtype=np.float32),
                np.asarray([str(x) for x in feats["instances"]]),
                task["train_idx"], transform="mlp", dim=128, seed=seed, device=device,
            )
            ev = eval_rep(proj, task, f"dino_mlp128_s{seed}", seeds=tuple(args.seeds), device=device)
            results[f"dino_mlp128_s{seed}"] = _strip_one(ev)
    except Exception as exc:  # noqa: BLE001
        results["dino_error"] = repr(exc)

    _write_json(out / "encoder.json", {"data": args.data, "regime": args.regime, "arms": results})
    return results


# --------------------------------------------------------------------------- #
# serialization helpers
# --------------------------------------------------------------------------- #
def _strip_one(rep: dict) -> dict:
    return {k: {kk: vv for kk, vv in v.items() if kk not in ("probs", "pred")}
            for k, v in rep.items() if isinstance(v, dict)}


def _strip_arrays(arms: dict) -> dict:
    return {tag: _strip_one(rep) for tag, rep in arms.items()}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _safe(obj):
    if isinstance(obj, dict):
        return {str(k): _safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _safe(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, torch.Tensor):
        return _safe(obj.detach().cpu().numpy())
    return obj


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["grid", "grid_prod", "bilevel", "encoder", "all"], default="grid")
    ap.add_argument("--data", default="data/category_pretrain_instance8")
    ap.add_argument("--experts", default="checkpoints/category_pretrain_instance8_experts")
    ap.add_argument("--init-projections", default="checkpoints/category_pretrain_instance8_projection",
                    help="projection checkpoint used to initialize both bilevel arms")
    ap.add_argument("--regime", choices=["category", "view-instance"], default="view-instance")
    ap.add_argument("--out", default="eval/results/attribution_v2")
    ap.add_argument("--transforms", nargs="+", default=["raw", "pca", "rand", "linear", "mlp", "res"])
    ap.add_argument("--dims", nargs="+", type=int, default=[128, 512])
    ap.add_argument("--levels", nargs="+", default=["category", "instance"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--open-set", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    if args.stage in ("grid", "all"):
        run_grid(args)
    if args.stage in ("grid_prod", "all"):
        run_grid_prod(args)
    if args.stage in ("bilevel", "all"):
        run_bilevel(args)
    if args.stage in ("encoder", "all"):
        run_encoder(args)


if __name__ == "__main__":
    main()
