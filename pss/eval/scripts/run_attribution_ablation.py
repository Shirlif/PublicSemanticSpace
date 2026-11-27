"""CLI: attribution ablation for the public semantic space.

Implements §5 of ``docs/architecture/PSS-信息论定位.md``: decompose recognition
error into (encoder) + (projection) + (reasoning) on one frozen encoder stack.

Arms. Same train/val/test split, same class set (seen categories):

  raw + LR / MLP          readout on frozen expert features  -> encoder info
  proj + LR / MLP         readout on the existing PSS heads  -> projection loss
  PSS prototype fusion    prototype bank on projected feats  -> reasoning layer
  learned bottleneck(D)   supervised projection, then proto  -> co-adaptation gap

Closed-set accuracy on ``test``; open-set AUROC on ``test_category``; ECE/NLL
for calibration. Bootstrap CIs on the key deltas.

Example:
    python -m pss.eval.scripts.run_attribution_ablation \
        --data data/category_pretrain \
        --experts checkpoints/category_pretrain_experts \
        --projections checkpoints/phase45_projection \
        --out eval/results/attribution_ablation
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from pss._console import ensure_utf8
from pss.encoders.tactile import load_tactile_expert
from pss.encoders.vision import load_vision_expert
from pss.projection.features import load_paired_features
from pss.projection.heads import load_projection
from pss.prototypes.bank import PrototypeBank, l2

MODALITIES = ("vision", "tactile", "geometry")
DEFAULT_WEIGHTS = {"vision": 1.0, "tactile": 0.5, "geometry": 1.0}
SEEDS = (0, 1, 2)


# --------------------------------------------------------------------------- #
# small torch models
# --------------------------------------------------------------------------- #
class MLPClassifier(nn.Module):
    """in -> hidden -> num_classes."""

    def __init__(self, in_dim: int, n_classes: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_classes)
        )

    def forward(self, x):
        return self.net(x)


class Bottleneck(nn.Module):
    """in -> dim -> num_classes; exposes the dim-dimensional activations."""

    def __init__(self, in_dim: int, n_classes: int, dim: int):
        super().__init__()
        self.body = nn.Linear(in_dim, dim)
        self.head = nn.Linear(dim, n_classes)

    def features(self, x):
        return F.relu(self.body(x))

    def forward(self, x):
        return self.head(self.features(x))


class GatedFusion(nn.Module):
    """Learned reliability fusion: per-modality embedding + softmax gate + head.

    This replaces the fixed cosine-fusion weights ``{vision:1, tactile:0.5,
    geometry:1}`` with a learned per-sample gate, which is the natural
    "learnable fusion head" upgrade of the PSS readout.
    """

    def __init__(self, dims, n_classes: int, d: int = 128, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.dims = tuple(int(x) for x in dims)
        self.embs = nn.ModuleList([nn.Linear(m, d) for m in self.dims])
        self.gates = nn.ModuleList([nn.Linear(d, 1) for _ in self.dims])
        self.head = nn.Sequential(
            nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, n_classes)
        )

    def forward(self, x):
        parts = torch.split(x, self.dims, dim=1)
        es = [F.relu(emb(p)) for emb, p in zip(self.embs, parts)]
        scores = torch.cat([gate(e) for gate, e in zip(self.gates, es)], dim=1)  # (B, M)
        w = torch.softmax(scores, dim=1)
        z = sum(w[:, i : i + 1] * es[i] for i in range(len(es)))
        return self.head(z)


class ConcatFusion(nn.Module):
    """Per-modality embedding + concat + head (capacity-preserving learned fusion).

    Unlike the gated sum, this keeps every modality's embedding in the fused
    vector, so fine-grained instance cues are not averaged away.
    """

    def __init__(self, dims, n_classes: int, d: int = 128, hidden: int = 256, dropout: float = 0.1, relu: bool = True):
        super().__init__()
        self.dims = tuple(int(x) for x in dims)
        self.relu = bool(relu)
        self.embs = nn.ModuleList([nn.Linear(m, d) for m in self.dims])
        self.head = nn.Sequential(
            nn.Linear(d * len(self.dims), hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        parts = torch.split(x, self.dims, dim=1)
        if self.relu:
            z = torch.cat([F.relu(emb(p)) for emb, p in zip(self.embs, parts)], dim=1)
        else:
            z = torch.cat([emb(p) for emb, p in zip(self.embs, parts)], dim=1)
        return self.head(z)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _fit_torch(model, Xtr, ytr, Xva, yva, *, epochs=300, lr=1e-3, wd=1e-4) -> nn.Module:
    """Train with Adam, keep the best-val-accuracy state."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss()
    Xtr_t = torch.as_tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.as_tensor(ytr, dtype=torch.long)
    Xva_t = torch.as_tensor(Xva, dtype=torch.float32)
    yva_t = torch.as_tensor(yva, dtype=torch.long)
    best, best_state = -1.0, {k: v.clone() for k, v in model.state_dict().items()}
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
            best, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model


def _probs(model, X) -> np.ndarray:
    with torch.no_grad():
        logits = model(torch.as_tensor(X, dtype=torch.float32))
    return torch.softmax(logits, dim=1).numpy()


def _features(model, X) -> np.ndarray:
    with torch.no_grad():
        z = model.features(torch.as_tensor(X, dtype=torch.float32))
    return z.numpy()


# --------------------------------------------------------------------------- #
# prototype fusion (the PSS readout)
# --------------------------------------------------------------------------- #
def fused_scores(
    Z: dict[str, np.ndarray],
    bank: PrototypeBank,
    rows: np.ndarray,
    weights: dict[str, float],
) -> np.ndarray:
    """(n, C) weighted cosine fusion, matching ``recognize`` levels 3-4."""
    total = sum(weights[m] for m in MODALITIES)
    scores = np.zeros((len(rows), len(bank.categories)), dtype=np.float64)
    for m in MODALITIES:
        z = np.asarray([l2(Z[m][i]) for i in rows], dtype=np.float64)
        scores += weights[m] * (z @ bank.protos[m].T.astype(np.float64))
    return scores / total


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


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def _acc(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(pred == y))


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


def _bootstrap_delta(pred_a, pred_b, y, n=2000, seed=0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    a = (pred_a == y).astype(float)
    b = (pred_b == y).astype(float)
    deltas = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        deltas.append(a[idx].mean() - b[idx].mean())
    deltas = np.asarray(deltas)
    return float(a.mean() - b.mean()), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def build_bank_labels(Z, train_rows, labels, classes) -> PrototypeBank:
    """Prototype bank over an arbitrary string label array (category or instance)."""
    protos = {}
    for m in MODALITIES:
        rows = [l2(Z[m][[i for i in train_rows if labels[i] == c]].mean(0)) for c in classes]
        protos[m] = np.stack(rows).astype(np.float32)
    return PrototypeBank(categories=list(classes), protos=protos, stats={}, version="labels")


def run_fewshot(raw, projected, labels, splits, classes, cls_to_i, val_idx, test_idx, ks=(8, 4)):
    """Same decomposition under a few-shot train regime (headroom for deltas).

    Prototype readouts are the classic few-shot method; a linear/MLP probe is
    the generic readout. If the public space is the bottleneck, both stay
    equally low; if reasoning helps, PSS wins at small K.
    """
    yva = np.array([cls_to_i[str(labels[i])] for i in val_idx])
    yte = np.array([cls_to_i[str(labels[i])] for i in test_idx])
    train_all = np.where(splits == "train")[0]
    out = {}
    for K in ks:
        rng = np.random.default_rng(1000 + K)
        parts = []
        for c in classes:
            rows = np.array([i for i in train_all if str(labels[i]) == c])
            if len(rows) == 0:
                continue
            rng.shuffle(rows)
            parts.append(rows[:K])
        tr = np.sort(np.concatenate(parts))
        ytr = np.array([cls_to_i[str(labels[i])] for i in tr])
        raw_std = {m: StandardScaler().fit(raw[m][tr]).transform(raw[m]) for m in MODALITIES}

        def conc(rep, rows):
            return np.concatenate([rep[m][rows] for m in MODALITIES], axis=1)

        Xr_tr, Xr_va, Xr_te = conc(raw_std, tr), conc(raw_std, val_idx), conc(raw_std, test_idx)
        Xp_tr, Xp_va, Xp_te = conc(projected, tr), conc(projected, val_idx), conc(projected, test_idx)
        res = {}
        for name, Xtr_, Xte_ in (("raw_lr", Xr_tr, Xr_te), ("proj_lr", Xp_tr, Xp_te)):
            clf = LogisticRegression(max_iter=3000, C=1.0).fit(Xtr_, ytr)
            res[name] = _acc(clf.predict(Xte_), yte)
        for name, Xtr_, Xva_, Xte_ in (
            ("raw_mlp", Xr_tr, Xr_va, Xr_te), ("proj_mlp", Xp_tr, Xp_va, Xp_te)
        ):
            accs = []
            for seed in SEEDS:
                set_seed(seed)
                model = _fit_torch(MLPClassifier(Xtr_.shape[1], len(classes), 256), Xtr_, ytr, Xva_, yva)
                accs.append(_acc(_probs(model, Xte_).argmax(1), yte))
            res[name] = float(np.mean(accs))
        bank = build_bank_labels(projected, tr, labels, classes)
        T = _fit_temperature(fused_scores(projected, bank, val_idx, DEFAULT_WEIGHTS), yva)
        res["pss_proto"] = _acc(fused_scores(projected, bank, test_idx, DEFAULT_WEIGHTS).argmax(1), yte)
        out[str(K)] = res
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
@dataclass
class Arm:
    name: str
    family: str  # encoder / projection / reasoning / coadapt
    acc: float
    acc_std: float = 0.0
    ece: float | None = None
    nll: float | None = None
    pred: np.ndarray | None = None
    probs: np.ndarray | None = None


def run(args) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vision = load_vision_expert(Path(args.experts) / "vision.pt", device=device)
    tactile = load_tactile_expert(Path(args.experts) / "tactile.pt", device=device)
    feats = load_paired_features(Path(args.data), vision, tactile)

    raw = {"vision": feats.vision, "tactile": feats.tactile, "geometry": feats.geometry}
    heads = {m: load_projection(Path(args.projections) / f"{m}.pt", device=device) for m in MODALITIES}
    projected = {}
    for m in MODALITIES:
        heads[m].eval()
        with torch.no_grad():
            projected[m] = heads[m](
                torch.as_tensor(raw[m], dtype=torch.float32, device=device)
            ).cpu().numpy().astype(np.float64)

    splits = np.array(feats.splits)
    views = np.array(feats.views)
    regime = getattr(args, "regime", "category")
    held_idx = np.where(splits == "test_category")[0]
    if regime == "view-instance":
        # Closed-set instance identity: same object in every split, views
        # partitioned so train/val/test see disjoint viewpoints.
        labels = np.array([str(x) for x in feats.instances])
        seen_mask = np.isin(splits, ["train", "val", "test"])
        vmax = int(views.max()) + 1
        if vmax < 3:
            raise ValueError("view-instance regime needs >= 3 views per sample")
        train_idx = np.where(seen_mask & (views <= vmax - 3))[0]
        val_idx = np.where(seen_mask & (views == vmax - 2))[0]
        test_idx = np.where(seen_mask & (views == vmax - 1))[0]
        classes = sorted({str(labels[i]) for i in train_idx})
    else:
        labels = np.array([str(x) for x in feats.categories])
        train_idx = np.where(splits == "train")[0]
        val_idx = np.where(splits == "val")[0]
        test_idx = np.where(splits == "test")[0]
        classes = sorted({str(labels[i]) for i in train_idx})
    cls_to_i = {c: i for i, c in enumerate(classes)}
    ytr = np.array([cls_to_i[str(labels[i])] for i in train_idx])
    yva = np.array([cls_to_i[str(labels[i])] for i in val_idx])
    yte = np.array([cls_to_i[str(labels[i])] for i in test_idx])

    # Standardize each modality on train (fit once, reuse for every arm).
    scalers, raw_std = {}, {}
    for m in MODALITIES:
        sc = StandardScaler().fit(raw[m][train_idx])
        scalers[m] = sc
        raw_std[m] = sc.transform(raw[m]).astype(np.float64)

    def concat(rep: dict[str, np.ndarray], rows) -> np.ndarray:
        return np.concatenate([rep[m][rows] for m in MODALITIES], axis=1)

    Xraw_tr, Xraw_va, Xraw_te = concat(raw_std, train_idx), concat(raw_std, val_idx), concat(raw_std, test_idx)
    Xprj_tr = concat(projected, train_idx)
    Xprj_va = concat(projected, val_idx)
    Xprj_te = concat(projected, test_idx)

    arms: dict[str, Arm] = {}

    # ---- linear probes (deterministic) --------------------------------- #
    for rep_name, Xtr, Xva, Xte in (("raw", Xraw_tr, Xraw_va, Xraw_te), ("proj", Xprj_tr, Xprj_va, Xprj_te)):
        clf = LogisticRegression(max_iter=3000, C=1.0).fit(Xtr, ytr)
        pred = clf.predict(Xte)
        probs = clf.predict_proba(Xte)
        arms[f"{rep_name}_lr_fused"] = Arm(
            f"{rep_name}_lr_fused", "encoder" if rep_name == "raw" else "projection",
            _acc(pred, yte), probs=probs, pred=pred,
            ece=_ece(probs, yte), nll=_nll(probs, yte),
        )

    # ---- per-modality linear probes (projection retention by modality) -- #
    per_modality = {}
    for m in MODALITIES:
        for rep_name, rep in (("raw", raw_std), ("proj", projected)):
            clf = LogisticRegression(max_iter=3000, C=1.0).fit(rep[m][train_idx], ytr)
            per_modality[f"{m}_{rep_name}"] = _acc(clf.predict(rep[m][test_idx]), yte)

    # ---- nonlinear probes / learned bottlenecks ------------------------ #
    def run_mlp(name, family, Xtr, Xva, Xte, *, bottleneck=False, dim=256):
        accs, prob_list, model = [], [], None
        for seed in SEEDS:
            set_seed(seed)
            if bottleneck:
                model = Bottleneck(Xtr.shape[1], len(classes), dim)
            else:
                model = MLPClassifier(Xtr.shape[1], len(classes), hidden=256)
            model = _fit_torch(model, Xtr, ytr, Xva, yva)
            probs = _probs(model, Xte)
            accs.append(_acc(probs.argmax(1), yte))
            prob_list.append(probs)
        ens = np.mean(prob_list, axis=0)
        pred = ens.argmax(1)
        arms[name] = Arm(name, family, _acc(pred, yte), float(np.std(accs)),
                         ece=_ece(ens, yte), nll=_nll(ens, yte), pred=pred, probs=ens)
        return model

    for rep_name, Xtr, Xva, Xte in (("raw", Xraw_tr, Xraw_va, Xraw_te), ("proj", Xprj_tr, Xprj_va, Xprj_te)):
        run_mlp(f"{rep_name}_mlp_fused", "encoder" if rep_name == "raw" else "projection",
                Xtr, Xva, Xte)

    # ---- PSS prototype fusion on the frozen projection ----------------- #
    bank = build_bank_labels(projected, train_idx, labels, classes)
    weights = DEFAULT_WEIGHTS
    val_scores = fused_scores(projected, bank, val_idx, weights)
    temperature = _fit_temperature(val_scores, yva)
    test_scores = fused_scores(projected, bank, test_idx, weights)
    pss_pred = test_scores.argmax(1)
    pss_probs = _softmax(test_scores, temperature)
    arms["pss_proto_frozen"] = Arm(
        "pss_proto_frozen", "reasoning", _acc(pss_pred, yte),
        ece=_ece(pss_probs, yte), nll=_nll(pss_probs, yte), pred=pss_pred, probs=pss_probs,
    )

    # ---- learned fusion heads (PSS readout upgrade) -------------------- #
    proj_dims = [projected[m].shape[1] for m in MODALITIES]

    def train_reasoning_head(name, factory):
        accs, prob_list = [], []
        for seed in SEEDS:
            set_seed(seed)
            model = _fit_torch(factory(), Xprj_tr, ytr, Xprj_va, yva)
            probs = _probs(model, Xprj_te)
            accs.append(_acc(probs.argmax(1), yte))
            prob_list.append(probs)
        ens = np.mean(prob_list, axis=0)
        pred = ens.argmax(1)
        arms[name] = Arm(name, "reasoning", _acc(pred, yte), float(np.std(accs)),
                         ece=_ece(ens, yte), nll=_nll(ens, yte), pred=pred, probs=ens)

    train_reasoning_head("learned_fusion_gated", lambda: GatedFusion(proj_dims, len(classes), dropout=0.0))
    train_reasoning_head("learned_fusion_concat", lambda: ConcatFusion(proj_dims, len(classes), dropout=0.0))
    train_reasoning_head("learned_fusion_concat_lin", lambda: ConcatFusion(proj_dims, len(classes), dropout=0.0, relu=False))

    # ---- co-adaptation / capacity ablation ----------------------------- #
    for dim in (64, 128, 256, 512):
        model = run_mlp(f"learned_bottleneck_{dim}", "coadapt", Xraw_tr, Xraw_va, Xraw_te,
                        bottleneck=True, dim=dim)
        # reuse the learned D-dim space with the PSS prototype readout
        Ztr = _features(model, Xraw_tr)   # rows align with train_idx / ytr
        Zva = _features(model, Xraw_va)
        Zte = _features(model, Xraw_te)
        protos = [l2(Ztr[ytr == ci].mean(0)) for ci in range(len(classes))]
        proto_mat = np.stack(protos)      # (C, D)
        scores_va = Zva @ proto_mat.T
        scores_te = Zte @ proto_mat.T
        T = _fit_temperature(scores_va, yva)
        pred = scores_te.argmax(1)
        probs = _softmax(scores_te, T)
        arms[f"learned_bottleneck_{dim}_proto"] = Arm(
            f"learned_bottleneck_{dim}_proto", "coadapt", _acc(pred, yte),
            ece=_ece(probs, yte), nll=_nll(probs, yte), pred=pred, probs=probs,
        )

    # ---- open-set AUROC ------------------------------------------------- #
    def auroc_seen_vs_unseen(score_seen: np.ndarray, score_unseen: np.ndarray) -> float:
        y = np.concatenate([np.zeros(len(score_seen)), np.ones(len(score_unseen))])
        s = np.concatenate([-score_seen, -score_unseen])  # lower similarity = more novel
        return float(roc_auc_score(y, s))

    openset = {}
    if regime == "category":
        seen_scores = test_scores.max(1)
        held_scores = fused_scores(projected, bank, held_idx, DEFAULT_WEIGHTS).max(1)
        openset["pss_proto_frozen"] = auroc_seen_vs_unseen(seen_scores, held_scores)
        for rep_name in ("raw", "proj"):
            if rep_name == "raw":
                Xtr, Xva, Xte_local = Xraw_tr, Xraw_va, Xraw_te
                Xheld = concat(raw_std, held_idx)
            else:
                Xtr, Xva, Xte_local = Xprj_tr, Xprj_va, Xprj_te
                Xheld = concat(projected, held_idx)
            clf = LogisticRegression(max_iter=3000, C=1.0).fit(Xtr, ytr)
            openset[f"{rep_name}_lr_fused"] = auroc_seen_vs_unseen(
                clf.predict_proba(Xte_local).max(1), clf.predict_proba(Xheld).max(1)
            )
        # Learned heads as OOD scorers: max softmax prob (what you get if you
        # replace the prototype bank with a discriminative head).
        held_prj = concat(projected, held_idx)
        set_seed(0)
        mlp_head = _fit_torch(
            MLPClassifier(Xprj_tr.shape[1], len(classes), 256), Xprj_tr, ytr, Xprj_va, yva
        )
        openset["proj_mlp_fused"] = auroc_seen_vs_unseen(
            _probs(mlp_head, Xprj_te).max(1), _probs(mlp_head, held_prj).max(1)
        )
        set_seed(0)
        gated = _fit_torch(
            GatedFusion(proj_dims, len(classes), dropout=0.0), Xprj_tr, ytr, Xprj_va, yva
        )
        openset["learned_fusion_gated"] = auroc_seen_vs_unseen(
            _probs(gated, Xprj_te).max(1), _probs(gated, held_prj).max(1)
        )
        set_seed(0)
        cat = _fit_torch(
            ConcatFusion(proj_dims, len(classes), dropout=0.0), Xprj_tr, ytr, Xprj_va, yva
        )
        openset["learned_fusion_concat"] = auroc_seen_vs_unseen(
            _probs(cat, Xprj_te).max(1), _probs(cat, held_prj).max(1)
        )

    # ---- decomposition -------------------------------------------------- #
    def acc(name):
        return arms[name].acc

    raw_ceiling = max(acc("raw_lr_fused"), acc("raw_mlp_fused"))
    proj_best = max(acc("proj_lr_fused"), acc("proj_mlp_fused"))
    pss = acc("pss_proto_frozen")
    fused_learned = acc("learned_fusion_gated")
    fused_concat = acc("learned_fusion_concat")
    coadapt_best = max(v.acc for k, v in arms.items() if k.startswith("learned_bottleneck_") and k.endswith("_proto"))
    decomposition = {
        "encoder_ceiling_raw": raw_ceiling,
        "projection_best": proj_best,
        "pss_reasoning": pss,
        "learned_fusion": fused_learned,
        "learned_fusion_concat": fused_concat,
        "projection_loss": raw_ceiling - proj_best,
        "reasoning_gain_vs_proj_mlp": pss - acc("proj_mlp_fused"),
        "reasoning_gain_vs_proj_lr": pss - acc("proj_lr_fused"),
        "reasoning_gain_learned_vs_proj_mlp": fused_learned - acc("proj_mlp_fused"),
        "reasoning_gain_learned_vs_proto": fused_learned - pss,
        "reasoning_gain_concat_vs_proj_mlp": fused_concat - acc("proj_mlp_fused"),
        "reasoning_gain_concat_vs_proto": fused_concat - pss,
        "coadapt_best": coadapt_best,
        "coadapt_gap_vs_pss": coadapt_best - pss,
    }

    deltas = {}
    for a, b in (
        ("raw_mlp_fused", "proj_mlp_fused"),
        ("pss_proto_frozen", "proj_mlp_fused"),
        ("pss_proto_frozen", "proj_lr_fused"),
        ("learned_fusion_gated", "proj_mlp_fused"),
        ("learned_fusion_gated", "pss_proto_frozen"),
        ("learned_fusion_concat", "proj_mlp_fused"),
        ("learned_fusion_concat", "pss_proto_frozen"),
        ("learned_fusion_concat_lin", "proj_mlp_fused"),
        ("learned_fusion_concat_lin", "learned_fusion_concat"),
    ):
        pa, pb = arms[a], arms[b]
        if pa.pred is not None and pb.pred is not None:
            mean, lo, hi = _bootstrap_delta(pa.pred, pb.pred, yte)
            deltas[f"{a}_minus_{b}"] = {"mean": mean, "ci95": [lo, hi]}

    payload = {
        "data": str(args.data),
        "experts": str(args.experts),
        "projections": str(args.projections),
        "device": device,
        "regime": regime,
        "counts": {
            "train": len(train_idx), "val": len(val_idx),
            "test": len(test_idx), "test_category": len(held_idx),
        },
        "classes": classes,
        "pss_weights": DEFAULT_WEIGHTS,
        "pss_temperature": temperature,
        "arms": {
            k: {
                "family": v.family, "acc": v.acc, "acc_std": v.acc_std,
                "ece": v.ece, "nll": v.nll,
            }
            for k, v in arms.items()
        },
        "open_set_auroc": openset,
        "per_modality": per_modality,
        "fewshot": (
            run_fewshot(raw, projected, labels, splits, classes, cls_to_i, val_idx, test_idx)
            if regime == "category" else {}
        ),
        "decomposition": decomposition,
        "bootstrap": deltas,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "attribution_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "ATTRIBUTION_REPORT.md").write_text(render_report(payload), encoding="utf-8")
    return payload


def _fmt(v, d=3):
    if v is None:
        return "n/a"
    try:
        return f"{float(v):.{d}f}"
    except (TypeError, ValueError):
        return str(v)


def render_report(p: dict) -> str:
    dec = p["decomposition"]
    arms = p["arms"]
    lines = [
        "# 归因消融报告：公共语义空间的\"信息密度\"到底差在哪一段",
        "",
        f"- data: `{p['data']}`",
        f"- experts: `{p['experts']}`  ·  projections: `{p['projections']}`",
        f"- device: `{p['device']}`  ·  regime: `{p.get('regime','category')}`  ·  classes: {len(p['classes'])}",
        f"- counts: {p['counts']}",
        f"- PSS 融合权重: `{p['pss_weights']}`  ·  温度 T={_fmt(p['pss_temperature'],2)}",
        "",
        "## 1. 主表（closed-set test accuracy）",
        "",
        "| arm | family | acc | ±std | ECE | NLL |",
        "|---|---|---:|---:|---:|---:|",
    ]
    order = [
        "raw_lr_fused", "raw_mlp_fused",
        "proj_lr_fused", "proj_mlp_fused",
        "pss_proto_frozen", "learned_fusion_gated", "learned_fusion_concat", "learned_fusion_concat_lin",
        "learned_bottleneck_64_proto", "learned_bottleneck_128_proto",
        "learned_bottleneck_256_proto", "learned_bottleneck_512_proto",
    ]
    for name in order:
        if name not in arms:
            continue
        a = arms[name]
        lines.append(
            f"| `{name}` | {a['family']} | {_fmt(a['acc'])} | {_fmt(a['acc_std'])} "
            f"| {_fmt(a['ece'])} | {_fmt(a['nll'])} |"
        )
    pm = p.get("per_modality", {})
    if pm:
        lines += [
            "", "## 1b. 分模态线性探针（raw vs projected）", "",
            "| modality | raw | projected | retention |", "|---|---:|---:|---:|",
        ]
        for m in MODALITIES:
            r, q = pm.get(f"{m}_raw"), pm.get(f"{m}_proj")
            ret = (q / r) if (r and r > 1e-9) else None
            lines.append(f"| {m} | {_fmt(r)} | {_fmt(q)} | {_fmt(ret)} |")

    lines += [
        "",
        "## 2. 分解",
        "",
        "| 量 | 值 | 含义 |",
        "|---|---:|---|",
        f"| encoder ceiling (raw best) | {_fmt(dec['encoder_ceiling_raw'])} | 冻结编码器上强读出能到的上界 |",
        f"| projection best (proj readout) | {_fmt(dec['projection_best'])} | 投影后同族读出的精度 |",
        f"| **projection loss** | **{_fmt(dec['projection_loss'])}** | raw − projected，投影/维度丢掉的精度 |",
        f"| PSS reasoning | {_fmt(dec['pss_reasoning'])} | 原型融合读出 |",
        f"| **reasoning gain vs proj_mlp** | **{_fmt(dec['reasoning_gain_vs_proj_mlp'])}** | PSS − 同表示的通用非线性读出 |",
        f"| reasoning gain vs proj_lr | {_fmt(dec['reasoning_gain_vs_proj_lr'])} | PSS − 同表示的线性读出 |",
        f"| learned fusion (gated) | {_fmt(dec.get('learned_fusion'))} | 可学门控融合读出 |",
        f"| **reasoning gain (learned) vs proj_mlp** | **{_fmt(dec.get('reasoning_gain_learned_vs_proj_mlp'))}** | 可学融合 − 通用非线性读出 |",
        f"| reasoning gain (learned) vs proto | {_fmt(dec.get('reasoning_gain_learned_vs_proto'))} | 可学融合 − 原型融合 |",
        f"| learned fusion (concat) | {_fmt(dec.get('learned_fusion_concat'))} | 分模态嵌入+拼接+头 |",
        f"| **reasoning gain (concat) vs proj_mlp** | **{_fmt(dec.get('reasoning_gain_concat_vs_proj_mlp'))}** | 可学拼接融合 − 通用非线性读出 |",
        f"| reasoning gain (concat) vs proto | {_fmt(dec.get('reasoning_gain_concat_vs_proto'))} | 可学拼接融合 − 原型融合 |",
        f"| co-adapt best (learned proto) | {_fmt(dec['coadapt_best'])} | 监督学习的投影 + 原型 |",
        f"| **co-adapt gap vs PSS** | **{_fmt(dec['coadapt_gap_vs_pss'])}** | 联合训练带来的增益（协同适应损失） |",
        "",
        "## 3. 关键 delta 的 bootstrap 95% CI",
        "",
        "| delta | mean | 95% CI |",
        "|---|---:|---|",
    ]
    for k, v in p["bootstrap"].items():
        lines.append(f"| `{k}` | {_fmt(v['mean'])} | [{_fmt(v['ci95'][0])}, {_fmt(v['ci95'][1])}] |")
    if p.get("open_set_auroc"):
        lines += [
            "",
            "## 4. Open-set AUROC（seen test vs held-out categories）",
            "",
            "| arm | AUROC |",
            "|---|---:|",
        ]
        for k, v in p["open_set_auroc"].items():
            lines.append(f"| `{k}` | {_fmt(v)} |")

    fs = p.get("fewshot", {})
    if fs:
        ks = sorted(fs, key=int, reverse=True)
        full_map = {
            "raw_lr": arms["raw_lr_fused"]["acc"], "raw_mlp": arms["raw_mlp_fused"]["acc"],
            "proj_lr": arms["proj_lr_fused"]["acc"], "proj_mlp": arms["proj_mlp_fused"]["acc"],
            "pss_proto": arms["pss_proto_frozen"]["acc"],
        }
        header = "| arm | full | " + " | ".join(f"K={k}" for k in ks) + " |"
        sep = "|---|---:|" + "---:|" * len(ks)
        lines += ["", "## 5. Few-shot 归因（每类 K 个训练样本，同一 test）", "", header, sep]
        for arm_name in ("raw_lr", "raw_mlp", "proj_lr", "proj_mlp", "pss_proto"):
            cells = [f"| `{arm_name}`", f"{_fmt(full_map[arm_name])}"]
            for k in ks:
                cells.append(f"{_fmt(fs[k].get(arm_name))}")
            lines.append(" | ".join(cells) + " |")

    judge_num = 4 + int(bool(p.get("open_set_auroc"))) + int(bool(p.get("fewshot")))
    lines += [
        "",
        f"## {judge_num}. 判读",
        "",
        "- **reasoning gain vs proj_mlp > 0**：公共空间不是瓶颈，PSS 的显式读出从同一表示里\n"
        "  多拿了信息 → 天花板由 reasoning 容量决定。",
        "- **projection loss > reasoning gain**：投影/维度是主要损失来源 → 该扩维度 / 上 residual / 双层训练。",
        "- **projection loss < 0**：投影/对齐**提升**了可读信息 → 公共空间不是瓶颈，反而是增益；\n"
        "  此时该查 reasoning 的读出容量，而不是换编码器。",
        "- **co-adapt gap ≈ 0**：冻结投影没有明显吃亏，协同适应不是主因。",
        "- **co-adapt gap ≫ 0**：监督联合训练显著更高 → 量化了协同适应损失，值得做 bilevel。",
        "- **few-shot 表**：若 `proj_*` 与 `raw_*` 同水平，投影不是瓶颈；若 `pss_proto` 在 K 小时\n"
        "  高于 `proj_lr`/`proj_mlp`，说明结构化原型读出在小数据下更强——这是 PSS 的可辩护优势。",
        "- 注意：closed-set 接近饱和（≈0.98）时 delta 的方差大，必须看 bootstrap CI 是否跨 0。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/category_pretrain")
    ap.add_argument("--experts", default="checkpoints/category_pretrain_experts")
    ap.add_argument("--projections", default="checkpoints/phase45_projection")
    ap.add_argument("--out", default="eval/results/attribution_ablation")
    ap.add_argument("--regime", choices=["category", "view-instance"], default="category",
                    help="category: closed-set category labels (default); "
                         "view-instance: instance identity with train/test on disjoint views")
    args = ap.parse_args()
    payload = run(args)
    print("Attribution ablation:")
    for name, a in payload["arms"].items():
        print(f"  {a['family']:>10}  {name:<32} acc={a['acc']:.3f}  ece={_fmt(a['ece'])}")
    print("Decomposition:", json.dumps(payload["decomposition"], ensure_ascii=False, indent=2))
    print(f"  report: {Path(args.out) / 'ATTRIBUTION_REPORT.md'}")


if __name__ == "__main__":
    main()
