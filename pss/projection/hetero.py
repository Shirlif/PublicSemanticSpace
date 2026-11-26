"""Width-agnostic projection, calibration and admission for heterogeneous experts (F4).

A heterogeneous expert arrives with its own native width.  The property suite
still needs a 128-d public space and the same admission contract, so the
adapter does three things per expert:

1. **Independent projection head.**  A :class:`~pss.projection.heads.ProjectionHead`
   with ``in_dim`` set to the expert width is trained on the expert's own
   *head-fit* split with intra-modal supervised contrastive loss.
2. **Independent calibration set.**  A disjoint set of views (or entries) is
   reserved for few-shot calibration and for the expert's own prototype bank.
3. **Two curves and one contract.**  The calibration curve maps ``k`` examples
   per class into the frozen public space (the E11 protocol); a few-shot
   prototype curve builds prototypes directly in the expert's projected space.
   The admission contract is evaluated in that projected space with the
   data-driven thresholds from :mod:`pss.prototypes.admission`.

Nothing here changes runtime code or the public API of ``pss.property``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from pss.experts import CategoryCodebook
from pss.metrics.property import zero_shot_compatibility
from pss.projection.heads import PUBLIC_DIM, ProjectionHead, save_projection
from pss.projection.losses import supcon_cross
from pss.prototypes.admission import calibrated_thresholds, pre_review
from pss.prototypes.bank import PrototypeBank, l2

CONTRACT_CASES = ("near_dup", "novel_class", "insufficient")
EXPECTED_CONTRACT = {
    "near_dup": "reject",
    "novel_class": "accept",
    "insufficient": "review",
}


@dataclass
class FamilySplit:
    """Index bookkeeping for one expert's independent head/calibration sets."""

    head_idx: list[int]
    calib_by_class: dict[str, list[int]]
    bank_idx: dict[str, list[int]]
    test_idx: list[int]
    holdout_by_class: dict[str, list[int]]
    present: list[str]
    head_views: list[int] | None = None
    calib_views: list[int] | None = None
    max_per_class: int | None = None


def ridge_map(X: np.ndarray, Y: np.ndarray, ridge: float = 1e-2) -> np.ndarray:
    """Solve ``X @ W ~= Y`` with a small ridge term."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    A = X.T @ X + float(ridge) * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ Y)


def split_family_indices(
    feats,
    classes,
    *,
    max_per_class: int | None = None,
    head_fraction: float = 0.5,
) -> FamilySplit:
    """Split one expert's features into head-fit, calibration, test and holdout.

    With multiple views, the head sees the first half of the view ids and the
    calibration set the second half; both are disjoint.  With a single view the
    per-class train entries are split by order instead.
    """
    cats = np.asarray(feats.categories, dtype=object)
    splits = np.asarray(feats.splits, dtype=object)
    views = np.asarray(feats.views, dtype=int)
    classes = list(classes)
    present = [c for c in classes if c in set(cats.tolist())]

    unique_views = sorted({int(v) for v in views.tolist()})
    if len(unique_views) >= 2:
        cut = max(1, int(round(len(unique_views) * float(head_fraction))))
        cut = min(cut, len(unique_views) - 1)
        head_views = unique_views[:cut]
        calib_views = unique_views[cut:]
        head_set = set(head_views)
    else:
        head_views = calib_views = None
        head_set = None

    head_idx: list[int] = []
    calib_by_class: dict[str, list[int]] = {c: [] for c in present}
    bank_idx: dict[str, list[int]] = {c: [] for c in present}
    holdout_by_class: dict[str, list[int]] = {c: [] for c in present}
    test_idx: list[int] = []
    test_counts: dict[str, int] = {}
    pending_head: dict[str, list[int]] = {c: [] for c in present}

    for i in range(len(cats)):
        c = str(cats[i])
        if c not in calib_by_class:
            continue
        split = str(splits[i])
        if split == "train":
            bank_idx[c].append(i)
            if head_set is None:
                pending_head[c].append(i)
            elif int(views[i]) in head_set:
                head_idx.append(i)
            else:
                calib_by_class[c].append(i)
        elif split in ("val", "test"):
            holdout_by_class[c].append(i)
            if split == "test":
                if max_per_class is not None and test_counts.get(c, 0) >= max_per_class:
                    continue
                test_counts[c] = test_counts.get(c, 0) + 1
                test_idx.append(i)

    if head_set is None:
        for c in present:
            entries = pending_head[c]
            cut = max(1, int(round(len(entries) * float(head_fraction))))
            cut = min(cut, max(len(entries) - 1, 1))
            head_idx.extend(entries[:cut])
            calib_by_class[c].extend(entries[cut:])

    return FamilySplit(
        head_idx=sorted(head_idx),
        calib_by_class=calib_by_class,
        bank_idx=bank_idx,
        test_idx=test_idx,
        holdout_by_class=holdout_by_class,
        present=present,
        head_views=head_views,
        calib_views=calib_views,
        max_per_class=max_per_class,
    )


def train_expert_head(
    X: np.ndarray,
    labels: np.ndarray,
    head_idx,
    in_dim: int,
    *,
    out_dim: int = PUBLIC_DIM,
    hidden: int = 256,
    epochs: int = 80,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    temp: float = 0.07,
    seed: int = 0,
    device: str = "cpu",
) -> ProjectionHead:
    """Train a fresh projection head on one expert's head-fit split."""
    head_idx = np.asarray(list(head_idx), dtype=np.int64)
    if len(head_idx) < 2:
        raise ValueError("need at least two head-fit samples")
    labels = np.asarray(labels, dtype=np.int64)
    if len(set(labels[head_idx].tolist())) < 2:
        raise ValueError("head-fit split needs at least two classes")

    torch.manual_seed(int(seed))
    head = ProjectionHead(int(in_dim), int(out_dim), hidden=int(hidden)).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    Xt = torch.as_tensor(np.asarray(X, dtype=np.float32))
    yt = torch.as_tensor(labels, dtype=torch.long)
    idx_t = torch.as_tensor(head_idx, dtype=torch.long)

    head.train()
    for _epoch in range(int(epochs)):
        perm = torch.randperm(len(head_idx))
        for start in range(0, len(head_idx), int(batch_size)):
            ii = idx_t[perm[start : start + int(batch_size)]]
            if len(ii) < 2:
                continue
            z = head(Xt[ii].to(device))
            loss = supcon_cross(z, z, yt[ii].to(device), temp=float(temp))
            opt.zero_grad()
            loss.backward()
            opt.step()
    head.eval()
    return head


def project_features(
    head: ProjectionHead,
    X: np.ndarray,
    device: str = "cpu",
    batch_size: int = 256,
) -> np.ndarray:
    """Project native features through a head, returning ``(N, out_dim)``."""
    head.eval()
    outs = []
    with torch.no_grad():
        for start in range(0, len(X), int(batch_size)):
            xb = torch.as_tensor(
                np.asarray(X[start : start + int(batch_size)], dtype=np.float32),
                device=device,
            )
            outs.append(head(xb).cpu().numpy())
    if not outs:
        return np.zeros((0, int(head.out_dim)), dtype=np.float32)
    return np.concatenate(outs, axis=0).astype(np.float32)


def few_shot_ridge_curve(
    Z: np.ndarray,
    calib_by_class: dict[str, list[int]],
    test_idx,
    gt,
    codebook: CategoryCodebook,
    classes,
    ks,
    *,
    ridge: float = 1e-2,
) -> dict[int, float]:
    """E11-style calibration: map ``k`` examples per class into the frozen bank."""
    Z = np.asarray(Z, dtype=np.float64)
    classes = list(classes)
    test_idx = list(test_idx)
    gt = list(gt)
    out: dict[int, float] = {}
    for k in ks:
        k = int(k)
        if k == 0:
            W = np.eye(codebook.dim, dtype=np.float64)
        else:
            rows, targets = [], []
            for c in classes:
                for i in calib_by_class.get(c, [])[:k]:
                    rows.append(Z[i])
                    targets.append(codebook.vector(c))
            if len(rows) < 2:
                out[k] = float("nan")
                continue
            W = ridge_map(np.stack(rows), np.stack(targets), ridge=ridge)
        preds = [codebook.predict(Z[i] @ W) for i in test_idx]
        out[k] = float(zero_shot_compatibility(preds, gt))
    return out


def few_shot_prototype_curve(
    Z: np.ndarray,
    calib_by_class: dict[str, list[int]],
    test_idx,
    gt,
    classes,
    ks,
    *,
    codebook: CategoryCodebook | None = None,
) -> dict[int, float]:
    """Build prototypes from ``k`` calibration examples per class and evaluate.

    ``k=0`` has no calibration examples, so it falls back to the frozen base
    codebook when one is supplied (the zero-shot reference); otherwise it is
    ``nan``.
    """
    Z = np.asarray(Z, dtype=np.float64)
    classes = list(classes)
    test_idx = list(test_idx)
    gt = list(gt)
    out: dict[int, float] = {}
    for k in ks:
        k = int(k)
        if k == 0:
            if codebook is None:
                out[k] = float("nan")
                continue
            preds = [codebook.predict(Z[i]) for i in test_idx]
        else:
            centroids = []
            for c in classes:
                idx = calib_by_class.get(c, [])[:k]
                if not idx:
                    centroids = []
                    break
                centroids.append(l2(Z[idx].mean(0)))
            if len(centroids) != len(classes):
                out[k] = float("nan")
                continue
            cents = np.stack(centroids).astype(np.float64)
            preds = []
            for i in test_idx:
                z = Z[i] / max(float(np.linalg.norm(Z[i])), 1e-12)
                preds.append(classes[int(np.argmax(cents @ z))])
        out[k] = float(zero_shot_compatibility(preds, gt))
    return out


def build_expert_bank(
    Z: np.ndarray,
    bank_idx: dict[str, list[int]],
    classes,
    modality: str,
    *,
    version: str = "hetero_width_v1",
) -> PrototypeBank:
    """Build a per-expert prototype bank in the expert's projected space."""
    Z = np.asarray(Z, dtype=np.float64)
    protos, stats = [], {}
    for c in classes:
        idx = list(bank_idx.get(c, []))
        if not idx:
            raise ValueError(f"no bank samples for category {c!r}")
        rows = Z[idx]
        centroid = l2(rows.mean(0))
        protos.append(centroid)
        stats[c] = {
            modality: {
                "sample_count": int(len(rows)),
                "variance": float(np.mean(1.0 - rows @ centroid)),
            }
        }
    return PrototypeBank(
        categories=list(classes),
        protos={modality: np.stack(protos).astype(np.float32)},
        stats=stats,
        version=version,
    )


def _admission_case(samples, bank, modality, *, min_samples, max_variance, threshold) -> dict:
    result = pre_review(
        {modality: np.asarray(samples, dtype=np.float64)},
        bank,
        min_samples=int(min_samples),
        max_variance=float(max_variance),
        min_inter_distance=float(threshold),
    )
    metrics = result.metrics.get(modality, {})
    return {
        "decision": result.decision,
        "reasons": list(result.reasons),
        "sample_count": int(metrics.get("sample_count", len(samples))),
        "variance": float(metrics.get("variance", float("nan"))),
        "min_inter_distance": float(metrics.get("min_inter_distance", float("nan"))),
    }


def admission_contract(
    Z: np.ndarray,
    bank_idx: dict[str, list[int]],
    holdout_by_class: dict[str, list[int]],
    present,
    modality: str,
    *,
    far_class: str = "plate",
    near_class: str = "block",
    min_samples: int = 20,
    max_variance: float = 0.35,
) -> dict:
    """Run the same admission contract in every expert's projected space.

    Three probes, all from the expert's own held-out samples:

    - ``near_dup``: a class already in the bank (expected ``reject``);
    - ``novel_class``: a class removed from the bank (expected ``accept``);
    - ``insufficient``: too few samples of the novel class (expected ``review``).

    Thresholds come from :func:`pss.prototypes.admission.calibrated_thresholds`,
    i.e. the contract's own data-driven mode, so the same rules apply at every
    width instead of a single hard-coded distance.
    """
    present = list(present)
    if near_class not in present:
        near_class = present[0]
    if far_class not in present:
        far_class = present[-1]
    if far_class == near_class:
        if len(present) < 2:
            raise ValueError("admission contract needs at least two classes")
        far_class = present[1] if present[0] == near_class else present[0]

    near_bank = build_expert_bank(Z, bank_idx, present, modality)
    far_bank = build_expert_bank(
        Z, bank_idx, [c for c in present if c != far_class], modality
    )
    near_threshold = float(
        calibrated_thresholds(near_bank)["per_modality"][modality]["distance_threshold"]
    )
    far_threshold = float(
        calibrated_thresholds(far_bank)["per_modality"][modality]["distance_threshold"]
    )

    Z = np.asarray(Z, dtype=np.float64)
    near = Z[list(holdout_by_class[near_class])]
    far = Z[list(holdout_by_class[far_class])]
    insufficient_n = max(1, min(5, int(min_samples) - 1))

    near_case = _admission_case(
        near, near_bank, modality,
        min_samples=min_samples, max_variance=max_variance, threshold=near_threshold,
    )
    far_case = _admission_case(
        far, far_bank, modality,
        min_samples=min_samples, max_variance=max_variance, threshold=far_threshold,
    )
    ins_case = _admission_case(
        far[:insufficient_n], far_bank, modality,
        min_samples=min_samples, max_variance=max_variance, threshold=far_threshold,
    )
    return {
        "modality": modality,
        "near_class": near_class,
        "far_class": far_class,
        "near_threshold": near_threshold,
        "far_threshold": far_threshold,
        "min_samples": int(min_samples),
        "max_variance": float(max_variance),
        "near_dup": near_case,
        "novel_class": far_case,
        "insufficient": ins_case,
        "consistent_with_expected": all(
            {
                "near_dup": near_case["decision"],
                "novel_class": far_case["decision"],
                "insufficient": ins_case["decision"],
            }[case] == EXPECTED_CONTRACT[case]
            for case in CONTRACT_CASES
        ),
    }


def run_width_family(
    feats,
    spec,
    base_bank: PrototypeBank,
    *,
    ks=(0, 1, 5, 20),
    seed: int = 0,
    device: str = "cpu",
    epochs: int = 80,
    hidden: int = 256,
    ridge: float = 1e-2,
    max_per_class: int | None = 40,
    far_class: str = "plate",
    min_samples: int = 20,
    max_variance: float = 0.35,
    save_head_path: str | Path | None = None,
) -> dict:
    """Run the full width-agnostic protocol for one expert spec."""
    modality = str(spec.modality)
    if modality == "vision":
        X = np.asarray(feats.vision, dtype=np.float32)
    elif modality == "tactile":
        X = np.asarray(feats.tactile, dtype=np.float32)
    else:
        raise ValueError(f"unsupported modality {modality!r}")
    if X.ndim != 2:
        raise ValueError(f"expected 2-D features, got {X.shape}")
    if X.shape[1] != int(spec.in_dim):
        raise ValueError(
            f"{spec.name}: declared in_dim={spec.in_dim}, features have {X.shape[1]}"
        )

    classes = list(base_bank.categories)
    split = split_family_indices(feats, classes, max_per_class=max_per_class)
    if len(split.present) < 2:
        raise ValueError(f"{spec.name}: need at least two shared classes")
    if not split.test_idx:
        raise ValueError(f"{spec.name}: no test samples for the shared classes")

    label_index = {c: i for i, c in enumerate(split.present)}
    labels = np.array(
        [label_index.get(str(c), -1) for c in feats.categories], dtype=np.int64
    )

    head = train_expert_head(
        X,
        labels,
        split.head_idx,
        int(spec.in_dim),
        hidden=hidden,
        epochs=epochs,
        seed=seed,
        device=device,
    )
    Z = project_features(head, X, device=device)
    if save_head_path is not None:
        Path(save_head_path).parent.mkdir(parents=True, exist_ok=True)
        save_projection(save_head_path, head)

    codebook = CategoryCodebook.from_vectors(
        base_bank.categories, base_bank.protos["vision"]
    )
    gt = [str(feats.categories[i]) for i in split.test_idx]
    ridge_curve = few_shot_ridge_curve(
        Z, split.calib_by_class, split.test_idx, gt, codebook, split.present, ks, ridge=ridge
    )
    proto_curve = few_shot_prototype_curve(
        Z,
        split.calib_by_class,
        split.test_idx,
        gt,
        split.present,
        ks,
        codebook=codebook,
    )
    admission = admission_contract(
        Z,
        split.bank_idx,
        split.holdout_by_class,
        split.present,
        modality,
        far_class=far_class,
        min_samples=min_samples,
        max_variance=max_variance,
    )

    return {
        **spec.summary(),
        "head": {
            "in_dim": int(spec.in_dim),
            "out_dim": int(PUBLIC_DIM),
            "hidden": int(hidden),
            "epochs": int(epochs),
            "seed": int(seed),
        },
        "counts": {
            "head_fit": int(len(split.head_idx)),
            "calib_per_class": {
                c: int(len(split.calib_by_class.get(c, []))) for c in split.present
            },
            "test": int(len(split.test_idx)),
            "present_classes": list(split.present),
            "head_views": split.head_views,
            "calib_views": split.calib_views,
        },
        "curves": {
            "ridge_to_frozen": {str(k): float(v) for k, v in ridge_curve.items()},
            "few_shot_prototype": {str(k): float(v) for k, v in proto_curve.items()},
        },
        "admission": admission,
    }


__all__ = [
    "CONTRACT_CASES",
    "EXPECTED_CONTRACT",
    "FamilySplit",
    "admission_contract",
    "build_expert_bank",
    "few_shot_prototype_curve",
    "few_shot_ridge_curve",
    "project_features",
    "ridge_map",
    "run_width_family",
    "split_family_indices",
    "train_expert_head",
]
