"""Phase 5 prototype package operations: incremental add and rollback.

The versioned, hash-checked package store already exists in
``pss.learning.l2_package``. This module adapts the Phase 5
``PrototypeBank`` to that store:

1. initialize a base package from the current bank;
2. admit a new category from projected samples, using the Phase 5
   ``pre_review`` gate;
3. append the category's prototypes, install and activate a new version;
4. evaluate a disjoint holdout half;
5. roll back to the parent version and verify integrity.

No model parameter is trained or updated: features are projected with
``torch.no_grad`` and the package store is numpy-only. Expert/projection/bank
hashes are compared before and after as evidence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from pss.contracts.schema import SPACE_VERSION
from pss.encoders.tactile import load_tactile_expert
from pss.encoders.vision import load_vision_expert
from pss.experts.adapter import admit_encoder
from pss.learning.l2_package import PackageManager, PrototypePackage
from pss.projection.features import load_paired_features
from pss.projection.heads import load_projection
from pss.prototypes.admission import calibrated_thresholds, pre_review
from pss.prototypes.bank import PrototypeBank, l2
from pss.prototypes.recognize import recognize

MODALITIES = ("vision", "tactile", "geometry")


def _hash_files(paths: list[Path]) -> dict[str, str]:
    return {
        str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in paths
        if Path(path).exists()
    }


def _project(head, X: np.ndarray, device: str) -> np.ndarray:
    head.eval()
    with torch.no_grad():
        xb = torch.as_tensor(X, dtype=torch.float32, device=device)
        return head(xb).cpu().numpy()


def load_manager(root: str | Path) -> PackageManager:
    """Rebuild an in-memory manager from the on-disk store."""
    root = Path(root)
    manager = PackageManager(root)
    packages_dir = root / "packages"
    if packages_dir.exists():
        for path in sorted(packages_dir.iterdir()):
            if path.is_dir() and (path / "manifest.json").exists():
                manager.packages[path.name] = path
    active_file = root / "ACTIVE"
    if active_file.exists():
        active = active_file.read_text(encoding="utf-8").strip()
        if active in manager.packages:
            manager.active = active
            active_pkg = PrototypePackage.load(manager.packages[active])
            if active_pkg.parent and active_pkg.parent in manager.packages:
                manager.history = [active_pkg.parent]
    return manager


def init_package(
    bank: PrototypeBank, root: str | Path, version: str = "v1"
) -> tuple[PrototypePackage, bool]:
    manager = load_manager(root)
    if version in manager.packages:
        manager.activate(version)
        return manager.active_package(), False
    package = PrototypePackage(
        version=version,
        categories=list(bank.categories),
        protos={m: np.asarray(v, dtype=np.float32) for m, v in bank.protos.items()},
        stats=dict(bank.stats),
        parent=None,
        changelog=["base Phase 5 prototype bank"],
    )
    manager.install(package)
    manager.activate(version)
    return package, True


def _load_weights(prototypes: Path) -> dict[str, float]:
    metrics_path = Path(prototypes) / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        return dict(metrics.get("weights", {m: 1.0 for m in MODALITIES}))
    return {m: 1.0 for m in MODALITIES}


def add_category(
    data_dir: str | Path,
    experts: str | Path,
    projections: str | Path,
    prototypes: str | Path,
    root: str | Path,
    category: str,
    version: str = "v2",
    force: bool = False,
    split: str = "test_category",
) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path(data_dir)
    experts = Path(experts)
    projections = Path(projections)
    prototypes = Path(prototypes)
    manager = load_manager(root)
    if manager.active is None:
        raise RuntimeError("no active package; run init first")
    active = manager.active_package()
    if active is None:
        raise RuntimeError("active package could not be loaded")
    bank = PrototypeBank(active.categories, active.protos, active.stats, active.version)
    if category in active.categories:
        raise ValueError(f"category {category!r} already exists")

    vision = admit_encoder(
        load_vision_expert(experts / "vision.pt", device=device),
        modality="vision",
        version=f"smallcnn@{SPACE_VERSION}",
    ).encoder
    tactile = admit_encoder(
        load_tactile_expert(experts / "tactile.pt", device=device),
        modality="tactile",
        version=f"mlp@{SPACE_VERSION}",
    ).encoder
    feats = load_paired_features(data_dir, vision, tactile)
    raw = {"vision": feats.vision, "tactile": feats.tactile, "geometry": feats.geometry}
    heads = {
        m: load_projection(projections / f"{m}.pt", device=device) for m in MODALITIES
    }
    projected = {m: _project(heads[m], raw[m], device) for m in MODALITIES}

    splits = np.array(feats.splits)
    categories = np.array(feats.categories)
    candidate_idx = [
        int(i)
        for i in np.where(splits == split)[0]
        if categories[i] == category
    ]
    if len(candidate_idx) < 4:
        raise ValueError(f"not enough samples for category {category!r}")
    half = max(1, len(candidate_idx) // 2)
    review_idx = candidate_idx[:half]
    holdout_idx = candidate_idx[half:]

    samples = {m: projected[m][review_idx] for m in MODALITIES}
    calibrated = calibrated_thresholds(bank)["per_modality"]
    thresholds = {m: calibrated[m]["distance_threshold"] for m in MODALITIES}
    review = pre_review(samples, bank, min_inter_distance=thresholds)
    if review.decision == "reject" and not force:
        return {
            "added": False,
            "category": category,
            "decision": review.decision,
            "reasons": review.reasons,
            "admission": review.metrics,
            "holdout_acc": None,
            "version": active.version,
        }

    new_protos = {
        m: l2(np.asarray(samples[m], dtype=np.float64).mean(0)) for m in MODALITIES
    }
    new_stats = {
        m: {
            "sample_count": int(len(review_idx)),
            "variance": float(np.mean(1.0 - np.asarray(samples[m]) @ new_protos[m])),
        }
        for m in MODALITIES
    }
    package = active.append_category(
        category,
        new_protos,
        new_stats,
        version=version,
        changelog=f"add {category} from {len(review_idx)} samples; admission={review.decision}",
    )
    manager.install(package)
    manager.activate(version)
    new_bank = PrototypeBank(
        package.categories, package.protos, package.stats, package.version
    )
    weights = _load_weights(prototypes)
    correct = 0
    for i in holdout_idx:
        query = {m: projected[m][i] for m in MODALITIES}
        result = recognize(
            query, new_bank, weights=weights, use_tree=False
        )
        correct += int(
            bool(result.ranked) and result.ranked[0][0] == category
        )
    holdout_acc = correct / len(holdout_idx) if holdout_idx else float("nan")
    return {
        "added": True,
        "category": category,
        "decision": review.decision,
        "reasons": review.reasons,
        "admission": review.metrics,
        "holdout_acc": float(holdout_acc),
        "holdout_n": len(holdout_idx),
        "version": version,
        "parent": active.version,
        "categories": list(package.categories),
    }


def rollback(root: str | Path, to_version: str | None = None) -> dict:
    manager = load_manager(root)
    if manager.active is None:
        raise RuntimeError("no active package")
    before = manager.active
    target = to_version or manager.rollback()
    if to_version:
        manager.activate(to_version)
        target = to_version
    package = manager.active_package()
    if package is None:
        raise RuntimeError("rolled-back package could not be loaded")
    return {
        "from": before,
        "to": target,
        "categories": list(package.categories),
        "integrity": package.verify(manager.packages[target]),
    }


def verify(root: str | Path) -> dict:
    manager = load_manager(root)
    out = {"active": manager.active, "packages": {}}
    for version, path in manager.packages.items():
        try:
            package = PrototypePackage.load(path)
            out["packages"][version] = {
                "ok": True,
                "parent": package.parent,
                "categories": package.categories,
            }
        except Exception as exc:  # integrity failure is a finding
            out["packages"][version] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    return out


def run_package_demo(
    data_dir: str | Path,
    experts: str | Path,
    projections: str | Path,
    prototypes: str | Path,
    root: str | Path,
    bank: str | Path,
    category: str = "sheet",
    base_version: str = "v1",
    new_version: str = "v2",
    split: str = "test_category",
) -> dict:
    root = Path(root)
    bank_obj = PrototypeBank.load(Path(bank) / "prototypes.npz")
    frozen_paths = [
        Path(experts) / "vision.pt",
        Path(experts) / "tactile.pt",
        Path(projections) / "vision.pt",
        Path(projections) / "tactile.pt",
        Path(projections) / "geometry.pt",
        Path(bank) / "prototypes.npz",
        Path(bank) / "tree.json",
    ]
    hashes_before = _hash_files(frozen_paths)

    base, created = init_package(bank_obj, root, version=base_version)
    add = add_category(
        data_dir,
        experts,
        projections,
        prototypes,
        root,
        category,
        version=new_version,
        split=split,
    )
    after_add = verify(root)
    rolled = rollback(root, to_version=base_version)
    after_rollback = verify(root)

    hashes_after = _hash_files(frozen_paths)
    no_backprop = hashes_before == hashes_after
    report = {
        "root": str(root),
        "categories": {
            "base_version": base_version,
            "new_version": new_version,
            "added_category": category,
        },
        "init": {"created": created, "version": base.version, "categories": base.categories},
        "add": add,
        "verify_after_add": after_add,
        "rollback": rolled,
        "verify_after_rollback": after_rollback,
        "frozen_hashes_before": hashes_before,
        "frozen_hashes_after": hashes_after,
        "no_backprop": bool(no_backprop),
        "ok": bool(
            add.get("added")
            and rolled.get("to") == base_version
            and rolled.get("integrity")
            and after_rollback.get("active") == base_version
            and no_backprop
        ),
    }
    out = Path(root) / "PACKAGE_REPORT.md"
    out.write_text(_render_report(report), encoding="utf-8")
    (Path(root) / "package_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def _render_report(report: dict) -> str:
    add = report["add"]
    rolled = report["rollback"]
    lines = [
        "# Phase 5 prototype package report",
        "",
        f"- package root: `{report['root']}`",
        f"- base version: `{report['categories']['base_version']}`",
        f"- new version: `{report['categories']['new_version']}`",
        f"- added category: `{report['categories']['added_category']}`",
        "",
        "## Add",
        "",
        f"- added: **{add.get('added')}**",
        f"- admission: {add.get('decision')} ({', '.join(add.get('reasons', [])) or '—'})",
        f"- holdout accuracy: {add.get('holdout_acc')}",
        f"- package categories: {len(add.get('categories', []))}",
        "",
        "## Rollback",
        "",
        f"- from `{rolled.get('from')}` to `{rolled.get('to')}`",
        f"- integrity: **{rolled.get('integrity')}**",
        f"- active after rollback: `{report['verify_after_rollback'].get('active')}`",
        "",
        "## No-backprop evidence",
        "",
        f"- frozen expert/projection/bank hashes unchanged: **{report['no_backprop']}**",
        "",
        "## Verdict",
        "",
        f"- package add/rollback: **{'PASS' if report['ok'] else 'FAIL'}**",
        "",
    ]
    return "\n".join(lines)
