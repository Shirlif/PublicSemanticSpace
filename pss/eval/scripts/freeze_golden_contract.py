"""Freeze the current 3-modality PSS behaviour as a regression fixture (W0).

The fixture is deliberately synthetic and tiny: it exercises the two pieces
that the compatibility refactor will touch first — the per-modality prototype
bank and the joint readout — without depending on checkpoints or a GPU.

Run once after any intentional behaviour change:

    python -m pss.eval.scripts.freeze_golden_contract --write

The test ``tests/test_golden_contract.py`` rebuilds the same deterministic
input and compares against the stored arrays.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pss._console import ensure_utf8
from pss.prototypes.bank import build_bank
from pss.prototypes.joint_readout import JointReadout
from pss.prototypes.recognize import recognize

FIXTURE = Path("tests/fixtures/golden_3modality.npz")
# Explicit, not the registry default: the golden contract must keep testing
# exactly these three modalities even after a fourth one is registered.
GOLDEN_MODALITIES = ("vision", "tactile", "geometry")
DEFAULT_WEIGHTS = {"vision": 1.0, "tactile": 0.5, "geometry": 0.95}
DIMS = {"vision": 16, "tactile": 8, "geometry": 4}
CLASSES = ("alpha", "beta", "gamma")
PER_CLASS = 12
TRAIN_PER_CLASS = 8


def _unit(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def make_fixture(seed: int = 0) -> dict:
    """Build deterministic synthetic features shared by the generator and test."""
    rng = np.random.default_rng(seed)
    centers = {
        modality: _unit(rng.normal(size=(len(CLASSES), DIMS[modality])))
        for modality in GOLDEN_MODALITIES
    }
    blocks = {modality: [] for modality in GOLDEN_MODALITIES}
    categories: list[str] = []
    sample_ids: list[str] = []
    instances: list[str] = []
    for class_index, category in enumerate(CLASSES):
        for view in range(PER_CLASS):
            categories.append(category)
            sample_ids.append(f"{category}_v{view:03d}")
            instances.append(f"{category}_i{view:03d}")
            for modality in GOLDEN_MODALITIES:
                noise = rng.normal(scale=0.35, size=DIMS[modality])
                blocks[modality].append(centers[modality][class_index] + noise)
    Z = {modality: _unit(np.stack(rows)) for modality, rows in blocks.items()}
    features = SimpleNamespace(
        categories=categories,
        sample_ids=sample_ids,
        instances=instances,
    )
    train_idx = np.array(
        [
            i
            for i, category in enumerate(categories)
            if (i % PER_CLASS) < TRAIN_PER_CLASS
        ],
        dtype=int,
    )
    test_idx = np.array(
        [
            i
            for i, category in enumerate(categories)
            if (i % PER_CLASS) >= TRAIN_PER_CLASS
        ],
        dtype=int,
    )
    return {
        "Z": Z,
        "features": features,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "classes": list(CLASSES),
    }


def build_payload(seed: int = 0) -> dict:
    data = make_fixture(seed)
    Z, features = data["Z"], data["features"]
    train_idx, test_idx = data["train_idx"], data["test_idx"]
    classes = data["classes"]

    bank = build_bank(features, Z, train_idx, classes, version="golden_v1")
    y_train = np.array([classes.index(features.categories[i]) for i in train_idx])
    joint = JointReadout.fit(
        {modality: Z[modality][train_idx] for modality in GOLDEN_MODALITIES},
        y_train,
        classes,
        modalities=GOLDEN_MODALITIES,
    )

    payload: dict[str, np.ndarray] = {
        "classes": np.array(json.dumps(classes, ensure_ascii=False)),
        "weights_json": np.array(json.dumps(DEFAULT_WEIGHTS)),
        "modalities_json": np.array(json.dumps(list(GOLDEN_MODALITIES))),
        "train_idx": train_idx,
        "test_idx": test_idx,
    }
    for modality in GOLDEN_MODALITIES:
        payload[f"proto_{modality}"] = np.asarray(bank.protos[modality])
    payload["joint_scores"] = np.asarray(
        joint.scores_batch({m: Z[m][test_idx] for m in GOLDEN_MODALITIES})
    )

    top1: list[str] = []
    ranked_flat: list[str] = []
    for i in test_idx:
        query = {m: Z[m][i] for m in GOLDEN_MODALITIES}
        result = recognize(
            query,
            bank,
            weights=DEFAULT_WEIGHTS,
            use_tree=False,
            top_k=3,
        )
        ranked = [category for category, _ in result.ranked]
        top1.append(ranked[0] if ranked else "")
        ranked_flat.extend(ranked + [""] * (3 - len(ranked)))
    payload["proto_top1"] = np.array(top1)
    payload["proto_ranked"] = np.array(ranked_flat).reshape(-1, 3)
    return payload


def write_fixture(path: Path = FIXTURE, seed: int = 0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **build_payload(seed))
    return path


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--out", default=str(FIXTURE))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not args.write:
        print("dry run; pass --write to overwrite the fixture")
        return
    path = write_fixture(Path(args.out), args.seed)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
