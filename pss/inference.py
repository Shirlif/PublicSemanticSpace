"""Minimal inference API for the public semantic space.

Input: one object image + one tactile map (geometry optional).
Pipeline: frozen experts -> per-modality MLP heads -> 128-d public space ->
prototype bank -> top-k.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from pss.contracts.modality import get_modalities, get_spec
from pss.contracts.schema import SPACE_VERSION
from pss.encoders.data import normalize_image
from pss.encoders.geometry import geometry_features
from pss.encoders.tactile.wtmm import WTMMTactileExpert, load_wtmm_tactile
from pss.encoders.vision import load_vision_expert
from pss.experts.adapter import admit_encoder
from pss.projection import load_projection
from pss.projection.calibration import load_calibration
from pss.projection.features import load_paired_features
from pss.prototypes import (
    JointReadout,
    PrototypeBank,
    build_bank,
    recognize,
    recognize_joint,
)

MODALITIES = get_modalities()
DEFAULT_WEIGHTS = {m: get_spec(m).reliability_prior for m in MODALITIES}


def resolve_weights(
    modalities,
    calibration: dict | None = None,
) -> dict[str, float]:
    """Registry priors, overridden by a calibration record when present."""
    weights = {m: get_spec(m).reliability_prior for m in modalities}
    if calibration:
        for modality, record in calibration.items():
            if modality in weights:
                weights[modality] = float(record.reliability)
    return weights


def load_available_heads(
    projection_dir: str | Path,
    device: str = "cpu",
    modalities: tuple[str, ...] | None = None,
) -> tuple[dict, list[str]]:
    """Load the projection heads that exist; report the ones that do not.

    This is the dynamic part of ``PublicSemanticSpace``: a modality without a
    projection file is simply inactive, not a construction error.
    """
    heads: dict = {}
    inactive: list[str] = []
    for modality in modalities or MODALITIES:
        path = Path(projection_dir) / f"{modality}.pt"
        if path.exists():
            heads[modality] = load_projection(path, device=device)
        else:
            inactive.append(modality)
    return heads, inactive


class PublicSemanticSpace:
    def __init__(
        self,
        vision_ckpt: str | Path = "checkpoints/expert_v1/vision.pt",
        tactile_ckpt: str | Path = "checkpoints/tactile_wtmm/tactile_wtmm.pt",
        projection_dir: str | Path = "checkpoints/projection_wtmm",
        bank_path: str | Path = "checkpoints/public_space_v1/bank.npz",
        device: str | None = None,
        calibration_path: str | Path | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Both experts are admitted against the same contract the registry and
        # property suite use; a malformed checkpoint fails here, not at the
        # first inference call.
        self.vision = admit_encoder(
            load_vision_expert(vision_ckpt, device=self.device),
            modality="vision",
            version=f"smallcnn@{SPACE_VERSION}",
        ).encoder
        self.tactile = admit_encoder(
            WTMMTactileExpert(
                encoder=load_wtmm_tactile(tactile_ckpt, device=self.device),
                device=self.device,
                input_dim=64,
            ),
            modality="tactile",
            version=f"wtmm@{SPACE_VERSION}",
        ).encoder
        self.heads, self.inactive_modalities = load_available_heads(
            projection_dir, device=self.device
        )
        self.modalities = tuple(self.heads)
        # Optional per-modality calibration artifact.  It overrides the
        # registry priors with validation-fitted reliability (and keeps the
        # temperatures available for callers that calibrate scores).
        if calibration_path is None:
            candidate = Path(projection_dir) / "calibration.json"
        else:
            candidate = Path(calibration_path)
        self.calibration_path = candidate if candidate.exists() else None
        self.calibration = (
            load_calibration(candidate) if self.calibration_path is not None else {}
        )
        self.weights = resolve_weights(self.modalities, self.calibration)
        self.bank_path = Path(bank_path)
        self.bank = PrototypeBank.load(self.bank_path) if self.bank_path.exists() else None
        self.joint_path = self.bank_path.parent / "joint_readout.npz"
        self.joint = (
            JointReadout.load(self.joint_path) if self.joint_path.exists() else None
        )
        self.readout_choice = (
            str(self.joint.metadata.get("readout_choice", "prototype"))
            if self.joint is not None
            else "prototype"
        )

    def describe(self) -> dict:
        """Report the active/inactive modalities and their registry versions."""
        return {
            "space_version": SPACE_VERSION,
            "modalities": list(self.modalities),
            "inactive_modalities": list(self.inactive_modalities),
            "weights": dict(self.weights),
            "calibration_path": (
                str(self.calibration_path) if self.calibration_path else None
            ),
            "calibration": {
                modality: record.to_dict()
                for modality, record in self.calibration.items()
            },
            "readout": self.readout_choice,
            "modality_versions": {
                modality: get_spec(modality).version for modality in self.modalities
            },
        }

    @torch.no_grad()
    def encode_image(self, image) -> np.ndarray:
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        image = image.resize((self.vision.image_size, self.vision.image_size), Image.BILINEAR)
        x = normalize_image(np.asarray(image, dtype=np.uint8))
        tensor = torch.as_tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
        feature = self.vision.encode(tensor)
        return self.heads["vision"](feature).cpu().numpy()[0]

    @torch.no_grad()
    def encode_tactile(self, tactile) -> np.ndarray:
        if isinstance(tactile, (str, Path)):
            tactile = np.load(tactile)
        arr = np.asarray(tactile, dtype=np.float32)
        if arr.size and arr.max() > 1.5:
            arr = arr / arr.max()
        tensor = torch.as_tensor(arr, dtype=torch.float32, device=self.device)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        feature = torch.as_tensor(
            self.tactile.encode(tensor), dtype=torch.float32, device=self.device
        )
        return self.heads["tactile"](feature).cpu().numpy()[0]

    @torch.no_grad()
    def encode_geometry(self, depth, seg, geom_id: int) -> np.ndarray:
        vec = geometry_features(np.asarray(depth), np.asarray(seg), int(geom_id))
        tensor = torch.as_tensor(vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.heads["geometry"](tensor).cpu().numpy()[0]

    def build_bank(self, data_dir: str | Path = "data/expert_v1") -> PrototypeBank:
        feats = load_paired_features(data_dir, self.vision, self.tactile)
        with torch.no_grad():
            Z = {
                m: self.heads[m](
                    torch.as_tensor(getattr(feats, m), dtype=torch.float32, device=self.device)
                ).cpu().numpy()
                for m in self.modalities
            }
        splits = np.array(feats.splits)
        classes = sorted({feats.categories[i] for i in np.where(splits == "train")[0]})
        train_idx = np.array(
            [i for i in np.where(splits == "train")[0] if feats.categories[i] in classes],
            dtype=int,
        )
        bank = build_bank(
            feats,
            Z,
            train_idx,
            classes,
            version="public_space_v1",
            modality_versions={m: get_spec(m).version for m in Z},
        )
        self.bank_path.parent.mkdir(parents=True, exist_ok=True)
        bank.save(self.bank_path)
        self.bank = bank

        # Joint discriminative readout on the same 128-d features.  The mask
        # models let a query with only a subset of modalities still work.
        y_train = np.array([classes.index(feats.categories[i]) for i in train_idx])
        modalities = tuple(Z)
        mask_strategy = "all" if len(modalities) <= 3 else "single_plus_all"
        self.joint = JointReadout.fit(
            {m: Z[m][train_idx] for m in modalities},
            y_train,
            classes,
            modalities=modalities,
            mask_strategy=mask_strategy,
        )
        val_idx = np.array(
            [i for i in np.where(splits == "val")[0] if feats.categories[i] in classes],
            dtype=int,
        )
        self.readout_choice = "prototype"
        if len(val_idx):
            raw_scores = self.joint.scores_batch(
                {m: Z[m][val_idx] for m in modalities}
            )
            y_val = np.array([classes.index(feats.categories[i]) for i in val_idx])
            joint_acc = float(np.mean(raw_scores.argmax(axis=1) == y_val))
            self.joint.threshold = float(np.percentile(raw_scores.max(axis=1), 5))

            weights = {m: self.weights[m] for m in modalities}
            proto_correct = 0
            for i in val_idx:
                query = {m: Z[m][i] for m in modalities}
                result = recognize(
                    query, bank, weights=weights, use_tree=False, top_k=1
                )
                proto_correct += bool(
                    result.ranked and result.ranked[0][0] == feats.categories[i]
                )
            proto_acc = proto_correct / len(val_idx)
            # Only switch when the joint head is actually better on val; the
            # prototype stays the safer open-set/hot-update head otherwise.
            self.readout_choice = (
                "joint" if joint_acc > proto_acc + 0.01 else "prototype"
            )
            self.joint.metadata["readout_choice"] = self.readout_choice
            self.joint.metadata["val_acc"] = {
                "prototype": float(proto_acc),
                "joint": joint_acc,
            }
        self.joint_path.parent.mkdir(parents=True, exist_ok=True)
        self.joint.save(self.joint_path)
        return bank

    def recognize(
        self,
        image=None,
        tactile=None,
        geometry=None,
        top_k: int = 3,
        readout: str = "auto",
    ) -> dict:
        if self.bank is None:
            raise RuntimeError("prototype bank missing; call build_bank() first")
        query = {}
        if image is not None:
            query["vision"] = self.encode_image(image)
        if tactile is not None:
            query["tactile"] = self.encode_tactile(tactile)
        if geometry is not None:
            query["geometry"] = self.encode_geometry(*geometry)
        if not query:
            raise ValueError("provide at least one modality value")

        weights = {m: self.weights.get(m, 1.0) for m in query}
        use_joint = readout == "joint" or (
            readout == "auto" and self.readout_choice == "joint"
        )
        if self.joint is not None and use_joint:
            result = recognize_joint(
                query,
                self.joint,
                bank=self.bank,
                top_k=top_k,
                weights={m: weights[m] for m in query},
                threshold=self.joint.threshold,
            )
            used = "joint"
        else:
            result = recognize(
                query,
                self.bank,
                weights={m: weights[m] for m in query},
                use_tree=False,
                top_k=top_k,
            )
            used = "prototype"
        return {
            "ranked": [(c, float(s)) for c, s in result.ranked],
            "per_modality_top": result.modality_top,
            "conflict": result.conflict,
            "unknown": result.unknown,
            "modalities": list(query),
            "readout": used,
        }
