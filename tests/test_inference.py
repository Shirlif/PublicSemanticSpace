from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("mujoco")

from pss.inference import PublicSemanticSpace

DATA = Path("data/expert_v1")
VISION = Path("checkpoints/expert_v1/vision.pt")
TACTILE = Path("checkpoints/tactile_wtmm/tactile_wtmm.pt")
PROJ = Path("checkpoints/projection_wtmm")


@pytest.mark.skipif(
    not (DATA.exists() and VISION.exists() and TACTILE.exists() and PROJ.exists()),
    reason="run WT-MM tactile + projection training first",
)
def test_single_image_and_tactile():
    from pss.encoders.data import crop_object
    from pss.sim.dataset import load_sample

    space = PublicSemanticSpace()
    if space.bank is None:
        space.build_bank(DATA)
    sample = load_sample(DATA / "category" / "cat_cup_02.npz")
    crop = crop_object(
        sample["rgb"][0], sample["segmentation"][0], int(sample["geom_id"]), size=64
    )
    result = space.recognize(image=crop, tactile=sample["tactile"][0])
    assert result["modalities"] == ["vision", "tactile"]
    assert result["ranked"]
    assert result["ranked"][0][0] == "cup"
    assert 0.0 <= result["ranked"][0][1] <= 1.0


@pytest.mark.skipif(
    not (DATA.exists() and VISION.exists() and TACTILE.exists() and PROJ.exists()),
    reason="run WT-MM tactile + projection training first",
)
def test_single_modality_is_allowed():
    from pss.encoders.data import crop_object
    from pss.sim.dataset import load_sample

    space = PublicSemanticSpace()
    if space.bank is None:
        space.build_bank(DATA)
    sample = load_sample(DATA / "category" / "cat_cup_02.npz")
    crop = crop_object(
        sample["rgb"][0], sample["segmentation"][0], int(sample["geom_id"]), size=64
    )
    vision_only = space.recognize(image=crop)
    assert vision_only["modalities"] == ["vision"]
    tactile_only = space.recognize(tactile=sample["tactile"][0])
    assert tactile_only["modalities"] == ["tactile"]
