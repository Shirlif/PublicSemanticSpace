import pytest

torch = pytest.importorskip("torch")

from pss.encoders.tactile.physiclear import (
    RealTactileDecoder,
    RealTactileEncoder,
    load_physiclear_constants,
    object_name_from_video,
)


def test_object_name_from_video():
    assert object_name_from_video("dataset/apple_0.mov") == "apple"
    assert (
        object_name_from_video("dataset/bread_knife_blade_12.mov")
        == "bread_knife_blade"
    )


def test_real_tactile_encoder_shape():
    encoder = RealTactileEncoder(out_dim=128)
    out = encoder(torch.zeros(2, 3, 64, 64))
    assert tuple(out.shape) == (2, 128)


def test_real_tactile_decoder_shape():
    decoder = RealTactileDecoder(out_dim=128, size=64)
    out = decoder(torch.zeros(2, 128))
    assert tuple(out.shape) == (2, 3, 64, 64)


def test_load_constants_without_executing(tmp_path):
    utils = tmp_path / "utils"
    utils.mkdir()
    (utils / "constants.py").write_text(
        "TRAIN_OBJECTS = ['physiclear_apple']\n"
        "VAL_OBJECTS = []\n"
        "TEST_OBJECTS = []\n"
        "HARDNESS_RANK = {'physiclear_apple': 0}\n"
        "ROUGHNESS_RANK = {'physiclear_apple': 1}\n"
        "TEXTURE_RANK = {'physiclear_apple': 2}\n",
        encoding="utf-8",
    )
    values = load_physiclear_constants(tmp_path)
    assert values["TRAIN_OBJECTS"] == ["physiclear_apple"]
    assert values["HARDNESS_RANK"]["physiclear_apple"] == 0
