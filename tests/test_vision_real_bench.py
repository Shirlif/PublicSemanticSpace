import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("PIL")

from PIL import Image

from pss.eval.scripts.run_vision_real_bench import _crop_front, _fit_probe, _tao_crop


def test_tao_crop_clips_to_frame_and_rejects_empty():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    crop = _tao_crop(frame, (180, 90, 40, 40))
    assert crop is not None and crop.size > 0
    assert crop.shape[0] <= 100 and crop.shape[1] <= 200
    assert _tao_crop(frame, (300, 300, 10, 10)) is None


def test_crop_front_stays_inside_image(tmp_path):
    path = tmp_path / "scene.png"
    Image.fromarray(np.zeros((48, 64, 3), dtype=np.uint8)).save(path)
    scene = {
        "objects": [
            {"pixel_coords": [60, 44, 1.0], "shape": "sphere", "material": "metal"},
        ]
    }
    crop = _crop_front(scene, path, crop=32)
    assert crop.shape == (32, 32, 3)


def test_fit_probe_separates_two_clusters():
    rng = np.random.default_rng(0)
    train_x = np.concatenate(
        [rng.normal(0.0, 0.1, (20, 4)), rng.normal(3.0, 0.1, (20, 4))]
    )
    train_y = ["a"] * 20 + ["b"] * 20
    clf = _fit_probe(train_x, train_y)
    assert list(clf.predict([[0.0] * 4, [3.0] * 4])) == ["a", "b"]
