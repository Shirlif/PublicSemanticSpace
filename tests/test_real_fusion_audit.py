"""Tests for the real-data CLEVR-CoGenT fusion audit helpers."""

import numpy as np
import pytest

pytest.importorskip("PIL")
pytest.importorskip("sklearn")

from pss.eval.scripts.run_real_fusion_audit import (
    LABEL_TASKS,
    _front,
    _majority,
    geometry_features,
    handcrafted_image_features,
    scene_labels,
)


def _obj(px, py, pz, *, shape="cube", color="red", material="metal", size="large",
         x=0.0, y=0.0, z=0.7, rot=0.0):
    return {
        "pixel_coords": [px, py, pz],
        "3d_coords": [x, y, z],
        "rotation": rot,
        "shape": shape,
        "color": color,
        "material": material,
        "size": size,
    }


def _scene(objects):
    return {"objects": objects}


def test_front_picks_smallest_depth_then_smallest_y():
    objs = [_obj(10, 50, 12.0, shape="cube"), _obj(20, 50, 8.0, shape="sphere")]
    assert _front(objs)["shape"] == "sphere"
    tied = [_obj(10, 90, 8.0, shape="cube"), _obj(20, 40, 8.0, shape="sphere")]
    assert _front(tied)["shape"] == "sphere"


def test_majority_tie_breaks_alphabetically():
    objs = [_obj(0, 0, 10, color="red"), _obj(0, 0, 11, color="blue")]
    assert _majority(objs, "color") == "blue"
    objs.append(_obj(0, 0, 12, color="red"))
    assert _majority(objs, "color") == "red"


def test_scene_labels_covers_every_task():
    objs = [
        _obj(10, 10, 9.0, shape="cube", color="red", material="metal"),
        _obj(20, 20, 10.0, shape="cube", color="blue", material="rubber"),
        _obj(30, 30, 11.0, shape="sphere", color="red", material="metal"),
    ]
    labels = scene_labels(_scene(objs))
    assert set(labels) == set(LABEL_TASKS)
    assert labels["front_shape"] == "cube"  # smallest depth
    assert labels["majority_color"] == "red"
    assert labels["majority_shape"] == "cube"
    assert labels["distinct_shape"] == "2"
    assert labels["distinct_color"] == "2"


def test_geometry_features_ignore_semantic_attributes():
    base = _obj(100, 100, 10.0, rot=45.0)
    other = _obj(100, 100, 10.0, rot=45.0, shape="sphere", color="blue",
                 material="rubber", size="small")
    a = geometry_features(_scene([base, _obj(200, 100, 12.0)]))
    b = geometry_features(_scene([other, _obj(200, 100, 12.0)]))
    assert a.shape == (132,)
    assert np.array_equal(a, b)


def test_geometry_features_depend_on_position():
    a = geometry_features(_scene([_obj(100, 100, 10.0)]))
    b = geometry_features(_scene([_obj(300, 100, 10.0)]))
    assert not np.array_equal(a, b)


def test_handcrafted_image_features_shape_and_determinism():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)
    first = handcrafted_image_features(image)
    second = handcrafted_image_features(image.copy())
    assert first.shape == (88,)
    assert np.array_equal(first, second)
    other = np.zeros_like(image)
    assert not np.array_equal(first, handcrafted_image_features(other))
