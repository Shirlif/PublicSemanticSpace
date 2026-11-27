"""Tests for the geometry / absolute-position validation helpers."""

from __future__ import annotations

import numpy as np

from pss.eval.scripts.run_geometry_validation import (
    PAIR_TYPE_EXPERIMENTS,
    _absolute_labels,
    _extra_rows,
    _grid,
    _isolation,
    _pair_type,
    _subset_names,
)


def test_grid_is_centered_and_complete():
    coords = _grid(7)
    assert len(coords) == 49
    values = np.stack(list(coords.values()))
    assert values[:, 0].min() == -3.0
    assert values[:, 0].max() == 3.0


def test_absolute_labels_use_subject_position():
    assert _absolute_labels((-1.0, 0.0), (1.0, 0.0)) == [True, False, False, False]
    assert _absolute_labels((1.0, 0.0), (-1.0, 0.0)) == [False, True, False, False]
    assert _absolute_labels((0.0, 1.0), (0.0, -1.0)) == [False, False, True, False]


def test_subset_sizes_and_isolation_directions():
    coords = _grid(7)
    block = _subset_names(coords, "block3")
    strip_x = _subset_names(coords, "strip_x")
    strip_y = _subset_names(coords, "strip_y")
    diag = _subset_names(coords, "diag")
    assert (len(block), len(strip_x), len(strip_y), len(diag)) == (9, 7, 7, 7)

    assert _isolation(block, coords) == {"x": True, "y": True}
    assert _isolation(strip_x, coords) == {"x": True, "y": False}
    assert _isolation(strip_y, coords) == {"x": False, "y": True}
    assert _isolation(diag, coords) == {"x": False, "y": False}


def test_pair_type_on_larger_grid():
    coords = _grid(7)
    assert _pair_type(coords["23"], coords["43"]) == "horizontal"
    assert _pair_type(coords["32"], coords["34"]) == "vertical"
    assert _pair_type(coords["22"], coords["44"]) == "diagonal"


def test_extra_rows_isolate_the_requested_axis():
    extent = (-3.0, -3.0, 3.0, 3.0)
    horizontal = _extra_rows(np.random.default_rng(0), "x", 8, "absolute", extent, noise=0.0)
    for features, _labels in horizontal:
        assert abs(features[5]) < 1e-9
        assert abs(features[4]) >= 0.5

    vertical = _extra_rows(np.random.default_rng(0), "y", 8, "relative", extent, noise=0.0)
    for features, _labels in vertical:
        assert len(features) == 4
        assert abs(features[0]) < 1e-9
        assert abs(features[1]) >= 0.5


def test_extra_rows_diagonal_varies_both_axes():
    extent = (-3.0, -3.0, 3.0, 3.0)
    rows = _extra_rows(
        np.random.default_rng(0), "diagonal", 8, "relative", extent, noise=0.0
    )
    for features, _labels in rows:
        assert abs(features[0]) >= 0.5
        assert abs(features[1]) >= 0.5


def test_pair_type_experiments_cover_the_four_arms():
    arms = {tuple(exp["augment"]) for exp in PAIR_TYPE_EXPERIMENTS}
    assert arms == {(), ("y",), ("y", "diagonal"), ("diagonal",)}
