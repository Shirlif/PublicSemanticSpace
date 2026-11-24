import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")

from pss.encoders.scripts.bench_expert_zoo import (
    OctopiTactileAdapter,
    RawView,
    RealTactileAdapter,
    _cell,
    _json_safe,
    _spearman,
    _stratified_subset,
    cross_modal,
    spec_image_size,
)


def test_spearman_monotonic():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([10.0, 20.0, 30.0, 40.0])
    assert _spearman(a, b) == pytest.approx(1.0)


def test_stratified_subset_keeps_classes():
    y = np.array([0] * 10 + [1] * 10 + [2] * 10)
    idx = _stratified_subset(y, 0, frac=0.8)
    assert len(idx) == 24
    assert set(y[idx].tolist()) == {0, 1, 2}


def test_cross_modal_recovers_linear_relation():
    rng = np.random.default_rng(0)
    samples = 16
    views = 4
    n = samples * views
    sample_ids = np.array([f"s{i // views}" for i in range(n)])
    splits = np.array(["train"] * (n // 2) + ["test"] * (n // 2))
    # Views of one sample share a base vector; per-view noise keeps them distinct.
    base = rng.normal(size=(samples, 8))
    a = np.repeat(base, views, axis=0) + 0.01 * rng.normal(size=(n, 8))
    w = rng.normal(size=(8, 6))
    b = a @ w + 0.01 * rng.normal(size=(n, 6))
    out = cross_modal(a, b, {"splits": splits, "sample_ids": sample_ids})
    assert out["method"] == "ridge_linear_probe"
    assert out["positives"] == (samples // 2) * views * (views - 1)
    assert out["r1"] >= 0.75


def test_json_safe_converts_nan():
    out = _json_safe({"a": float("nan"), "b": [np.float32(1.5)]})
    assert out["a"] is None
    assert out["b"] == [1.5]


def test_cell_keeps_table_row_on_one_line():
    assert _cell("a|b\nc") == "a\\|b c"


def test_spec_image_size():
    assert spec_image_size({"input_schema": {"rgb": [3, 224, 224]}}) == 224
    assert spec_image_size({}) == 64


def test_octopi_batch_is_clip_ready():
    adapter = OctopiTactileAdapter.__new__(OctopiTactileAdapter)
    adapter.image_size = 224
    adapter.frames = 3
    view = RawView(
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        depth=np.zeros((8, 8), dtype=np.float32),
        seg=np.zeros((8, 8, 2), dtype=np.int32),
        geom_id=0,
        tactile=np.zeros(64, dtype=np.float32),
        geometry=np.zeros(16, dtype=np.float32),
        attrs=np.zeros(5, dtype=np.float32),
        category="cup",
        split="train",
        instance="cup_0",
        sample_id="cat_cup_0",
        view=0,
    )
    x = adapter._batch([view])
    assert tuple(x.shape) == (1, 3, 3, 224, 224)


def test_real_tactile_batch_is_encoder_ready():
    adapter = RealTactileAdapter.__new__(RealTactileAdapter)
    adapter.size = 64
    view = RawView(
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        depth=np.zeros((8, 8), dtype=np.float32),
        seg=np.zeros((8, 8, 2), dtype=np.int32),
        geom_id=0,
        tactile=np.zeros(64, dtype=np.float32),
        geometry=np.zeros(16, dtype=np.float32),
        attrs=np.zeros(5, dtype=np.float32),
        category="cup",
        split="train",
        instance="cup_0",
        sample_id="cat_cup_0",
        view=0,
    )
    x = adapter._batch([view])
    assert tuple(x.shape) == (1, 3, 64, 64)
