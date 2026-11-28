import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")

from pss.encoders.scripts.bench_expert_zoo import (
    HandcraftedTactileAdapter,
    HandcraftedVisionAdapter,
    OctopiTactileAdapter,
    RawView,
    RealTactileAdapter,
    _cell,
    _json_safe,
    _spearman,
    _stratified_subset,
    build_adapter,
    cross_modal,
    fusion_audit,
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


def _vision_view() -> RawView:
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    seg = np.ones((64, 64, 2), dtype=np.int32) * 7
    return RawView(
        rgb=rgb,
        depth=np.ones((64, 64), dtype=np.float32),
        seg=seg,
        geom_id=7,
        tactile=np.zeros(64, dtype=np.float32),
        geometry=np.zeros(16, dtype=np.float32),
        attrs=np.zeros(5, dtype=np.float32),
        category="cup",
        split="train",
        instance="cup_0",
        sample_id="cat_cup_0",
        view=0,
    )


def test_handcrafted_adapter_matches_spec_dim():
    adapter = build_adapter(
        {
            "name": "vision_handcrafted_v0",
            "input_schema": {"rgb": [3, 64, 64]},
        },
        device="cpu",
        allow_download=False,
    )
    assert isinstance(adapter, HandcraftedVisionAdapter)
    feats = adapter.encode([_vision_view()], preserve_aspect=False)
    assert tuple(feats.shape) == (1, adapter.out_dim)
    assert adapter.params() == 0
    assert np.isfinite(feats).all()


def test_handcrafted_tactile_adapter_is_zero_param():
    adapter = build_adapter(
        {"name": "tactile_handcrafted_v0", "input_schema": {"contact_grid": [8, 8]}},
        device="cpu",
        allow_download=False,
    )
    assert isinstance(adapter, HandcraftedTactileAdapter)
    view = _vision_view()
    view.tactile = np.full(64, 0.5, dtype=np.float32)
    feats = adapter.encode([view], preserve_aspect=False)
    assert feats.shape[0] == 1 and feats.shape[1] > 0
    assert adapter.params() == 0
    assert np.isfinite(feats).all()


def _fusion_meta(n_per_class: int = 50, n_val: int = 10, n_test: int = 10):
    """Four balanced classes with train/val/test splits."""
    cats, splits = [], []
    for cls in range(4):
        for split, count in (("train", n_per_class), ("val", n_val), ("test", n_test)):
            cats.extend([f"c{cls}"] * count)
            splits.extend([split] * count)
    return {
        "categories": np.asarray(cats),
        "splits": np.asarray(splits),
        "instances": np.asarray(cats),
        "sample_ids": np.asarray([f"s{i}" for i in range(len(cats))]),
        "views": np.zeros(len(cats), dtype=np.int64),
        "attrs": np.zeros((len(cats), 5), dtype=np.float32),
    }


def _block_features(meta, means, rng, noise: float = 0.05):
    cats = np.asarray(meta["categories"])
    out = np.empty((len(cats), len(means[0])), dtype=np.float64)
    for cls in range(len(means)):
        idx = np.where(cats == f"c{cls}")[0]
        out[idx] = np.asarray(means[cls]) + noise * rng.normal(size=(len(idx), len(means[0])))
    return out.astype(np.float32)


def test_fusion_audit_detects_complementary_experts():
    rng = np.random.default_rng(0)
    meta = _fusion_meta()
    # Expert A separates {0,1}; expert B separates {2,3}. Neither is complete,
    # but the concat is.
    feat_a = _block_features(meta, [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (2.0, 0.0)], rng)
    feat_b = _block_features(meta, [(0.0, 0.0), (0.0, 0.0), (1.0, 0.0), (2.0, 0.0)], rng)
    out = fusion_audit({"a": feat_a, "b": feat_b}, meta)
    assert out["all_concat_test"] > 0.95
    assert out["gap_vs_best_single"] > 0.15
    assert out["oracle_any"] >= out["all_concat_test"]
    assert out["mean_disagreement"] > 0.0
    for key in ("vote_lda_test", "weighted_vote_test", "gate_top1_test", "gate_soft_test"):
        assert 0.0 <= out[key] <= 1.0


def test_fusion_audit_redundant_experts_have_no_gap():
    rng = np.random.default_rng(1)
    meta = _fusion_meta()
    feat = _block_features(meta, [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0), (6.0, 0.0)], rng)
    out = fusion_audit({"a": feat, "b": feat.copy()}, meta)
    assert out["all_concat_test"] >= 0.95
    assert abs(out["gap_vs_best_single"]) < 0.02
    assert out["mean_error_corr"] == pytest.approx(1.0, abs=1e-6) or np.isnan(
        out["mean_error_corr"]
    )
