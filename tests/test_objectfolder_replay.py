import numpy as np
import pytest

pytest.importorskip("torch")

from pss.rd2.objectfolder_replay import (
    CrossModalBindingAdapter,
    CrossModalBindingProperty,
    PassthroughAdapter,
    binding_metrics,
    build_stream,
    discover_renders,
    load_object_metadata,
)
from pss_eval.properties import IdentityRetentionProperty


def _separable(n: int = 10, dim: int = 16, noise: float = 0.01, seed: int = 0):
    rng = np.random.default_rng(seed)
    vision = rng.normal(size=(n, dim))
    vision /= np.linalg.norm(vision, axis=1, keepdims=True)
    touch = vision + noise * rng.normal(size=(n, dim))
    names = [f"obj{i}" for i in range(n)]
    return vision, touch, names


def test_binding_metrics_separable():
    # 20 objects in 4-d: the half-split Ridge fit is well determined and should
    # recover the identity map, unlike a 16-d/5-sample underdetermined fit.
    vision, touch, names = _separable(n=20, dim=4, noise=0.001)
    metrics = binding_metrics(vision, touch, names, seed=0, permutations=50)
    assert metrics["n"] == 20
    assert metrics["chance_r1"] == pytest.approx(0.05)
    assert metrics["zero_shot"]["vision_to_touch"]["r1"] >= 0.9
    assert metrics["zero_shot"]["touch_to_vision"]["r1"] >= 0.9
    assert metrics["ridge_half_split"]["vision_to_touch"]["r1"] >= 0.9
    assert metrics["cosine"]["gap"] > 0.0
    assert metrics["permutation"]["p_value"] <= 0.05


def test_binding_metrics_anti_correlated():
    vision, _, names = _separable()
    metrics = binding_metrics(vision, -vision, names, seed=0, permutations=50)
    # Anti-correlated pairs make same-object retrieval systematically wrong.
    assert metrics["zero_shot"]["vision_to_touch"]["r1"] == 0.0


def test_build_stream_orders_modalities_and_labels():
    vision, touch, names = _separable(n=5, dim=8)
    stream = build_stream(vision, touch, names, seed=0)
    assert len(stream) == 10
    assert stream.dim == 8
    modalities = [sample.observation.modality for sample in stream.samples]
    assert modalities == ["vision"] * 5 + ["touch"] * 5
    for index, name in enumerate(names):
        assert stream.samples[index].observation.observation_id == f"{name}:vision"
        assert stream.samples[5 + index].observation.observation_id == f"{name}:touch"
        assert stream.identity_labels[f"{name}:vision"] == f"object:{name}"
        assert stream.identity_labels[f"{name}:touch"] == f"object:{name}"


def test_cross_modal_binding_property_passes_on_separable():
    vision, touch, names = _separable()
    stream = build_stream(vision, touch, names)
    result = CrossModalBindingProperty().evaluate(
        lambda: PassthroughAdapter(), stream, seed=0
    )
    assert result.status == "pass"
    assert result.metrics["r1"] >= 0.9
    assert result.metrics["chance"] == pytest.approx(0.1)


def test_cross_modal_binding_property_fails_on_anti_correlated():
    vision, _, names = _separable()
    stream = build_stream(vision, -vision, names)
    result = CrossModalBindingProperty().evaluate(
        lambda: PassthroughAdapter(), stream, seed=0
    )
    assert result.status == "fail"
    assert result.metrics["r1"] == 0.0


def test_nn_binding_adapter_exposes_consistent_identity_assignments():
    vision, touch, names = _separable()
    stream = build_stream(vision, touch, names)
    result = IdentityRetentionProperty().evaluate(
        lambda: CrossModalBindingAdapter(), stream, seed=0
    )
    assert result.status == "pass"
    assert result.metrics["identity_retention"] >= 0.9


def test_discover_renders_and_metadata(tmp_path):
    csv_path = tmp_path / "objects.csv"
    csv_path.write_text(
        "1,Cabinet,2.36,Wood,https://example/1\n"
        "2,Barrel,0.88,Steel,https://example/2\n",
        encoding="utf-8",
    )
    metadata = load_object_metadata(csv_path)
    assert metadata["1"]["name"] == "Cabinet"
    assert metadata["2"]["material"] == "Steel"

    renders = tmp_path / "renders"
    for object_id in ("1", "2", "3"):
        (renders / object_id / "vision").mkdir(parents=True)
        (renders / object_id / "touch").mkdir(parents=True)
        (renders / object_id / "vision" / "000.png").write_bytes(b"")
        (renders / object_id / "touch" / "0.png").write_bytes(b"")

    objects = discover_renders(renders, limit=2, metadata=metadata)
    assert [obj.object_id for obj in objects] == ["1", "2"]
    assert objects[0].name == "Cabinet"
    assert objects[0].material == "Wood"
    assert len(objects[0].vision) == 1
    assert len(objects[0].touch) == 1
