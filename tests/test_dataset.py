import numpy as np
import pytest

pytest.importorskip("mujoco")

from pss.sim.dataset import load_sample, save_sample
from pss.sim.generator import generate_category_sample


def test_save_load_roundtrip(tmp_path):
    sample = generate_category_sample("cup", 0, seed=0, width=32, height=32, views=1)
    path = tmp_path / "sample.npz"
    save_sample(path, sample)
    loaded = load_sample(path)

    for key in ("rgb", "depth", "segmentation", "tactile"):
        assert np.allclose(loaded[key], sample[key])
    assert loaded["category"] == sample["category"]
    assert loaded["instance_id"] == sample["instance_id"]
    assert loaded["split"] == sample["split"]
    assert loaded["material"] == sample["material"]


def test_occlusion_v3_is_fully_occluded_and_objects_move():
    from pss.sim.generator import generate_occlusion_v3_sequence

    sample = generate_occlusion_v3_sequence(
        seed=0, width=64, height=64, frames=10, visible_head=2, visible_tail=2
    )
    assert sample["occlusion_start"] == 2
    assert sample["occlusion_end"] == 8
    assert sample["fully_occluded"]
    assert all(v > 0 for v in sample["visible_object_pixels"]["head"])
    assert all(v > 0 for v in sample["visible_object_pixels"]["tail"])

    start = np.asarray(sample["object_start_positions"], dtype=float)
    end = np.asarray(sample["object_end_positions"], dtype=float)
    motion = np.linalg.norm(start[:, :2] - end[:, :2], axis=1)
    assert float(motion.max()) > 0.05


def test_occlusion_v3_same_category_instances():
    from pss.sim.generator import generate_occlusion_v3_sequence

    sample = generate_occlusion_v3_sequence(
        seed=1,
        width=64,
        height=64,
        frames=10,
        visible_head=2,
        visible_tail=2,
        category="cup",
        variation_specs=True,
    )
    assert sample["same_category"] == "cup"
    assert set(sample["categories"]) == {"cup"}
    assert len(set(sample["instance_ids"])) == 3
    assert sample["fully_occluded"]


def test_occlusion_v3_near_identical_specs():
    from pss.sim.generator import generate_occlusion_v3_sequence

    sample = generate_occlusion_v3_sequence(
        seed=2,
        width=64,
        height=64,
        frames=10,
        visible_head=2,
        visible_tail=2,
        category="cup",
        near_identical=True,
    )
    assert sample["near_identical"]
    assert set(sample["categories"]) == {"cup"}
    specs = sample["object_specs"]
    assert len({s["instance_id"] for s in specs}) == 3
    first = specs[0]
    for spec in specs[1:]:
        assert spec["size"] == first["size"]
        assert spec["rgba"] == first["rgba"]
        assert spec["material"] == first["material"]
