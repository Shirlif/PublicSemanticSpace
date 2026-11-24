from pathlib import Path

from pss.encoders.data import load_index
from pss.sim.tracking_bench import generate_tracking_bench


def test_generate_tracking_bench_splits(tmp_path):
    summary = generate_tracking_bench(
        tmp_path,
        train_count=4,
        val_count=4,
        test_count=4,
        seed=0,
        width=128,
        height=128,
        frames=24,
    )
    assert summary["entries"] == 12
    assert summary["train"] == 4
    assert summary["val"] == 4
    assert summary["test"] == 4

    index = load_index(tmp_path)
    subsets = {e["subset"] for e in index["entries"]}
    assert subsets == {"mixed", "samecat", "near_identical"}
    for entry in index["entries"]:
        assert entry["fully_occluded"]
        assert entry["occlusion_end"] - entry["occlusion_start"] == 14
