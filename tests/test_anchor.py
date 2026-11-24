import numpy as np
import pytest

pytest.importorskip("torch")

from pss.projection.anchor import _hash_files, _material_bins, _retrieval


def test_hash_files_changes_with_content(tmp_path):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"one")
    before = _hash_files([path])
    path.write_bytes(b"two")
    after = _hash_files([path])
    assert before != after


def test_material_bins_range():
    rng = np.random.default_rng(0)
    attrs = rng.random((20, 5))
    bins = _material_bins(attrs, np.arange(20), n_bins=3)
    assert bins.shape == (20,)
    assert set(bins.tolist()) <= set(range(9))


def test_retrieval_identity():
    z = np.eye(4)
    labels = ["a", "b", "c", "d"]
    idx = np.arange(4)
    assert _retrieval(z, z, labels, idx, "category") == 1.0
    assert _retrieval(z, z, labels, idx, "pair") == 1.0
