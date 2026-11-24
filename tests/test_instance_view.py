import numpy as np
import pytest

pytest.importorskip("torch")

from pss.projection.instance_view import (
    _category_retrieval,
    _instance_retrieval,
    _instance_retrieval_qg,
)


def test_instance_retrieval_identity():
    z = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    instances = ["a", "a", "b", "b"]
    idx = np.arange(4)
    result = _instance_retrieval(z, z, instances, idx)
    assert result["r1"] == 1.0
    assert result["r5"] == 1.0
    assert result["positives"] == 4


def test_category_retrieval_identity():
    z = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    categories = ["x", "x", "y", "y"]
    idx = np.arange(4)
    assert _category_retrieval(z, z, categories, idx) == 1.0


def test_instance_retrieval_query_gallery_identity():
    query = np.array([[1.0, 0.0], [0.0, 1.0]])
    gallery = np.array([[1.0, 0.0], [0.0, 1.0]])
    instances = ["a", "b"]
    result = _instance_retrieval_qg(query, gallery, instances, [0, 1], [0, 1])
    assert result["r1"] == 1.0
    assert result["positives"] == 2
