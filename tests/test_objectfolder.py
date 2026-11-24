import numpy as np
import pytest

pytest.importorskip("torch")

from pss.rd2.objectfolder import _retrieval, obj_vertices, write_touch_queries


def test_retrieval_identity():
    z = np.array([[1.0, 0.0], [0.0, 1.0]])
    result = _retrieval(z, z)
    assert result["r1"] == 1.0
    assert result["r5"] == 1.0
    assert result["chance"] == 0.5


def test_obj_vertices_and_touch_queries(tmp_path):
    obj = tmp_path / "model.obj"
    obj.write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 2 3\n", encoding="utf-8"
    )
    vertices = obj_vertices(obj)
    assert vertices.shape == (4, 3)

    queries = write_touch_queries(tmp_path, tmp_path / "queries", touches=2, seed=0)
    assert queries is not None
    vertices_path, gelinfo_path = queries
    assert np.load(vertices_path).shape == (2, 3)
    assert np.load(gelinfo_path).shape == (2, 3)
