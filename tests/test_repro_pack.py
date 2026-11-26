from __future__ import annotations

import pandas as pd

from pss.eval.scripts.run_gap_l2_memory import _has_embedding, _valid_summary


def test_has_embedding_detects_non_empty_rows(tmp_path):
    path = tmp_path / "obs_detections.parquet"
    pd.DataFrame({"emb": [[0.1, 0.2], [0.3, 0.4]]}).to_parquet(path, index=False)
    assert _has_embedding(path)


def test_has_embedding_rejects_empty_vectors(tmp_path):
    path = tmp_path / "obs_detections.parquet"
    pd.DataFrame({"emb": [[], []]}).to_parquet(path, index=False)
    assert not _has_embedding(path)


def test_has_embedding_rejects_missing_column(tmp_path):
    path = tmp_path / "obs_detections.parquet"
    pd.DataFrame({"step": [0, 1]}).to_parquet(path, index=False)
    assert not _has_embedding(path)


def test_valid_summary_excludes_zero_retention_bundles():
    metrics = {
        "variants": ["full", "no_memory", "no_entity"],
        "rows": [
            {"bundle": "a", "variant": "full", "id_retention": 1.0, "id_switches": 0.0},
            {"bundle": "a", "variant": "no_memory", "id_retention": 0.4, "id_switches": 1.0},
            {"bundle": "a", "variant": "no_entity", "id_retention": 0.2, "id_switches": 3.0},
            {"bundle": "b", "variant": "full", "id_retention": 0.0, "id_switches": 0.0},
            {"bundle": "b", "variant": "no_memory", "id_retention": 0.0, "id_switches": 0.0},
            {"bundle": "b", "variant": "no_entity", "id_retention": 0.0, "id_switches": 0.0},
        ],
    }
    summary = _valid_summary(metrics)
    assert summary["full"]["n"] == 1
    assert summary["full"]["id_retention"] == 1.0
    assert summary["no_memory"]["id_retention"] == 0.4
    assert summary["no_entity"]["id_switches"] == 3.0
