"""Tests for the Phase A evaluation-suite skeleton."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pss.eval.suite.adapter import BenchmarkUnavailable
from pss.eval.suite.adapters.bfcl import BFCLAdapter
from pss.eval.suite.adapters.clevr_cogent import CLEVRCoGenTAdapter
from pss.eval.suite.adapters.tao import TAOAdapter
from pss.eval.suite.adapters.openood import OpenOODAdapter
from pss.eval.suite.adapters.zsre import ZsREAdapter
from pss.eval.suite.metrics import auroc, fpr_at_95tpr
from pss.eval.suite.predictors import predict_episode
from pss.eval.suite.registry import get_adapter, list_adapters
from pss.eval.suite.runner import run


def _check(adapter):
    episodes = adapter.sample_episodes()
    assert episodes
    predictions = [predict_episode(e) for e in episodes]
    results = adapter.evaluate(predictions)
    assert results
    for result in results:
        assert result.n > 0
        assert np.isfinite(result.value)
        assert 0.0 <= result.value <= 1.0


def test_bfcl_smoke():
    _check(BFCLAdapter())


def test_openood_smoke():
    _check(OpenOODAdapter())


def test_zsre_smoke():
    _check(ZsREAdapter())


def test_clevr_smoke():
    _check(CLEVRCoGenTAdapter())


def test_tao_smoke():
    _check(TAOAdapter())


def test_tao_official_annotations_if_present():
    root = "data/public/tao"
    if not Path(root).exists():
        pytest.skip("TAO annotations not downloaded")
    adapter = TAOAdapter(root=root)
    try:
        adapter.prepare()
    except BenchmarkUnavailable:
        pytest.skip("TAO annotations not present")
    episodes = list(adapter.episodes("test", max_videos=2))
    assert episodes
    predictions = [predict_episode(e) for e in episodes]
    results = adapter.evaluate(predictions)
    assert results
    assert all(np.isfinite(r.value) for r in results if r.value is not None)


def test_registry_and_unavailable():
    assert "bfcl" in list_adapters()
    with pytest.raises(KeyError):
        get_adapter("no_such_benchmark")
    with pytest.raises(BenchmarkUnavailable):
        BFCLAdapter(root=None).prepare()


def test_metrics_auroc_and_fpr():
    scores = [0.9, 0.8, 0.2, 0.1]
    positive = [True, True, False, False]
    assert auroc(scores, positive) == 1.0
    assert fpr_at_95tpr(scores, positive) == 0.0


def test_runner_smoke(tmp_path):
    payload = run(
        ["bfcl", "openood", "zsre"], smoke=True, out=str(tmp_path), allow_missing=True
    )
    assert set(payload["benchmarks"]) == {"bfcl", "openood", "zsre"}
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["smoke"] is True
    assert (tmp_path / "report.md").exists()
