"""The public inference API must reject malformed experts at construction."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("PIL")

import pss.inference as inference_mod
from pss.experts.contract import ExpertValidationError


class _BadVision:
    out_dim = 6
    image_size = 8

    def encode(self, images):
        return np.zeros((len(images), 5))


def test_inference_rejects_malformed_vision(monkeypatch):
    monkeypatch.setattr(
        inference_mod, "load_vision_expert", lambda *a, **k: _BadVision()
    )
    with pytest.raises(ExpertValidationError, match="returned dim 5"):
        inference_mod.PublicSemanticSpace()


def test_inference_rejects_missing_width(monkeypatch):
    class _NoDim:
        image_size = 8

        def encode(self, images):
            return np.zeros((len(images), 6))

    monkeypatch.setattr(inference_mod, "load_vision_expert", lambda *a, **k: _NoDim())
    with pytest.raises(ExpertValidationError, match="declares no output width"):
        inference_mod.PublicSemanticSpace()
