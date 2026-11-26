"""Real-bundle stream tests.

These skip cleanly when the development bundles are not present (the public
repository does not ship data).  The PSS adapter never opens ``gt.parquet``;
ground truth is only attached to ``Stream.identity_labels`` for the harness.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pss_eval.adapters.pss_runtime import PSSRuntimeAdapter
from pss_eval.bundles import (
    DEFAULT_BUNDLE_NAMES,
    DEFAULT_BUNDLE_ROOT,
    find_bundle_dirs,
    load_bundle_stream,
    load_bundle_streams,
)
from pss_eval.properties import ContractProperty, IdentityRetentionProperty

BUNDLE_ROOT = Path(DEFAULT_BUNDLE_ROOT)
BUNDLES_AVAILABLE = len(find_bundle_dirs(BUNDLE_ROOT)) == len(DEFAULT_BUNDLE_NAMES)

pytestmark = pytest.mark.skipif(
    not BUNDLES_AVAILABLE, reason=f"development bundles not found under {BUNDLE_ROOT}"
)


def _first_stream():
    streams = load_bundle_streams(BUNDLE_ROOT, max_frames=8)
    name = next(iter(streams))
    return name, streams[name]


def test_four_bundles_load_with_embeddings():
    streams = load_bundle_streams(BUNDLE_ROOT, max_frames=4)
    assert len(streams) == len(DEFAULT_BUNDLE_NAMES)
    for name, stream in streams.items():
        assert len(stream) == 4, name
        assert stream.dim == 384
        assert stream.identity_labels, name
        for sample in stream:
            detections = sample.observation.metadata["detections"]
            assert detections
            assert all(
                detection["embedding"].shape == (384,) for detection in detections
            )


def test_observations_carry_no_ground_truth():
    _, stream = _first_stream()
    forbidden = {"gt_instance", "instance_id", "label", "gt_category", "gt_id"}
    for sample in stream:
        assert set(sample.observation.metadata) == {
            "frame_index",
            "step",
            "t_s",
            "detections",
        }
        for detection in sample.observation.metadata["detections"]:
            assert not (set(detection) & forbidden)
        # Ground truth lives only on the harness-side sample/stream.
        assert sample.label is None


def test_identity_labels_are_opt_in():
    bundle_dir = find_bundle_dirs(BUNDLE_ROOT)[0]
    without = load_bundle_stream(bundle_dir, max_frames=4, with_gt_identities=False)
    assert without.identity_labels == {}
    with_labels = load_bundle_stream(bundle_dir, max_frames=4, with_gt_identities=True)
    assert with_labels.identity_labels


def test_pss_adapter_contract_passes_on_a_real_bundle():
    _, stream = _first_stream()
    factory = lambda: PSSRuntimeAdapter(  # noqa: E731
        input_dim=stream.dim, public_dim=32, budget_ms=None
    )
    result = ContractProperty().evaluate(factory, stream, seed=0)
    assert result.status == "pass"
    assert all(value == 1.0 for value in result.metrics.values())


def test_identity_retention_runs_on_a_real_bundle():
    _, stream = _first_stream()
    factory = lambda: PSSRuntimeAdapter(  # noqa: E731
        input_dim=stream.dim, public_dim=32, budget_ms=None
    )
    result = IdentityRetentionProperty().evaluate(factory, stream, seed=0)
    assert result.status in {"pass", "fail"}
    retention = result.metrics["identity_retention"]
    assert 0.0 <= retention <= 1.0
    assert result.metrics["n_detections"] > 0


def test_pss_adapter_rejects_wrong_embedding_dim():
    _, stream = _first_stream()
    system = PSSRuntimeAdapter(input_dim=stream.dim, public_dim=32, budget_ms=None)
    with pytest.raises(ValueError):
        system.ingest(stream.samples[0].observation.__class__(
            observation_id="bad",
            modality="vision",
            features=np.zeros(8),
            metadata={
                "detections": (
                    {
                        "ref": "bad",
                        "pos": np.zeros(3),
                        "quat": np.array([1.0, 0.0, 0.0, 0.0]),
                        "score": 1.0,
                        "category": "cube",
                        "embedding": np.zeros(8),
                    },
                )
            },
        ))
