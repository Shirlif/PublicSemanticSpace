"""Per-modality contract checks for every registered active modality."""

import numpy as np
import pytest

from pss.contracts import PUBLIC_DIM, Entity, get_modalities, get_spec
from pss.memory.perception import ModalityObservation


@pytest.mark.parametrize("modality", get_modalities())
def test_registered_modality_spec_is_valid(modality):
    spec = get_spec(modality)
    assert spec.name == modality
    assert spec.kind
    assert spec.preprocess
    assert spec.normalize in {"l2", "none", "standardize"}
    assert spec.time_unit in {"frame", "ms", "s"}
    assert spec.missing_policy in {"skip", "impute", "raise"}
    assert spec.calibration in {"temperature", "none"}
    assert 0.0 <= spec.reliability_prior <= 1.0
    assert spec.max_latency_ms >= 0
    assert spec.version


@pytest.mark.parametrize("modality", get_modalities())
def test_entity_accepts_one_public_vector_per_modality(modality):
    vector = np.ones(PUBLIC_DIM) / np.sqrt(PUBLIC_DIM)
    Entity("obj0", raw={modality: vector}, bound={modality: vector}).validate()


@pytest.mark.parametrize("modality", get_modalities())
def test_modality_observation_timing_fields(modality):
    stream = ModalityObservation(
        modality=modality,
        vector=np.zeros(4),
        t_capture=1.0,
        t_arrival=1.01,
        sensor_id=f"{modality}-0",
        quality=0.9,
    )
    stream.validate()
    assert stream.clock == "monotonic"
    assert stream.sensor_id == f"{modality}-0"


def test_experimental_modality_is_explicitly_opt_in():
    assert "audio" not in get_modalities()
    assert "audio" in get_modalities(include_experimental=True)
    assert get_spec("audio").status == "experimental"


def test_unknown_modality_is_rejected():
    with pytest.raises(Exception):
        get_spec("sonar")
