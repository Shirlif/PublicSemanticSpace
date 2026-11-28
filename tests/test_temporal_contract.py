import numpy as np
import pytest

from pss.memory.perception import ModalityObservation, Observation


def _obs(**kwargs) -> Observation:
    defaults = dict(
        gt_id="obj0",
        pos=np.zeros(3),
        quat=np.array([1.0, 0.0, 0.0, 0.0]),
        appearance=np.zeros(8),
        raw={"vision": np.ones(128)},
        confidence=0.9,
        category="cup",
    )
    defaults.update(kwargs)
    return Observation(**defaults)


def test_legacy_observation_without_streams_works():
    obs = _obs()
    assert obs.valid_raw() == {"vision": obs.raw["vision"]}
    assert obs.temporal_summary()["streams"] == {}


def test_synchronous_streams_are_not_stale():
    obs = _obs()
    obs.add_stream(
        ModalityObservation("vision", np.ones(128), t_capture=10.0, t_arrival=10.01),
        t_ref=10.0,
    )
    obs.add_stream(
        ModalityObservation("tactile", np.ones(128), t_capture=10.0, t_arrival=10.01),
        t_ref=10.0,
    )
    assert obs.stale_modalities == []
    assert set(obs.valid_raw()) == {"vision", "tactile"}
    assert obs.time_skew_ms == {"vision": 0.0, "tactile": 0.0}


def test_delayed_tactile_becomes_stale():
    obs = _obs()
    obs.add_stream(
        ModalityObservation("vision", np.ones(128), t_capture=10.0, t_arrival=10.01),
        t_ref=10.0,
    )
    obs.add_stream(
        ModalityObservation("tactile", np.ones(128), t_capture=9.95, t_arrival=10.01),
        t_ref=10.0,
    )
    stale = obs.refresh_staleness({"vision": 50.0, "tactile": 20.0})
    assert stale == ["tactile"]
    assert "tactile" not in obs.valid_raw()
    assert "vision" in obs.valid_raw()


def test_small_delay_is_within_budget():
    obs = _obs()
    obs.add_stream(
        ModalityObservation("tactile", np.ones(128), t_capture=9.99, t_arrival=10.0),
        t_ref=10.0,
    )
    obs.refresh_staleness({"tactile": 20.0})
    assert obs.stale_modalities == []
    assert "tactile" in obs.valid_raw()


def test_arrival_before_capture_is_rejected():
    stream = ModalityObservation("vision", np.ones(128), t_capture=10.0, t_arrival=9.0)
    with pytest.raises(ValueError):
        stream.validate()
    obs = _obs()
    with pytest.raises(ValueError):
        obs.add_stream(stream, t_ref=10.0)


def test_invalid_stream_is_excluded_even_if_fresh():
    obs = _obs()
    obs.add_stream(
        ModalityObservation(
            "tactile", np.ones(128), t_capture=10.0, t_arrival=10.0, valid=False
        ),
        t_ref=10.0,
    )
    assert obs.stale_modalities == ["tactile"]
    assert obs.valid_raw() == {}


def test_registry_budget_is_used_by_default():
    obs = _obs()
    # tactile budget in contracts/modalities.yaml is 20 ms.
    obs.add_stream(
        ModalityObservation("tactile", np.ones(128), t_capture=9.9, t_arrival=10.0),
        t_ref=10.0,
    )
    obs.refresh_staleness()
    assert obs.stale_modalities == ["tactile"]
