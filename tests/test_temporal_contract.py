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


def _timed_obs(tactile_time: float) -> Observation:
    raw = {
        "vision": np.ones(128),
        "geometry": np.ones(128),
        "tactile": np.ones(128),
    }
    obs = _obs(raw=raw)
    obs.add_stream(
        ModalityObservation("vision", raw["vision"], 10.0, 10.0), t_ref=10.0
    )
    obs.add_stream(
        ModalityObservation("geometry", raw["geometry"], 10.0, 10.0), t_ref=10.0
    )
    obs.add_stream(
        ModalityObservation("tactile", raw["tactile"], tactile_time, 10.0),
        t_ref=10.0,
    )
    obs.refresh_staleness({"vision": 50.0, "geometry": 50.0, "tactile": 20.0})
    return obs


def test_entity_tokens_exclude_stale_streams():
    from pss.memory.learned_metric import pad_observations

    fresh = _timed_obs(10.0)
    stale = _timed_obs(8.0)
    assert stale.stale_modalities == ["tactile"]

    tokens_stale, mask_stale = pad_observations([stale], include_tactile=True)
    tokens_fresh, mask_fresh = pad_observations([fresh], include_tactile=True)
    assert mask_stale[0] == mask_fresh[0] == 1.0
    assert not np.allclose(tokens_stale[0], tokens_fresh[0])
    # The stale token must equal the fresh token with the tactile block zeroed.
    tactile_dim = 128  # pss.memory.learned_metric.TACTILE_DIM
    prefix = tokens_fresh.shape[1] - tactile_dim
    assert np.allclose(tokens_stale[0][:prefix], tokens_fresh[0][:prefix])
    assert np.allclose(tokens_stale[0][prefix:], 0.0)


def test_stale_tactile_does_not_change_entity_identity():
    from pss.runtime.layers import EntityLayer

    layer = EntityLayer(use_default_relations=False, use_default_active=False)
    first = _timed_obs(10.0)
    layer.ingest([first], 0)
    track_id = next(iter(layer.graph.tracks))
    assert "tactile" in layer.graph.tracks[track_id].raw

    second = _timed_obs(8.0)
    second.frame = 1
    layer.ingest([second], 1)
    assert next(iter(layer.graph.tracks)) == track_id
    assert "tactile" not in layer.graph.tracks[track_id].raw

