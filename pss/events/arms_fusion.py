"""Fusion arms for the contact-event benchmark.

Arms
----
``fusion``
    Calibrated logistic fusion over the clean channels (see
    ``pss.events.fusion``), including the clean, oracle-like force channel.
``fusion_abstain``
    ``fusion`` plus a calibration-set abstain threshold.
``force_noisy``
    The force channel alone with the ``NoisyForceExtractor`` degradation
    (relative noise, delay, quantization, dropout).  GT stays clean, so this
    is a non-oracle single-channel baseline.
``fusion_noisy``
    Fusion of ``force_noisy`` with the remaining physical channels
    (geometry, tactile).  Audio is excluded because the synthetic audio proxy
    is derived from the *clean* force and would leak the oracle; vision is a
    zero placeholder.  Parameters are loaded from the noisy calibration file.

Parameters are loaded once per path from ``PSS_FUSION_PARAMS`` /
``PSS_FUSION_NOISY_PARAMS`` when set, otherwise from the C2 result
directories.  Missing files fall back to ``FusionParams`` defaults.
"""

from __future__ import annotations

import os
from pathlib import Path

from .bench_api import register_arm, run_channel
from .fusion import DEFAULT_PARAMS_PATH, FusionParams, fuse_events
from .sensor_models import NoisyForceExtractor

DEFAULT_NOISY_PARAMS_PATH = (
    Path(__file__).resolve().parents[2]
    / "eval"
    / "results"
    / "contact_event"
    / "c2_noisy"
    / "fusion_noisy_params.json"
)

_CACHE: dict[str, FusionParams] = {}


def _params_path() -> str:
    return os.environ.get("PSS_FUSION_PARAMS", str(DEFAULT_PARAMS_PATH))


def _noisy_params_path() -> str:
    return os.environ.get("PSS_FUSION_NOISY_PARAMS", str(DEFAULT_NOISY_PARAMS_PATH))


def get_params() -> FusionParams:
    path = _params_path()
    if path not in _CACHE:
        _CACHE[path] = FusionParams.load(path)
    return _CACHE[path]


def get_noisy_params() -> FusionParams:
    path = _noisy_params_path()
    if path not in _CACHE:
        _CACHE[path] = FusionParams.load(path)
    return _CACHE[path]


def reset_cache() -> None:
    _CACHE.clear()


def _noisy_force(config) -> NoisyForceExtractor:
    return NoisyForceExtractor(seed=int(config.seed))


@register_arm("fusion")
def fusion_arm(scenario, config):
    return fuse_events(scenario, config, get_params())


@register_arm("fusion_abstain")
def fusion_abstain_arm(scenario, config):
    return fuse_events(scenario, config, get_params(), abstain=True)


@register_arm("force_noisy")
def force_noisy_arm(scenario, config):
    return run_channel(scenario, config, channel=_noisy_force(config))


@register_arm("fusion_noisy")
def fusion_noisy_arm(scenario, config):
    return fuse_events(
        scenario,
        config,
        get_noisy_params(),
        force_extractor=_noisy_force(config),
    )


__all__ = [
    "DEFAULT_NOISY_PARAMS_PATH",
    "force_noisy_arm",
    "fusion_arm",
    "fusion_abstain_arm",
    "fusion_noisy_arm",
    "get_noisy_params",
    "get_params",
    "reset_cache",
]
