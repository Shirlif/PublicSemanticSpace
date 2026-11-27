"""Tests for the paired Version B protocol."""

from __future__ import annotations

import numpy as np

from pss.eval.relational_bench import TEST_PAIRS
from pss.eval.scripts.run_version_b_paired import (
    build_episodes,
    evaluate_pss,
    paired_bootstrap,
)


def test_build_episodes_is_deterministic():
    first = build_episodes(TEST_PAIRS, 3, seed=7)
    second = build_episodes(TEST_PAIRS, 3, seed=7)
    assert first == second
    assert len(first) == len(TEST_PAIRS) * 3


def test_build_episodes_use_distinct_states_per_pair():
    episodes = build_episodes(TEST_PAIRS, 4, seed=7)
    assert all(len(ep["state"]) == 3 for ep in episodes)
    assert {ep["rel1"] for ep in episodes} == {p[0] for p in TEST_PAIRS}


def test_paired_bootstrap_zero_delta():
    ci = paired_bootstrap(np.zeros(50), n_boot=500)
    assert ci["mean"] == 0.0
    assert ci["lo"] == 0.0
    assert ci["hi"] == 0.0
    assert ci["n"] == 50


def test_paired_bootstrap_constant_delta_excludes_zero():
    ci = paired_bootstrap(np.ones(50), n_boot=500)
    assert ci["mean"] == 1.0
    assert ci["lo"] == 1.0
    assert ci["hi"] == 1.0


def test_paired_bootstrap_is_paired_not_independent():
    # Halves: +1 on the first 25 episodes, -1 on the last 25.  The mean is 0,
    # but a paired bootstrap still sees a wide interval around 0.
    delta = np.concatenate([np.ones(25), -np.ones(25)])
    ci = paired_bootstrap(delta, n_boot=2000, seed=3)
    assert ci["mean"] == 0.0
    assert ci["lo"] < 0.0 < ci["hi"]


def test_evaluate_pss_solves_constructed_episodes():
    episodes = build_episodes(TEST_PAIRS, 3, seed=11)
    ok = evaluate_pss(None, episodes)
    assert ok.shape == (len(episodes),)
    assert ok.all()
