"""Wave 0 contract tests: the calibration pipeline seam is frozen.

These tests do not check behaviour -- no stage is implemented yet.  They pin
the interface so the five Wave 1 tracks can develop against it in parallel and
so an accidental signature change fails loudly.

See ``docs/plans/PSS-校准抗干扰融合设计.md``.
"""

from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from pss.projection.calibration_pipeline import (
    PIPELINE_ORDER,
    CalibrationBatch,
    Evidence,
    GateDecision,
    Params,
    SampleWeights,
    State,
    bound_step,
    classify_state,
    fit_params,
    gate,
    score_samples,
    source_health,
)
from pss.projection.calibration_preregistration import (
    EXPECTED_ROBUSTNESS,
    check_robustness_preregistration,
)


def _batch(n: int = 3) -> CalibrationBatch:
    return CalibrationBatch(
        embeddings={"vision": np.zeros((n, 4)), "geometry": np.zeros((n, 4))},
        prototypes={"vision": np.zeros((2, 4)), "geometry": np.zeros((2, 4))},
        labels=np.zeros(n, dtype=np.int64),
        sources=["a"] * n,
        sample_ids=[f"s{i}" for i in range(n)],
    )


def test_pipeline_order_is_frozen() -> None:
    # gate must run after bound_step: it validates what will actually ship.
    assert PIPELINE_ORDER == (
        "source_health",
        "score_samples",
        "fit_params",
        "bound_step",
        "gate",
    )


def test_state_enum_values() -> None:
    assert {s.value for s in State} == {
        "normal",
        "quarantine",
        "recalibrate",
        "abstain",
    }


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Params(temperature={"vision": 1.0}, reliability={"vision": 0.9}),
        lambda: Evidence(
            cross_modal_agreement=1.0,
            label_consensus_agreement=1.0,
            drift_score=0.0,
            suspect_mass=0.0,
        ),
        lambda: SampleWeights(per_sample=np.ones(3)),
        lambda: GateDecision(
            accept=True, integrity_ok=True, adaptation_ok=True, reason="ok"
        ),
    ],
)
def test_contract_dataclasses_are_frozen(factory) -> None:
    value = factory()
    field_name = next(iter(value.__dataclass_fields__))
    with pytest.raises(FrozenInstanceError):
        setattr(value, field_name, None)


def test_batch_rejects_misaligned_sources() -> None:
    with pytest.raises(ValueError):
        CalibrationBatch(
            embeddings={"vision": np.zeros((3, 4))},
            prototypes={"vision": np.zeros((2, 4))},
            labels=np.zeros(3, dtype=np.int64),
            sources=["a", "b"],  # too short
            sample_ids=["s0", "s1", "s2"],
        )


def test_batch_rejects_misaligned_embeddings() -> None:
    with pytest.raises(ValueError):
        CalibrationBatch(
            embeddings={"vision": np.zeros((2, 4))},
            prototypes={"vision": np.zeros((2, 4))},
            labels=np.zeros(3, dtype=np.int64),
            sources=["a", "b", "c"],
            sample_ids=["s0", "s1", "s2"],
        )


def test_expected_parameter_names() -> None:
    expected = {
        source_health: ["batch", "min_source_n"],
        score_samples: ["batch", "params", "source_trust", "state"],
        fit_params: ["batch", "weights"],
        bound_step: ["old", "new", "evidence"],
        gate: ["old", "new", "gate_set", "window_holdout", "thresholds"],
        classify_state: ["evidence", "thresholds"],
    }
    for fn, names in expected.items():
        actual = list(inspect.signature(fn).parameters)
        assert actual == names, f"{fn.__name__}: {actual} != {names}"


def test_stages_are_implemented() -> None:
    # Wave 0 pinned NotImplementedError; Wave 1 lands the bodies.  The seam is
    # still frozen by the signature tests above.
    batch = _batch()
    params = Params(temperature={"vision": 1.0}, reliability={"vision": 0.9})
    evidence = Evidence(1.0, 1.0, 0.0, 0.0)

    trust = source_health(batch)
    weights = score_samples(batch, params, trust, state=State.NORMAL)
    assert len(weights.per_sample) == len(batch.labels)
    assert set(fit_params(batch, weights).temperature) == {"vision", "geometry"}
    assert set(bound_step(params, params, evidence).temperature) == {"vision"}
    assert gate(params, params, batch, batch).accept in (True, False)
    assert isinstance(classify_state(evidence), State)


def test_preregistration_covers_required_scenarios() -> None:
    required = {
        "clean",
        "random_label_flip@0.05",
        "random_label_flip@0.20",
        "targeted_agree@0.10",
        "targeted_single_class@0.20",
        "source_outlier@0.30",
        "drift_sigma@0.30",
        "ood_unseen",
    }
    assert required <= set(EXPECTED_ROBUSTNESS)


def test_preregistration_caps_are_positive() -> None:
    for scenario, exp in EXPECTED_ROBUSTNESS.items():
        for cap in (exp.ece_map_max, exp.signed_gap_max, exp.recovery_samples_max):
            assert cap is None or cap > 0, f"{scenario}: non-positive cap"
        if exp.accuracy_min is not None:
            assert 0.0 < exp.accuracy_min <= 1.0, f"{scenario}: bad accuracy floor"


def test_clean_expectation_is_the_reference() -> None:
    clean = EXPECTED_ROBUSTNESS["clean"]
    assert clean.detected is False
    assert clean.abstain is False
    # Poison scenarios must not be more permissive than clean.
    assert clean.ece_map_max <= EXPECTED_ROBUSTNESS["random_label_flip@0.20"].ece_map_max


def test_preregistration_targets_the_confidence_channel() -> None:
    # Temperature-ECE is no longer the calibration claim; the map is.
    for scenario, exp in EXPECTED_ROBUSTNESS.items():
        if exp.ece_map_max is not None:
            assert exp.ece_map_max > 0, scenario
    assert EXPECTED_ROBUSTNESS["drift_sigma@0.30"].ece_map_max > (
        EXPECTED_ROBUSTNESS["clean"].ece_map_max
    )


def test_check_accepts_conforming_observation() -> None:
    observed = {
        "clean": {
            "detected": False,
            "abstain": False,
            "ece_map_fused": 0.020,
            "signed_gap_map_fused": -0.010,
            "fused_accuracy": 0.950,
        },
        "random_label_flip@0.20": {
            "detected": True,
            "accept_update": False,
            "ece_map_fused": 0.030,
        },
        "drift_sigma@0.30": {
            "detected": True,
            "ece_map_fused": 0.100,
            "fused_accuracy": 0.650,
            "recovery_samples": 22,
        },
    }
    result = check_robustness_preregistration(observed)
    assert result["ok"] is True, result["mismatches"]


def test_check_flags_accepting_a_poisoned_update() -> None:
    observed = {
        "random_label_flip@0.20": {
            "detected": True,
            "accept_update": True,  # pre-registered as False
            "ece_map_fused": 0.030,
        }
    }
    result = check_robustness_preregistration(observed)
    assert result["ok"] is False
    assert any("accept_update" in m for m in result["mismatches"])


def test_check_flags_ece_over_cap() -> None:
    observed = {
        "random_label_flip@0.20": {
            "detected": True,
            "accept_update": False,
            "ece_map_fused": 0.400,
        }
    }
    result = check_robustness_preregistration(observed)
    assert result["ok"] is False
    assert any("ece_map_fused" in m for m in result["mismatches"])


def test_check_flags_accuracy_below_floor() -> None:
    observed = {
        "drift_sigma@0.30": {
            "detected": True,
            "ece_map_fused": 0.100,
            "fused_accuracy": 0.200,  # floor is 0.60
        }
    }
    result = check_robustness_preregistration(observed)
    assert result["ok"] is False
    assert any("fused_accuracy" in m for m in result["mismatches"])


def test_check_flags_signed_gap() -> None:
    observed = {
        "clean": {
            "detected": False,
            "ece_map_fused": 0.020,
            "signed_gap_map_fused": 0.300,  # cap is 0.05
        }
    }
    result = check_robustness_preregistration(observed)
    assert result["ok"] is False
    assert any("signed_gap" in m for m in result["mismatches"])


def test_check_ignores_unregistered_scenarios() -> None:
    observed = {"some_new_scenario": {"detected": True, "ece_map_fused": 99.0}}
    assert check_robustness_preregistration(observed)["ok"] is True
