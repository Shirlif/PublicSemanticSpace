import numpy as np
import pytest

from pss.experts import (
    CategoryCodebook,
    NaNExpert,
    NonNormalizedExpert,
    OracleExpert,
    RandomExpert,
    WrongDimExpert,
    make_quality_expert,
    manipulation_check,
    validate_expert,
)
from pss.observations import make_scene


def test_oracle_satisfies_contract():
    scene = make_scene(8, dim=128, seed=0)
    expert = OracleExpert(scene.codebook, seed=0)
    assert validate_expert(expert, 128, probe=scene.observations()[:4])
    assert expert.health().calibration_ece == 0.0


def test_invalid_modules_are_rejected():
    scene = make_scene(4, dim=128, seed=1)
    obs = scene.observations()[:1]
    for expert in (
        WrongDimExpert(scene.codebook),
        NonNormalizedExpert(scene.codebook),
        NaNExpert(scene.codebook),
    ):
        with pytest.raises(Exception):
            validate_expert(expert, 128, probe=obs)


def test_quality_levels_measure_close_to_target():
    scene = make_scene(200, dim=128, seed=2)
    obs = scene.observations()
    checks = {
        "Q0": (1.00, 0.02),
        "Q1": (0.90, 0.10),
        "Q2": (0.75, 0.12),
        "Q3": (0.55, 0.14),
    }
    for level, (target, tol) in checks.items():
        expert = make_quality_expert(level, scene.codebook, seed=0)
        result = manipulation_check(expert, obs, scene.codebook)
        assert abs(result["accuracy"] - target) <= tol, (level, result)


def test_random_is_near_chance():
    scene = make_scene(200, dim=128, seed=3)
    obs = scene.observations()
    expert = RandomExpert(scene.codebook, seed=0)
    result = manipulation_check(expert, obs, scene.codebook)
    chance = 1.0 / len(scene.codebook.categories)
    assert result["accuracy"] <= chance * 3


def test_codebook_roundtrip_and_unit_norm():
    cb = CategoryCodebook(["a", "b", "c"], dim=128, seed=4)
    for category in cb.categories:
        vec = cb.vector(category)
        assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-6)
        assert cb.predict(vec) == category
