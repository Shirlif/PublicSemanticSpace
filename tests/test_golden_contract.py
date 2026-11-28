import json

import numpy as np
import pytest

from pss.eval.scripts.freeze_golden_contract import (
    FIXTURE,
    GOLDEN_MODALITIES,
    build_payload,
)


pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(),
    reason="golden fixture missing; run python -m pss.eval.scripts.freeze_golden_contract --write",
)


def _load():
    return np.load(FIXTURE, allow_pickle=False)


def test_golden_modal_ids_are_the_three_incumbents():
    with _load() as f:
        assert json.loads(str(f["modalities_json"])) == list(GOLDEN_MODALITIES)


def test_prototypes_unchanged():
    payload = build_payload(0)
    with _load() as f:
        for modality in GOLDEN_MODALITIES:
            np.testing.assert_allclose(
                payload[f"proto_{modality}"],
                f[f"proto_{modality}"],
                rtol=0.0,
                atol=1e-6,
                err_msg=f"prototype drift for {modality}",
            )


def test_joint_readout_scores_unchanged():
    payload = build_payload(0)
    with _load() as f:
        np.testing.assert_allclose(
            payload["joint_scores"],
            f["joint_scores"],
            rtol=0.0,
            atol=1e-6,
            err_msg="joint readout score drift",
        )


def test_prototype_recognition_unchanged():
    payload = build_payload(0)
    with _load() as f:
        assert list(payload["proto_top1"]) == list(f["proto_top1"])
        assert payload["proto_ranked"].tolist() == f["proto_ranked"].tolist()
