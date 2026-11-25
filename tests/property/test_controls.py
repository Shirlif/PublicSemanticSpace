from pss.property.controls import run_negative_controls
from pss.property.sandbox import run_interaction_suite


def test_negative_controls_behave_as_expected():
    report = run_negative_controls(seed=0)
    assert report["all_passed"], [
        c["id"] for c in report["controls"] if not c["passed"]
    ]


def test_interaction_suite_passes():
    result = run_interaction_suite(seed=0)
    assert result.metrics["interaction_pass_rate"] == 1.0
    assert result.metrics["HK_rollback"] == 1.0
    assert result.metrics["HC_ok"] == 1.0
    assert result.metrics["KC_ok"] == 1.0
    assert result.metrics["HKC_ok"] == 1.0
