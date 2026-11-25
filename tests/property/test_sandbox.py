from pss.property.sandbox import run_all_l0


def test_l0_property_suite_passes():
    report = run_all_l0(n_observations=60, seed=0)
    failed = [name for name, ok in report["pass"].items() if not ok]
    assert not failed, failed
    assert report["metrics"]["SSR"] == 1.0
    assert report["metrics"]["RBE"] == 1.0
    assert report["metrics"]["BAR"] == 1.0
    assert report["metrics"]["LOC"] == 0.0
    assert report["metrics"]["RBE-KB"] == 1.0
