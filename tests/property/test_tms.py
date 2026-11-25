from pss.knowledge import TruthMaintenanceSystem


def test_default_reasoning_is_nonmonotonic():
    tms = TruthMaintenanceSystem()
    tms.assume(("Tweety", "bird", "yes"))
    tms.justify(
        ("Tweety", "flies", "yes"),
        in_nodes=[("Tweety", "bird", "yes")],
        out_nodes=[("Tweety", "abnormal", "yes")],
        source="default",
        rule="birds_fly",
    )
    assert tms.label(("Tweety", "flies", "yes")) is True

    tms.assume(("Tweety", "abnormal", "yes"))
    assert tms.label(("Tweety", "flies", "yes")) is False

    tms.unassume(("Tweety", "abnormal", "yes"))
    assert tms.label(("Tweety", "flies", "yes")) is True


def test_retract_source_propagates_to_dependents():
    tms = TruthMaintenanceSystem()
    tms.assume(("A", "v", "1"))
    tms.justify(("B", "v", "1"), in_nodes=[("A", "v", "1")], source="r1")
    tms.justify(("C", "v", "1"), in_nodes=[("B", "v", "1")], source="r2")
    assert tms.label(("C", "v", "1")) is True

    tms.remove_source("r1")
    assert tms.label(("B", "v", "1")) is False
    assert tms.label(("C", "v", "1")) is False
    # A is an assumption and survives; only the dependent chain is retracted.
    assert tms.label(("A", "v", "1")) is True


def test_conflict_keys_reports_multiple_supported_values():
    tms = TruthMaintenanceSystem()
    tms.assume(("cup", "color", "red"))
    tms.justify(("cup", "color", "blue"), source="sensor_b")
    conflicts = tms.conflict_keys()
    assert ("cup", "color") in conflicts
    assert set(conflicts[("cup", "color")]) == {"red", "blue"}


def test_cycles_are_not_well_founded():
    tms = TruthMaintenanceSystem()
    tms.justify(("A", "v", "1"), in_nodes=[("B", "v", "1")], source="s1")
    tms.justify(("B", "v", "1"), in_nodes=[("A", "v", "1")], source="s2")
    assert tms.label(("A", "v", "1")) is False
    assert tms.label(("B", "v", "1")) is False


def test_rollback_restores_previous_state():
    tms = TruthMaintenanceSystem()
    tms.assume(("A", "v", "1"))
    tms.justify(("B", "v", "1"), in_nodes=[("A", "v", "1")], source="r1")
    assert tms.label(("B", "v", "1")) is True
    tms.rollback(0)
    assert tms.label(("A", "v", "1")) is False
    assert tms.label(("B", "v", "1")) is False
