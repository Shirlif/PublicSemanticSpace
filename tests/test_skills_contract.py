from __future__ import annotations

from pss.isaac.faults import FAULT_KINDS, FaultInjector
from pss.isaac.skills import ARM_JOINTS, STATUSES, SkillResult


def test_fault_engine_covers_ten_kinds():
    assert len(FAULT_KINDS) == 10
    assert "occlusion_insert" in FAULT_KINDS
    assert "path_blocked" in FAULT_KINDS


def test_unknown_fault_is_rejected():
    injector = FaultInjector(stage=None, assets={}, executor=None)
    try:
        injector.inject("nope")
    except ValueError:
        return
    raise AssertionError("unknown fault should raise")


def test_skill_contract():
    assert len(ARM_JOINTS) == 7
    assert "ok" in STATUSES and "unsupported" in STATUSES
    result = SkillResult("reach", "ok", 3, "err=0.01")
    assert result.name == "reach" and result.status == "ok" and result.steps == 3
