from __future__ import annotations

from pathlib import Path

from pss.isaac.runner import _sample_layout
from pss.spec import load_spec

REPO = Path(__file__).resolve().parents[1]


def test_fixed_policy_does_not_move_assets():
    spec = load_spec(REPO / "specs" / "s1_occlusion.yaml")
    assert _sample_layout(spec, 0, 0.77) == {}


def test_lhs_layout_is_unique_inside_workspace_and_seed_dependent():
    spec = load_spec(REPO / "specs" / "s1_occlusion_20.yaml")
    layout0 = _sample_layout(spec, 0, 0.77)
    layout1 = _sample_layout(spec, 1, 0.77)
    assert len(layout0) == 20
    assert len({tuple(round(v, 6) for v in pos) for pos in layout0.values()}) == 20
    for pos in layout0.values():
        assert -0.30 <= pos[0] <= 0.30
        assert -0.25 <= pos[1] <= 0.25
        assert pos[2] > 0.77
    assert layout0 != layout1
    assert layout0 == _sample_layout(spec, 0, 0.77)
