"""Tests for the relation-coverage accounting."""

from __future__ import annotations

import pytest

from pss.reasoning.relations.coverage import (
    axis_isolation,
    convex_hull,
    coverage_features,
    coverage_report,
    direction_coverage,
    hull_distance,
    ordered_pairs,
    pairs_within,
    point_in_convex_hull,
    sign_pattern,
    sign_pattern_coverage,
    subsets_by_size,
    unseen_pairs,
)

REGIONS = ("A", "B", "C", "D")
GRID = {
    "A": (-1.0, 0.0),
    "B": (1.0, 0.0),
    "C": (0.0, 1.0),
    "D": (0.0, -1.0),
    "G": (0.0, 0.0),
}


def test_ordered_pairs_excludes_self_pairs():
    pairs = ordered_pairs(REGIONS)
    assert len(pairs) == 12
    assert ("A", "A") not in pairs
    assert ("A", "B") in pairs


def test_pairs_within_uses_both_endpoints():
    pairs = pairs_within(("A", "B"))
    assert pairs == [("A", "B"), ("B", "A")]


def test_coverage_report_support_and_fractions():
    report = coverage_report(REGIONS, pairs_within(("A", "B")))
    assert report.region_support == {"A": 2, "B": 2, "C": 0, "D": 0}
    assert report.region_fraction == 0.5
    assert abs(report.fraction - 2 / 12) < 1e-9
    assert report.total_pairs == 12


def test_coverage_report_counts_support_per_pair():
    report = coverage_report(REGIONS, [("A", "B")], support_per_pair=300)
    assert report.pair_support[("A", "B")] == 300
    assert report.region_support["A"] == 300


def test_unseen_pairs_are_the_complement():
    seen = {("A", "B"), ("B", "A")}
    complement = unseen_pairs(REGIONS, ("A", "B"))
    assert set(complement) == set(ordered_pairs(REGIONS)) - seen


def test_subsets_by_size_preserves_region_order():
    subsets = subsets_by_size(REGIONS, [2])[2]
    assert len(subsets) == 6
    assert subsets[0] == ("A", "B")
    assert subsets[-1] == ("C", "D")
    assert all(tuple(sorted(s, key=REGIONS.index)) == s for s in subsets)


# --------------------------------------------------------------------------- #
# B1: coordinate-span coverage
# --------------------------------------------------------------------------- #
def test_convex_hull_contains_centre_and_rejects_outside_point():
    hull = convex_hull(GRID[r] for r in ("A", "B", "C", "D"))
    assert len(hull) == 4
    assert point_in_convex_hull((0.0, 0.0), hull)
    assert not point_in_convex_hull((2.0, 0.0), hull)


def test_hull_distance_is_zero_inside_and_positive_outside():
    hull = convex_hull(GRID[r] for r in ("A", "B", "C", "D"))
    assert hull_distance((0.0, 0.0), hull) == 0.0
    assert hull_distance((2.0, 0.0), hull) == pytest.approx(1.0)


def test_axis_isolation_separates_acg_from_bcd():
    # ACG has A-G (varies x, y fixed) and C-G (varies y, x fixed).
    assert axis_isolation(("A", "C", "G"), GRID) == 2
    # BCD has C-D (varies y) but no pair that varies x with y fixed.
    assert axis_isolation(("B", "C", "D"), GRID) == 1


def test_direction_coverage_reports_span_and_sides():
    dc = direction_coverage(("A", "C", "G"), GRID)
    assert dc.x_span == pytest.approx(1.0)
    assert dc.y_span == pytest.approx(1.0)
    assert dc.x_negative and not dc.x_positive
    assert dc.y_positive and not dc.y_negative
    assert dc.axis_isolation == 2


def test_sign_pattern_marks_dead_zone():
    assert sign_pattern("A", "G", GRID) == (1, 0)
    assert sign_pattern("C", "G", GRID) == (0, -1)


def test_sign_pattern_coverage_detects_unseen_diagonal():
    # ACG trains on (+,+), (-,-), (+,0), (-,0), (0,+), (0,-); it never sees
    # (+,-) or (-,+).  B->C is (-,+) and C->B is (+,-), both unseen.
    pairs = [("B", "C"), ("C", "B")]
    assert sign_pattern_coverage(("A", "C", "G"), pairs, GRID) == 0.0
    # A-B is (+, 0), which ACG does see through A-G.
    assert sign_pattern_coverage(("A", "C", "G"), [("A", "B")], GRID) == 1.0


def test_coverage_features_shape_and_hull_counts():
    features = coverage_features(("B", "C", "D"), tuple(GRID), GRID)
    assert features["n_regions"] == 3
    assert features["axis_isolation"] == 1
    assert features["n_held_out"] == 2
    # A is outside the BCD triangle; G sits on the C-D edge.
    assert features["n_outside_hull"] == 1
    assert features["max_hull_distance"] == pytest.approx(1.0)
    assert 0.0 <= features["sign_pattern_seen"] <= 1.0
