"""Region coverage accounting for learned relation predicates.

A learned relation head generalises to a region only if that region appeared
in the pairs it was trained on.  The P5b result makes this concrete: a head
trained on regions A/B/C reaches exact-4 0.571 and planner OOD 0.682, while the
same planner with a full-coverage head reaches 1.000.

This module keeps the accounting explicit and benchmark-free: given a region
list and the ordered pairs used for training it reports per-pair support,
per-region support and the coverage fraction.  The sweep over region subsets
lives in ``pss/eval/scripts/run_relation_coverage.py``.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TypeAlias

Coord: TypeAlias = tuple[float, float]

__all__ = [
    "Coord",
    "CoverageReport",
    "DirectionCoverage",
    "axis_isolation",
    "convex_hull",
    "coverage_features",
    "coverage_report",
    "direction_coverage",
    "hull_area",
    "hull_distance",
    "ordered_pairs",
    "pairs_within",
    "point_in_convex_hull",
    "sign_pattern",
    "sign_pattern_coverage",
    "subsets_by_size",
    "unseen_pairs",
]


def ordered_pairs(regions) -> list[tuple[str, str]]:
    """All ordered region pairs ``(a, b)`` with ``a != b``."""

    regions = list(regions)
    return [(a, b) for a in regions for b in regions if a != b]


def pairs_within(regions) -> list[tuple[str, str]]:
    """Ordered pairs where both endpoints are in ``regions``."""

    return ordered_pairs(regions)


def unseen_pairs(regions, train_regions) -> list[tuple[str, str]]:
    """Ordered pairs with at least one endpoint outside ``train_regions``."""

    train = set(train_regions)
    return [(a, b) for a, b in ordered_pairs(regions) if a not in train or b not in train]


@dataclass(frozen=True)
class CoverageReport:
    regions: tuple[str, ...]
    pair_support: dict[tuple[str, str], int]
    region_support: dict[str, int]
    covered_pairs: tuple[tuple[str, str], ...]
    total_pairs: int

    @property
    def fraction(self) -> float:
        return len(self.covered_pairs) / max(self.total_pairs, 1)

    @property
    def region_fraction(self) -> float:
        covered = {r for pair in self.covered_pairs for r in pair}
        return len(covered) / max(len(self.regions), 1)


def coverage_report(
    regions,
    train_pairs,
    support_per_pair: int = 1,
) -> CoverageReport:
    """Build the support matrix for one training configuration.

    ``train_pairs`` is the ordered region pairs the head was trained on.
    ``support_per_pair`` is the number of labelled samples per pair (the P5b
    generator uses a fixed count, so a scalar is enough).
    """

    regions = tuple(regions)
    train_pairs = [tuple(p) for p in train_pairs]
    pair_support = {pair: int(support_per_pair) for pair in train_pairs}
    region_support = {r: 0 for r in regions}
    for a, b in train_pairs:
        region_support[a] = region_support.get(a, 0) + int(support_per_pair)
        region_support[b] = region_support.get(b, 0) + int(support_per_pair)
    return CoverageReport(
        regions=regions,
        pair_support=pair_support,
        region_support=region_support,
        covered_pairs=tuple(train_pairs),
        total_pairs=len(ordered_pairs(regions)),
    )


def subsets_by_size(regions, sizes) -> dict[int, list[tuple[str, ...]]]:
    """All region subsets of each requested size, in the input region order."""

    regions = tuple(regions)
    order = {r: i for i, r in enumerate(regions)}
    out: dict[int, list[tuple[str, ...]]] = {}
    for size in sizes:
        size = int(size)
        out[size] = [
            tuple(sorted(combo, key=order.__getitem__))
            for combo in itertools.combinations(regions, size)
        ]
    return out


# --------------------------------------------------------------------------- #
# coordinate-span coverage (B1)
# --------------------------------------------------------------------------- #
#
# Region count is a coarse proxy: the B0 sweep shows two 3/5 subsets landing at
# OOD 1.000 and 0.280.  These helpers describe *where* the training regions sit,
# not how many there are.  They are benchmark-free: callers pass a coordinate
# table (``region -> (x, y)``).


def _cross(o: Coord, a: Coord, b: Coord) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def convex_hull(points: Iterable[Coord]) -> tuple[Coord, ...]:
    """Counter-clockwise convex hull (monotone chain), collinear points dropped."""

    pts = sorted({(float(x), float(y)) for x, y in points})
    if len(pts) <= 2:
        return tuple(pts)
    lower: list[Coord] = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[Coord] = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return (pts[0], pts[-1])
    return tuple(hull)


def _on_segment(p: Coord, a: Coord, b: Coord, tol: float = 1e-9) -> bool:
    if abs(_cross(a, b, p)) > tol:
        return False
    return (
        min(a[0], b[0]) - tol <= p[0] <= max(a[0], b[0]) + tol
        and min(a[1], b[1]) - tol <= p[1] <= max(a[1], b[1]) + tol
    )


def point_in_convex_hull(point: Coord, hull, tol: float = 1e-9) -> bool:
    """True if ``point`` is inside or on the boundary of ``hull``."""

    hull = tuple(hull)
    p = (float(point[0]), float(point[1]))
    if len(hull) == 1:
        return abs(p[0] - hull[0][0]) <= tol and abs(p[1] - hull[0][1]) <= tol
    if len(hull) == 2:
        return _on_segment(p, hull[0], hull[1], tol)
    sign = 0
    for i, a in enumerate(hull):
        b = hull[(i + 1) % len(hull)]
        c = _cross(a, b, p)
        if abs(c) <= tol:
            continue
        s = 1 if c > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True


def hull_distance(point: Coord, hull, tol: float = 1e-9) -> float:
    """Distance from ``point`` to the hull: 0 inside, positive outside."""

    hull = tuple(hull)
    if point_in_convex_hull(point, hull, tol):
        return 0.0
    if not hull:
        return float("nan")
    if len(hull) == 1:
        return float(((point[0] - hull[0][0]) ** 2 + (point[1] - hull[0][1]) ** 2) ** 0.5)
    best = float("inf")
    edges = (
        [(hull[0], hull[1])]
        if len(hull) == 2
        else [(hull[i], hull[(i + 1) % len(hull)]) for i in range(len(hull))]
    )
    for a, b in edges:
        abx, aby = b[0] - a[0], b[1] - a[1]
        denom = abx * abx + aby * aby
        t = 0.0 if denom == 0 else ((point[0] - a[0]) * abx + (point[1] - a[1]) * aby) / denom
        t = max(0.0, min(1.0, t))
        dx = point[0] - (a[0] + t * abx)
        dy = point[1] - (a[1] + t * aby)
        best = min(best, (dx * dx + dy * dy) ** 0.5)
    return float(best)


def hull_area(hull) -> float:
    """Shoelace area of a convex hull; 0 for degenerate hulls."""

    hull = tuple(hull)
    if len(hull) < 3:
        return 0.0
    total = 0.0
    for i, a in enumerate(hull):
        b = hull[(i + 1) % len(hull)]
        total += a[0] * b[1] - b[0] * a[1]
    return abs(total) / 2.0


def _has_isolating_pair(regions, coords: Mapping[str, Coord], axis: str, eps: float) -> bool:
    """Is there an ordered pair that varies ``axis`` while the other stays put?"""

    for a, b in ordered_pairs(regions):
        dx = coords[b][0] - coords[a][0]
        dy = coords[b][1] - coords[a][1]
        if axis == "x" and abs(dx) > eps and abs(dy) <= eps:
            return True
        if axis == "y" and abs(dy) > eps and abs(dx) <= eps:
            return True
    return False


@dataclass(frozen=True)
class DirectionCoverage:
    """Coordinate-span view of one training subset.

    ``x_negative`` .. ``y_positive`` say whether the subset has a point on each
    side of the reference (the centroid of all regions by default).
    ``x_isolating`` / ``y_isolating`` say whether a training pair varies one
    axis while the other stays in the dead zone.  A relation head needs those
    isolating pairs to learn ``left/right`` and ``above/below`` separately;
    without them the predicate is entangled with the other axis.
    """

    regions: tuple[str, ...]
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    x_negative: bool
    x_positive: bool
    y_negative: bool
    y_positive: bool
    x_isolating: bool
    y_isolating: bool

    @property
    def x_span(self) -> float:
        return self.x_max - self.x_min

    @property
    def y_span(self) -> float:
        return self.y_max - self.y_min

    @property
    def axis_isolation(self) -> int:
        return int(self.x_isolating) + int(self.y_isolating)

    @property
    def directions(self) -> int:
        return sum((self.x_negative, self.x_positive, self.y_negative, self.y_positive))


def direction_coverage(
    regions,
    coords: Mapping[str, Coord],
    reference: Coord | None = None,
    eps: float = 1e-9,
) -> DirectionCoverage:
    """Coordinate-span signature of ``regions`` under ``coords``."""

    regions = tuple(regions)
    if not regions:
        raise ValueError("direction_coverage needs at least one region")
    points = [tuple(map(float, coords[r])) for r in regions]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    if reference is None:
        all_points = [tuple(map(float, p)) for p in coords.values()]
        reference = (
            sum(p[0] for p in all_points) / len(all_points),
            sum(p[1] for p in all_points) / len(all_points),
        )
    rx, ry = float(reference[0]), float(reference[1])
    return DirectionCoverage(
        regions=regions,
        x_min=min(xs),
        x_max=max(xs),
        y_min=min(ys),
        y_max=max(ys),
        x_negative=any(x < rx - eps for x in xs),
        x_positive=any(x > rx + eps for x in xs),
        y_negative=any(y < ry - eps for y in ys),
        y_positive=any(y > ry + eps for y in ys),
        x_isolating=_has_isolating_pair(regions, coords, "x", eps),
        y_isolating=_has_isolating_pair(regions, coords, "y", eps),
    )


def axis_isolation(regions, coords: Mapping[str, Coord], eps: float = 1e-9) -> int:
    """Number of axes (0-2) with an axis-isolating training pair."""

    return direction_coverage(regions, coords, eps=eps).axis_isolation


def sign_pattern(
    a: str, b: str, coords: Mapping[str, Coord], eps: float = 1e-9
) -> tuple[int, int]:
    """Sign of ``(dx, dy) = coords[b] - coords[a]`` with a dead zone at zero."""

    dx = coords[b][0] - coords[a][0]
    dy = coords[b][1] - coords[a][1]
    sx = 1 if dx > eps else (-1 if dx < -eps else 0)
    sy = 1 if dy > eps else (-1 if dy < -eps else 0)
    return sx, sy


def sign_pattern_coverage(
    train_regions,
    test_pairs,
    coords: Mapping[str, Coord],
    eps: float = 1e-9,
) -> float:
    """Fraction of ``test_pairs`` whose (sign dx, sign dy) was seen in training.

    ``train_regions`` is the set the head was trained on; all ordered pairs
    within it count as seen.  Empty ``test_pairs`` returns 1.0 by convention.
    """

    train_pairs = ordered_pairs(train_regions)
    seen = {sign_pattern(a, b, coords, eps) for a, b in train_pairs}
    test_pairs = list(test_pairs)
    if not test_pairs:
        return 1.0
    hits = sum(1 for a, b in test_pairs if sign_pattern(a, b, coords, eps) in seen)
    return hits / len(test_pairs)


def coverage_features(
    regions,
    all_regions,
    coords: Mapping[str, Coord],
    eps: float = 1e-9,
) -> dict:
    """All coverage descriptors for one subset, as JSON-friendly scalars.

    This is the input to the offline re-analysis in
    ``pss/eval/scripts/run_relation_coverage.py``: it lets the B0 sweep be
    re-ranked by coordinate geometry without retraining anything.
    """

    regions = tuple(regions)
    all_regions = tuple(all_regions)
    train = set(regions)
    held_out = [r for r in all_regions if r not in train]
    all_points = [tuple(map(float, coords[r])) for r in all_regions]
    reference = (
        sum(p[0] for p in all_points) / len(all_points),
        sum(p[1] for p in all_points) / len(all_points),
    )
    dc = direction_coverage(regions, coords, reference=reference, eps=eps)
    hull = convex_hull([coords[r] for r in regions])
    distances = [hull_distance(coords[r], hull, eps) for r in held_out]
    test_pairs = [
        (a, b)
        for a, b in ordered_pairs(all_regions)
        if a not in train or b not in train
    ]

    gx = [p[0] for p in all_points]
    gy = [p[1] for p in all_points]
    subset_points = [coords[r] for r in regions]
    extremes = sum(
        (
            any(abs(p[0] - min(gx)) <= eps for p in subset_points),
            any(abs(p[0] - max(gx)) <= eps for p in subset_points),
            any(abs(p[1] - min(gy)) <= eps for p in subset_points),
            any(abs(p[1] - max(gy)) <= eps for p in subset_points),
        )
    )

    return {
        "n_regions": len(regions),
        "region_fraction": len(regions) / max(len(all_regions), 1),
        "x_span": float(dc.x_span),
        "y_span": float(dc.y_span),
        "x_negative": dc.x_negative,
        "x_positive": dc.x_positive,
        "y_negative": dc.y_negative,
        "y_positive": dc.y_positive,
        "directions": int(dc.directions),
        "x_isolating": dc.x_isolating,
        "y_isolating": dc.y_isolating,
        "axis_isolation": int(dc.axis_isolation),
        "extremes": int(extremes),
        "hull_area": float(hull_area(hull)),
        "n_held_out": len(held_out),
        "n_outside_hull": int(sum(1 for d in distances if d > eps)),
        "max_hull_distance": float(max(distances) if distances else 0.0),
        "sum_hull_distance": float(sum(distances)),
        "sign_pattern_seen": float(
            sign_pattern_coverage(regions, test_pairs, coords, eps)
        ),
        "n_test_pairs": len(test_pairs),
    }
