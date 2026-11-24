"""Action operators: soft preconditions + add/remove effects (Phase 7).

v1 is hand-authored. The laundry task uses the same 12-step structure as the
6.7 demo, expressed as operators so a planner can search instead of replaying a
script.
"""

from __future__ import annotations

from dataclasses import dataclass

Fact = tuple


@dataclass(frozen=True)
class Operator:
    name: str
    preconditions: frozenset[Fact]
    add: frozenset[Fact]
    remove: frozenset[Fact]
    cost: float = 1.0

    def applicable(self, facts: frozenset) -> bool:
        return self.preconditions <= facts

    def apply(self, facts: frozenset) -> frozenset:
        return frozenset((facts - self.remove) | self.add)


class OperatorLibrary:
    def __init__(self, operators: list[Operator]):
        self.operators = {op.name: op for op in operators}

    def get(self, name: str) -> Operator:
        return self.operators[name]

    def applicable(
        self,
        facts: frozenset,
        confidence: dict[Fact, float] | None = None,
        threshold: float = 0.5,
    ) -> list[tuple[Operator, float]]:
        out = []
        for op in self.operators.values():
            if not op.preconditions <= facts:
                continue
            score = 1.0
            if confidence:
                confs = [confidence.get(p, 1.0) for p in op.preconditions]
                if confs:
                    if min(confs) < threshold:
                        continue
                    score = sum(confs) / len(confs)
            out.append((op, score))
        return sorted(out, key=lambda x: (-x[1], x[0].name))

    def apply(self, op: Operator, facts: frozenset) -> frozenset:
        return op.apply(facts)


def laundry_library() -> OperatorLibrary:
    F = frozenset
    return OperatorLibrary(
        [
            Operator(
                "pick_clothes",
                F({("clothes", "in", "basket")}),
                F({("clothes", "held")}),
                F({("clothes", "in", "basket")}),
            ),
            Operator(
                "pick_detergent",
                F({("detergent", "on", "counter")}),
                F({("detergent", "held")}),
                F({("detergent", "on", "counter")}),
            ),
            Operator(
                "pick_detergent_floor",
                F({("detergent", "on", "floor")}),
                F({("detergent", "held")}),
                F({("detergent", "on", "floor")}),
            ),
            Operator(
                "open_washer",
                F({("door", "closed"), ("machine", "stopped")}),
                F({("door", "open")}),
                F({("door", "closed")}),
            ),
            Operator(
                "put_in_clothes",
                F({("clothes", "held"), ("door", "open")}),
                F({("clothes", "in", "washer")}),
                F({("clothes", "held")}),
            ),
            Operator(
                "pour_detergent",
                F({("detergent", "held"), ("door", "open")}),
                F({("detergent", "in", "washer")}),
                F({("detergent", "held")}),
            ),
            Operator(
                "close_washer",
                F({("door", "open")}),
                F({("door", "closed")}),
                F({("door", "open")}),
            ),
            Operator(
                "fill_water",
                F({("door", "closed"), ("water", "drained"), ("clothes", "in", "washer")}),
                F({("water", "in", "washer")}),
                F({("water", "drained")}),
            ),
            Operator(
                "press_start",
                F(
                    {
                        ("door", "closed"),
                        ("water", "in", "washer"),
                        ("clothes", "in", "washer"),
                        ("detergent", "in", "washer"),
                        ("machine", "stopped"),
                    }
                ),
                F({("machine", "running")}),
                F({("machine", "stopped")}),
            ),
            Operator(
                "wait",
                F({("machine", "running")}),
                F({("machine", "stopped"), ("water", "drained"), ("clothes", "clean")}),
                F({("machine", "running"), ("water", "in", "washer"), ("clothes", "dry")}),
            ),
            Operator(
                "take_out",
                F({("clothes", "in", "washer"), ("door", "open")}),
                F({("clothes", "held")}),
                F({("clothes", "in", "washer")}),
            ),
            Operator(
                "hang",
                F({("clothes", "held"), ("drying_rack", "empty")}),
                F({("clothes", "on", "drying_rack")}),
                F({("clothes", "held"), ("drying_rack", "empty")}),
            ),
        ]
    )


def initial_laundry_facts() -> frozenset:
    return frozenset(
        {
            ("clothes", "in", "basket"),
            ("clothes", "dry"),
            ("detergent", "on", "counter"),
            ("door", "closed"),
            ("machine", "stopped"),
            ("water", "drained"),
            ("drying_rack", "empty"),
        }
    )


def laundry_goal() -> frozenset:
    # Require both the wash result and the final location; otherwise the
    # planner can trivially pick clothes from the basket and hang them.
    return frozenset({("clothes", "on", "drying_rack"), ("clothes", "clean")})


def hand_written_plan() -> list[str]:
    return [
        "pick_clothes",
        "open_washer",
        "put_in_clothes",
        "pick_detergent",
        "pour_detergent",
        "close_washer",
        "fill_water",
        "press_start",
        "wait",
        "open_washer",
        "take_out",
        "hang",
    ]


def simulate_plan(plan: list[str], initial: frozenset, library: OperatorLibrary, anomaly_step=None, anomaly_fn=None):
    facts = frozenset(initial)
    log = []
    for i, name in enumerate(plan):
        op = library.get(name)
        if not op.preconditions <= facts:
            return facts, log, False
        facts = op.apply(facts)
        if anomaly_step is not None and i == anomaly_step and anomaly_fn is not None:
            facts = frozenset(anomaly_fn(facts))
        log.append((i, name, facts))
    return facts, log, True


def spill_anomaly(facts: frozenset) -> frozenset:
    return frozenset(
        (facts - {("detergent", "in", "washer")}) | {("detergent", "on", "floor")}
    )
