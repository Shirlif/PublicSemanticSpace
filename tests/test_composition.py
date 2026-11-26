"""Tests for the typed composition layer and the minimal loop."""

from __future__ import annotations

from pss.reasoning.composition import (
    Candidate,
    ConstraintAtom,
    PickAtom,
    PlanCache,
    ProjectAtom,
    SelectAtom,
    Signature,
    TypeMatchAtom,
    best_first,
    instantiate,
    plan_template,
)


class Op:
    def __init__(self, name, input_type, output_type, effect=None):
        self.name = name
        self.signature = Signature(input_type, output_type, cost=1.0)
        self.effect = effect or (lambda payload: None)

    def applicable(self, payload):
        return True

    def apply(self, payload):
        out = dict(payload)
        out["type"] = self.signature.output_type
        self.effect(out)
        return out


def test_typed_search_finds_the_chain():
    atoms = [
        Op("a", "start", "mid"),
        Op("b", "mid", "final"),
    ]

    def goal(payload):
        return payload.get("type") == "final"

    result = best_first({"type": "start"}, atoms, goal)
    assert result.success
    assert result.steps == ["a", "b"]


def test_type_check_rejects_an_invalid_sequence():
    def mark_ok(payload):
        payload["ok"] = True

    def mark_bad(payload):
        payload["ok"] = False

    atoms = [
        Op("good", "start", "mid", mark_bad),
        Op("skip", "start", "wrong", mark_ok),
        Op("finish", "mid", "final"),
    ]

    def goal(payload):
        return payload.get("type") == "final" and payload.get("ok") is True

    typed = best_first({"type": "start"}, atoms, goal, check_types=True)
    untyped = best_first({"type": "start"}, atoms, goal, check_types=False)

    assert typed.success is False
    assert untyped.success is True
    assert untyped.steps == ["skip", "finish"]


def test_plan_template_round_trip():
    template = plan_template(["frame:viewer", "place:s0", "move:cup", "check"], subject="cup")
    assert template == ("frame:viewer", "place:s0", "move:$subject", "check")
    assert instantiate(template, subject="mug") == [
        "frame:viewer",
        "place:s0",
        "move:mug",
        "check",
    ]


def test_cache_replay_rejects_stale_plan():
    class Pick:
        name = "pick"
        signature = Signature("state", "state")

        def applicable(self, payload):
            return "item" in payload["facts"]

        def apply(self, payload):
            out = dict(payload)
            out["facts"] = payload["facts"] | {"held"}
            return out

    atom = Pick()
    cache = PlanCache()
    key = ("goal", "held")
    goal = lambda payload: "held" in payload["facts"]  # noqa: E731

    initial = {"type": "state", "facts": {"item"}}
    result = best_first(initial, [atom], goal)
    assert result.success
    cache.put(key, result.steps)

    assert cache.get(key, {"pick": atom}, initial, goal) is not None
    assert cache.get(key, {"pick": atom}, {"type": "state", "facts": set()}, goal) is None
    assert cache.stats.stale == 1


def test_composition_loop_smoke():
    from pss.eval.scripts.run_composition_loop import run

    metrics = run(n_laundry=2, n_spatial=3, seed=0)
    typed = metrics["strategies"]["typed"]["overall"]
    direct = metrics["strategies"]["direct"]["overall"]
    assert typed["success_rate"] >= direct["success_rate"]
    assert typed["false_claim_rate"] == 0.0
    assert metrics["tasks"]["laundry"] >= 1
    assert metrics["cache"]["stores"] >= 0


def test_selection_pipeline_requires_every_filter():
    candidates = [
        Candidate("target", {"category": "cup", "color": "red", "size": 0.10, "occupied": False}),
        Candidate("blue", {"category": "cup", "color": "blue", "size": 0.10, "occupied": False}),
        Candidate("bowl", {"category": "bowl", "color": "red", "size": 0.10, "occupied": False}),
        Candidate("big", {"category": "cup", "color": "red", "size": 0.30, "occupied": False}),
        Candidate("used", {"category": "cup", "color": "red", "size": 0.10, "occupied": True}),
    ]
    atoms = [
        SelectAtom("cup", "category", "cup"),
        SelectAtom("red", "color", "red"),
        ProjectAtom("size", lambda candidate: 0.08 <= float(candidate.get("size")) <= 0.12),
        ConstraintAtom("free", lambda candidate: not bool(candidate.get("occupied"))),
        PickAtom("pick", require_unique=True),
    ]
    result = best_first(
        {"type": "candidates", "candidates": candidates},
        atoms,
        lambda payload: payload.get("selected") == "target",
        max_steps=6,
    )
    assert result.success
    assert len(result.steps) == 5


def test_type_match_atom_filters_by_membership():
    class Constraint:
        def membership(self, query):
            return float(query.get("sim", 0.0))

    candidates = [
        Candidate("a", {"embedding": {"sim": 0.9}}),
        Candidate("b", {"embedding": {"sim": 0.2}}),
    ]
    atom = TypeMatchAtom("match", Constraint(), threshold=0.5)
    out = atom.apply({"type": "candidates", "candidates": candidates})
    assert [candidate.id for candidate in out["candidates"]] == ["a"]


def test_cache_stores_only_verified_plans():
    cache = PlanCache()
    key = ("k",)
    cache.record(key, ["a"], verified=False)
    assert cache.entries[key].steps == ()
    assert cache.stats.rejected == 1

    cache.record(key, ["a"], verified=True)
    assert cache.entries[key].steps == ("a",)
    assert cache.entries[key].support == 1
    cache.record(key, ["a"], verified=True)
    assert cache.confidence(key) > 0.5


def test_cache_coverage_and_rollback():
    cache = PlanCache()
    cache.record(("k",), ["a"], verified=True)
    snapshot = cache.snapshot()
    cache.record(("j",), ["b"], verified=True)

    assert abs(cache.coverage([("k",), ("j",), ("missing",)]) - 2 / 3) < 1e-9
    cache.rollback(snapshot)
    assert ("k",) in cache.entries
    assert ("j",) not in cache.entries
