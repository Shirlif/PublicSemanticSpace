"""Spatial suite sanity tests: determinism, leakage, label consistency."""

from __future__ import annotations

import json

from pss.eval.spatial import predicates as P
from pss.eval.spatial.baselines import ExactSolver, RandomBaseline, solve_exact
from pss.eval.spatial.generator import ALL_FAMILIES, generate_suite
from pss.eval.spatial.schema import parse_answer, parse_final_answer, scene_payload, scene_to_json


def test_suite_is_deterministic():
    a = generate_suite(n_per_family=4, seed=7, families=ALL_FAMILIES)
    b = generate_suite(n_per_family=4, seed=7, families=ALL_FAMILIES)
    assert [x.to_record() for x in a] == [x.to_record() for x in b]


def test_scene_payload_has_no_answer_or_relation_fields():
    items = generate_suite(n_per_family=4, seed=3)
    for item in items:
        payload = scene_payload(item.scene)
        assert "answer" not in json.dumps(payload)
        assert "relations" not in payload
        assert "split" not in payload
        assert "ood" not in payload
        assert "scene_id" not in payload
        # No pairwise object entries.
        for obj in payload["objects"]:
            assert "relations" not in obj
            assert "distance" not in obj


def test_exact_solver_matches_generator_on_all_families():
    items = generate_suite(n_per_family=6, seed=11)
    solver = ExactSolver()
    mismatches = []
    for item in items:
        predicted = solver(item)
        if predicted != item.query.answer:
            mismatches.append((item.query.family, item.query.question, item.query.answer, predicted))
    assert not mismatches, mismatches[:5]


def test_each_family_is_generated():
    items = generate_suite(n_per_family=5, seed=19)
    families = {item.query.family for item in items}
    assert families == set(ALL_FAMILIES)


def test_bool_answers_are_balanced_per_family():
    items = generate_suite(n_per_family=6, seed=23)
    by_family = {}
    for item in items:
        if item.query.kind != "bool":
            continue
        counts = by_family.setdefault(item.query.family, {"yes": 0, "no": 0})
        counts[item.query.answer] += 1
    for family, counts in by_family.items():
        assert counts["yes"] > 0 and counts["no"] > 0, (family, counts)


def test_count_and_ordinal_answers_are_valid():
    items = generate_suite(n_per_family=8, seed=29)
    for item in items:
        if item.query.kind == "count":
            assert 0 <= int(item.query.answer) <= len(item.scene.objects) - 1
        if item.query.kind == "object":
            obj = item.scene.obj(item.query.answer)
            assert obj.category == item.query.meta["category"]


def test_parse_answer_is_strict():
    assert parse_answer("yes", "bool") == "yes"
    assert parse_answer("No.", "bool") == "no"
    assert parse_answer("maybe", "bool") is None
    assert parse_answer("3", "count") == "3"
    assert parse_answer("3 objects", "count") is None
    assert parse_answer("obj_2", "object") == "obj_2"
    assert parse_answer("the answer is obj_2", "object") is None


def test_parse_final_answer_handles_cot():
    assert parse_final_answer("First, x is greater.\nAnswer: yes", "bool") == "yes"
    assert parse_final_answer("There are 3.\n\nAnswer: 3", "count") == "3"
    assert parse_final_answer("Answer: obj_1", "object") == "obj_1"
    assert parse_final_answer("maybe", "bool") is None


def test_predicates_basic_geometry():
    from pss.eval.spatial.schema import Camera, ObjectSpec, Scene

    cam = Camera("cam0", (0.0, -1.0, 0.08), (0.0, 0.0, 0.0))
    a = ObjectSpec("obj_0", "cup", (0.1, 0.0, 0.05), (0.04, 0.04, 0.05))
    b = ObjectSpec("obj_1", "cup", (-0.1, 0.0, 0.05), (0.04, 0.04, 0.05))
    scene = Scene("s", (a, b), (cam,), (-0.3, 0.3, -0.2, 0.2))
    assert P.left_of(scene, b, a, "world")
    assert P.right_of(scene, a, b, "world")


def test_occlusion_and_containment_labels():
    from pss.eval.spatial.schema import Camera, ObjectSpec, Scene

    cam = Camera("cam0", (0.0, -1.0, 0.08), (0.0, 0.0, 0.0))
    front = ObjectSpec("obj_0", "box", (0.0, 0.0, 0.035), (0.06, 0.045, 0.035))
    back = ObjectSpec("obj_1", "cup", (0.0, 0.15, 0.05), (0.04, 0.04, 0.05))
    scene = Scene("s", (front, back), (cam,), (-0.3, 0.3, -0.2, 0.2))
    assert P.occluded(scene, back, front, "cam0")

    container = ObjectSpec(
        "obj_0", "bin", (0.0, 0.0, 0.06), (0.08, 0.08, 0.06), container=True, inner_radius=0.06
    )
    small = ObjectSpec("obj_1", "block", (0.0, 0.0, 0.036), (0.035, 0.035, 0.035))
    scene = Scene("s2", (container, small), (cam,), (-0.3, 0.3, -0.2, 0.2))
    assert P.inside(small, container)
    assert P.fits(small, container)


def test_random_baseline_is_serializable():
    items = generate_suite(n_per_family=2, seed=31)
    baseline = RandomBaseline(seed=0)
    for item in items[:5]:
        answer = baseline(item)
        assert isinstance(answer, str)
