import numpy as np

from pss.reasoning.relations import relations_from_frame
from pss.reasoning.relations.contact import contact_pairs


def _box(x, half=0.05):
    return {
        "geom_type": "box",
        "size": [half, half, half],
    }, np.array([x, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])


def test_box_box_contact_and_separation():
    a, pa = _box(0.0)
    b, pb = _box(0.10)
    assert contact_pairs([a, b], np.stack([pa, pb])) == [(0, 1)]

    b_far, pb_far = _box(0.11)
    assert contact_pairs([a, b_far], np.stack([pa, pb_far])) == []


def test_capsule_box_contact():
    cap = {"geom_type": "capsule", "size": [0.01, 0.06]}
    box = {"geom_type": "box", "size": [0.05, 0.05, 0.05]}
    # Capsule axis is local x. Endpoint reaches x=0.10-0.06=0.04, box face at 0.05.
    poses = np.array(
        [
            [0.10, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ]
    )
    assert contact_pairs([cap, box], poses) == [(0, 1)]


def test_sphere_box_contact_and_gap():
    sphere = {"geom_type": "sphere", "size": [0.02]}
    box = {"geom_type": "box", "size": [0.05, 0.05, 0.05]}
    touching = np.array(
        [
            [0.069, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ]
    )
    assert contact_pairs([sphere, box], touching) == [(0, 1)]

    gap = touching.copy()
    gap[0, 0] = 0.071
    assert contact_pairs([sphere, box], gap) == []


def test_relations_from_frame_uses_collision_geometry():
    sample = {
        "instance_ids": ["a", "b"],
        "categories": ["box", "box"],
        "poses": np.array(
            [
                [
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.10, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                ]
            ]
        ),
        "object_specs": [
            {"instance_id": "a", "geom_type": "box", "size": [0.05, 0.05, 0.05]},
            {"instance_id": "b", "geom_type": "box", "size": [0.05, 0.05, 0.05]},
        ],
    }
    records = relations_from_frame(sample, 0, {"box": (0.05, 0.05, 0.05)})
    contacts = {r.as_tuple() for r in records if r.predicate == "contact"}
    assert contacts == {("a", "contact", "b"), ("b", "contact", "a")}
    assert all(r.source == "collision_geometry" for r in records if r.predicate == "contact")
