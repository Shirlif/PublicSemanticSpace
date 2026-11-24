import numpy as np

from pss.binding import (
    build_entity,
    make_slots,
    match_modality,
    rebuild_for_slot,
    unbind_modality,
)


def _unit(rng, d=128):
    v = rng.normal(size=d)
    return v / np.linalg.norm(v)


def test_per_modality_binding_is_separate():
    rng = np.random.default_rng(0)
    slots = make_slots("hadamard", 128, 2)
    feats = {
        "vision": _unit(rng),
        "tactile": _unit(rng),
        "geometry": _unit(rng),
    }
    entity = build_entity("obj0", feats, slots[0], slot_index=0)
    entity.validate()

    for name, true in feats.items():
        assert np.allclose(unbind_modality(entity, name, slots[0]), true, atol=1e-9)

    # The bound vectors are per modality, not a cross-modality sum.
    assert not np.allclose(entity.bound["vision"], entity.bound["tactile"])
    total = entity.bound["vision"] + entity.bound["tactile"] + entity.bound["geometry"]
    assert not np.allclose(entity.bound["vision"], total)


def test_wrong_slot_does_not_recover_feature():
    rng = np.random.default_rng(1)
    slots = make_slots("hadamard", 128, 2)
    feats = {"vision": _unit(rng)}
    entity = build_entity("obj0", feats, slots[0], slot_index=0)
    recovered = unbind_modality(entity, "vision", slots[1])
    assert not np.allclose(recovered, feats["vision"])


def test_rebuild_for_slot_preserves_raw_features():
    rng = np.random.default_rng(2)
    slots = make_slots("hadamard", 128, 2)
    feats = {"vision": _unit(rng), "tactile": _unit(rng)}
    entity = build_entity("obj0", feats, slots[0], slot_index=0)
    moved = rebuild_for_slot(entity, slots[1], slot_index=1)
    assert moved.slot == 1
    for name, true in feats.items():
        assert np.allclose(unbind_modality(moved, name, slots[1]), true, atol=1e-9)


def test_match_modality_ranks_prototypes():
    rng = np.random.default_rng(3)
    slots = make_slots("hadamard", 128, 1)
    prototypes = {"A": _unit(rng), "B": _unit(rng), "C": _unit(rng)}
    entity = build_entity("obj0", {"vision": prototypes["A"]}, slots[0], slot_index=0)
    ranked = match_modality(entity, "vision", prototypes, slots[0])
    assert ranked[0][0] == "A"
    assert ranked[0][1] > 0.99
