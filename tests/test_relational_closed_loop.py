from pss.eval.relational_closed_loop import _gt_state, _make_world, _place_at


def test_place_at_updates_ground_truth():
    scene, _rig, _tracker, state = _make_world(0)
    free = next(i for i in range(5) if i not in state)
    _place_at(scene, 0, free)
    assert _gt_state(scene)[0] == free
