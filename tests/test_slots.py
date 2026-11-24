from pss.binding import SlotPool, make_slots


def test_acquire_release_and_eviction():
    pool = SlotPool(make_slots("hadamard", 128, 4))
    for i in range(4):
        idx, evicted = pool.acquire(f"e{i}", priority=float(i))
        assert idx == i
        assert evicted is None
    assert len(pool) == 4

    idx, evicted = pool.acquire("e4", priority=10.0)
    assert evicted == "e0"  # lowest priority
    assert pool.is_active("e4")
    assert not pool.is_active("e0")
    assert pool.slot_index("e4") == idx

    released = pool.release("e1")
    assert released == 1
    assert not pool.is_active("e1")


def test_reacquire_reuses_slot():
    pool = SlotPool(make_slots("hadamard", 128, 4))
    idx, _ = pool.acquire("e0")
    idx2, evicted = pool.acquire("e0", priority=5.0)
    assert idx2 == idx
    assert evicted is None
    assert pool.active() == {"e0": idx}
