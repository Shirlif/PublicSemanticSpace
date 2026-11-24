from pss.knowledge import MultiSourceFactStore, Rule, VersionedRuleEngine


def test_multisource_priority_and_rollback():
    store = MultiSourceFactStore()
    store.assert_fact("cup", "material", "glass", "a", priority=0)
    store.assert_fact("cup", "material", "metal", "b", priority=2)
    assert store.resolve("cup", "material") == "metal"
    v = store.version
    store.delete_entity("cup")
    assert store.resolve("cup", "material") is None
    store.rollback(v)
    assert store.resolve("cup", "material") == "metal"
    assert store.conflicts() == {("cup", "material"): ["glass", "metal"]}


def test_versioned_rule_update_and_rollback():
    engine = VersionedRuleEngine(
        [Rule("fragile", lambda s, e: s.get(e, "material") == "glass", "fragile")]
    )
    from pss.knowledge import FactStore

    store = FactStore({("cup", "material"): "ceramic"})
    assert engine.answer(store, "cup", "fragile") == "no"
    engine.update(
        Rule("fragile", lambda s, e: s.get(e, "material") in ("glass", "ceramic"), "fragile")
    )
    assert engine.answer(store, "cup", "fragile") == "yes"
    engine.rollback(0)
    assert engine.answer(store, "cup", "fragile") == "no"
