from pss.knowledge import FactStore, Rule, RuleEngine


def test_fact_update_rollback_and_audit():
    store = FactStore({("cup", "material"): "glass"})
    store.update("cup", "material", "metal", reason="test")
    assert store.get("cup", "material") == "metal"
    assert store.version == 1
    store.rollback(0)
    assert store.get("cup", "material") == "glass"
    assert any(entry.action == "rollback" for entry in store.audit)


def test_rule_engine_derives():
    store = FactStore(
        {
            ("cup", "material"): "glass",
            ("cup", "location"): "shelf",
        }
    )
    engine = RuleEngine(
        [
            Rule("fragile", lambda s, e: s.get(e, "material") == "glass", "fragile"),
            Rule(
                "needs_padding",
                lambda s, e: s.get(e, "material") == "glass"
                and s.get(e, "location") == "shelf",
                "needs_padding",
            ),
        ]
    )
    assert engine.answer(store, "cup", "fragile") == "yes"
    assert engine.answer(store, "cup", "needs_padding") == "yes"
