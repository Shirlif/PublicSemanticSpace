from hypothesis import given, settings, strategies as st

from pss.knowledge import TruthMaintenanceSystem
from pss.property.kb import VersionedKnowledgeBase


FACT = st.tuples(
    st.text(min_size=1, max_size=8),
    st.text(min_size=1, max_size=8),
    st.text(min_size=1, max_size=8),
    st.text(min_size=1, max_size=8),
)


@settings(max_examples=50, deadline=None)
@given(st.lists(FACT, max_size=8))
def test_assert_retract_identity_hypothesis(facts):
    kb = VersionedKnowledgeBase()
    before = kb.state_hash()
    for i, (entity, attribute, value, source) in enumerate(facts):
        kb.assert_fact(entity, attribute, value, source=f"{source}-{i}")
    for i, (entity, attribute, value, source) in enumerate(facts):
        kb.retract(entity, attribute, source=f"{source}-{i}")
    assert kb.state_hash() == before


@settings(max_examples=50, deadline=None)
@given(
    st.lists(
        st.tuples(
            st.sampled_from(["assert", "update", "retract"]),
            st.text(min_size=1, max_size=6),
            st.text(min_size=1, max_size=6),
            st.text(min_size=1, max_size=6),
            st.text(min_size=1, max_size=6),
        ),
        max_size=12,
    ),
    st.integers(0, 20),
)
def test_rollback_exactness_hypothesis(operations, target):
    kb = VersionedKnowledgeBase()
    hashes = [(0, kb.state_hash())]
    for action, entity, attribute, value, source in operations:
        if action == "assert":
            kb.assert_fact(entity, attribute, value, source=source)
        elif action == "update":
            kb.update(entity, attribute, value, source=source)
        else:
            kb.retract(entity, attribute, source=source)
        hashes.append((kb.snapshot()["index"], kb.state_hash()))
    recorded = [i for i, _ in hashes]
    index = min(target, recorded[-1])
    if index not in recorded:
        index = max(i for i in recorded if i <= index)
    expected = dict(hashes)[index]
    kb.rollback(index)
    assert kb.state_hash() == expected


@settings(max_examples=50, deadline=None)
@given(
    st.lists(
        st.tuples(st.integers(0, 5), st.integers(0, 5)),
        max_size=8,
    ),
    st.sets(st.integers(0, 5), max_size=3),
)
def test_tms_support_matches_reachability(edges, assumptions):
    # Keep only forward edges so the graph is a DAG.
    edges = [(i, j) for i, j in edges if i < j]
    tms = TruthMaintenanceSystem()
    for i in assumptions:
        tms.assume((f"n{i}", "v", "1"))
    for i, j in edges:
        tms.justify(
            (f"n{j}", "v", "1"),
            in_nodes=[(f"n{i}", "v", "1")],
            source="rule",
        )

    # Reachability from assumptions along forward edges.
    reachable = set(assumptions)
    changed = True
    while changed:
        changed = False
        for i, j in edges:
            if i in reachable and j not in reachable:
                reachable.add(j)
                changed = True

    for i in range(6):
        assert tms.label((f"n{i}", "v", "1")) == (i in reachable)


@settings(max_examples=50, deadline=None)
@given(
    st.lists(st.tuples(st.integers(0, 5), st.integers(0, 5)), max_size=8),
    st.sets(st.integers(0, 5), max_size=3),
)
def test_tms_source_retraction(edges, assumptions):
    edges = [(i, j) for i, j in edges if i < j]
    tms = TruthMaintenanceSystem()
    for i in assumptions:
        tms.assume((f"n{i}", "v", "1"))
    for i, j in edges:
        tms.justify(
            (f"n{j}", "v", "1"),
            in_nodes=[(f"n{i}", "v", "1")],
            source="rule",
        )
    tms.remove_source("rule")
    for i in range(6):
        assert tms.label((f"n{i}", "v", "1")) == (i in assumptions)
