import numpy as np
import pytest

from pss.contracts import PUBLIC_DIM, Entity, PrototypeNode, Relation, l2_normalize


def _unit() -> np.ndarray:
    v = np.ones(PUBLIC_DIM)
    return v / np.linalg.norm(v)


def test_entity_valid():
    e = Entity(
        "obj0",
        bound={"vision": _unit()},
        raw={"vision": _unit()},
        conf={"vision": 0.9},
    )
    e.validate()


def test_entity_wrong_dim():
    e = Entity("obj0", bound={"vision": np.ones(8)})
    with pytest.raises(ValueError):
        e.validate()


def test_entity_bad_status():
    e = Entity("obj0", bound={"vision": _unit()}, status="unknown")
    with pytest.raises(ValueError):
        e.validate()


def test_relation_requires_three_parts():
    r = Relation("a", "in", "b", bound_parts={"subj": _unit()})
    with pytest.raises(ValueError):
        r.validate()


def test_relation_valid():
    r = Relation(
        "a",
        "in",
        "b",
        bound_parts={"subj": _unit(), "rel": _unit(), "obj": _unit()},
        fact=_unit(),
    )
    r.validate()


def test_prototype_unit_norm():
    PrototypeNode("p", "苹果", modality_protos={"vision": _unit()}).validate()
    with pytest.raises(ValueError):
        PrototypeNode("p2", "苹果", modality_protos={"vision": np.ones(PUBLIC_DIM)}).validate()


def test_l2_normalize_rejects_zero():
    with pytest.raises(ValueError):
        l2_normalize(np.zeros(4))
