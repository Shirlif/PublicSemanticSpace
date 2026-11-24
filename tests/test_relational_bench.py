import numpy as np
import torch

from pss.eval.relational_bench import (
    ACTIONS,
    NAMES,
    RelationalPolicy,
    goal_ok,
    plan,
)


def test_plan_reaches_goal():
    start = (0, 1, 2)
    solution = plan(start, "left", "above")
    assert solution is not None
    state = list(start)
    for obj, region in solution:
        state[obj] = region
    assert goal_ok(tuple(state), "left", "above")


def test_compositional_pairs_have_solutions():
    for rel1, rel2 in (("left", "below"), ("right", "above")):
        assert plan((0, 1, 2), rel1, rel2) is not None


def test_policy_forward():
    model = RelationalPolicy(d_model=32, layers=1, heads=2)
    states = torch.as_tensor([[0, 1, 2]], dtype=torch.long)
    logits = model(states, torch.tensor([0]), torch.tensor([1]))
    assert logits.shape == (1, len(ACTIONS))
