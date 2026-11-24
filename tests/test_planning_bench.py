import numpy as np
import torch

from pss.eval.planning_bench import (
    GOAL,
    NAMES,
    RegionPolicy,
    clear_mask,
    library_from_mask,
    plan_from,
    reachable,
    run,
)


def test_mask_and_plan():
    mask = clear_mask([])
    assert mask.shape == (len(NAMES), len(NAMES))
    assert np.allclose(mask, mask.T)
    assert reachable(mask, "A")
    plan = plan_from(mask, "A")
    assert plan and plan[-1] == GOAL
    library = library_from_mask(mask)
    assert "move_A_G" in library.operators


def test_policy_forward_shape():
    model = RegionPolicy(d_model=16, layers=1, heads=2)
    mask = torch.as_tensor(clear_mask([])[None], dtype=torch.float32)
    logits = model(mask, torch.tensor([0]), torch.tensor([NAMES.index(GOAL)]))
    assert logits.shape == (1, len(NAMES))


def test_planning_run_smoke():
    metrics = run(n_train=8, n_val=3, n_test=4, epochs=2, seeds=1)
    agg = metrics["runs"][0]
    assert agg["id"]["pss"]["success"] >= 0.0
    assert agg["id"]["n"] == 4
