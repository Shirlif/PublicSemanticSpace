import numpy as np

from pss.eval.relational_perception_bench import REGION_COORDS, _nearest_region


def test_nearest_region_roundtrip():
    for i, xy in enumerate(REGION_COORDS):
        assert _nearest_region(np.asarray(xy)) == i
