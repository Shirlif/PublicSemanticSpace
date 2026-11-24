"""Smoke tests for the detector-replacement evaluation paths."""

from __future__ import annotations

import numpy as np
import torch

from pss.eval.scripts.run_tao_detector import _iou_track, _match


def test_centernet_shapes_and_depth_target():
    from pss.eval.scripts.train_clevr_detector import HM, CenterNetLite, _targets

    model = CenterNetLite().eval()
    with torch.no_grad():
        heat, depth = model(torch.zeros(2, 3, 256, 256))
    assert heat.shape == (2, 1, HM, HM)
    assert depth.shape == (2, 1, HM, HM)

    hm, dep = _targets(np.array([[32.0, 32.0]], dtype=np.float32), np.array([10.0], dtype=np.float32))
    assert hm.shape == (HM, HM)
    assert float(hm.max()) > 0.99
    assert float(dep.sum()) == 1.0


def test_center_matching_respects_threshold():
    from pss.eval.scripts.train_clevr_detector import _match_centers

    pred = [[10.0, 10.0], [100.0, 100.0]]
    gts = [[11.0, 10.0], [180.0, 180.0]]
    matched = _match_centers(pred, gts, threshold=14.0)
    assert matched == {0: (0, 1.0)}


def test_tao_detector_iou_tracker_keeps_ids():
    records = [
        {"image_id": "1", "boxes": [[10.0, 10.0, 20.0, 20.0]]},
        {"image_id": "2", "boxes": [[12.0, 10.0, 20.0, 20.0]]},
        {"image_id": "3", "boxes": [[100.0, 100.0, 20.0, 20.0]]},
    ]
    assignments = _iou_track(records)
    assert assignments["1"]["det_0"] == assignments["2"]["det_0"]
    assert assignments["3"]["det_0"] != assignments["1"]["det_0"]


def test_tao_detection_gt_match():
    dets = [[10.0, 10.0, 20.0, 20.0], [50.0, 50.0, 10.0, 10.0]]
    gts = [[11.0, 10.0, 20.0, 20.0]]
    matched = _match(dets, gts, threshold=0.5)
    assert matched == {0: (0, matched[0][1])}
    assert matched[0][1] > 0.8
