"""Tests for the C0 contact-event foundation."""

from __future__ import annotations

import numpy as np

from pss.events.bench_api import (
    ContactConfig,
    ContactScenario,
    build_scenario,
    evaluate_events,
    run_channel,
)
from pss.events.extractors import ForceExtractor
from pss.events.gt import events_from_contacts
from pss.events.model import ContactEvent, ContactFrame
from pss.events.segmenter import Interval, segment_signal, temporal_iou
from pss.sim.scene import SceneConfig, build_scene
from pss.sim.sensors import SensorRig


def _frame(t, force=0.0, a="a", b="b"):
    contacts = []
    if force > 0:
        contacts.append(
            {
                "instance1": a,
                "instance2": b,
                "normal_force": force,
                "pos": np.array([0.0, 0.0, 0.0]),
            }
        )
    return ContactFrame(
        t=t,
        contacts=contacts,
        tactile={a: np.zeros((4, 4)), b: np.zeros((4, 4))},
        poses={a: np.zeros(3), b: np.zeros(3)},
    )


def test_segment_signal_merges_short_gaps():
    times = np.arange(6) * 0.01
    scores = np.array([0.0, 0.6, 0.7, 0.0, 0.6, 0.0])
    intervals = segment_signal(
        times, scores, threshold=0.5, max_gap=0.02, min_duration=0.001
    )
    assert len(intervals) == 1
    assert intervals[0].start == 0.01
    assert intervals[0].end == 0.04


def test_segment_signal_min_duration():
    times = np.arange(3) * 0.01
    scores = np.array([0.0, 0.9, 0.0])
    assert segment_signal(times, scores, threshold=0.5, min_duration=0.05) == []


def test_events_from_contacts():
    frames = [
        _frame(0.00),
        _frame(0.01, 2.0),
        _frame(0.02, 3.0),
        _frame(0.03, 1.5),
        _frame(0.04),
    ]
    events = events_from_contacts(
        frames, threshold=1.0, min_duration=0.005, max_gap=0.02
    )
    assert len(events) == 1
    event = events[0]
    assert event.pair == ("a", "b")
    assert event.onset == 0.01
    assert event.offset == 0.03
    assert event.intensity == 3.0


def test_evaluate_empty_and_matching():
    gt = [
        ContactEvent("g", "a", "b", onset=0.0, offset=0.1, intensity=2.0)
    ]
    pred = [
        ContactEvent("p", "a", "b", onset=0.01, offset=0.11, intensity=1.8)
    ]
    m = evaluate_events(pred, gt, iou_threshold=0.3)
    assert m["tp"] == 1 and m["fp"] == 0 and m["fn"] == 0
    assert m["f1"] == 1.0
    assert evaluate_events([], [])["f1"] == 1.0
    assert evaluate_events(pred, [])["f1"] == 0.0


def test_temporal_iou():
    assert np.isclose(
        temporal_iou(Interval(0.0, 0.1), Interval(0.05, 0.15)), 1 / 3
    )
    assert temporal_iou(Interval(0.0, 0.1), Interval(0.2, 0.3)) == 0.0


def test_force_extractor_is_normalized():
    frames = [_frame(0.0, 1.0), _frame(0.01, 3.0), _frame(0.02)]
    scores = ForceExtractor(f0=1.0).pair_score(frames, ("a", "b"))
    assert np.isclose(scores[0], 0.5)
    assert np.isclose(scores[1], 0.75)
    assert scores[2] == 0.0


def test_near_miss_scenario_has_no_gt():
    config = ContactConfig(max_steps=40, capture_every=5, render=False, seed=0)
    scenario = build_scenario("near_miss", config)
    assert isinstance(scenario, ContactScenario)
    assert scenario.gt_events == []


def test_strict_metric_rejects_long_interval():
    gt = [ContactEvent("g", "a", "b", onset=0.0, offset=0.1, intensity=2.0)]
    pred = [ContactEvent("p", "a", "b", onset=0.0, offset=1.0, intensity=2.0)]
    loose = evaluate_events(
        pred, gt, iou_threshold=0.3, onset_tolerance=0.05, strict=False
    )
    strict = evaluate_events(
        pred, gt, iou_threshold=0.5, onset_tolerance=0.05,
        offset_tolerance=0.1, strict=True,
    )
    assert loose["f1"] == 1.0
    assert strict["f1"] == 0.0


def test_tactile_maps_ignore_unknown_endpoint():
    scene = build_scene(
        SceneConfig(n_objects=2, seed=0, width=32, height=32, bins=0)
    )
    rig = SensorRig(scene, 32, 32)
    a, b = scene.objects[0].instance_id, scene.objects[1].instance_id
    try:
        rig.contacts = lambda: [
            {
                "instance1": a,
                "instance2": "unknown",
                "normal_force": 5.0,
                "pos": np.zeros(3),
            }
        ]
        maps = rig.tactile_maps(size=4)
        assert np.allclose(maps[a], 0.0)
        assert np.allclose(maps[b], 0.0)

        rig.contacts = lambda: [
            {
                "instance1": a,
                "instance2": b,
                "normal_force": 5.0,
                "pos": np.zeros(3),
            }
        ]
        maps = rig.tactile_maps(size=4)
        assert maps[a].sum() > 0.0
        assert maps[b].sum() > 0.0
    finally:
        rig.close()


def test_channel_arm_runs():
    config = ContactConfig(max_steps=60, capture_every=5, render=False, seed=0)
    scenario = build_scenario("drop", config)
    events = run_channel(scenario, config, channel="force")
    # The drop may or may not produce a contact within 60 steps; the call is
    # the smoke test, not a performance claim.
    assert isinstance(events, list)
