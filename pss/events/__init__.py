"""Contact-event layer: model, channels, scenarios, benchmark."""

from .bench_api import (
    ARMS,
    SCENARIO_BUILDERS,
    ContactConfig,
    ContactScenario,
    build_scenario,
    evaluate_events,
    register_arm,
    register_scenario,
    run_bench,
    run_channel,
)
from .model import ContactEvent, ContactEvidence, ContactFrame
from .segmenter import Interval, segment_signal, temporal_iou

# Import for side effects (channel/scenario registration).
from . import extractors, scenarios  # noqa: F401,E402

# Optional modules register extra arms/scenarios when they exist.
for _module in (
    "arms_channels",
    "arms_fusion",
    "scenarios_conflict",
    "scenarios_active",
):
    try:
        __import__(f"pss.events.{_module}")
    except ModuleNotFoundError:
        pass

__all__ = [
    "ARMS",
    "SCENARIO_BUILDERS",
    "ContactConfig",
    "ContactEvent",
    "ContactEvidence",
    "ContactFrame",
    "ContactScenario",
    "Interval",
    "build_scenario",
    "evaluate_events",
    "register_arm",
    "register_scenario",
    "run_bench",
    "run_channel",
    "segment_signal",
    "temporal_iou",
]
