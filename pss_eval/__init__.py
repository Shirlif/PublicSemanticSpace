"""``pss_eval`` -- a standalone property-evaluation protocol for PSS.

The protocol layer (:mod:`pss_eval.protocol`) imports only the standard library
and NumPy.  Adapters are optional and live under :mod:`pss_eval.adapters`; only
the E5 adapter imports PSS internals.

Typical use::

    from pss_eval import make_default_systems, default_properties
    from pss_eval.fixtures import make_gaussian_stream
    from pss_eval.runner import run_suite

    stream = make_gaussian_stream(n=32, dim=32, seed=0)
    report = run_suite(
        make_default_systems(dim=32),
        default_properties(),
        stream,
        seeds=(0, 1, 2),
    )
"""

from .fixtures import LabeledSample, Stream, make_gaussian_stream
from .preregistration import (
    EXPECTED_CONTRACT_METRICS,
    EXPECTED_STATUS,
    check_preregistration,
)
from .properties import (
    ContractProperty,
    DeterminismProperty,
    Property,
    PropertyResult,
    ResetExactnessProperty,
    StateHygieneProperty,
    default_properties,
)
from .protocol import (
    AuditCapable,
    Observation,
    Prediction,
    SnapshotCapable,
    StateSnapshot,
    SwapCapable,
    SystemUnderTest,
    capabilities_of,
    predictions_equal,
    state_digest,
    state_size_bytes,
)
from .runner import SystemSpec, make_default_systems, run, run_suite, write_report

__version__ = "0.1.0"

__all__ = [
    "AuditCapable",
    "ContractProperty",
    "DeterminismProperty",
    "EXPECTED_CONTRACT_METRICS",
    "EXPECTED_STATUS",
    "LabeledSample",
    "Observation",
    "Prediction",
    "Property",
    "PropertyResult",
    "ResetExactnessProperty",
    "SnapshotCapable",
    "StateHygieneProperty",
    "StateSnapshot",
    "Stream",
    "SwapCapable",
    "SystemSpec",
    "SystemUnderTest",
    "capabilities_of",
    "check_preregistration",
    "default_properties",
    "make_default_systems",
    "make_gaussian_stream",
    "predictions_equal",
    "run",
    "run_suite",
    "state_digest",
    "state_size_bytes",
    "write_report",
]
