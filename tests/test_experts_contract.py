"""Contract checks for the expert registration list.

``contracts/experts.yaml`` is a registration/review list, not the runtime
source of truth: production builds experts through
``pss.encoders.registry`` / ``pss.experts`` by kind.  These checks catch the
drift that matters — a modality that no longer exists, a missing license, or a
non-positive width.
"""

from pathlib import Path

import pytest
import yaml

from pss.contracts.modality import get_modalities

EXPERTS = Path("contracts/experts.yaml")
KNOWN_STATUS = {"active", "planned", "deferred", "retired"}


def _entries():
    data = yaml.safe_load(EXPERTS.read_text(encoding="utf-8")) or {}
    return data.get("experts", [])


def test_experts_file_exists_and_has_entries():
    assert EXPERTS.exists()
    assert _entries()


@pytest.mark.parametrize("spec", _entries(), ids=lambda s: s.get("name", "?"))
def test_expert_entry_is_consistent(spec):
    assert spec.get("name")
    assert spec.get("modality")
    assert spec.get("license")
    assert int(spec.get("output_dim", 0)) > 0
    assert spec.get("status") in KNOWN_STATUS
    assert spec.get("source")


def test_expert_modalities_are_registered():
    registered = set(
        get_modalities(include_experimental=True, include_retired=True)
    )
    for spec in _entries():
        assert spec["modality"] in registered, (
            f"{spec['name']} uses unregistered modality {spec['modality']!r}"
        )
