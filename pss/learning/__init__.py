"""Learning permissions and package management (Phase 9).

``review`` depends on the prototype/torch stack and is imported explicitly by
the Phase 9 CLI so that L1/L2/OTA stay lightweight.
"""

from .l1_store import L1Record, L1Store
from .l2_package import PackageIntegrityError, PackageManager, PrototypePackage
from .ota import (
    AdapterRegistry,
    Registry,
    RuleRegistry,
    VersionedPackage,
    adapter_package,
    rule_package,
)

__all__ = [
    "AdapterRegistry",
    "L1Record",
    "L1Store",
    "PackageIntegrityError",
    "PackageManager",
    "PrototypePackage",
    "Registry",
    "RuleRegistry",
    "VersionedPackage",
    "adapter_package",
    "rule_package",
]
