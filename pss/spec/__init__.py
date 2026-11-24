"""World spec package (A1).

``WorldSpec`` is the declarative description of a scene family; its canonical
SHA-256 is the key used by USD builds, bundles and scorecards.
"""

from .loader import (
    DEFAULT_REGISTRY,
    lint_file,
    load_spec,
    load_yaml,
    register,
)
from .schema import (
    LintResult,
    SpecValidationError,
    WorldSpec,
    hash_spec,
    json_schema,
    validate,
)

__all__ = [
    "DEFAULT_REGISTRY",
    "LintResult",
    "SpecValidationError",
    "WorldSpec",
    "hash_spec",
    "json_schema",
    "lint_file",
    "load_spec",
    "load_yaml",
    "register",
    "validate",
]
