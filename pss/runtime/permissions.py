"""Permission profiles for the PSS runtime.

The runtime already has versions, snapshots and rollback. This module adds the
missing authority layer: who may read, ingest, write facts, plan, act, or
roll back. Denied actions are audited before the exception is raised.

Profiles are intentionally coarse and explicit:

- ``read_only``: read state/summary/explain only;
- ``service``: read + ingest perception;
- ``operator``: service + write facts + plan + act;
- ``admin``: everything, including rollback.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

ACTIONS = ("read", "ingest", "write_facts", "plan", "act", "rollback")


@dataclass(frozen=True)
class PermissionProfile:
    name: str
    read: bool = True
    ingest: bool = False
    write_facts: bool = False
    plan: bool = False
    act: bool = False
    rollback: bool = False

    def allows(self, action: str) -> bool:
        if action not in ACTIONS:
            return False
        return bool(getattr(self, action))


PROFILES = {
    "read_only": PermissionProfile("read_only"),
    "service": PermissionProfile("service", ingest=True),
    "operator": PermissionProfile(
        "operator", ingest=True, write_facts=True, plan=True, act=True
    ),
    "admin": PermissionProfile(
        "admin",
        ingest=True,
        write_facts=True,
        plan=True,
        act=True,
        rollback=True,
    ),
}


class PermissionDenied(RuntimeError):
    pass


class PermissionService:
    def __init__(self, profile: str | PermissionProfile = "admin"):
        if isinstance(profile, str):
            if profile not in PROFILES:
                raise KeyError(f"unknown permission profile {profile!r}")
            profile = PROFILES[profile]
        self.profile = profile

    def allows(self, action: str) -> bool:
        return self.profile.allows(action)

    def require(self, action: str) -> None:
        if not self.allows(action):
            raise PermissionDenied(
                f"profile {self.profile.name!r} cannot perform {action!r}"
            )

    def describe(self) -> dict:
        return asdict(self.profile)
