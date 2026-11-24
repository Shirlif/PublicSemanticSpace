"""Audit snapshots and replay (Phase 9)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import json

import numpy as np


@dataclass
class AuditRecord:
    observation: dict
    entities: list
    emotion: dict
    motivation: dict
    candidates: list
    decision: str
    scores: dict
    value_vector: dict
    guardrail: dict
    health: str
    guardrail_inputs: dict | None = None
    plan_audit: dict | None = None
    chain_audit: dict | None = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    )

    def to_dict(self) -> dict:
        return asdict(self)


class AuditLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, record: AuditRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    def replay(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    @staticmethod
    def verify(records: list[dict]) -> bool:
        for r in records:
            weights = r["value_vector"]
            best_score = -1e9
            best_id = None
            for candidate in r["candidates"]:
                score = (
                    float(candidate.get("similarity", 1.0))
                    * float(weights.get(candidate.get("value_domain", "task_fidelity"), 0.5))
                    * float(candidate.get("safety_coeff", 1.0))
                    - float(candidate.get("cost_penalty", 0.0))
                )
                if score > best_score:
                    best_score = score
                    best_id = candidate["id"]
            if best_id != r["decision"]:
                return False
            stored = r.get("scores", {}).get("final")
            if stored is not None and abs(float(stored) - best_score) > 1e-6:
                return False

            if r.get("guardrail_inputs"):
                from . import GuardrailEvaluator

                inputs = r["guardrail_inputs"]
                result = GuardrailEvaluator().evaluate(
                    inputs["action"], inputs["state"], inputs["candidate"]
                )
                recorded = r.get("guardrail", {})
                if bool(recorded.get("hard_veto")) != bool(result.hard_veto):
                    return False
                if "allowed" in recorded and bool(recorded["allowed"]) != bool(result.allowed):
                    return False

            if r.get("plan_audit"):
                from pss.world_model.actions import (
                    laundry_goal,
                    laundry_library,
                    simulate_plan,
                )

                plan = r["plan_audit"]
                if plan.get("library") != "laundry":
                    return False
                try:
                    initial = frozenset(tuple(x) for x in plan["initial_facts"])
                    goal = frozenset(tuple(x) for x in plan["goal"])
                    facts, _log, ok = simulate_plan(
                        plan["plan"], initial, laundry_library()
                    )
                except Exception:
                    return False
                if not ok or not (goal <= facts):
                    return False

            if r.get("chain_audit"):
                from pss.reasoning.relations import relations_from_frame
                from pss.sim.scene import CATEGORY_SPECS

                chain = r["chain_audit"]
                try:
                    sample = {
                        "poses": np.asarray(chain["poses"], dtype=float),
                        "instance_ids": list(chain["instance_ids"]),
                        "categories": list(chain["categories"]),
                        "object_specs": list(chain["object_specs"]),
                    }
                    sizes = {c: spec["size"] for c, spec in CATEGORY_SPECS.items()}
                    computed = {
                        fact.as_tuple()
                        for fact in relations_from_frame(sample, 0, sizes)
                    }
                    recorded = {tuple(fact) for fact in chain["relations"]}
                except Exception:
                    return False
                if computed != recorded:
                    return False
        return True
