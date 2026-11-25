"""Twin/counterfactual pairing for property experiments.

A twin pair holds a control and a treated measurement produced from the same
spec/seed/scene; only the architecture variant or the operation differs.  The
paired delta is the unit of causal analysis in the plan (§4.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pss.property.stats import mean_ci


@dataclass
class TwinPair:
    pair_id: str
    spec_hash: str
    seed: int
    control_variant: str
    treated_variant: str
    control: dict
    treated: dict

    def validate(self) -> list[str]:
        errors = []
        if self.control_variant == self.treated_variant:
            errors.append("control and treated variants must differ")
        if not self.spec_hash:
            errors.append("spec_hash is required")
        for metric in set(self.control) & set(self.treated):
            if not np.isfinite(float(self.control[metric])) or not np.isfinite(
                float(self.treated[metric])
            ):
                errors.append(f"non-finite metric {metric}")
        return errors

    def delta(self, metric: str) -> float:
        if metric not in self.control or metric not in self.treated:
            return float("nan")
        return float(self.treated[metric]) - float(self.control[metric])


@dataclass
class TwinRunner:
    spec_hash: str
    seed: int
    pairs: list[TwinPair] = field(default_factory=list)

    def run(
        self,
        pair_id: str,
        control_variant: str,
        treated_variant: str,
        control: dict,
        treated: dict,
    ) -> TwinPair:
        pair = TwinPair(
            pair_id=pair_id,
            spec_hash=self.spec_hash,
            seed=int(self.seed),
            control_variant=control_variant,
            treated_variant=treated_variant,
            control=dict(control),
            treated=dict(treated),
        )
        errors = pair.validate()
        if errors:
            raise ValueError(f"invalid twin {pair_id}: {errors}")
        self.pairs.append(pair)
        return pair

    def summary(self, metrics) -> dict:
        out = {}
        for metric in metrics:
            deltas = [p.delta(metric) for p in self.pairs if np.isfinite(p.delta(metric))]
            if not deltas:
                out[metric] = {"n": 0, "mean_delta": None, "ci95": [None, None]}
                continue
            ci = mean_ci(deltas, n_boot=5000, seed=self.seed)
            out[metric] = {
                "n": len(deltas),
                "mean_delta": ci["mean"],
                "ci95": ci["ci"],
                "values": deltas,
            }
        return {
            "spec_hash": self.spec_hash,
            "seed": int(self.seed),
            "pairs": [p.__dict__ for p in self.pairs],
            "deltas": out,
        }
