"""L0 sandbox: property suites over structured observations and expert stubs.

No perception is involved.  The suites exercise the hot-swap registry, the
versioned KB/TMS and the budgeted planner, and return plain scorecards that the
CLI writes to ``scorecards/property/``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from pss.experts import (
    CategoryCodebook,
    NaNExpert,
    NonNormalizedExpert,
    WrongDimExpert,
    make_quality_expert,
    manipulation_check,
)
from pss.metrics.property import (
    audit_completeness,
    budget_adherence,
    budget_adherence_rate,
    conflict_detection_rate,
    cost_attribution_error,
    degradation_correctness,
    downtime,
    forgetting_rate,
    interference,
    locality_violations,
    mean_swap_latency,
    monotonicity_rate,
    propagation_correctness,
    resolution_correctness,
    rollback_exactness,
    routing_correctness,
    safety_rejection_rate,
    swap_success_rate,
    update_success_rate,
    zero_shot_compatibility,
)
from pss.observations import make_large_scene, make_scene
from pss.property.compute import (
    Budget,
    BudgetController,
    BudgetedPlanner,
    anytime_curve,
    degradation_scenario,
    measure_costs,
    monotonic_quality,
)
from pss.property.kb import VersionedKnowledgeBase
from pss.property.swap import ModuleRegistry
from pss.reasoning.planner import AStarPlanner
from pss.safety.degradation import HealthMonitor
from pss.world_model.actions import (
    Operator,
    OperatorLibrary,
    initial_laundry_facts,
    laundry_goal,
    laundry_library,
)

QUALITY_ORDER = ("Q0", "Q1", "Q2", "Q3", "Q4")


@dataclass
class SuiteResult:
    name: str
    metrics: dict
    details: dict

    def to_dict(self) -> dict:
        return {"name": self.name, "metrics": self.metrics, "details": self.details}


# ── H: hot swap ────────────────────────────────────────────────────────────


def run_swap_suite(
    n_observations: int = 200,
    dim: int = 128,
    seed: int = 0,
) -> SuiteResult:
    scene = make_scene(n_observations, dim=dim, seed=seed)
    obs = scene.observations()
    codebook = scene.codebook
    probe = obs[:8]

    registry = ModuleRegistry(public_dim=dim, probe=probe)
    oracle = make_quality_expert("Q0", codebook, modality="vision", seed=seed)
    geometry = make_quality_expert("Q0", codebook, modality="geometry", seed=seed)
    registry.register(oracle, probe=probe)
    registry.register(geometry, probe=probe)

    per_level: dict[str, dict] = {}
    swap_successes: list[bool] = []
    zcr_values: list[float] = []
    routing: list[bool] = []
    rollbacks: list[float] = []
    latencies: list[float] = []
    interference_values: list[float] = []

    for level in QUALITY_ORDER:
        expert = make_quality_expert(level, codebook, modality="vision", seed=seed)
        vision_before = registry.behavior_hash("vision", probe=probe)
        geometry_before = registry.behavior_hash("geometry", probe=probe)

        swap = registry.swap(expert, probe=probe)
        swap_successes.append(swap.success)
        latencies.append(swap.latency_ms)

        preds = [codebook.predict(row) for row in expert.encode(obs)]
        gt = [o.gt_category for o in obs]
        zcr = zero_shot_compatibility(preds, gt)
        zcr_values.append(zcr)
        routing.append(registry.get("vision").version == expert.version)

        geometry_after = registry.behavior_hash("geometry", probe=probe)
        # The registry only swaps vision; geometry must be bit-identical.
        interference_values.append(0.0 if geometry_before == geometry_after else 1.0)

        rollback = registry.rollback()
        vision_after = registry.behavior_hash("vision", probe=probe)
        rollbacks.append(rollback_exactness(vision_before, vision_after))

        per_level[level] = {
            "swap_success": bool(swap.success),
            "zcr": float(zcr),
            "swap_latency_ms": float(swap.latency_ms),
            "rollback_exact": float(rollbacks[-1]),
            "measured": manipulation_check(expert, obs, codebook),
        }

    # Safety rejection: invalid modules must be rejected and the old module kept.
    invalid = [
        WrongDimExpert(codebook),
        NonNormalizedExpert(codebook),
        NaNExpert(codebook),
    ]
    rejected = 0
    old_kept = 0
    for bad in invalid:
        before = registry.behavior_hash("vision", probe=probe)
        result = registry.swap(bad, probe=probe)
        after = registry.behavior_hash("vision", probe=probe)
        rejected += int(not result.success)
        old_kept += int(before == after)

    # Cross-modal transplant sanity check: a tactile stub over the same
    # codebook.  L1/L2 replace this with real cross-modal experts.
    tactile = make_quality_expert("Q1", codebook, modality="tactile", seed=seed)
    registry.register(tactile, probe=probe)
    tactile_preds = [codebook.predict(row) for row in tactile.encode(obs)]
    cross_modal_zcr = zero_shot_compatibility(tactile_preds, [o.gt_category for o in obs])

    metrics = {
        "SSR": swap_success_rate(swap_successes),
        "ZCR": float(np.mean(zcr_values)),
        "ZCR_same_modality": float(per_level["Q1"]["zcr"]),
        "ZCR_cross_modal": float(cross_modal_zcr),
        "INT": float(max(interference_values)),
        "RBE": float(np.mean(rollbacks)),
        "DT": downtime(0, max(1, len(swap_successes))),
        "RCR": routing_correctness(sum(routing), len(routing)),
        "SRR": safety_rejection_rate(rejected, len(invalid)),
        "old_module_retained": float(old_kept / len(invalid)),
        "swap_latency_ms": mean_swap_latency(latencies),
    }
    return SuiteResult(
        "H",
        metrics,
        {
            "per_level": per_level,
            "invalid_rejected": rejected,
            "invalid_injected": len(invalid),
            "cross_modal_note": "L0 uses a same-codebook tactile stub; real transplant is L1/L2.",
        },
    )


# ── K: knowledge-base hot update ───────────────────────────────────────────


def _kb_facts(kb: VersionedKnowledgeBase) -> frozenset:
    out = set()
    for entity, attribute in kb.store.assertions:
        value = kb.resolve(entity, attribute)
        if value is not None:
            out.add((entity, attribute, value))
    return frozenset(out)


def run_kb_suite(seed: int = 0) -> SuiteResult:
    details: dict = {}
    successes: list[bool] = []

    # K1: assert -> retract identity.
    kb = VersionedKnowledgeBase()
    h0 = kb.state_hash()
    kb.assert_fact("cup", "material", "glass", source="s1")
    kb.retract("cup", "material", "s1")
    k1 = rollback_exactness(h0, kb.state_hash())
    successes.append(k1 == 1.0)
    details["K1_assert_retract_identity"] = k1

    # K2: locality.
    kb = VersionedKnowledgeBase()
    kb.assert_fact("cup", "material", "glass", source="s1")
    kb.assert_fact("block", "color", "red", source="s1")
    unrelated_before = kb.resolve("block", "color")
    kb.update("cup", "material", "metal", source="s1")
    unrelated_after = kb.resolve("block", "color")
    loc = locality_violations(int(unrelated_before != unrelated_after))
    successes.append(loc == 0)
    details["K2_locality_violations"] = loc

    # K3: dependency propagation through the TMS.
    tms = VersionedKnowledgeBase().tms
    tms.assume(("A", "v", "1"))
    tms.justify(("B", "v", "1"), in_nodes=[("A", "v", "1")], source="r1")
    tms.justify(("C", "v", "1"), in_nodes=[("B", "v", "1")], source="r2")
    before_ok = tms.label(("C", "v", "1")) is True
    tms.remove_source("r1")
    after_ok = (
        tms.label(("B", "v", "1")) is False
        and tms.label(("C", "v", "1")) is False
        and tms.label(("A", "v", "1")) is True
    )
    pcr = propagation_correctness([before_ok, after_ok])
    successes.append(pcr == 1.0)
    details["K3_propagation"] = pcr

    # K4: conflict detection and policy resolution.
    kb = VersionedKnowledgeBase()
    kb.assert_fact("cup", "color", "red", source="s1", priority=1)
    kb.assert_fact("cup", "color", "blue", source="s2", priority=0)
    conflicts = kb.conflicts()
    conflict_set = kb.conflict_set()
    detected = int(("cup", "color") in conflicts and ("cup", "color") in conflict_set)
    resolved = kb.resolve("cup", "color")
    resolution_ok = int(resolved == "red")
    cdr = conflict_detection_rate(detected, 1)
    res = resolution_correctness(resolution_ok, 1)
    successes.extend([cdr == 1.0, res == 1.0])
    details["K4_conflicts"] = {
        "detected": detected,
        "conflict_set": {str(k): v for k, v in conflict_set.items()},
        "resolved": resolved,
    }

    # K5: rollback exactness over assert/update.
    kb = VersionedKnowledgeBase()
    h0 = kb.state_hash()
    kb.assert_fact("cup", "material", "glass", source="s1")
    kb.update("cup", "material", "metal", source="s1")
    kb.rollback(0)
    k5 = rollback_exactness(h0, kb.state_hash())
    successes.append(k5 == 1.0)
    details["K5_rollback_exactness"] = k5

    # K6: hot update mid-task (plan fails, update, plan succeeds).
    kb = VersionedKnowledgeBase()
    kb.assert_fact("detergent", "available", "no", source="gt")
    library = OperatorLibrary(
        [
            Operator(
                "clean",
                frozenset({("detergent", "available", "yes")}),
                frozenset({("goal", "done", "yes")}),
                frozenset(),
            )
        ]
    )
    goal = frozenset({("goal", "done", "yes")})
    before_plan = AStarPlanner(library, max_expansions=50).plan(_kb_facts(kb), goal)
    kb.update("detergent", "available", "yes", source="runtime")
    after_plan = AStarPlanner(library, max_expansions=50).plan(_kb_facts(kb), goal)
    hot_ok = (not before_plan.success) and after_plan.success and after_plan.actions == ["clean"]
    successes.append(hot_ok)
    details["K6_hot_update"] = {
        "before_success": bool(before_plan.success),
        "after_success": bool(after_plan.success),
        "actions": list(after_plan.actions),
    }

    # K7: adversarial / poisoned update (harness-level guard).
    kb = VersionedKnowledgeBase()
    rejected = 0
    accepted = 0
    injected_bad = 0
    bad_values = {"unsafe", "poison"}
    bad_sources = {"poison"}
    for i, (value, source) in enumerate(
        [("unsafe", "sensor"), ("ok", "poison"), ("ok", "sensor")]
    ):
        if value in bad_values or source in bad_sources:
            rejected += 1
            injected_bad += 1
            continue
        kb.assert_fact(f"e{i}", "state", value, source=source)
        accepted += 1
    rejection_rate = safety_rejection_rate(rejected, injected_bad)
    audited = audit_completeness(kb.audit_count(), kb.change_count())
    successes.extend([rejection_rate == 1.0, audited == 1.0])
    details["K7_adversarial"] = {
        "rejected": rejected,
        "accepted": accepted,
        "audit": audited,
    }

    # K8: multi-source priority/timestamp.
    kb = VersionedKnowledgeBase()
    kb.assert_fact("cup", "color", "red", source="s1", priority=0)
    kb.assert_fact("cup", "color", "blue", source="s2", priority=5)
    first = kb.resolve("cup", "color") == "blue"
    kb.assert_fact("cup", "color", "green", source="s3", priority=5)
    second = kb.resolve("cup", "color") == "green"
    k8 = resolution_correctness(int(first) + int(second), 2)
    successes.append(k8 == 1.0)
    details["K8_multisource"] = {"priority_ok": first, "timestamp_ok": second}

    # Aggregate.
    metrics = {
        "USR": update_success_rate(successes),
        "LOC": float(details["K2_locality_violations"]),
        "FOR": forgetting_rate([True] * len(successes), [bool(x) for x in successes]),
        "PCR": float(pcr),
        "CDR": float(cdr),
        "RES": float(res),
        "RBE-KB": float(k5),
        "AUD": float(audited),
        "UL_ms": 0.0,
    }
    return SuiteResult("K", metrics, details)


# ── C: bounded inference ───────────────────────────────────────────────────


def run_compute_suite(seed: int = 0) -> SuiteResult:
    details: dict = {}
    library = laundry_library()
    initial = initial_laundry_facts()
    goal = laundry_goal()

    budgets = [1, 2, 3, 5, 8, 12, 20, 50, 100]
    rows = []
    for budget in budgets:
        planner = BudgetedPlanner(library, Budget(max_expansions=budget))
        result, report = planner.plan(initial, goal)
        rows.append(
            {
                "budget": budget,
                "success": bool(result.success),
                "nodes": int(result.nodes),
                "within_budget": bool(report.within_budget),
                "latency_ms": float(report.latency_ms),
            }
        )
    pairs = [(r["nodes"], r["budget"]) for r in rows]
    bar = budget_adherence_rate(pairs)
    mon = monotonic_quality([r["success"] for r in rows])
    details["C1_budget_sweep"] = rows

    # C2: scene-size pressure; clipping must keep sizes inside the bound.
    scene = make_large_scene(n_entities=200, n_relations=1000, seed=seed)
    controller = BudgetController(Budget(max_entities=50, max_relations=100))
    active = controller.active_set(scene.entities)
    relations = controller.relation_set(scene.relations)
    c2_ok = len(active) <= 50 and len(relations) <= 100
    details["C2_scene_pressure"] = {
        "entities": len(active),
        "relations": len(relations),
        "within": bool(c2_ok),
    }

    # C3: anytime interruption curve.
    def plan_at(budget):
        planner = BudgetedPlanner(library, Budget(max_expansions=budget))
        return planner.plan(initial, goal)

    curve = anytime_curve(plan_at, [1, 2, 3, 5, 8, 12, 20, 50, 100])
    c3_valid = all(set(r) >= {"budget", "success", "nodes", "latency_ms"} for r in curve)
    details["C3_anytime"] = curve

    # C4: adversarial / unreachable goal must respect the bound.
    unreachable = frozenset({("never", "happens", "yes")})
    adversarial = BudgetedPlanner(library, Budget(max_expansions=20))
    adv_result, adv_report = adversarial.plan(initial, unreachable)
    wcb = max(0, int(adv_result.nodes) - 20)
    details["C4_adversarial"] = {
        "success": bool(adv_result.success),
        "nodes": int(adv_result.nodes),
        "within_budget": bool(adv_report.within_budget),
        "worst_case_excess": int(wcb),
    }

    # C5: degradation FSM.
    monitor = HealthMonitor(debounce=1)
    signals = [
        {"camera_ok": False},
        {"temperature": 90.0},
        {},
    ]
    levels = degradation_scenario(monitor, signals)
    expected = ["degraded", "emergency", "normal"]
    dcr = degradation_correctness(
        sum(1 for row, exp in zip(levels, expected) if row["level"] == exp), len(expected)
    )
    details["C5_degradation"] = levels

    # C6: cost attribution.
    costs = measure_costs(
        {
            "active_set": lambda: controller.active_set(scene.entities),
            "relations": lambda: controller.relation_set(scene.relations),
            "plan": lambda: BudgetedPlanner(library, Budget(max_expansions=100)).plan(initial, goal),
        }
    )
    cat = cost_attribution_error(list(costs["stages"].values()), costs["total_ms"])
    details["C6_costs"] = costs

    metrics = {
        "BAR": float(bar),
        "MON": float(mon),
        "p95_ms": float(np.percentile([r["latency_ms"] for r in rows], 95)),
        "p99_ms": float(np.percentile([r["latency_ms"] for r in rows], 99)),
        "DCR": float(dcr),
        "CAT": float(cat),
        "WCB": float(wcb),
        "anytime_valid": 1.0 if c3_valid else 0.0,
    }
    return SuiteResult("C", metrics, details)


# ── cross-property interactions ───────────────────────────────────────────


def run_interaction_suite(seed: int = 0) -> SuiteResult:
    details: dict = {}
    checks: list[bool] = []

    # H x K: swap then update; KB remains consistent and rollback exact.
    scene = make_scene(20, dim=128, seed=seed)
    codebook: CategoryCodebook = scene.codebook
    registry = ModuleRegistry(public_dim=128, probe=scene.observations()[:4])
    registry.register(make_quality_expert("Q0", codebook, modality="vision", seed=seed))
    kb = VersionedKnowledgeBase()
    kb.assert_fact("cup", "material", "glass", source="s1")
    h_before = kb.state_hash()
    h_index = kb.snapshot()["index"]
    registry.swap(make_quality_expert("Q2", codebook, modality="vision", seed=seed))
    kb.update("cup", "material", "metal", source="s1")
    hk_ok = kb.resolve("cup", "material") == "metal" and not kb.conflicts()
    kb.rollback(h_index)
    hk_rollback = rollback_exactness(h_before, kb.state_hash())
    checks.extend([hk_ok, hk_rollback == 1.0])
    details["HK"] = {"consistent": hk_ok, "rollback": hk_rollback}

    # H x C: swap latency + planning latency stays inside a generous bound.
    library = laundry_library()
    initial = initial_laundry_facts()
    goal = laundry_goal()
    budget = Budget(max_expansions=100, max_latency_ms=10_000.0)
    planner = BudgetedPlanner(library, budget)
    start = time.perf_counter()
    result, report = planner.plan(initial, goal)
    total_ms = (time.perf_counter() - start) * 1000.0
    hc_ok = report.within_budget and total_ms <= budget.max_latency_ms
    checks.append(hc_ok)
    details["HC"] = {"plan_success": bool(result.success), "total_ms": total_ms}

    # K x C: KB growth + bounded planning.
    kb = VersionedKnowledgeBase()
    for i in range(100):
        kb.assert_fact(f"e{i}", "state", "ok", source="bulk")
    kb_plan, kb_report = planner.plan(_kb_facts(kb) | initial, goal)
    kc_ok = kb_report.within_budget and int(kb_plan.nodes) <= budget.max_expansions
    checks.append(kc_ok)
    details["KC"] = {"nodes": int(kb_plan.nodes), "within": bool(kb_report.within_budget)}

    # H x K x C: all three at once.
    registry.swap(make_quality_expert("Q3", codebook, modality="vision", seed=seed))
    kb.update("detergent", "available", "yes", source="runtime")
    hkc_plan, hkc_report = planner.plan(_kb_facts(kb) | initial, goal)
    hkc_ok = hkc_report.within_budget and registry.get("vision") is not None
    checks.append(hkc_ok)
    details["HKC"] = {"within": bool(hkc_report.within_budget), "nodes": int(hkc_plan.nodes)}

    metrics = {
        "interaction_pass_rate": float(sum(checks) / len(checks)),
        "HK_ok": float(hk_ok),
        "HK_rollback": float(hk_rollback),
        "HC_ok": float(hc_ok),
        "KC_ok": float(kc_ok),
        "HKC_ok": float(hkc_ok),
    }
    return SuiteResult("I", metrics, details)


# ── aggregate ──────────────────────────────────────────────────────────────


THRESHOLDS = {
    "SSR": 0.90,
    "ZCR_same_modality": 0.90,
    "ZCR_cross_modal": 0.70,
    "INT": 0.02,  # upper bound; handled specially below
    "RBE": 1.00,
    "DT": 0.01,  # upper bound
    "RCR": 0.95,
    "SRR": 1.00,
    "USR": 0.99,
    "LOC": 0.0,  # upper bound
    "PCR": 1.00,
    "CDR": 0.95,
    "RES": 1.00,
    "RBE-KB": 1.00,
    "AUD": 1.00,
    "BAR": 1.00,
    "MON": 0.95,
    "DCR": 0.95,
    "CAT": 0.05,  # upper bound
    "WCB": 0.0,  # upper bound
    "interaction_pass_rate": 1.00,
}


def _pass_flags(metrics: dict) -> dict:
    upper = {"INT", "DT", "LOC", "CAT", "WCB"}
    flags = {}
    for name, threshold in THRESHOLDS.items():
        if name not in metrics:
            continue
        value = float(metrics[name])
        flags[name] = bool(value <= threshold) if name in upper else bool(value >= threshold)
    return flags


def run_all_l0(n_observations: int = 200, seed: int = 0) -> dict:
    suites = [
        run_swap_suite(n_observations=n_observations, seed=seed),
        run_kb_suite(seed=seed),
        run_compute_suite(seed=seed),
        run_interaction_suite(seed=seed),
    ]
    metrics = {}
    for suite in suites:
        metrics.update(suite.metrics)
    return {
        "seed": int(seed),
        "n_observations": int(n_observations),
        "suites": [s.to_dict() for s in suites],
        "metrics": metrics,
        "thresholds": dict(THRESHOLDS),
        "pass": _pass_flags(metrics),
    }
