"""Phase 9 CLI: learning permissions, OTA, guardrails, degradation, audit."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pss._console import ensure_utf8
from pss.learning import (
    AdapterRegistry,
    L1Record,
    L1Store,
    PackageManager,
    PrototypePackage,
    RuleRegistry,
    adapter_package,
    rule_package,
)
from pss.learning.review import (
    aggregate_proposals,
    bank_from_package,
    load_human_decision,
    review_proposal,
    write_human_decision,
)
from pss.learning.signing import generate_keypair
from pss.prototypes import PrototypeBank
from pss.prototypes.build import extract_projected_features
from pss.safety import (
    DEFAULT_VALUE_VECTOR,
    GuardrailEvaluator,
    ScenarioValuePrototype,
    ScenarioValueStore,
    ValueVector,
    calibrate_min_score,
)
from pss.safety.audit import AuditLog, AuditRecord
from pss.safety.degradation import HealthMonitor
from pss.world_model.actions import (
    hand_written_plan,
    initial_laundry_facts,
    laundry_goal,
)


def _faults_json(status) -> list[dict]:
    return [{"name": f.name, "severity": f.severity} for f in status.faults]


def _package_from_bank(bank: PrototypeBank, version: str) -> PrototypePackage:
    return PrototypePackage(
        version=version,
        categories=list(bank.categories),
        protos={m: bank.protos[m].copy() for m in bank.protos},
        stats=bank.stats,
        changelog=[f"base from {bank.version}"],
    )


def _run_learning(data_dir, expert_dir, projection_dir, prototype_dir, out: Path):
    bank = PrototypeBank.load(Path(prototype_dir) / "prototypes.npz")
    base = _package_from_bank(bank, "v1.0")
    private_key, public_key = generate_keypair()
    keys_dir = out / "keys"
    keys_dir.mkdir(parents=True, exist_ok=True)
    (keys_dir / "private.pem").write_bytes(private_key)
    (keys_dir / "public.pem").write_bytes(public_key)
    manager = PackageManager(
        out / "packages",
        trusted_public_key=public_key,
        private_key=private_key,
    )
    manager.install(base)
    manager.activate("v1.0")
    base_categories = manager.active_package().categories

    feats, Z = extract_projected_features(data_dir, expert_dir, projection_dir, device="cpu")
    cloth_idx = [
        i
        for i, split in enumerate(feats.splits)
        if split == "test_category" and feats.categories[i] == "cloth"
    ]
    if len(cloth_idx) < 10:
        return None
    n_prop = int(len(cloth_idx) * 0.6)
    prop_idx, hold_idx = cloth_idx[:n_prop], cloth_idx[n_prop:]

    store = L1Store(out / "l1.jsonl")
    for i in prop_idx:
        store.append(
            L1Record(
                kind="observation",
                category="cloth",
                vectors={m: Z[m][i].tolist() for m in Z},
                confidence=1.0,
                environment={"lighting": "normal", "device_class": "sim"},
                health="normal",
            )
        )
    store.save()
    proposal = store.make_proposal("cloth", health="normal")

    aggregated = aggregate_proposals([proposal])[0]
    samples = {m: np.stack([Z[m][i] for i in prop_idx]) for m in Z}
    holdout = {m: np.stack([Z[m][i] for i in hold_idx]) for m in Z}
    reviewer_private, reviewer_public = generate_keypair()
    (keys_dir / "reviewer_private.pem").write_bytes(reviewer_private)
    (keys_dir / "reviewer_public.pem").write_bytes(reviewer_public)
    write_human_decision(
        out / "human_review.json",
        proposal_id="cloth",
        reviewer="simulated_reviewer",
        decision="accept",
        note="regression gate passed",
        private_key=reviewer_private,
    )
    human_accept = load_human_decision(
        out / "human_review.json", trusted_public_key=reviewer_public
    )
    review = review_proposal(
        samples,
        aggregated["vectors"],
        holdout,
        bank,
        "cloth",
        proposal["sample_count"],
        human_decision=human_accept,
        require_human=True,
    )
    write_human_decision(
        out / "human_review_reject.json",
        proposal_id="cloth",
        reviewer="simulated_reviewer",
        decision="reject",
        note="veto test",
        private_key=reviewer_private,
    )
    human_reject = load_human_decision(
        out / "human_review_reject.json", trusted_public_key=reviewer_public
    )
    review_veto = review_proposal(
        samples,
        aggregated["vectors"],
        holdout,
        bank,
        "cloth",
        proposal["sample_count"],
        human_decision=human_reject,
        require_human=True,
    )
    _, wrong_reviewer_public = generate_keypair()
    wrong_reviewer_key_rejected = False
    try:
        load_human_decision(
            out / "human_review.json", trusted_public_key=wrong_reviewer_public
        )
    except Exception:
        wrong_reviewer_key_rejected = True

    new_version = "v1.1"
    active_categories = base_categories
    rolled_back_to = None
    if review.accepted:
        stats = {
            m: {
                "sample_count": proposal["sample_count"],
                "variance": proposal["variance"].get(m, 0.0),
            }
            for m in aggregated["vectors"]
        }
        package = base.append_category(
            "cloth", aggregated["vectors"], stats, version=new_version, changelog="add cloth"
        )
        manager.install(package)
        manager.activate(new_version)
        active_categories = manager.active_package().categories
        rolled_back_to = manager.rollback()

    corrupt = manager.copy_package(new_version, out / "corrupt_package")
    npz = corrupt / "prototypes.npz"
    npz.write_bytes(npz.read_bytes() + b"corrupt")
    integrity_rejected = False
    try:
        PrototypePackage.load(corrupt)
    except Exception:
        integrity_rejected = True

    signature_verified = manager.verify(new_version)
    wrong_private, wrong_public = generate_keypair()
    wrong_key_rejected = False
    try:
        PrototypePackage.load(
            manager.packages[new_version], trusted_public_key=wrong_public
        )
    except Exception:
        wrong_key_rejected = True

    unsigned_dir = out / "unsigned_package"
    base.save(unsigned_dir)
    unsigned_rejected = False
    try:
        PrototypePackage.load(unsigned_dir, trusted_public_key=public_key)
    except Exception:
        unsigned_rejected = True

    return {
        "base_version": base.version,
        "base_categories": base_categories,
        "new_version": new_version,
        "signing": {
            "public_key": str(keys_dir / "public.pem"),
            "signature_verified": signature_verified,
            "wrong_key_rejected": wrong_key_rejected,
            "unsigned_rejected": unsigned_rejected,
        },
        "l1": {
            "records": len(store.records),
            "proposal_samples": proposal["sample_count"],
            "scrubbed": proposal["scrubbed"],
            "proposal_category": proposal["category"],
        },
        "review": {
            "pre_review": review.pre_review,
            "pre_review_thresholds": review.pre_review_thresholds,
            "pre_review_metrics": review.pre_review_metrics,
            "regression": review.regression,
            "accepted": review.accepted,
            "reasons": review.reasons,
        },
        "human_review": {
            "reviewer": human_accept["reviewer"],
            "decision": human_accept["decision"],
            "accepted": review.accepted,
            "veto_accepted": review_veto.accepted,
            "veto_reasons": review_veto.reasons,
            "signature_verified": True,
            "wrong_reviewer_key_rejected": wrong_reviewer_key_rejected,
        },
        "package": {
            "active_categories": active_categories,
            "rollback_to": rolled_back_to,
            "integrity_rejected": integrity_rejected,
        },
        "manager": manager,
    }


def _run_registries(out: Path) -> dict:
    def digest(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    adapter_root = out / "l3"
    rule_root = out / "l4"
    v1 = b'{"backbone":"smallcnn","dim":128,"adapter":"a1"}'
    v2 = b'{"backbone":"smallcnn","dim":128,"adapter":"a2"}'
    adapters = AdapterRegistry(adapter_root)
    adapters.install(
        adapter_package(
            "a1",
            "projection",
            {"vision.pt": digest(v1)},
            artifacts={"vision.pt": v1},
        )
    )
    adapters.install(
        adapter_package(
            "a2",
            "projection",
            {"vision.pt": digest(v2)},
            parent="a1",
            artifacts={"vision.pt": v2},
        )
    )
    adapters.activate("a1")
    adapters.activate("a2")
    adapter_active = adapters.active
    adapter_rollback = adapters.rollback()

    r1 = b'{"blacklist":["human_in_zone"],"version":"r1"}'
    r2 = b'{"blacklist":["human_in_zone","sharp_tool_fast"],"version":"r2"}'
    rules = RuleRegistry(rule_root)
    rules.install(
        rule_package(
            "r1",
            DEFAULT_VALUE_VECTOR,
            ["human_in_zone"],
            artifacts={"rules.json": r1},
        )
    )
    rules.install(
        rule_package(
            "r2",
            {**DEFAULT_VALUE_VECTOR, "human_safety": 0.9},
            ["human_in_zone", "sharp_tool_fast"],
            parent="r1",
            artifacts={"rules.json": r2},
        )
    )
    rules.activate("r1")
    rules.activate("r2")
    rule_active = rules.active
    rule_rollback = rules.rollback()

    (adapter_root / "a2" / "files" / "vision.pt").write_bytes(b"tampered")
    adapter_tamper_rejected = not adapters.verify("a2")
    return {
        "adapter_active": adapter_active,
        "adapter_rollback": adapter_rollback,
        "rule_active": rule_active,
        "rule_rollback": rule_rollback,
        "adapter_files": [
            str(adapter_root / "a1" / "files" / "vision.pt"),
            str(adapter_root / "a2" / "files" / "vision.pt"),
        ],
        "rule_files": [str(rule_root / "r1" / "files" / "rules.json")],
        "adapter_tamper_rejected": adapter_tamper_rejected,
    }


def _run_guardrails(min_score: float = 0.2) -> dict:
    scenarios = ScenarioValueStore(
        [
            ScenarioValuePrototype(
                "night_mode", {"human_comfort": 0.10, "exploration_restraint": 0.20}
            )
        ]
    )
    guard = GuardrailEvaluator(scenario_store=scenarios, min_score=min_score)
    candidate = {
        "id": "task",
        "similarity": 0.9,
        "value_domain": "task_fidelity",
        "safety_coeff": 1.0,
        "cost_penalty": 0.1,
    }
    safe_action = {"kind": "push", "params": {"force": 2.0}, "target_is_human": False}
    safe_state = {"human_distance": 1.0, "target_category": "cup", "temperature": 25.0}
    hard_action = {"kind": "push", "params": {"force": 20.0}, "target_is_human": True}
    hard_state = {"human_distance": 0.2, "target_category": "tool", "temperature": 25.0}

    results = {
        "hard_veto": guard.evaluate(hard_action, hard_state, candidate),
        "safe": guard.evaluate(safe_action, safe_state, candidate),
        "unsafe_candidate": guard.evaluate(
            safe_action, safe_state, {**candidate, "safety_coeff": 0.0}
        ),
        "comfort_base": guard.evaluate(
            safe_action, safe_state, {**candidate, "value_domain": "human_comfort"}
        ),
        "comfort_night": guard.evaluate(
            safe_action,
            safe_state,
            {**candidate, "value_domain": "human_comfort"},
            scenario="night_mode",
        ),
        "llm_injection": guard.evaluate(
            {**hard_action, "source": "llm"}, hard_state, candidate
        ),
    }
    return {
        name: {
            "allowed": r.allowed,
            "hard_veto": r.hard_veto,
            "score": r.score,
            "reasons": r.reasons,
        }
        for name, r in results.items()
    }


def _run_degradation() -> list[dict]:
    monitor = HealthMonitor(debounce=2)
    sequence = []

    def step(signals: dict):
        status = monitor.check(signals)
        sequence.append(
            {
                "signals": signals,
                "level": status.level.value,
                "proposal": status.proposal_allowed,
                "motion": status.motion_allowed,
                "exploration": status.exploration_allowed,
                "manual": status.manual_required,
                "offline": status.offline,
                "faults": _faults_json(status),
            }
        )

    step({})
    step({"projection_drift": 0.2})
    step({"projection_drift": 0.2})
    step({"camera_ok": False})
    step({"camera_ok": False})
    step({"temperature": 90})
    step({"temperature": 90})
    step({})
    step({})
    step({"offline": True})
    return sequence


def _run_audit(out: Path) -> dict:
    candidates = [
        {
            "id": "task",
            "similarity": 0.9,
            "value_domain": "task_fidelity",
            "safety_coeff": 1.0,
            "cost_penalty": 0.1,
        },
        {
            "id": "comfort",
            "similarity": 0.8,
            "value_domain": "human_comfort",
            "safety_coeff": 1.0,
            "cost_penalty": 0.0,
        },
    ]
    value_vector = ValueVector()
    scores = {c["id"]: value_vector.score(c) for c in candidates}
    decision = max(scores, key=scores.get)
    selected = next(c for c in candidates if c["id"] == decision)
    action = {"kind": "push", "params": {"force": 2.0}}
    state = {"human_distance": 1.0, "target_category": "cup"}
    guard = GuardrailEvaluator().evaluate(action, state, selected)

    from pss.reasoning.relations import relations_from_frame
    from pss.sim.scene import CATEGORY_SPECS

    sizes = {c: spec["size"] for c, spec in CATEGORY_SPECS.items()}
    box_size = list(CATEGORY_SPECS["box"]["size"])
    chain_sample = {
        "poses": np.array(
            [
                [
                    [0.0, 0.0, box_size[2], 1.0, 0.0, 0.0, 0.0],
                    [0.10, 0.0, box_size[2], 1.0, 0.0, 0.0, 0.0],
                ]
            ],
            dtype=float,
        ),
        "instance_ids": ["box_00", "box_01"],
        "categories": ["box", "box"],
        "object_specs": [
            {"geom_type": "box", "size": box_size},
            {"geom_type": "box", "size": box_size},
        ],
    }
    chain_relations = [
        list(fact.as_tuple())
        for fact in relations_from_frame(chain_sample, 0, sizes)
    ]
    chain_audit = {
        "poses": chain_sample["poses"].tolist(),
        "instance_ids": chain_sample["instance_ids"],
        "categories": chain_sample["categories"],
        "object_specs": chain_sample["object_specs"],
        "relations": chain_relations,
    }
    record = AuditRecord(
        observation={"scene": "tabletop", "task": "put_away"},
        entities=["obj_0001", "obj_0002"],
        emotion={"valence": 0.6, "arousal": 0.2},
        motivation={"goal": "put_away", "priority": 0.7},
        candidates=candidates,
        decision=decision,
        scores={"final": scores[decision], "per_candidate": scores},
        value_vector=value_vector.weights,
        guardrail={
            "hard_veto": guard.hard_veto,
            "allowed": guard.allowed,
            "score": guard.score,
            "reasons": guard.reasons,
            "scenario": None,
        },
        health="normal",
        guardrail_inputs={"action": action, "state": state, "candidate": selected},
        plan_audit={
            "initial_facts": sorted([list(f) for f in initial_laundry_facts()]),
            "plan": hand_written_plan(),
            "goal": sorted([list(f) for f in laundry_goal()]),
            "library": "laundry",
        },
        chain_audit=chain_audit,
    )
    log = AuditLog(out / "audit.jsonl")
    log.append(record)
    replayed = log.replay()
    return {
        "records": len(replayed),
        "verified": AuditLog.verify(replayed),
        "decision": decision,
        "scores": scores,
        "guardrail_hard_veto": guard.hard_veto,
        "guardrail_allowed": guard.allowed,
        "plan_actions": len(hand_written_plan()),
        "plan_library": "laundry",
        "chain_relations": len(chain_relations),
    }


def _write_report(path: Path, metrics: dict) -> None:
    learning = metrics["learning"]
    review = learning["review"]
    human = learning["human_review"]
    signing = learning["signing"]
    package = learning["package"]
    registries = metrics["registries"]
    guard = metrics["guardrails"]
    calibration = metrics["guardrail_calibration"]
    seq = metrics["degradation"]
    audit = metrics["audit"]
    lines = [
        "# Phase 9 report: learning permissions, safety, degradation",
        "",
        "## Learning loop (L1 -> review -> L2 OTA)",
        "",
        "| item | value |",
        "|---|---|",
        f"| L1 records | {learning['l1']['records']} |",
        f"| proposal samples | {learning['l1']['proposal_samples']} |",
        f"| scrubbed | {learning['l1']['scrubbed']} |",
        f"| pre-review | {review['pre_review']} |",
        f"| pre-review thresholds | {review['pre_review_thresholds']} |",
        f"| human reviewer | {human['reviewer']} |",
        f"| human decision | {human['decision']} |",
        f"| human signature verified | {human['signature_verified']} |",
        f"| wrong reviewer key rejected | {human['wrong_reviewer_key_rejected']} |",
        f"| human veto accepted | {human['veto_accepted']} |",
        f"| regression new acc | {review['regression']['new_acc']:.3f} |",
        f"| holdout | {review['regression']['n_holdout']} |",
        f"| accepted | {review['accepted']} ({', '.join(review['reasons']) or 'ok'}) |",
        f"| active categories | {', '.join(package['active_categories'])} |",
        f"| rollback to | {package['rollback_to']} |",
        f"| corrupted package rejected | {package['integrity_rejected']} |",
        f"| signature verified | {signing['signature_verified']} |",
        f"| wrong key rejected | {signing['wrong_key_rejected']} |",
        f"| unsigned rejected | {signing['unsigned_rejected']} |",
        "",
        "## L3 / L4 registries",
        "",
        "| layer | active | rollback |",
        "|---|---|---|",
        f"| L3 adapters | {metrics['registries']['adapter_active']} | {metrics['registries']['adapter_rollback']} |",
        f"| L4 rules | {metrics['registries']['rule_active']} | {metrics['registries']['rule_rollback']} |",
        "",
        f"- L3 payload files: {registries['adapter_files']}",
        f"- L4 payload files: {registries['rule_files']}",
        f"- L3 tamper rejected: {registries['adapter_tamper_rejected']}",
        "",
        "## Guardrails",
        "",
        f"- calibrated min_score: {calibration['min_score']:.3f} "
        f"(balanced accuracy {calibration['balanced_accuracy']:.3f}, "
        f"{calibration['n_cases']} synthetic cases)",
        "",
        "| case | allowed | hard veto | score | reasons |",
        "|---|---|---|---:|---|",
    ]
    for name, r in guard.items():
        lines.append(
            f"| {name} | {r['allowed']} | {r['hard_veto']} | {r['score']:.3f} | "
            f"{', '.join(r['reasons']) or '—'} |"
        )
    lines += [
        "",
        "## Degradation",
        "",
        "| step | signals | level | proposal | motion | exploration | manual |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, row in enumerate(seq):
        lines.append(
            f"| {i} | {row['signals']} | {row['level']} | {row['proposal']} | "
            f"{row['motion']} | {row['exploration']} | {row['manual']} |"
        )
    lines += [
        "",
        "## Audit",
        "",
        f"- records: {audit['records']}",
        f"- replay verified: {audit['verified']}",
        f"- decision: {audit['decision']}",
        f"- guardrail replay: hard_veto={audit['guardrail_hard_veto']} "
        f"allowed={audit['guardrail_allowed']}",
        f"- plan replay: {audit['plan_actions']} actions via `{audit['plan_library']}` library",
        f"- chain replay: {audit['chain_relations']} relation facts recomputed from entity states",
        "",
        "## Caveats",
        "",
        "- L2 signing is Ed25519; sha256 remains the integrity check.",
        "- L3/L4 install real payload files and verify hashes; the payloads are small configs, not trained weights.",
        "- Pre-review thresholds are calibrated per modality from the bank; regression accuracy remains the decisive gate.",
        "- Human review is an Ed25519-signed file record; there is no UI, identity provider, or hardware token.",
        "- Guardrail min_score is calibrated on synthetic cases; value-vector weights remain design values.",
        "- LLM injection is simulated as a rule violation; hard safety vetoes it regardless of source.",
        "- Degradation is an FSM + chaos sequence, not real hardware fault injection.",
        "- Audit replay recomputes value scores, guardrail verdicts, the plan, and the entity→relation chain; it does not replay raw pixels→entities.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_phase9(
    data_dir: str | Path,
    expert_dir: str | Path,
    projection_dir: str | Path,
    prototype_dir: str | Path,
    out_dir: str | Path,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    learning = _run_learning(data_dir, expert_dir, projection_dir, prototype_dir, out)
    if learning is None:
        raise RuntimeError("not enough unseen-category samples for the Phase 9 loop")
    calibration = calibrate_min_score()
    metrics = {
        "data_dir": str(data_dir),
        "learning": {k: v for k, v in learning.items() if k != "manager"},
        "registries": _run_registries(out),
        "guardrails": _run_guardrails(calibration["min_score"]),
        "guardrail_calibration": calibration,
        "degradation": _run_degradation(),
        "audit": _run_audit(out),
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    (out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_report(out / "REPORT.md", metrics)
    return metrics


def main() -> None:
    ensure_utf8()
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/expert_v1")
    ap.add_argument("--experts", default="checkpoints/expert_v1")
    ap.add_argument("--projections", default="checkpoints/projection_v1")
    ap.add_argument("--prototypes", default="checkpoints/prototypes_v1")
    ap.add_argument("--out", default="checkpoints/phase9")
    args = ap.parse_args()
    metrics = run_phase9(
        args.data, args.experts, args.projections, args.prototypes, args.out
    )
    learning = metrics["learning"]
    print("Phase 9 complete:")
    print(f"  proposal samples: {learning['l1']['proposal_samples']}")
    print(f"  review: accepted={learning['review']['accepted']} "
          f"new_acc={learning['review']['regression']['new_acc']:.3f}")
    print(f"  active categories: {learning['package']['active_categories']}")
    print(f"  rollback: {learning['package']['rollback_to']}")
    print(f"  corrupted rejected: {learning['package']['integrity_rejected']}")
    print(f"  audit verified: {metrics['audit']['verified']}")
    print(f"  output: {args.out}")


if __name__ == "__main__":
    main()
