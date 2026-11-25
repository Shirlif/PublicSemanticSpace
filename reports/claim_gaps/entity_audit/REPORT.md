# Claim-gap E1: entity-state audit

> seeds: 5, dim: 128
> L0 structured observations; real `Entity` / `build_entity` / `rebuild_for_slot` paths.

| property | mean | min | std | threshold | pass |
|---|---:|---:|---:|---:|---|
| provenance_completeness | 1.000 | 1.000 | 0.000 | 1.0 | True |
| retraction_correctness | 1.000 | 1.000 | 0.000 | 1.0 | True |
| id_rollback_exactness | 1.000 | 1.000 | 0.000 | 1.0 | True |
| rebind_identity | 1.000 | 1.000 | 0.000 | 1.0 | True |
| tamper_detection | 1.000 | 1.000 | 0.000 | 1.0 | True |
| audit_chain_valid | 1.000 | 1.000 | 0.000 | 1.0 | True |
| stale_provenance_rate | 0.000 | 0.000 | 0.000 | n/a | n/a |

**All pass: True**

## Per-seed

| seed | provenance | retraction | rollback | rebind | tamper | chain | entries |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 36 |
| 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 36 |
| 2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 36 |
| 3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 36 |
| 4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 36 |

## Negative controls

| control | targeted metric | mean | detected |
|---|---|---:|---|
| no_audit | provenance_completeness | 0.000 | True |
| no_chain | tamper_detection | 0.000 | True |
| shallow_rollback | id_rollback_exactness | 0.000 | True |

**Controls valid: True**

## What this closes

The cognitive-runtime claim says entity state is inspectable. This suite measures it:
every state change carries provenance, retractions are exact, rebind preserves identity,
rollback is bit-exact, and an edited audit entry is detected. It does not measure
perception; it measures the state layer.
