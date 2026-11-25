# Claim-gap validation

> generated: 2026-11-25 15:07:35 UTC
> design: `notes/CLAIM_GAP_EXPERIMENTS.md`

## Verdict table

| gap | experiment | key result | status |
|---|---|---|---|
| C1-a auditable state | E1 entity audit | provenance/retraction/rollback/tamper all min=1.00 | pass |
| C1-b tactile confound | E2 tactile ablation | tactile +0.117, state +0.489 | measured |
| C2 hybrid relation | E3 hybrid under noise | hybrid_diag vs geometry(tau) at noise=0.15: 0.857 vs 0.828 | supported |
| C3 fact active set | E4 query-conditioned set | k=4: hot 0.237, old 0.241, query 1.000/1.000 | fixed |

## E1: entity-state audit

| property | mean | min | pass |
|---|---:|---:|---|
| provenance_completeness | 1.000 | 1.000 | True |
| retraction_correctness | 1.000 | 1.000 | True |
| id_rollback_exactness | 1.000 | 1.000 | True |
| rebind_identity | 1.000 | 1.000 | True |
| tamper_detection | 1.000 | 1.000 | True |
| audit_chain_valid | 1.000 | 1.000 | True |
| stale_provenance_rate | 0.000 | 0.000 | inverse metric |

| control | targeted metric | mean | detected |
|---|---|---:|---|
| no_audit | provenance_completeness | 0.000 | True |
| no_chain | tamper_detection | 0.000 | True |
| shallow_rollback | id_rollback_exactness | 0.000 | True |

## E2: tactile binding ablation

| variant | ID retention | switches | reappearance |
|---|---:|---:|---:|
| graph_tactile | 0.989 | 0.02 | 0.978 |
| graph_tactile_zero | 0.872 | 0.26 | 0.744 |
| graph_vision_zero | 0.961 | 0.08 | 0.922 |
| simple_tracker | 0.917 | 0.17 | 0.833 |
| text_state_proxy | 0.500 | 1.00 | 0.000 |

| delta | value |
|---|---:|
| tactile_contribution | +0.117 |
| vision_contribution | +0.028 |
| graph_contribution | +0.072 |
| state_contribution | +0.489 |

## E3: hybrid relation under noise

| method | noise=0.0 | noise=0.03 | noise=0.08 | noise=0.15 |
|---|---:|---:|---:|---:|
| geometry_eps0.1 | 1.000 | 0.987 | 0.730 | 0.537 |
| geometry_eps0.25 | 1.000 | 1.000 | 0.980 | 0.828 |
| learned | 0.595 | 0.580 | 0.572 | 0.532 |
| hybrid | 0.595 | 0.580 | 0.567 | 0.503 |
| hybrid_diag | 1.000 | 1.000 | 0.985 | 0.857 |

## E4: query-conditioned fact active set

| method | k=4 | k=8 | k=16 |
|---|---:|---:|---:|
| full | 1.000 | 1.000 | 1.000 |
| recency | 0.074 | 0.259 | 0.313 |
| hot | 0.237 | 0.421 | 0.473 |
| random | 0.001 | 0.004 | 0.008 |
| learned_old | 0.241 | 0.379 | 0.604 |
| query_recency | 1.000 | 1.000 | 1.000 |
| learned_query | 1.000 | 1.000 | 1.000 |

## What changes for the paper

- **C1**: add an entity-audit property (E1). State the persistence mechanism and the
  signal decomposition separately: E2 shows tactile is the largest sensory signal,
  while the state layer's gain over a no-memory proxy is much larger. Do not claim
  the state layer alone carries identity.
- **C2**: narrow the hybrid claim. Learned relations are a calibration channel only
  inside the regime they were trained on; on aligned/held-out axes, exact geometry
  owns the decision. The naive hybrid (learned on every ambiguous axis) fails.
- **C3**: the old negative result was a query-conditioning bug, not a learning limit.
  Report that the simple query-recency rule matches the learned policy; the claim is
  about the query-conditioned interface, not about learned scoring.
- **C5 (deferred)**: the property suite has not been run on a non-PSS baseline.

## Files

- `reports/claim_gaps/entity_audit/`
- `reports/claim_gaps/tactile_binding/`
- `reports/claim_gaps/hybrid_relations/`
- `reports/claim_gaps/fact_active/`
