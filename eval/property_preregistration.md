# Property-Level Preregistration (PSS)

**Status:** frozen before formal runs
**Date:** 2026-11-25
**Plan:** `eval/EXPERIMENT_PLAN_PROPERTIES.md`
**Scope:** validate three architectural properties independently of sensory-expert quality.

---

## 1. Claims under test

- **CL1 (hot-swap):** A contract-compliant expert can replace another without retraining other modules, and the shared-space contract survives.
- **CL2 (KB hot update):** A versioned KB update is atomic, local, propagatable, conflict-detectable, rollback-exact and auditable.
- **CL3 (bounded inference):** The runtime respects configured compute bounds, returns valid output at interruption, and degrades gracefully.
- **CL4 (expert invariance):** The three property metrics are explained primarily by architecture, not expert quality.
- **CL5 (test validity):** Negative controls fail as expected; random experts pass property tests but fail capability tests.

---

## 2. Hypotheses and kill criteria

| ID | Hypothesis | Kill criterion |
|---|---|---|
| PH1 | Same-modality zero-shot swap success is independent of expert quality (slope ≈ 0). | Slope significantly negative over Q1–Q4, and Q2 success < 0.70. |
| PH2 | Cross-modal transplant works through the shared space. | Cross-modal success ≤ chance. |
| PH3 | Versioned KB beats flat KB on locality/rollback. | Versioned KB locality violations > 0 or rollback exactness < 1.00. |
| PH4 | The budget controller respects the bound in adversarial scenes. | Any scene budget adherence < 1.00 or p99 > 2× bound. |
| PH5 | Architecture effect variance ≥ 3× expert effect variance. | Variance ratio < 1.5, or property metric decreases monotonically with expert quality. |
| PH6 | Negative controls fail as expected. | Any negative control passes. |

---

## 3. Primary metrics (frozen)

### Hot-swap

- **SSR** — swap success rate.
- **ZCR** — zero-shot compatibility (top-1 agreement after swap).
- **INT** — interference: max absolute change in unrelated-module metric.
- **RBE** — rollback exactness (1.00 required).
- **DT** — downtime during swap.

### KB update

- **USR** — update success rate.
- **LOC** — locality violations (0 required).
- **PCR** — propagation correctness.
- **CDR** — conflict detection rate.
- **RBE-KB** — rollback exactness (business state, excluding audit/version).
- **AUD** — audit completeness.

### Bounded inference

- **BAR** — budget adherence rate (1.00 required).
- **MON** — monotonicity rate of quality vs. budget.
- **p99** — 99th percentile latency.
- **DCR** — degradation-FSM correctness.
- **CAT** — cost attribution error.

Secondary metrics (exploratory): swap latency, update latency, forgetting rate, resolution correctness, anytime quality curve, worst-case bound.

---

## 4. Thresholds (frozen)

| Metric | Threshold |
|---|---:|
| SSR (same-modality) | ≥ 0.90 |
| ZCR (same-modality) | ≥ 0.90 |
| ZCR (cross-modal) | ≥ 0.70 |
| INT | ≤ 0.02 |
| RBE / RBE-KB | = 1.00 |
| DT | ≤ 1% |
| USR | ≥ 0.99 |
| LOC | = 0 |
| PCR | = 1.00 |
| CDR | ≥ 0.95 |
| AUD | = 1.00 |
| BAR | = 1.00 |
| MON | ≥ 0.95 |
| p99 | ≤ configured bound |
| DCR | ≥ 0.95 |
| CAT | ≤ 5% |
| Variance ratio (PH5) | ≥ 3.0 |
| TOST equivalence bound δ | 0.05 absolute or 5% of baseline |

Any change to a primary metric or threshold after this point must be recorded as an amendment and labelled exploratory.

---

## 5. Design

### Levels

- **L0:** structured synthetic observations + expert stubs; no perception.
- **L1:** frozen real experts + controlled quality injection; perception in the loop.
- **L2:** real experts; external validity only.

Primary evidence: L0 + L1. L2 does not overturn a property claim unless it reveals a contract violation.

### Factors

- Expert quality: Q0 (oracle), Q1 (0.90), Q2 (0.75), Q3 (0.55), Q4 (random), Q5 (adversarial).
- Architecture: A0 full, A1 per-modality, A2 flat KB, A3 unbounded, A4 naive swap, A5 retrain-all, A6 no active set, A7 flat retrieval, A8 no degradation.
- Difficulty: T1 easy, T2 medium, T3 hard.

### Pairing

Each treated run has a control twin with the same spec hash, seed, scene and episode; only the architecture/operation differs. Paired differences are the unit of causal analysis.

### Sample size

- L0: exhaustive / seeded random, 1,000+ operation sequences per property.
- L1: 5 seeds × 10 episodes per cell; pilot first, power analysis before full factorial.
- If power < 0.80 for d = 0.50, increase seeds before episodes.

---

## 6. Analysis plan

### Models

Mixed-effects model per primary metric:

```
y ~ expert_quality * architecture * difficulty + (1 | scene) + (1 | seed)
```

Report marginal R², conditional R², and variance shares.

### Expert invariance

- Slope of property metric vs. expert quality (with 95% bootstrap CI).
- TOST with δ as above.
- Variance ratio `Var(architecture) / Var(expert)`.

### Multiplicity

- Primary metrics: Holm–Bonferroni.
- Exploratory metrics: Benjamini–Hochberg FDR.
- Report effect sizes and CIs, not only p-values.

### Exclusions

- Runs with corrupted bundles, NaN metrics, or crashed runtimes are reported separately; they are never silently dropped.
- Crash rate is itself a property metric.
- No manual episode selection.

---

## 7. Negative controls (must fail/pass as specified)

| ID | Control | Expected |
|---|---|---|
| N1 | A2 flat KB on K2/K5 | fail (LOC > 0 / RBE < 1) |
| N2 | A3 unbounded on C1/C4 | fail (BAR < 1) |
| N3 | A4 naive swap on H1/H7 | fail (low ZCR) |
| N4 | Q4 random on capability | fail (chance accuracy) |
| N5 | Q4 random on property | pass (no crash, contract holds) |
| N6 | no-op update | identity |
| N7 | shuffled embeddings | capability fails, property passes |
| N8 | deliberately over-budget controller | BAR catches it |

If any negative control does not behave as specified, the test suite is considered invalid and must be fixed before formal runs.

---

## 8. Reporting

- All preregistered conditions, including null results.
- Tables: metric × expert quality; ablation matrix; negative controls; variance decomposition.
- Figures: invariance plot; KB rollback/state hash; quality-vs-budget curve.
- Raw JSON scorecards under `scorecards/property/`.
- Report under `reports/property_validation/`.

---

## 9. Deviations

Record every deviation here with date, reason, and whether it is exploratory.

| Date | Deviation | Reason | Exploratory? |
|---|---|---|---|
| 2026-11-25 | ZCR reclassified as capability-linked; PH1 uses SSR/RBE/INT/RCR, and ZCR is reported as a ratio over the expert-quality baseline. | L1 showed ZCR decreases monotonically with injected quality (Q0 0.969 → Q3 0.587, 3 seeds) while swap/rollback/interference stay invariant. Absolute ZCR therefore measures capability, not the swap contract. | No — interpretation clarified. |
| 2026-11-25 | L2 bundle latency bound 10 ms was not met (BAR 0.9993). | 2/14 bundle traces contain 52–63 ms single-frame spikes; entity-count budget held at 100%. Reported as a partial result and a need for a latency guard. | No — reported. |
| 2026-11-26 | Added a post-hoc deadline guard (E21) and reported guarded BAR alongside the original wall-clock BAR. | The guard bounds the caller wait and sheds optional relation work; it does not replace the preregistered wall-clock metric and cannot prevent OS preemption of the guard thread. | Yes — exploratory. |
| 2026-11-25 | L1 was run with seed 0 only. | L1 loads two real expert families and is comparatively expensive; L0 covers seed variance. | Yes — exploratory. |
