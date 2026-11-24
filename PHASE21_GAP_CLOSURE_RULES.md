# Phase 21 — Gap closure 4: learnable, versioned relation rules

> Date: 2026-11-24
> Scope: `PSS-认知层架构定位.md` §5, row 4 (relations)
> Data: `data/phase7_v2` episodes (synthetic)
> Result artifacts: `eval/results/runtime_rules/`

---

## 1. What changed

Row 4 target: **typed object attention + learned rules**. The typed relation
heads were already the runtime default (Phase 12/18). This phase makes the
remaining hand-written geometric thresholds learnable and versioned:

| Component | Change |
|---|---|
| `RelationRuleSet` | `pss/reasoning/relations/rules.py`: near/contact/bin thresholds |
| fitting | grid search per predicate against simulator GT |
| version/rollback | `update()` increments version, `rollback(snapshot)` restores |
| runtime | `RelationFactLayer.add_geometry()` consumes the rule set |
| audit | `components.relation_rules` reports version/source/thresholds |

The hand thresholds remain the default/fallback; the learned set is opt-in and
stored at `checkpoints/relation_rules/rules.json`.

---

## 2. Results

Fit on the first half of the phase7 episode frames, evaluate on the second
half.

| threshold | hand | learned |
|---|---:|---:|
| `near_dist` | 0.120 | 0.120 |
| `contact_pad` | 0.010 | 0.072 |
| `bin_half` | 0.060 | 0.030 |
| `bin_z` | 0.140 | 0.140 |

| predicate | hand P/R/F1 | learned P/R/F1 |
|---|---:|---:|
| `in` (support 16) | 1.000 / 1.000 / **1.000** | 1.000 / 1.000 / **1.000** |
| `contact` (support 18) | 0.000 / 0.000 / **0.000** | **0.200 / 0.444 / 0.267** |
| `near` (support 0) | n/a | n/a |

- The learned `contact_pad` improves contact F1 from 0.000 to **0.267** while
  preserving `in` at 1.000.
- `near` has no ground-truth examples in this split; both rule sets predict no
  near relations, so it is n/a rather than a win/loss.
- The rule set is versioned: an edit increments the version and a snapshot
  restores the previous thresholds; the runtime reports
  `components.relation_rules`.

Runtime integration: `RelationFactLayer(rule_set=learned)` adds the learned
geometry relations to `CognitiveState`; the component record includes the
thresholds and version.

---

## 3. Findings

1. **Rule thresholds are learnable and auditable.** The rule set is a small,
   inspectable object with a version and a snapshot, not a black box.
2. **Contact improves modestly.** 0.000 → 0.267 is real but limited: the rule
   uses bounding spheres while the GT contact comes from collision geometry.
   The feature is the bottleneck, not the threshold search.
3. **`in` is preserved.** Learning does not disturb the predicate that already
   worked.
4. **The typed relation heads remain the stronger learned channel.** Phase 12
   measured spatial 0.957 / semantic 0.996 / tactile 0.970 on tracking pairs;
   the rule set is a calibration layer for the symbolic predicates, not a
   replacement.
5. **Version/rollback works.** Rule edits are reversible and appear in the
   runtime component record.

---

## 4. Boundaries

- `phase7_v2` has no `near` ground truth in this split; `near` is n/a.
- Contact GT uses collision geometry while the rule uses bounding spheres;
  the F1 ceiling is feature-limited.
- Small split (first half / second half of 40 frames).
- The rule set is threshold calibration, not a new relation model.
- No real data.

---

## 5. Next row

**Row 5 (planning): learning + search.** Keep A*/BFS as the search, add a
learned cost-to-go heuristic trained from A* labels, and report expansions /
latency / success against the hand heuristic. The planner operators remain
versioned from Phase 17.

Reproduce:

```bash
python -m pss.eval.scripts.run_runtime_rules
```
