# Phase 20 — Gap closure 3: object attention as a binding option

> Date: 2026-11-24
> Scope: `PSS-认知层架构定位.md` §5, row 3 (entity binding)
> Data: `tracking_bench_v2`, `tracking_bench_long` (synthetic)
> Result artifacts: `eval/results/runtime_binding/`

---

## 1. What changed

Row 3 target: **object attention / persistent slots**. Persistent slots were
already implemented in `EntityLayer`; this phase makes object attention a
first-class association option in the runtime.

| Component | Change |
|---|---|
| `ObjectAttentionReID` | `pss/memory/object_attention.py` adapter: attention scores -> tracker cost matrix |
| save/load | stores token dim, d_model, heads, layers, tactile flag |
| `EntityLayer(associator=...)` | builds `TrackerConfig(metric=...)`; records `associator_name` |
| `PSSRuntime(associator=...)` | passes the associator to the entity layer |
| `components()` | reports `associator: ObjectAttentionReID` / `hand` |

The hand cost and graph matcher remain selectable. The runtime default stays
`hand` unless an associator is supplied.

---

## 2. Results

Trained `ObjectAttentionMatcher` on `tracking_bench_v2` train/val; evaluated
on test pairs and through the tracker.

Pair re-ID (one seed):

| method | overall | near-identical |
|---|---:|---:|
| graph matcher | 1.000 | 1.000 |
| object attention | 1.000 | 1.000 |

Tracker retention / latency:

| dataset | associator | ID retention | switches | reappearance | ms/frame |
|---|---|---:|---:|---:|---:|
| v2 | hand | 1.000 | 0.00 | 1.000 | 3.4 |
| v2 | graph matcher | 1.000 | 0.00 | 1.000 | 2.3 |
| v2 | object attention | 1.000 | 0.00 | 1.000 | 2.5 |
| long | hand | 1.000 | 0.00 | 1.000 | 0.7 |
| long | graph matcher | 1.000 | 0.00 | 1.000 | 0.7 |
| long | object attention | 1.000 | 0.00 | 1.000 | 0.8 |

Runtime integration: `components.associator = "ObjectAttentionReID"` after
constructing `PSSRuntime(associator=...)`; the entity layer uses the attention
cost for re-ID.

---

## 3. Findings

1. **Object attention is now a runtime binding option**, not an offline
   experiment. The adapter shape matches the graph matcher, so the tracker and
   runtime do not special-case it.
2. **The benchmark is saturated.** One seed reaches 1.000 for graph matcher and
   object attention on both v2 and the 70-frame long bench. The prior 5-seed
   P1-15 result was 0.987 vs 0.989 (near 0.956 vs 0.956): a tie, not a win.
3. **No superiority claim.** This row closes the option/architecture gap. It
   does not show object attention beats the graph matcher.
4. **Persistent slots already existed.** The runtime `EntityLayer` owns track
   IDs, lifecycle, attributes, confidence and provenance; the new path adds
   the attention-based association.

---

## 4. Boundaries

- One seed; the benchmark saturates, so the numbers cannot distinguish the
  matchers.
- Object attention is trained on the same category/tactile tokens; no extra
  relation supervision in this run.
- The runtime default remains `hand`; object attention is opt-in. Making it the
  default is a separate decision that should wait for a benchmark where it
  actually differs.
- Synthetic data only.

---

## 5. Next row

**Row 4 (relations): learned/versioned rules.** Typed relation heads are
already the runtime default; the remaining hand-written part is the symbolic
rule thresholds and the planner operators. The next step is to learn/version
those rules while keeping exact geometry as a fallback.

Reproduce:

```bash
python -m pss.eval.scripts.run_runtime_binding --device cuda --sequences 8
```
