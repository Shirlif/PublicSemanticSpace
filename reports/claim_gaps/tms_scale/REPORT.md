# Claim-gap F7: TMS scale (100 / 1k / 10k facts)

> generated: 2026-11-26 08:30:54 UTC
> checkpoint interval: 256

## Runtime integration

`RelationFactLayer` (the runtime's fact layer; `EntityLayer` owns object
slots, not facts) now defaults to `TMSFactStore` with
`snapshot_mode='delta'`. The constructor still accepts an injected
`MultiSourceFactStore` or `TMSFactStore`, and `components()['fact_store']`
reports which store is active.

K3/K4/K5/AUD and the E12/E19/E20 regressions are recorded in
`REGRESSION.md` in this directory.

## Snapshot cost: build / update / rollback

`update` is one retract+assert of a single source value (the same path
as `VersionedKnowledgeBase.update`). Times are wall-clock medians over
the per-mode sample count shown in the table.

| facts | mode | updates | build ms | update ms (median) | update ms (p90) | rollback ms | post-rollback relabel ms | rollback exact |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 100 | full | 10 | 7.87 | 4.304 | 4.425 | 1.715 | 0.116 | 1 |
| 100 | cow | 10 | 0.48 | 0.035 | 0.050 | 0.079 | 0.104 | 1 |
| 100 | delta | 10 | 0.33 | 0.013 | 0.017 | 0.062 | 0.091 | 1 |
| 1000 | full | 10 | 27.04 | 45.720 | 48.729 | 15.004 | 0.870 | 1 |
| 1000 | cow | 10 | 3.35 | 0.229 | 0.244 | 0.757 | 0.827 | 1 |
| 1000 | delta | 10 | 3.31 | 0.012 | 0.016 | 0.506 | 0.838 | 1 |
| 10000 | full | 5 | 263.87 | 580.764 | 679.425 | 161.820 | 10.751 | 1 |
| 10000 | cow | 10 | 42.57 | 3.899 | 5.581 | 6.828 | 15.889 | 1 |
| 10000 | delta | 10 | 36.56 | 0.014 | 0.018 | 6.226 | 9.944 | 1 |

## Retained memory

Measured with `tracemalloc`; `update retained` is the additional
memory kept after the update loop, divided by the number of updates.

| facts | mode | store after build MB | update retained MB | per update KB | peak MB |
|---:|---|---:|---:|---:|---:|
| 100 | full | 0.23 | 2.05 | 204.6 | 2.48 |
| 100 | cow | 0.13 | 0.32 | 31.7 | 0.45 |
| 100 | delta | 0.13 | 0.05 | 5.0 | 0.18 |
| 1000 | full | 2.20 | 19.90 | 1989.7 | 22.98 |
| 1000 | cow | 1.20 | 2.47 | 247.2 | 3.67 |
| 1000 | delta | 1.20 | 0.05 | 5.0 | 1.31 |
| 10000 | full | 21.85 | 97.91 | 19581.9 | 127.38 |
| 10000 | cow | 11.85 | 21.43 | 2143.4 | 33.28 |
| 10000 | delta | 11.85 | 0.05 | 5.0 | 12.79 |

## Incremental vs full propagation

Each fact is an assertion with a two-level derived chain
(`assertion -> d1 -> d2`), so `nodes = 3 x facts`. One assertion is
updated per sample; the incremental path should touch only that
fact's chain.

| facts | nodes | initial propagate ms | incremental ms | full ms | speedup | labels equal |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 300 | 0.53 | 0.008 | 0.609 | 78.1x | 1 |
| 1000 | 3000 | 7.20 | 0.009 | 6.949 | 789.6x | 1 |
| 10000 | 30000 | 117.96 | 0.013 | 78.547 | 5905.8x | 1 |

## Where the superlinear cost comes from

- **Build.** `MultiSourceFactStore` copies the assertion table once and
  the TMS bulk-loads justifications without per-item history, so build
  is linear in the fact count for all three modes. The constructor's
  initial checkpoint(s) are the only O(N) snapshots.
- **Update, `full`.** Every mutation deep-copies the whole assertion
  table *and* the whole justification graph. An update is a retract plus
  an assert, so it pays four O(N) deep copies: two store snapshots and
  two TMS snapshots. This is the original cost and it is linear per
  update, hence quadratic over a sequence of updates.
- **Update, `cow`.** Same O(N) asymptotic because the snapshot still
  copies one pointer per node, but the constant is much smaller: no
  recursive deep copy of `Assertion` / `Justification` objects.
- **Update, `delta`.** O(changed keys + changed nodes) per update, with a
  COW checkpoint every `checkpoint_interval` changes. The periodic
  checkpoint is the only O(N) term and it amortises to O(N / interval)
  per update.
- **Rollback.** `full` / `cow` restore a snapshot directly (O(N));
  `delta` replays from the nearest checkpoint (O(interval + changed)).
  All modes then clear labels, so the first query after a rollback pays
  an O(N) relabel; the table reports that separately instead of hiding
  it in the rollback number.
- **Propagation.** `propagate_full()` re-evaluates every node, so its
  cost grows with the whole graph. Incremental `propagate()` evaluates
  only the changed nodes and their transitive dependents, so the cost is
  independent of total fact count for a fixed fan-out. This is the
  reason the 10k-fact update latency is dominated by snapshotting, not
  by truth maintenance.

## How to read this

The incremental propagation number uses `propagate(copy_labels=False)`.
Materialising every label with the default `propagate()` is an O(N) dict
copy by definition; it is not part of the dirty re-evaluation. A
single-node query (`label()`) also skips that copy.

The contract properties (K3/K4/K5/AUD) are unchanged; this suite only
moves cost. `delta` is the runtime default. The 10k-fact update cost is
a measured number, not an extrapolation from 1k.
