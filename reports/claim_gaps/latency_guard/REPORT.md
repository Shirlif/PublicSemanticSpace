# Claim-gap E21: deadline guard on L2 bundle replay

> generated: 2026-11-26 06:29:56 UTC
> bundles: 19 under `bundles\dev`
> budget: 10.0 ms; repeats: 3

## Aggregate

| mode | runs | BAR mean | BAR min | p99 ms | p99.9 ms | max ms | timeouts | degraded | ID retention | relation rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| off | 57 | 0.9993 | 0.9900 | 1.75 | 5.44 | 67.34 | 0 | 0 | 1.000 | 16520.9 |
| on | 57 | 0.9993 | 0.9895 | 2.53 | 6.41 | 80.18 | 6 | 6 | 1.000 | 16399.6 |
| off_iso | 57 | 0.9995 | 0.9900 | 1.75 | 4.75 | 59.67 | 0 | 0 | 1.000 | 16520.9 |
| on_iso | 57 | 0.9991 | 0.9600 | 2.93 | 4.53 | 72.62 | 2 | 2 | 1.000 | 16482.5 |
| off_slow | 57 | 0.9439 | 0.9333 | 31.31 | 33.91 | 59.94 | 0 | 0 | 1.000 | 16520.9 |
| on_slow | 57 | 0.9986 | 0.9600 | 8.88 | 11.08 | 64.91 | 241 | 241 | 1.000 | 14254.7 |
| off_slow_iso | 57 | 0.9444 | 0.9333 | 31.29 | 32.33 | 57.61 | 0 | 0 | 1.000 | 16520.9 |
| on_slow_iso | 57 | 0.9985 | 0.9667 | 9.14 | 13.25 | 69.16 | 240 | 240 | 1.000 | 14280.5 |

The `on` latency is the guarded caller wait; `worker_late_max_ms` and
`frame_max_ms` show the real work and the catch-up cost.  A guard is not
a claim that the underlying worker became fast.

## Baseline spike characterization

4 of 57 unguarded runs contain a frame above 10.0 ms.

| bundle | repeat | max ms | p99.9 ms | BAR |
|---|---:|---:|---:|---:|
| s1_occlusion_20/18c18c5e/seed3 | 1 | 67.34 | 61.01 | 0.9900 |
| s1_occlusion_20/18c18c5e/seed0 | 0 | 56.52 | 51.25 | 0.9900 |
| s1_occlusion_20/18c18c5e/seed1 | 2 | 54.33 | 49.27 | 0.9900 |
| s1_camera_real/b4bb5f0e/seed0 | 0 | 52.86 | 47.76 | 0.9900 |

## Guard mechanism test (injected slow stage)

Slow injection: 0.0 ms sleep, 30.0 ms busy-loop, every 20 frame(s).

| mode | BAR mean | max ms | timeouts | degraded | frame max ms | ID retention |
|---|---:|---:|---:|---:|---:|---:|
| off_slow | 0.9439 | 59.94 | 0 | 0 | 0.00 | 1.000 |
| on_slow | 0.9986 | 64.91 | 241 | 241 | 72.16 | 1.000 |
| off_slow_iso | 0.9444 | 57.61 | 0 | 0 | 0.00 | 1.000 |
| on_slow_iso | 0.9985 | 69.16 | 240 | 240 | 90.30 | 1.000 |

The guarded p99 stays near 8.9 ms while the unguarded BAR drops to
~0.944.  A residual `on_slow` max above the budget is a guard-thread
preemption event, not an undetected worker overrun; a hard wall-clock
bound needs OS-level real-time scheduling.

## How to read this

- `off` is the current contract: the caller waits for the whole ingest
  and a scheduler/GC spike becomes a BAR violation.
- `on` bounds the caller wait.  Timed-out frames carry empty events,
  enter a short work-shedding cooldown, and are joined before the next
  frame; the catch-up cost appears in `frame_ms`, not in the caller
  latency.
- If `on` loses ID retention or relation rows, that is the price of the
  deadline, not a free win.  Persistent overload should abstain rather
  than degrade forever.
- `off_iso` / `on_iso` add best-effort OS isolation (1 ms timer, highest
  thread priority on Windows).  It reduces guard-thread preemption but
  is not a hard real-time guarantee.
- Wall-clock p99.9/max are load-sensitive; run on a quiet machine for
  comparisons, and report the load conditions with the numbers.
