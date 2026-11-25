# Claim-gap E2: tactile binding ablation

> data: `data\tracking_bench_long`, sequences: 60, seeds: 3
> graph includes tactile=True, dino=False

| variant | ID retention | switches | reappearance |
|---|---:|---:|---:|
| graph_tactile | 0.989 | 0.02 | 0.978 |
| graph_tactile_zero | 0.872 | 0.26 | 0.744 |
| graph_vision_zero | 0.961 | 0.08 | 0.922 |
| simple_tracker | 0.917 | 0.17 | 0.833 |
| text_state_proxy | 0.500 | 1.00 | 0.000 |

## Contribution deltas (ID retention)

| delta | value |
|---|---:|
| tactile_contribution | +0.117 |
| vision_contribution | +0.028 |
| graph_contribution | +0.072 |
| state_contribution | +0.489 |

## Per subset

| variant | mixed | samecat | near-identical |
|---|---:|---:|---:|
| graph_tactile | 1.000 | 1.000 | 0.956 |
| graph_tactile_zero | 1.000 | 0.852 | 0.637 |
| graph_vision_zero | 1.000 | 0.933 | 0.911 |
| simple_tracker | 1.000 | 0.978 | 0.689 |
| text_state_proxy | 0.500 | 0.500 | 0.500 |

## How to read this

`tactile_contribution` is the drop when the tactile channel is zeroed with the
same tracker. If it is large, C1 must say: the state layer provides the persistence
mechanism, but the identity signal is carried by tactile. `state_contribution` is
the gain over the one-frame text proxy; it isolates persistence from perception.
