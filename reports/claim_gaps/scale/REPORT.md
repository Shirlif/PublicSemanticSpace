# Claim-gap E8: scale evidence

> seed: 0

## Representation dimension

| dim | SSR | RBE | INT | RCR | ZCR | swap ms |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1.000 | 1.000 | 0.000 | 1.000 | 0.674 | 0.096 |
| 256 | 1.000 | 1.000 | 0.000 | 1.000 | 0.674 | 0.127 |
| 512 | 1.000 | 1.000 | 0.000 | 1.000 | 0.667 | 0.118 |

## Fact count

| facts | build ms | update ms | rollback exact | audit | locality |
|---:|---:|---:|---:|---:|---:|
| 100 | 1.7 | 5.089 | 1 | 0 | 1 |
| 500 | 9.0 | 30.451 | 1 | 0 | 1 |
| 1000 | 17.1 | 52.928 | 1 | 0 | 1 |

## Scene size

| entities | relations | active entities | active relations | bounded | latency ms |
|---:|---:|---:|---:|---|---:|
| 50 | 100 | 50 | 100 | True | 0.004 |
| 200 | 1000 | 50 | 100 | True | 0.025 |
| 500 | 2000 | 50 | 100 | True | 0.036 |

## How to read this

The contract metrics should stay flat across dim and data size; latency is allowed
to grow. The fact sweep uses the versioned KB, whose snapshot-on-change design is
O(history) per update, so build time is expected to superlinear; that is a
known cost, not a property failure.
