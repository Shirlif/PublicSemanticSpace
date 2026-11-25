# PSS property scorecard — L0

- seed: `0`
- observations: `200`

## Metrics

| metric | value | threshold | pass |
|---|---:|---:|---|
| SSR | 1 | 0.9 | ✅ |
| ZCR | 0.674 |  |  |
| ZCR_same_modality | 0.9 | 0.9 | ✅ |
| ZCR_cross_modal | 0.9 | 0.7 | ✅ |
| INT | 0 | 0.02 | ✅ |
| RBE | 1 | 1 | ✅ |
| DT | 0 | 0.01 | ✅ |
| RCR | 1 | 0.95 | ✅ |
| SRR | 1 | 1 | ✅ |
| old_module_retained | 1 |  |  |
| swap_latency_ms | 0.22968 |  |  |
| USR | 1 | 0.99 | ✅ |
| LOC | 0 | 0 | ✅ |
| FOR | 0 |  |  |
| PCR | 1 | 1 | ✅ |
| CDR | 1 | 0.95 | ✅ |
| RES | 1 | 1 | ✅ |
| RBE-KB | 1 | 1 | ✅ |
| AUD | 1 | 1 | ✅ |
| UL_ms | 0 |  |  |
| BAR | 1 | 1 | ✅ |
| MON | 1 | 0.95 | ✅ |
| p95_ms | 0.21444 |  |  |
| p99_ms | 0.217608 |  |  |
| DCR | 1 | 0.95 | ✅ |
| CAT | 0 | 0.05 | ✅ |
| WCB | 0 | 0 | ✅ |
| anytime_valid | 1 |  |  |
| interaction_pass_rate | 1 | 1 | ✅ |
| HK_ok | 1 |  |  |
| HK_rollback | 1 |  |  |
| HC_ok | 1 |  |  |
| KC_ok | 1 |  |  |
| HKC_ok | 1 |  |  |

## Suites

### H

```json
{
  "SSR": 1.0,
  "ZCR": 0.674,
  "ZCR_same_modality": 0.9,
  "ZCR_cross_modal": 0.9,
  "INT": 0.0,
  "RBE": 1.0,
  "DT": 0.0,
  "RCR": 1.0,
  "SRR": 1.0,
  "old_module_retained": 1.0,
  "swap_latency_ms": 0.22968000266700983
}
```

### K

```json
{
  "USR": 1.0,
  "LOC": 0.0,
  "FOR": 0.0,
  "PCR": 1.0,
  "CDR": 1.0,
  "RES": 1.0,
  "RBE-KB": 1.0,
  "AUD": 1.0,
  "UL_ms": 0.0
}
```

### C

```json
{
  "BAR": 1.0,
  "MON": 1.0,
  "p95_ms": 0.2144400088582188,
  "p99_ms": 0.21760800504125655,
  "DCR": 1.0,
  "CAT": 0.0,
  "WCB": 0.0,
  "anytime_valid": 1.0
}
```

### I

```json
{
  "interaction_pass_rate": 1.0,
  "HK_ok": 1.0,
  "HK_rollback": 1.0,
  "HC_ok": 1.0,
  "KC_ok": 1.0,
  "HKC_ok": 1.0
}
```

