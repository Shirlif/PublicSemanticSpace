# Claim-gap E3: hybrid relation under coordinate noise

> train regions: ['A', 'B', 'C']; held-out pairs: 14
> tau=0.25, strict eps=0.1, seeds=3

| method | noise=0.0 | noise=0.03 | noise=0.08 | noise=0.15 |
|---|---:|---:|---:|---:|
| geometry_eps0.1 | 1.000 ± 0.000 | 0.987 ± 0.002 | 0.730 ± 0.008 | 0.537 ± 0.007 |
| geometry_eps0.25 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.980 ± 0.003 | 0.828 ± 0.008 |
| learned | 0.595 ± 0.034 | 0.580 ± 0.012 | 0.572 ± 0.004 | 0.532 ± 0.003 |
| hybrid | 0.595 ± 0.034 | 0.580 ± 0.012 | 0.567 ± 0.005 | 0.503 ± 0.004 |
| hybrid_diag | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.985 ± 0.003 | 0.857 ± 0.007 |

## Verdict

| noise | hybrid ≥ geometry | hybrid > geometry | hybrid ≥ learned | hybrid ≥ geometry(tau) | hybrid_diag > geometry(tau) |
|---:|---|---|---|---|---|
| 0.0 | False | False | True | False | False |
| 0.03 | False | False | True | False | False |
| 0.08 | False | False | False | False | True |
| 0.15 | False | False | False | False | True |

`geometry_eps0.25` is the key control: if hybrid matches a wider geometric dead zone,
the gain comes from the dead zone, not from the learned head. A hybrid claim is only
supported if `hybrid > geometry(tau)` at some noise level.
