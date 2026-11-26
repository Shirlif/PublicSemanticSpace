# L2 bundle replay reproduction

> generated: 2026-11-26 05:57:47 UTC
> root: `bundles/repro`
> bundles: 33 total, 19 valid

## E12 cognitive-layer ablation (valid bundles only)

| variant | n | ID retention | switches |
|---|---:|---:|---:|
| full | 19 | 1.000 | 0.00 |
| no_memory | 19 | 0.827 | 0.56 |
| no_entity | 19 | 0.270 | 47.47 |
| no_relations | 19 | 1.000 | 0.00 |

## Perception subset (non-empty detector embeddings)

Bundles: 4.

| variant | n | ID retention | switches |
|---|---:|---:|---:|
| full | 4 | 1.000 | 0.00 |
| no_memory | 4 | 1.000 | 0.00 |
| no_entity | 4 | 1.000 | 0.00 |
| no_relations | 4 | 1.000 | 0.00 |

The perception subset is the 4 bundles whose detection tables carry
GroundingDINO boxes and DINOv2 crop embeddings.  Their ablation is a
null result (all variants retain 1.000): these bundles exercise the
detector-to-runtime path, not the memory layer.  The memory signal comes
from the occlusion/oracle bundles in the full set.

## Check

**PASS** — published E12 numbers reproduced at the report's precision (retention 3 dp, switches 2 dp).

## Reproduce

```bash
python scripts/fetch_repro_bundles.py --url <URL> --out bundles/repro
python scripts/reproduce_l2.py --root bundles/repro --out reports/repro
```
