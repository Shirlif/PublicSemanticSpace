# s1_occlusion_20 ablation (dev, preliminary)

- episodes: 5  variants: 9

| variant | IDF1 | retention | switches | duplicate | reappearance |
|---|---|---|---|---|---|
| full | 1.0000 | 1.0000 | 0.000 | 0.0000 | 1.000 |
| no_entity | 0.0113 | 0.0114 | 87.370 | 0.9887 | 0.000 |
| no_relations | 1.0000 | 1.0000 | 0.000 | 0.0000 | 1.000 |
| no_memory | 1.0000 | 1.0000 | 0.000 | 0.0000 | 1.000 |
| no_active_set | 1.0000 | 1.0000 | 0.000 | 0.0000 | 1.000 |
| no_uncertainty | 1.0000 | 1.0000 | 0.000 | 0.0000 | 1.000 |
| no_audit | 1.0000 | 1.0000 | 0.000 | 0.0000 | 1.000 |
| flat_context | 0.0113 | 0.0114 | 87.370 | 0.9887 | 0.000 |
| no_tools | 1.0000 | 1.0000 | 0.000 | 0.0000 | 1.000 |
