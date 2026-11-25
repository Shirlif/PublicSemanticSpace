# S1 three-tier comparison (dev)

4 cubes, 1 seed, W1 tabletop with moving occluder. Detector tiers: oracle / noisy / frozen_real (GroundingDINO Swin-T + DINOv2-small embeddings).

| tier | IDF1 | retention | switches | duplicate | reappearance |
|---|---|---|---|---|---|
| oracle | 1.0000 | 1.0000 | 0.000 | 0.0000 | 1.000 |
| noisy | 0.9424 | 1.0000 | 0.000 | 0.0000 | 1.000 |
| frozen_real | 0.8942 | 1.0000 | 0.000 | 0.0000 | 1.000 |

- oracle is an upper bound, not capability evidence.
- noisy uses the default miss/pose/FP profile; frozen_real uses rendered RGB-D.
