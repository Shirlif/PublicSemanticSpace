# PSS v0.1.1 — L2 reproduction pack

Second tagged release of **Public Semantic Space (PSS)**. This release does not
change the technical report or the architecture claims; it makes the L2
bundle-replay evidence (E12) publicly reproducible.

## What's new

- **Public L2 bundle pack**: 33 bundles, 244 files, compressed to 5.5 MB.
  The pack contains the input tables needed by the E12 replay; rendered images
  and derived replay outputs are excluded.
- **Checksum-verified fetch**: `scripts/fetch_repro_bundles.py` verifies the
  archive SHA256, rejects unsafe tar members, and checks every file against
  `PACK_MANIFEST.json`.
- **One-command reproduction**: `scripts/reproduce_l2.py` reruns the E12
  ablation and the perception subset, compares against the published numbers,
  and exits non-zero on mismatch.
- **Docs**: `docs/reproducibility.md`, `docs/repro/PSS_BUNDLES_v0.1.1.json`,
  and `reports/repro/REPRODUCTION.md`.
- **Runner extension**: `run_gap_l2_memory.py` gains `--require-emb` and
  reusable valid-bundle helpers.

## Artifact

| Field | Value |
|---|---|
| Asset | `pss-l2-bundles-v0.1.1.tar.gz` |
| Size | 5,463,441 bytes |
| SHA256 | `9bd50f418d21e3372c3b77bdb300e9ec73926979a24b8cf70e2f324c34c7984e` |
| Bundles | 33 |
| Files | 244 |
| Bundles with detector embeddings | 4 |

Download: <https://github.com/Shirlif/PublicSemanticSpace/releases/download/v0.1.1/pss-l2-bundles-v0.1.1.tar.gz>

## Quick start

```bash
pip install -e ".[train,sim]"
python scripts/fetch_repro_bundles.py \
  --url https://github.com/Shirlif/PublicSemanticSpace/releases/download/v0.1.1/pss-l2-bundles-v0.1.1.tar.gz \
  --sha256 9bd50f418d21e3372c3b77bdb300e9ec73926979a24b8cf70e2f324c34c7984e \
  --out bundles/repro
python scripts/reproduce_l2.py --root bundles/repro --out reports/repro
```

Or, with the archive already downloaded:

```bash
python scripts/fetch_repro_bundles.py \
  --archive pss-l2-bundles-v0.1.1.tar.gz \
  --out bundles/repro
make reproduce-l2
```

## Expected results

E12, valid bundles only (19 of 33):

| variant | n | ID retention | switches |
|---|---:|---:|---:|
| full | 19 | 1.000 | 0.00 |
| no_memory | 19 | 0.827 | 0.56 |
| no_entity | 19 | 0.270 | 47.47 |
| no_relations | 19 | 1.000 | 0.00 |

The 4 bundles with detector embeddings (`s1_hard`, `s1_occlusion_real`,
`s2_twins`, `s1_camera_real`) show retention 1.000 for all variants. This is a
null memory-ablation result, not a reproduction failure: those bundles exercise
the detector-to-runtime path. The memory signal comes from the occlusion and
oracle bundles.

## Scope and limitations

- The pack reproduces **E12 bundle replay only**. It does not run fresh Isaac
  S4-PSS episodes.
- Images are excluded; re-running GroundingDINO requires
  `python scripts/pack_repro_bundles.py --include-images`.
- The 4 perception bundles do not exercise memory ablation; see
  `docs/reproducibility.md` §5.
- The technical report and its DOI are unchanged:
  `10.5281/zenodo.24549223`.

## Citation

Use `CITATION.cff` (version 0.1.1). The paper citation remains:

> Qi, J. (2026). *PSS: A Cognitive Runtime Above Perception* (Version 0.1.0).
> Zenodo. https://doi.org/10.5281/zenodo.24549223

## License

Apache License 2.0. See `LICENSE` and `NOTICE`; both are included in the
bundle pack.
