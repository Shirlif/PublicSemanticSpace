# arXiv submission sheet

> **Note:** A Zenodo version is already published at
> <https://zenodo.org/records/24549223> (DOI `10.5281/zenodo.24549223`).
> If you submit to arXiv later, add it as a related identifier.

This file is a copy-paste sheet for submitting the PSS technical report to
arXiv. The clean source archive is `dist/pss_arxiv_source.zip` (generated from
`paper/`; 0.96 MB, 22 files).

## Metadata

**Title**

```
PSS: A Cognitive Runtime Above Perception
```

The PDF title page also carries the subtitle "Contract-Level Modularity and
Property-Level Evaluation for Embodied AI"; leave it in the PDF.

**Authors**

```
Jiahe Qi
```

ORCID: `0009-0001-8379-631X` (add it in the arXiv author field if the form
offers an ORCID field).

**Primary category**

```
cs.RO (Robotics)
```

**Cross-list**

```
cs.AI (Artificial Intelligence)
```

**License**

```
CC BY 4.0
```

(Code is Apache-2.0; the paper text and figures can be CC BY 4.0.)

**Comments**

```
12 pages, 7 figures in the main text; supplementary material included.
Code and reports: https://github.com/Shirlif/PublicSemanticSpace
```

If the repository is still private at submission time, use:

```
12 pages, 7 figures in the main text; supplementary material included.
Code and reports will be available at
https://github.com/Shirlif/PublicSemanticSpace
```

**Journal reference / DOI**

Leave empty for the first submission.

**Abstract**

```
Embodied systems keep upgrading their perception stack, but their cognitive
state — identity, relations, memory, and audit — usually lives inside an
end-to-end model. Swapping an expert then forces retraining the whole stack,
knowledge updates cannot be inspected or rolled back, and inference cost has no
hard bound. We present PSS, a cognitive runtime above perception: a pluggable
expert layer governed by an admission contract, a public semantic space used as
a calibratable alignment interface rather than a zero-shot coordinate system,
and an explicit state layer that maintains persistent entity identity, typed
relations, active-set memory, a versioned truth-maintenance store, uncertainty
and abstention, and bounded inference. We evaluate PSS at the property level,
treating expert quality as a controlled factor, pairing architecture variants
as twins, and testing the metrics with negative controls. Across L0/L1
experiments the hot-swap contract (swap success, rollback exactness,
non-interference, routing) is invariant to expert quality, while semantic
compatibility is not: cross-family zero-shot swap falls to chance, and one-shot
calibration recovers vision but not tactile. Entity-state audit, typed
relations, active-set selection, and bounded inference are measured separately,
and a replay over the same detection stream isolates the contribution of the
memory layer from perception. We report boundary conditions and negative
results, including a +0.012 fair-attribution re-identification gain, a
closed-set accuracy that is invariant to the projection space, and a case where
retraining experts is no better than rebuilding prototypes on frozen features.
PSS is not a foundation model; it is a contract-level runtime and a
measurement protocol.
```

## Upload

1. Sign in to arXiv and start **New submission**.
2. Choose **TeX** as the source format.
3. Upload `dist/pss_arxiv_source.zip`. The archive root contains `main.tex`,
   `refs.bib`, `supplement.tex`, and `figures/`.
4. Set the compiler to **pdfLaTeX** (the default). arXiv runs bibtex
   automatically.
5. Compile and check the log:
   - expected page count: **12** for `main.tex`;
   - expected errors: **0**;
   - expected warnings: **0**;
   - `supplement.tex` is included as source but is not compiled as the main
     document. If you want a separate supplement PDF, compile it on Overleaf
     and upload it as an ancillary file.
6. Preview the PDF. Check page 1 (title/abstract) and page 6 (Table 2).
7. Fill the metadata above, then submit.

## If the compile fails

- **Citations show `[?]`**: download `output.bbl` from the Overleaf project,
  rename it `main.bbl`, and add it to the upload. Or re-run bibtex in the arXiv
  compile settings.
- **Missing package**: the paper uses `newtxtext`, `newtxmath`, `newtxtt`,
  `xurl`, `tikz`, `natbib`, `booktabs`, `hyperref`; all are in arXiv's TeX
  Live. If a package is missing, remove the corresponding font package and
  fall back to `times` (see git history).
- **Wrong main file**: make sure `main.tex` is selected as the main document,
  not `supplement.tex`.

## Endorsement

If arXiv asks for an endorsement for `cs.RO`, you need an endorser with a
`cs.RO` submission history. If that is not available, submit to `cs.AI` as the
primary category (the paper is also an AI systems/evaluation paper) and
cross-list `cs.RO` if allowed. Do not change the category to something the
paper does not fit just to bypass endorsement.

## After submission

- [ ] Make the GitHub repository public **before** the announcement if the
      paper's comments point to it.
- [ ] Add the arXiv ID/DOI to `CITATION.cff`, `README.md`, and
      `docs/PRIORITY.md`.
- [ ] Tag the repository `v0.1.0` (or the matching version) and, optionally,
      connect Zenodo for a software DOI.
