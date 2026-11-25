# PSS paper skeleton (arXiv)

This directory contains a LaTeX skeleton for the PSS paper, based on
`notes/PAPER_WRITING_MAP.md`. It is a starting point, not a submission-ready
manuscript. The text is in English so it compiles on arXiv without CJK packages.

## Files

- `main.tex` — title, abstract, sections, tables, figures, appendix.
- `refs.bib` — bibliography used by `main.tex`.
- `figures/` — generated figures (PDF for LaTeX, PNG for preview).

## Regenerate the figures

All figures except the TikZ architecture diagram are generated from the
claim-gap and property reports:

```bash
python scripts/make_paper_figures.py
```

This writes `paper/figures/fig_*.pdf` and `.png`. The figure plan, including
which report each figure comes from, is in
`notes/PAPER_WRITING_MAP.md` §4.


## Compile

There is no TeX toolchain in this environment. Use Overleaf or a local TeX Live:

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Or upload the whole `paper/` directory to arXiv/Overleaf.

## Before submitting

- [ ] Fill the author block (currently from `CITATION.cff`).
- [ ] Replace or remove TODO markers; add the missing prose.
- [ ] Verify every number against `reports/claim_gaps/` and
      `scorecards/property/`.
- [ ] Check `refs.bib` metadata; update arXiv IDs/venues if needed.
- [ ] Add the Zenodo/arXiv DOI once assigned.
- [ ] Reconcile the N3 label with `eval/property_preregistration.md`: the
      preregistration lists a naive-swap control, while the executed results
      table uses the no-admission-validation control. Either add a deviation
      entry to the preregistration or use the preregistered label in the paper.
- [ ] Check the target venue's preprint policy.
- [ ] Make sure `.tex` comments contain nothing sensitive or unpublished:
      arXiv archives the source, comments included.
- [ ] Keep the paper-map "do not write" list: no SOTA, no zero-shot public
      space, no architecture-dominates-capability, no real-robot claims.

## Where the numbers come from

- `reports/claim_gaps/REPORT.md` — E1–E4 (first batch).
- `reports/claim_gaps/FOLLOWUPS.md` — E5–E20 (§10 follow-ups).
- `reports/property_validation/SECTION15_RESULTS.md` — L0/L1 property results.
- `scorecards/property/` — raw scorecards and manifests.

## Notes

- The figures are copied for convenience; if you regenerate them, update
  `paper/figures/`.
- `\todo{}` is defined in the preamble but unused; use it for visible TODOs if
  you prefer.
- The skeleton deliberately reports negative results and limitations. Do not
  remove them to make the paper look stronger; they are part of the claim map.
