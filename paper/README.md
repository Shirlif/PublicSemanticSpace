# PSS paper skeleton (arXiv)

This directory contains a LaTeX skeleton for the PSS paper, based on
`notes/PAPER_WRITING_MAP.md`. It is a starting point, not a submission-ready
manuscript. The text is in English so it compiles on arXiv without CJK packages.

## Files

- `main.tex` — title, abstract, sections, tables, figures, appendix.
- `supplement.tex` — additional diagnostics (conflict/open-set, space/RAG,
  KB state), compiled separately.
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

## Compiled result

- Published record: <https://zenodo.org/records/24549223>
  — DOI `10.5281/zenodo.24549223` (concept DOI `10.5281/zenodo.24549222`).
- Drafting was done in a private Overleaf project; the public source of record
  is this directory.
- `main.tex`: **12 pages**, 0 errors, 0 warnings (pdfTeX, 2026-11-26).
- `supplement.tex`: additional figures and summary tables; compile separately
  if needed.
- Typeset with `newtxtext` / `newtxmath` / `newtxtt` (a consistent Times family);
  code and paths remain monospace by design.
- Tables use `\tabcolsep=8pt` and `\arraystretch=1.05` to avoid the cramped look
  noted in review; long paths use `xurl`/`\path`, so they break instead of
  producing an overfull line.

The full version with all appendices compiled to 16 pages. The 12-page main
moves the additional diagnostics, the claim–evidence table, and redundant
summary tables to `supplement.tex`, uses 0.75 in margins, and uses
0.80\linewidth for the two-panel figures. Keep the main text at 10–12 pages if
you edit it.



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
