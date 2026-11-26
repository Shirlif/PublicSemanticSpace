# PSS v0.1.0 — Technical Report + Property Validation

First tagged release of **Public Semantic Space (PSS)**, a contract-level modular
architecture for embodied AI.

## Technical report

- **PSS: A Cognitive Runtime Above Perception** — 12-page technical report
  (`paper/main.tex`; PDF and TeX source attached to this release).
- **DOI**: `10.5281/zenodo.24549223`
  (<https://zenodo.org/records/24549223>); concept DOI
  `10.5281/zenodo.24549222` (all versions).
- The report separates the architecture claim from expert quality with a
  property-level evaluation protocol, and reports boundary conditions and
  negative results.

## Highlights

- **Apache-2.0** licensed, with `CITATION.cff` (Jiahe Qi, ORCID 0009-0001-8379-631X).
- **Expert contract + hot-swap registry**: contract validation, atomic swap,
  routing, rollback, and invalid-module rejection.
- **Public semantic space**: per-modality projection heads and prototype banks
  as an interface for heterogeneous experts.
- **Versioned knowledge runtime**: `MultiSourceFactStore` + TMS-style
  justifications; assert/retract/update/rollback/audit.
- **Bounded inference**: entity/relation/expansion/retrieval caps, anytime
  behavior, degradation FSM, cost attribution.
- **Property-level evaluation suite**: L0 sandbox, L1 real-expert integration,
  L2 bundle replay + fresh Isaac S4-PSS; twin pairing, negative controls,
  Hypothesis tests, TOST/ANOVA, and reproducibility manifests.

## Results snapshot

- L0: 5 seeds; SSR=1.00, RBE=1.00, INT=0.00, LOC=0, RBE-KB=1.00, BAR=1.00, MON=1.00.
- L1: 3 seeds; contract metrics invariant across Q0–Q4; ZCR tracks expert quality
  (Q0 0.969 → Q4 0.142).
- Cross-family zero-shot swap: vision 0.20, tactile 0.067 (chance 0.167).
- Calibration: vision k=1 → 1.00; tactile best 0.40.
- Fresh Isaac S4-PSS: 5/5 seeds succeed, `pick → place`, wall mean 2.136 s.
- Negative controls N1–N8 all behave as expected.

## What is included

- `pss/experts/`, `pss/observations/`, `pss/property/`, `pss/knowledge/tms.py`,
  `pss/metrics/property.py`
- `tests/property/` (24 tests, including 4 Hypothesis properties)
- `scripts/run_property_*.py`, `scripts/make_*_manifest.py`
- `eval/EXPERIMENT_PLAN_PROPERTIES.md`, `eval/property_preregistration.md`
- `reports/property_validation/REPORT.md` and `SECTION15_RESULTS.md` with figures
- `reports/claim_gaps/` (E1–E20 evidence, `INDEX.md`, `FOLLOWUPS.md`)
- `paper/` (technical report source; `supplement.tex` for additional diagnostics)
- `scorecards/property/` (raw results, manifests, expert sha256)

## Known limitations

- ZCR is a **capability** metric, not a property metric; it is not a pass/fail
  criterion for hot-swap.
- The shared space is a **calibratable interface**, not a zero-shot coordinate
  system; cross-family zero-shot swap fails.
- Tactile remains expert-limited; calibration does not fully recover it.
- L2 covers S4-PSS with oracle GT boxes; H/K are not yet exercised at L2.
- A5 is an oracle/retrain upper bound, not a real retrain-all run.
- No real-robot evidence yet.
- A7/A8 runtime variants are not implemented.

## Citation

Use `CITATION.cff` or the "Cite this repository" button on GitHub.

Paper (APA):

> Qi, J. (2026). *PSS: A Cognitive Runtime Above Perception* (Version 0.1.0).
> Zenodo. https://doi.org/10.5281/zenodo.24549223

## License

Apache License 2.0. See `LICENSE`.
