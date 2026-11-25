# Priority and provenance record

> Purpose: a dated public record of what this project is, when its results were
> made public, and how to cite it. It is **not** a legal document. For legal
> questions (patents, trademarks, infringement), consult a lawyer in your
> jurisdiction.

## Project identity

- **Name**: Public Semantic Space (PSS)
- **Repository**: https://github.com/Shirlif/PublicSemanticSpace
- **License**: Apache-2.0 (`LICENSE`); attribution notice in `NOTICE`
- **Principal author**: Jiahe Qi (齐嘉禾), ORCID
  [0009-0001-8379-631X](https://orcid.org/0009-0001-8379-631X)
- **Citation metadata**: `CITATION.cff` (version 0.1.0, released 2026-11-25)
- **Primary public record**: this repository's commit history + release notes

## Timeline

| Date | Event | Evidence |
|---|---|---|
| 2026-11-25 | Repository initialized | commit `72906ae` |
| 2026-11-25 | Apache-2.0, `CITATION.cff`, paper writing map | commit `01fab9f` |
| 2026-11-25 | Claim-gap validation, first batch (E1–E4) | commit `8a2c929` |
| 2026-11-25 | v0.1.0 release notes added | commit `61e26ee` |
| 2026-11-26 | §10 follow-up experiments (E5–E18) | commit `5f51de9` |
| 2026-11-26 | L2 cognitive-layer completion (E19–E20) | commit `570d315` |

The commit timestamps and public release history are the first line of evidence
for priority. A citable DOI or preprint is stronger; see "Recommended next
steps".

## Releases and identifiers

| Version | Git tag | DOI | Date | Notes |
|---|---|---|---|---|
| 0.1.0 | not yet created | not yet assigned | 2026-11-25 | property validation suite + first claim-gap batch |
| 0.1.x | not yet created | not yet assigned | 2026-11-26 | §10 follow-ups + L2 completion |

Do not invent a DOI. Add it here and in `CITATION.cff` once Zenodo (or another
archive) assigns one.

## Recommended next steps

These make the priority record much stronger:

1. **Tag the release** (annotated tag; signed if you have a signing key):

   ```bash
   # choose the commit that matches the release notes
   git tag -a v0.1.0 61e26ee -m "PSS v0.1.0: property validation suite"
   git tag -a v0.1.1 570d315 -m "PSS v0.1.1: section 10 follow-ups + L2 completion"
   git push origin v0.1.0 v0.1.1
   ```

2. **Connect Zenodo (or another archive) to the GitHub release** and record the
   DOI here and in `CITATION.cff`. Update `CITATION.cff`'s `version` and
   `date-released` to match the tag you publish.

3. **Publish a technical report or preprint** (arXiv, OSF, or a lab page) with a
   DOI. State the claims, the evidence map, and the code release it refers to.
   This is the strongest practical priority signal in academic records.

4. **Keep the public timeline intact**: release notes, commit history, issue/PR
   timestamps, and experiment manifests (`scorecards/property/run_manifest.json`,
   `expert_manifest.json`). Do not rewrite published history.

## What this record is for

- If there is a priority or authorship dispute, these timestamps and identifiers
  show what was public, when, and in what form.
- If someone copies code, figures, tables, or text, the license/copyright and
  the public record make the derivation easier to demonstrate.
- If someone independently reimplements the ideas, priority is a scholarly
  judgment; a DOI/preprint is the clearest record of when the work existed.

## What this record cannot do

- It cannot prevent someone from re-describing the ideas in their own words and
  claiming authorship; that is an academic-integrity matter, handled by the
  venue and the authors' institutions.
- It cannot replace a patent filing, a trademark registration, or legal advice.
- It cannot make a GitHub timestamp equivalent to a peer-reviewed publication;
  a preprint/DOI is stronger.
