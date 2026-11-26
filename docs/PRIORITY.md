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
- **Citation metadata**: `CITATION.cff` (version 0.1.0, released 2026-11-26)
- **Paper DOI**: `10.5281/zenodo.24549223`
  (<https://zenodo.org/records/24549223>); concept DOI
  `10.5281/zenodo.24549222` (all versions)
- **Primary public record**: this repository + the Zenodo record

## Timeline

| Date | Event | Evidence |
|---|---|---|
| 2026-11-25 | Repository initialized | commit `72906ae` |
| 2026-11-25 | Apache-2.0, `CITATION.cff`, paper writing map | commit `01fab9f` |
| 2026-11-25 | Claim-gap validation, first batch (E1–E4) | commit `8a2c929` |
| 2026-11-25 | v0.1.0 release notes added | commit `61e26ee` |
| 2026-11-26 | §10 follow-up experiments (E5–E18) | commit `5f51de9` |
| 2026-11-26 | L2 cognitive-layer completion (E19–E20) | commit `570d315` |
| 2026-11-26 | GitHub repository made public | <https://github.com/Shirlif/PublicSemanticSpace> |
| 2026-11-26 | Technical report published on Zenodo | DOI `10.5281/zenodo.24549223` |
| 2026-11-26 | `v0.1.0` tag created and pushed | `git tag -n` |

The commit timestamps and public release history are the first line of evidence
for priority. A citable DOI or preprint is stronger; see "Recommended next
steps".

## Releases and identifiers

| Version | Git tag | DOI | Date | Notes |
|---|---|---|---|---|
| 0.1.0 | `v0.1.0` | `10.5281/zenodo.24549223` | 2026-11-26 | paper + TeX source on Zenodo; concept DOI `10.5281/zenodo.24549222` |

The Zenodo DOI above is the assigned record DOI for version 0.1.0. The
concept DOI (`10.5281/zenodo.24549222`) always resolves to the latest version;
use it when citing "all versions".

## Recommended next steps

These make the priority record much stronger:

1. **Release tag**: `v0.1.0` is created at the release commit and pushed to
   origin. Future releases should use `v0.1.1`, `v0.1.2`, etc.

2. **Optional: connect GitHub–Zenodo for the software artifact.** The paper
   record is already published (`10.5281/zenodo.24549223`); a separate software
   record can reference it. Update `CITATION.cff`'s `version` and
   `date-released` to match any later tag.

3. **Submit to arXiv later if endorsement becomes available.** The paper source
   is `paper/`; `dist/pss_arxiv_source.zip` is the upload archive. Add the
   Zenodo DOI as a related identifier so the two records point at each other.

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
