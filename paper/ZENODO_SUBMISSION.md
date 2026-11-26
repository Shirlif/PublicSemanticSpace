# Zenodo submission sheet

> **Published:** <https://zenodo.org/records/24549223> — version DOI
> `10.5281/zenodo.24549223`; concept DOI `10.5281/zenodo.24549222` (2026-11-26).

Zenodo is the fastest fallback when arXiv endorsement is unavailable: it
publishes immediately and mints a DOI. This record is for the **paper**. The
code repository can be archived separately later (GitHub–Zenodo integration).

## Prepared files

- `dist/pss_paper.pdf` — 12 pages, 0.36 MB,
  sha256 `46eae0d0ee5509ff8e3f5f166df1a4450c448f544d8780e9032026c83f1ed6aa`
- `dist/pss_arxiv_source.zip` — 0.96 MB, TeX source + figures + supplement

## Steps

1. Sign in at <https://zenodo.org> (GitHub or ORCID).
2. Click **New upload**.
3. **Files**: upload `pss_paper.pdf` as the main file, and
   `pss_arxiv_source.zip` as a supplementary file.
4. **Metadata**:

   **Resource type**: `Publication` → `Preprint`
   (use `Technical report` if you prefer the report label)

   **Title**

   ```
   PSS: A Cognitive Runtime Above Perception
   ```

   **Authors**

   ```
   Qi, Jiahe
   ```

   Affiliation: `Independent Researcher`
   ORCID: `0009-0001-8379-631X`

   **Description** (plain text; the abstract)

   ```
   Embodied systems keep upgrading their perception stack, but their cognitive
   state — identity, relations, memory, and audit — usually lives inside an
   end-to-end model. Swapping an expert then forces retraining the whole stack,
   knowledge updates cannot be inspected or rolled back, and inference cost has
   no hard bound. We present PSS, a cognitive runtime above perception: a
   pluggable expert layer governed by an admission contract, a public semantic
   space used as a calibratable alignment interface rather than a zero-shot
   coordinate system, and an explicit state layer that maintains persistent
   entity identity, typed relations, active-set memory, a versioned
   truth-maintenance store, uncertainty and abstention, and bounded inference.
   We evaluate PSS at the property level, treating expert quality as a
   controlled factor, pairing architecture variants as twins, and testing the
   metrics with negative controls. Across L0/L1 experiments the hot-swap
   contract (swap success, rollback exactness, non-interference, routing) is
   invariant to expert quality, while semantic compatibility is not:
   cross-family zero-shot swap falls to chance, and one-shot calibration
   recovers vision but not tactile. Entity-state audit, typed relations,
   active-set selection, and bounded inference are measured separately, and a
   replay over the same detection stream isolates the contribution of the
   memory layer from perception. We report boundary conditions and negative
   results, including a +0.012 fair-attribution re-identification gain, a
   closed-set accuracy that is invariant to the projection space, and a case
   where retraining experts is no better than rebuilding prototypes on frozen
   features. PSS is not a foundation model; it is a contract-level runtime and
   a measurement protocol.
   ```

   **License**: `CC BY 4.0`

   **Keywords**

   ```
   modular architecture; cognitive runtime; hot-swap; knowledge update;
   truth maintenance; bounded inference; property-level evaluation; embodied AI
   ```

   **Language**: English

   **Version**: `0.1.0`

   **Publication date**: `2026-11-26`

   **Related identifiers**

   ```
   https://github.com/Shirlif/PublicSemanticSpace
   Relation: isSupplementedBy
   Resource type: Software
   ```

   **Access rights**: Open

   **Publisher**: leave empty for a preprint.

5. Optional: click **Reserve DOI** if you want to put the DOI into the paper
   before publishing. Otherwise Zenodo assigns it on publish.
6. Click **Publish**, confirm. The record becomes public immediately and gets
   a DOI (`10.5281/zenodo.XXXXXXX`).
7. Zenodo creates a **version DOI** and a **concept DOI** (all versions). Use
   the version DOI in `CITATION.cff` for this release; the concept DOI is the
   stable one for the project.

## After publishing

- [ ] Put the DOI in `CITATION.cff`:
      `doi: 10.5281/zenodo.XXXXXXX`
- [ ] Put the DOI in `README.md` and `docs/PRIORITY.md`.
- [ ] Make the GitHub repository public, since the paper points to it.
- [ ] If you later submit to arXiv, add the Zenodo DOI as a related identifier
      and keep the versions aligned.
- [ ] Optional: connect GitHub–Zenodo for the repository itself to mint a
      software DOI; the paper record and the software record can reference each
      other.

## Notes

- Zenodo does not moderate content; publishing is immediate.
- You can edit the metadata after publishing, but changing files creates a new
  version with a new version DOI.
- The paper PDF is fixed. If you want the GitHub link to work on first reading,
  make the repository public **before** clicking Publish.
