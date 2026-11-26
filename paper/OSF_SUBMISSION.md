# OSF Preprints submission sheet

OSF Preprints is a good fallback when arXiv endorsement is not available: it
gives a DOI, accepts a PDF, and does not require an arXiv endorser. This sheet
is for the generic **OSF Preprints** server; **EngrXiv** is a topical
alternative for engineering/robotics.

## Before you start

- OSF account: <https://osf.io> (ORCID login works).
- PDF of the paper: **prepared at `dist/pss_paper.pdf`** (12 pages, 0.36 MB;
  sha256 `46eae0d0ee5509ff8e3f5f166df1a4450c448f544d8780e9032026c83f1ed6aa`).
  This is the v7 compile downloaded from Overleaf.
- Optional supplementary files: `dist/pss_arxiv_source.zip` (TeX source and
  figures) and, if you compile it, `supplement.pdf`.
- Decide the license: **CC BY 4.0** is recommended for the paper text. Code
  remains Apache-2.0.

## Steps

1. Sign in at <https://osf.io> and verify your email.
2. Open the preprint service:
   - generic: <https://osf.io/preprints/>
   - engineering: <https://engrxiv.org/> (uses OSF)
3. Click **Add a preprint** (or **Submit a preprint**).
4. Choose the server:
   - **OSF Preprints** for a general CS/systems audience;
   - **EngrXiv** if you want the engineering/robotics label.
5. Create a new OSF project when prompted (title: `PSS: A Cognitive Runtime
   Above Perception`). The preprint is attached to this project.
6. Upload the manuscript:
   - Primary file: `pss_paper.pdf`;
   - Additional files: `dist/pss_arxiv_source.zip`, optional `supplement.pdf`.
7. Fill the metadata:

   **Title**

   ```
   PSS: A Cognitive Runtime Above Perception
   ```

   **Authors**

   ```
   Jiahe Qi (Independent Researcher; ORCID 0009-0001-8379-631X)
   ```

   **Abstract** (plain text)

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

   **License**: CC BY 4.0.

   **Disciplines**: Computer Science → Robotics; Engineering.

   **Tags / keywords**: modular architecture; cognitive runtime; hot-swap;
   knowledge update; truth maintenance; bounded inference; property-level
   evaluation; embodied AI.

   **Data availability**: code and reports at
   `https://github.com/Shirlif/PublicSemanticSpace` (make the repository public
   before the preprint goes live, or use "will be available at").

   **Conflicts of interest**: none.

8. Preview the PDF and metadata.
9. Submit. OSF Preprints runs a moderation check; the preprint and DOI appear
   after it passes (usually a few business days).

## After submission

- [ ] Record the OSF DOI in `CITATION.cff`, `README.md`, and
      `docs/PRIORITY.md`.
- [ ] Make the GitHub repository public if the paper points to it.
- [ ] If you later obtain arXiv endorsement, you can still submit the same
      source (`dist/pss_arxiv_source.zip`) to arXiv; note the OSF preprint as a
      related identifier.

## If you need a DOI immediately

OSF Preprints has a moderation queue. If you need a DOI today, use Zenodo:
upload `pss_paper.pdf`, add the metadata above, publish, and you get a DOI
immediately. You can still post to OSF later.
