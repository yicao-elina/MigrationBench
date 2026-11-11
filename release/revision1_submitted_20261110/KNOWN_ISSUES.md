# Known issues in the submitted Revision 1 (found 2026-11-10)

These are open to fix at the next revision. None were changed in the submitted
sources, which are frozen at `334260c`.

## Scientific caveats to keep visible

1. **Canonical DFT-NEB reference is not converged by the force gate.** The three
   81-atom Cr QE NEB runs (Fig. 4, Fig. 5, Table S4) stop with maximum image forces of
   0.38–3.2 eV/Å, above the 0.03 eV/Å gate. The Rockfish ledger
   (`canonical_dft_barriers.tsv`) marks them `scf_warning=TRUE` and
   `EXCLUDE_FROM_PUBLICATION`, while the paper uses them as the reference (1.52 eV
   in-gap). Reconcile the ledger and the text.
2. **MD15:** 11 of 15 runs are `SCIENTIFIC_FAILURE` in `MD15_STATUS_AUDIT_R02.tsv`;
   foundation rows are `TARGET_REACHED_PROVISIONAL`, but Table S8 says "Target reached".
   Response Table R7 says "Scientific stop" where Table S8 says "Threshold reached".

## Text and figure inconsistencies

3. **Fig. S2 caption** describes the legacy G1 panel (FT-600K/Scratch/Scratch-5%, single
   MPA0, dashed 0.16 eV), but the submitted figure shows the canonical in-gap panel. The
   in-repo `paper/revision1/scripts/plot_grouped_retraining_overlap.py` is stale; the
   hash-matched generator is `items/si/FigS2_grouped_retraining_overlap/source/`.
4. **Table S3** values match `cr_three_path_coverage_audit_R02`, while Fig. 4 uses the R05
   audit.
5. **Table S5** dropped the convergence Status column; the status now appears only in the caption.
6. **Response A4 / Table R4** cites `SI-sec:endpoint_perturbation_campaign`, but the SI no
   longer inputs `appendix/endpoint_perturbation_appendix.tex`; the reference resolves only
   through the stale cached `sn-article-SI.aux`.
7. **Fig. 4 final asset** has the Canva typo "Self-relaxted".
8. `paper/revision1/data_processed/current_release/source_ledger.json` lists an older sha
   (`dcfb7f45`) for `fig_NEB-fixed-path_Cu-Li-Na-Cr.pdf`; the submitted one is `72263413`.

## Provenance gaps that need a person

9. **Canva sources:** final Figs. 1–7, the graphical abstract and the hi-res uploads are
   Canva exports (design `DAHZXpLtN6U`, connector needs re-authorisation). Export each page
   (PDF + editable link) into `items/main/<item>/canva/`. The graphical abstract has no
   source file anywhere else.
10. **Fig. S4** final PDF (sha `6f45a330`, cairo export 2026-11-03) was hand-edited from
    CANVA-P09; the edited intermediate was not found.
11. **Table R11:** the original script that wrote `shap_reproduction_check.json` was not
    archived. `tools/compare_reproduction.py` re-creates it, and a 2026-11-10 rerun in the reference
    environment reproduced the JSON exactly (resolved).
12. **Canva palette bundle:** 18 files differ between the repo copy (regenerated with stable
    hashes on 2026-11-05/06, kept) and the ChatGPT workspace copy
    (`imports/IMPORT_CONFLICTS.tsv`).
13. **iCloud:** part of the ChatGPT workspace is evicted to iCloud (`dataless`). Files still
    evicted at import time are listed in `imports/ICLOUD_DATALESS.tsv`; rerun
    `tools/import_external.py` after they download.
