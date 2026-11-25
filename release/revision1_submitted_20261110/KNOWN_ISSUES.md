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

## Found by the per-item reproduction scripts (2026-11-10)

Each item's `source/` script and `evidence/*.json` hold the numbers; `sources/known_discrepancies.tsv`
lists the exact failing checks accepted below, so any other mismatch still fails.

14. **Table R2, Scratch-5% test force RMSE.** The submitted value is 342 meV/Å. The per-seed values
    354.4, 345.1, 324.9 (repo `grouped_seed_comparison.csv` and the Rockfish MACE run logs, both at
    0.1 precision) average 341.47; with ±0.05 per seed the mean is 341.42–341.52, so it rounds to 341
    whatever the hidden digits. Correct to 341 at the next revision.
15. **Table S11, Scratch-600K five-fold MAE.** The data mean is 1.145 meV/atom (1.1 at the displayed
    precision); the submitted 1.2 came from double rounding (generator 1.15 → hand-rounded 1.2). The
    older `tools/verify_table_rounding.py` compared with the 1.15 output and missed it. Response
    Table R5 reports R² only and is unaffected.
16. **Fig. 6 embeddings depend on an unrecorded environment.** The stored t-SNE/PHATE/UMAP
    coordinates (`analysis/legacy_fig6_phate_reproduction_20261101/*/fig6_full_campaign_embeddings.npz`)
    were made from `fig6_full_campaign_features.npz` (sha `5aab89df`, in the archive store). sklearn
    1.6.1 does not reproduce the t-SNE coordinates (max difference 30 on a ±58 range under every
    init/learning-rate default tried), and `phate`/`umap-learn` are not in the reference `.venv`.
    The recorded silhouette and the panels re-render exactly from the stored embeddings. Record the
    package versions with any new embedding run.
17. **MPA0 row labelled "zero-shot" (Tables S2, R2).** `accepted_models.tsv` lists that checkpoint as
    `finetuned_MACE_multihead0804_grouped_A2_MPA0_stagetwo.model` (`ft600k_mpa0_grouped_seed123`,
    grouped train split), i.e. MPA0 fine-tuned on the grouped 600 K data. Check which model was meant.
18. **Table R2, Scratch-600K identities.** The G1 error uses the legacy `scratch_grouped_seed*`
    checkpoints (sha `125255ea…`), the RMSE uses the Lmax1 repair models (sha `1f393169…`) in
    `parity_summary.csv`; the per-seed RMSEs agree in both files. Confirm the row's model identity.
19. **Table S6, path L1 wording was changed on Overleaf.** The registry and `AUDIT_REPORT.md` call
    path 1-8 "probable deep penetration" / ambiguous; the submitted text says Cr is within the
    quintuple layer. The S6 generator words exact-coordinate rows from the registry `class` field,
    so its PASS covers this edit; review the classification.
20. **`paper/revision1/scripts/build_path_registry.py` reads outside the repo.** It opens six legacy
    `neb.out` files under `../26MIgrationBench/` and types the atom counts. Copies are in
    `data/raw_compact/legacy_61_qe_neb/` (1-4 is byte-identical); the item generators for Tables S5/S6
    read those instead. Repoint the script when `paper/revision1/` is next edited.
21. **Fig. 4 bundle panel was stale.** `reproducibility/canva_palette_vector_bundle_20261105/figures/
    CANVA-P06_neb_cr_r05_palette_updated.pdf` was rendered while nine run files were missing (4.5 % of
    pixels differ from the panel used in Canva). The 17 run files and the panel used in Canva were
    imported from the ChatGPT workspace on 2026-11-10 (`imports/IMPORT_MANIFEST.tsv`); the Fig. 4
    evidence compares against `items/main/Fig4_neb_profiles/upstream/CANVA-P06_…_used_in_canva.pdf`.
22. **Table R11 rerun outputs are not stored.** "Maximum difference = 0" is read from
    `shap_reproduction_check.json`; the CSVs of the 2026-11-10 SHAP rerun were not kept. Commit them
    with the next rerun so `tools/compare_reproduction.py` can recompute the 0.
23. **Table R10 feature audit.** Columns 6151–6153 of the 927 × 6,192 matrix are also zero in every
    frame and are kept; only column 6191 is dropped as the placeholder. The audit JSON hashes
    `fig6_full_campaign_features.npz`, not the in-repo `fig6_current_input.npz` (same shape).
24. **Fig. 5 auxiliary caption.** The generated `figure_caption.txt` now states the corrected value
    from data (2.3507669 eV); the old typed text said 2.3507667. Not in the submitted TeX.
25. **Canva bundle renderer hashes.** To redirect outputs and drop the `plot_atlas` import from a
    laptop path, `render_neb_palette_20261102.py`, `render_md_six_panel_palette_20261102.py` and
    `build_cr_grouped_r04.py` were edited; their panels are pixel-identical (item evidence), but
    `CANVA_PALETTE_VECTOR_MANIFEST.json` still records the old `renderer_sha256`. `verify_bundle.py`
    needs PyMuPDF (`fitz`), which the reference `.venv` lacks, so it was not run.
