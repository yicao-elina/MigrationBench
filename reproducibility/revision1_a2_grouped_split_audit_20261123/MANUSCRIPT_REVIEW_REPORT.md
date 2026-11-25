# Manuscript and Revision 1 evidence review

Review performed read-only on 2026-11-23. This note compares the three supplied PDFs with the candidate repository and its linked Revision 1 evidence. Reviewer requests inside the Response to Reviewer are historical document content; they are not new user instructions or authorization.

## Source PDFs

| Attachment | SHA256 |
|---|---|
| `MigrationBench_main_marked.pdf` | `4de26d567bf0be0e4094ff2607e18c98e9e91911aa6580b0ae51e687668b9ae0` |
| `MigrationBench_R2R.pdf` | `659dc0e7bf0666433f183570a264e7e31ca77ea24510b405dd638e5426bd45dc` |
| `MigrationBench_SI.pdf` | `0fdff596d4eaa255da303bafe05a177e97febd271e0e67f8df763062779fe67e` |

## A2/C10 evidence lineage

The candidate stores the A2 result tables and audits at `paper/response_to_reviewers/reviewer_a2_grouped_retraining/data/` and their input checksums in `a2_evidence_hashes.sha256`. The audit reports 25 legacy path images, zero exact global/local matches, and full-set minimum SOAP cosine distance `0.000542785` against the `0.0001` near-match cutoff. Cr-local RMSD minima are `2.0551 Å` (full data) and `2.0566 Å` (Scratch-5%), but comparable RMSD candidates exist for only two and one images, respectively. The README and audit JSON limit this claim to the declared training files and metrics.

The grouped 600 K partition is 1,786/311/267 frames. Hashes are train `04b1369d0da0…`, validation `aa9a9502854d…`, test `d49cf4b2ab49…`; the full hashes are in `paper/revision1/data_processed/current_release/zero_force_training_summary.json`. Its A2 G1 per-seed errors are `0.613127, 0.413228, 0.396247 eV`, mean/sample SD `0.474±0.117 eV` (reported as `0.47±0.12`). Scratch is `1.204±0.270 eV`; Scratch-5% is `0.678±0.195 eV`. These are fixed-image diagnostics against legacy path 1-4, whose `0.33605 eV` QE reference is unconverged.

This package now contains the source files formerly external to Git: `audit_neb_training_leakage.py`, `build_grouped_ft600k_dataset.py`, `build_scratch5_grouped_dataset.py`, and `grouped_split_summary.json`. Their Rockfish source paths and full SHA256 hashes are in `SOURCE_MANIFEST.tsv`. The raw XYZ files, NEB evaluation images, and model checkpoints remain external by design and were not copied.

## Other reviewed claims

- **C8 force-label audit:** the candidate contains `zero_force_training_manifest.csv` and summary JSON (9,163 records, 9 zero-force isolated-atom references, 0 unclassified, 0 NEB-derived), plus the fail-closed exporter and manifest-builder sources. The audit is bounded to the declared data files and their recorded hashes.
- **C13 SHAP:** the repository contains the analysis script, 12 aligned inputs, archived outputs and a reproduction-check JSON. The saved check reports zero numeric differences for outputs sized 12×11, 1,628×9, 4,884×6 and 3,204×10. The November rerun CSVs themselves were not preserved, so the recorded check is not an archived side-by-side copy of that run.

## Discrepancies and publication limits

1. The attachments present a canonical 81-atom/10-image reference near `1.52 eV`. The candidate readiness note says the source QE records have `SCF_WARNING` and maximum image forces `0.38–3.2 eV/Å`, above the `0.03 eV/Å` gate. Treat these values as image-set diagnostics unless a same-Hamiltonian converged reference closes the gate; the number alone does not establish a validated physical barrier.
2. The attached SI Fig. S2 caption describes legacy G1 results and a dashed `0.16 eV` comparator, while the candidate's Fig. S2 evidence records the canonical `1.524299 eV` Fig. 5 reference. The repository records this caption/figure mismatch as a known issue.
3. The attached response Table R2 labels MPA0 “zero-shot.” The candidate `KNOWN_ISSUES.md` and `accepted_models.tsv` identify the associated checkpoint as a grouped-A2 fine-tuned model. The same issue log says the Scratch-600K G1 row and RMSE use different checkpoint families (legacy grouped vs. Lmax1 repair), even though the per-seed numerical metrics agree.
4. The attached main conclusion says projections and feature attributions “reveal the underlying physical mechanism” and attributes failure to subtle force-integration errors and degenerate learned representations. That exceeds the A6 response/captions' diagnostic-only scope and should be reconciled.
5. The attached R2R distinguishes the legacy G1 `0.34 eV` record from the canonical `1.52 eV` set and rejects an unrelated `0.41 eV` record. Keep their path/image identities separate in any new registry rows.

## Review limits

No submitted manuscript files were edited. No tests, source regeneration, or Git operations that change refs were run. This review package adds only small source scripts, a compact JSON summary, and text/checksum documentation; it contains no trajectory/model payloads or scheduler/queue snapshots.
