PASS

# R03 final data review

Independent read-only review of the explicit R03 evidence files. No campaign data or plotting code was modified.

## Evidence

- MLFF collection: `COLLECTION_MANIFEST.json` reports `target_rows=108`, `rows_with_manifests=108`, `converged_rows=108`, `failure_rows=0`, and `attempt_rows=125`. The final collection hash is `0b56ec0a37bafe29e243bc05205da7d1b17d91ac7782c8e36ad8fa87a82844f9`.
- MLFF rows/profiles: `ROW_RESULTS.tsv` has 108 accepted rows; `IMAGE_PROFILES.tsv` has 1080 rows (10 images per row), all finite and `CONVERGED`; `CLASS_SUMMARY.tsv` has 36 class rows across 12 canonical paths and 3 families, with denominators 5+2+2=9 per element/path. Recorded hashes: ROW_RESULTS `e8e3cd32832549128ed87a84c8de36d8353b7d984426f320d8f1c6ce0463d1cd`; IMAGE_PROFILES `9327bd93104703a94ca3a69a2ec5d4c208685f1bb0fee4a30256bca517de8e5f`.
- Retry/provenance retention: `ATTEMPT_LEDGER.tsv` and `RAW_ARTIFACT_INDEX.tsv` each retain 125 attempts, including nonconverged retry lineage; hashes are `fcaf8ae748817b35b0adbfe72ecd2997e2d34b89cdf0f3973637784bcc86f2ad` and `f23468d05857f3d5587007dec65c6fefbe3244ef1d4e0479f7149058291a71fc`. The accepted ROW_RESULTS remain separate from the retained attempt/raw ledgers.
- Reference matrix: `GROUND_TRUTH_REFERENCE_PROFILES_R03.tsv` contains 120 finite points in 12 groups, exactly 10 per canonical element×path. Canonical IDs match the 12-path registry used by the current self-relaxed profiles. Plot image indices are normalized to 0–9 (`i/9`); Cr source indices are 0–9, while Cu/Li/Na source indices are 1–10 and are preserved separately.
- Reference statuses are retained: Cr has 30 `COMPLETED_BUT_SCF_WARNING` points; Cu/Li/Na have 90 `failed_scf_convergence` points. Reference profile SHA-256 is `25486368d49c24710e8326e8e2ebea75e483b312243b2eba8451b0becd381170`; manifest SHA-256 is `216745b34375922d85f04d1427d9c1184a700cc36faf14ce3196bebd985f3aeb`. The manifest binds the input/source hashes and the current self-profile hash, and declares `PROVISIONAL_DIAGNOSTIC_REFERENCE` with `unqualified_ground_truth_claim=false`.
- Visual/caption boundary: `VISUAL_QA_R03.json` binds the `DFT-NEB` legend abbreviation to the exact reference manifest/profile hashes, 120 points, both raw statuses, `same_geometry_dft_rescoring=false`, and the full label `QE/DFT reference · SCF warning; diagnostic`. The R03 caption states that these are canonical fixed-path QE/DFT single-point diagnostics, not same-geometry DFT rescoring, a validated DFT MEP, a converged DFT barrier, or unconditional ground truth. The removed `converged / included` legend item does not remove acceptance semantics: those remain in the collection, derived-profile, row, attempt, raw-artifact, and review manifests.

## Scientific boundary

The MLFF traces are Foundation self-relaxed endpoint-only NEB observables. The added DFT-NEB traces are contextual canonical fixed-path QE/DFT diagnostics only. They must not be used as same-geometry DFT rescoring of the MLFF-relaxed images, validated DFT minimum-energy paths, converged DFT barriers, or unqualified ground truth.

## Operational note

CloudFiles intermittently returned zero-byte reads for some large TSVs during a later retry, although their successful materialized reads and the immutable bookkeeping hashes above were obtained during this audit. Downstream handoff should keep those explicit files materialized; this does not change the data-gate result or any campaign file.
