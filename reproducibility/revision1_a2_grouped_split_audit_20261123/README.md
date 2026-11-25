# Revision 1 A2 grouped split and overlap evidence

This is a compact source package for the grouped 600 K split and Reviewer 1 A2 overlap audit. It preserves the small scripts and split summary needed to trace the method. It deliberately excludes model checkpoints and the `train.xyz`, `valid.xyz`, `test.xyz`, NEB image, and queue files. Those inputs remain at the Rockfish paths recorded in the manifests and audit reports.

## Contents

- `build_grouped_ft600k_dataset.py` and `build_scratch5_grouped_dataset.py`: dataset construction sources copied from the Rockfish campaign.
- `grouped_split_summary.json`: grouped full-data split metadata.
- `audit_neb_training_leakage.py`: exact/SOAP/RMSD overlap-audit source.
- `SOURCE_MANIFEST.tsv` and `SHA256SUMS.txt`: source paths, file sizes, and integrity hashes.
- `MANUSCRIPT_REVIEW_REPORT.md`: claim-level comparison of the supplied manuscript/rebuttal/SI against the Revision 1 evidence in this repository.

## Linked results and scope

The saved audit outputs are in `paper/response_to_reviewers/reviewer_a2_grouped_retraining/data/overlap_full_leakage_audit.json` and `overlap_scratch5_leakage_audit.json`; their source-data checksums are in the adjacent `a2_evidence_hashes.sha256`. The grouped G1 results are in `grouped_seed_comparison.csv` and `grouped_neb_barrier_summary.csv`. The figure/table reproduction checks live under `release/revision1_submitted_20261110/items/`.

The overlap audit covers 25 legacy migration images against the declared grouped 600 K and 89-frame Scratch-5% files. Exact fingerprints are zero; SOAP minima exceed the declared 1e-4 cutoff. Cr-local RMSD was comparable for only two full-data and one Scratch-5% candidate environments, so those RMSD minima are not a 25-image all-pairs statistic. The G1 errors are fixed-image diagnostics against path 1-4 (0.33605 eV); the QE source is unconverged and must not be described as a validated physical migration barrier.

The copied scripts require their original XYZ/evaluation inputs (and, for the overlap audit, the declared model checkpoint) to regenerate the outputs. Those payloads were intentionally not copied here. No commands were executed while assembling this package.
