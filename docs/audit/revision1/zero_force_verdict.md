# REV-C8: zero-force NEB exporter audit

Audit date: 2026-09-29. The audit was run read-only on Rockfish against the declared historical and grouped training splits. The executable is `scripts/build_zero_force_training_manifest.py`; the complete frame-level manifest and summary are in `data/processed/cluster/training_provenance/`.

## Result

PASS. The audit covered 9,163 frames across FT-600K, FT-MultiT, grouped FT-600K, and grouped Scratch-5% train/valid/test files. Every frame had a `forces:R:3` array. There were nine exact all-zero force frames, all nine explicitly marked `config_type=IsolatedAtom` and all one-atom Cr/Sb/Te isolated-atom reference records. There were **zero unclassified all-zero force frames**, zero NEB-derived zero-force frames, and no evidence that `neb_data_collector.py:442-443` or `neb_to_extended_xyz.py:278-282` supplied a training frame.

The isolated-atom references are legitimate E0 records used by the training configuration, not fabricated NEB labels. They are retained in the manifest and classified separately so that a future audit cannot silently mistake them for a clean dataset.

| dataset family | train | valid | test | unclassified zero-force frames |
|---|---:|---:|---:|---:|
| historical FT-600K | 1,891 (3 isolated) | 236 | 237 | 0 |
| historical FT-MultiT | 3,015 (3 isolated) | 376 | 377 | 0 |
| grouped FT-600K | 1,786 (3 isolated) | 311 | 267 | 0 |
| grouped Scratch-5% | 89 | 311 | 267 | 0 |

The historical FT-600K and FT-MultiT training logs point to `data_600K/` and `data_Multi_T/`, respectively; the grouped retraining scripts point to the hash-bound `data_grouped/` and `data_grouped_scratch5/` paths. No training script in the audited lineage points to NEB proposal/evaluation output.

## Reproduction

The Rockfish run used Python 3.6-compatible standard-library parsing and the following source/script identity:

```text
source_script = neb_data_collector.py:442-443; neb_to_extended_xyz.py:278-282
remote_audit_root = /scratch16/pclancy3/yi/revision1_migrationbench_runs/ft600k_grouped_A2/c8_zero_force_audit
```

The manifest records, per frame: dataset label, frame index, source file, source script label, source file SHA-256, atom count, force-array presence, exact-zero status, zero-force classification, maximum force, consecutive-zero run length, and the source header prefix containing trajectory/source metadata. The summary is fail-closed: `acceptance=PASS` only when the count of unclassified zero-force frames is zero.

## Reviewer response text

We hard-disabled the affected exporter behavior: an NEB frame without real QE forces cannot be written as a training-format frame. We then audited the hash-bound FT-600K and FT-MultiT train/validation/test files, including the grouped retraining splits. Across 9,163 frames, the only nine exact zero-force frames were explicitly labelled one-atom isolated-atom E0 references (three Cr/Sb/Te records in each full training split); no unclassified or NEB-derived zero-force frame was found. The frame-level manifest, fail-closed checker, input hashes, and result summary are released with this revision under `REV-C8`.
