# MigrationBench Revision 1 lineage overlay R3

R3 is an additive copy of the R2 SQLite overlay with the three observed MACE from-scratch 1 ns terminal instability failures registered as a separate partial campaign. The R2 base is immutable. R3 adds 3 run records, 3 terminal-state result records, 3 content-identified model records, 61 artifact records, 63 run/artifact links, and 3 result/run links.

## Scientific boundary

All three runs terminated at `instability_threshold` before 1,000,000 steps: 795,600 (training seed 123), 52,400 (training seed 234), and 83,000 (baseline training seed 345). Each run/result/model/artifact is marked `EXCLUDED` or `FAILURE_MODE; EXCLUDE_FROM_PUBLICATION`; these are preserved failure observations, not completed transport or conductivity evidence. Scheduler state is `UNKNOWN` because no full Slurm status snapshot was retained.

## Post-processing identity

The 15 Rockfish CSV products and 15 Git historical copies have equal data-row counts but different hashes and byte sizes. Each stays as its own artifact; Git CSV copies are marked `CONFLICT` against the Rockfish copy, without attributing the cause. The nine JSON files per side match by hash/size according to the frozen audit. Raw trajectories and checkpoints remain in their approved Rockfish personal roots and are referenced by exact path, size, and SHA256 rather than copied into Git.

## Frozen input correction

The original read-only evidence package is copied byte-for-byte in `source_audit/` and remains separately archived on Rockfish at `/scratch16/pclancy3/yi/MigrationBench_data_archive/storage_audits/mace_from_scratch_terminal_failure_lineage_20261125/`. One 63-character Git `baseline seed345` `vacf.csv` checksum cell in its artifact TSV was corrected in the separate `evidence_correction/` layer after recomputing the Git blob from pinned main commit `feb328843932e436f0d75e7f407b9b425319d96c`. See `evidence_correction/CORRECTION_NOTE.md`; the original audit payload remains unchanged.

## Rebuild

Supply byte-identical R2 SQLite and the frozen `evidence_correction/` directory:

```sh
python import_mace_terminal_failure_r3.py \
  --r2-db /path/to/migrationbench_r1_lineage_overlay_20261125_r2.sqlite \
  --evidence-dir /path/to/evidence_correction \
  --output-db /path/to/migrationbench_r1_lineage_overlay_20261125_r3.sqlite
```

The importer refuses to overwrite the destination, verifies frozen input hashes, checks SQLite `quick_check` and foreign keys, and writes `R3_MANIFEST.json`. Query examples QL13–QL17 are in `queries.sql`.

## Snapshot identity

`R3_MANIFEST.json` records exact base and input-package hashes, result DB SHA256 and size, counts, and integrity status. The SQLite payload is archived outside Git under `MigrationBench_data_archive/revision1_lineage_overlays/20261125_r3/`; the importer and query interface belong in the canonical Git repository.
