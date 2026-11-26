# Claim-matrix sidecar R2: Fig. 4 to R05 summary crosswalk

R2 is an additive version of the existing claim-matrix sidecar. It starts from the exact R1 SQLite file, retains the 639 claim rows and all R1 metadata unchanged, and adds one summary-grain table for 30 R4 Fig. 4 rows mapped to 29 current R05 profile rows plus one explicit `MISSING_OR_EXCLUDED` row. R1 remains preserved at its original archive path and SHA-256.

This does not add a federated SQLite attachment. The existing `claim_matrix` alias is repointed from R1 to the versioned R2 snapshot. QCE01–QCE04 continue to read only the unchanged `claim_evidence_row` and `snapshot_metadata` tables. QCE05 reads the new summary crosswalk.

## Row and source boundaries

The only mapping key is `(observable, path_key, group)`, aligned to an R4 `lineage_results` summary record and the R05 TSV row. Panels `a`–`c` map to `FIXED_PATH_SINGLE_POINT`, panels `d`–`f` to `SELF_RELAXED_ENDPOINT_ONLY_NEB`; panel paths are `a/d=in_gap`, `b/e=gap_to_ql`, and `c/f=within_ql`.

R05 uses the plotted-profile TSV for its 29 present group rows and the coverage TSV to retain the full 30-slot panel/group grid. Self-relaxed `within_ql × ft_multit` has no plotted-profile row: its `profile_data_line` is NULL and its coverage state remains `MISSING_OR_EXCLUDED` with zero listed models/profile rows. This is not interpreted as zero attempts.

The crosswalk records source line numbers, status fields, R4 result keys, and pinned file hashes. It does not link any individual run, attempt, scheduler job, model checkpoint, trajectory, or raw input/output. It does not validate the PDF or scientific acceptance; fixed-path single-point and endpoint-only self-relaxed MLIP-NEB protocols remain distinct.

## Build and query

The importer copies the pinned R1 SQLite snapshot to a new output path, verifies the R4/R05 source hashes, and adds only the two R2 tables. It refuses to overwrite an existing output. The versioned snapshot manifest and `FEDERATED_SOURCES.json` pin R2 while retaining R1 as a separately addressable historical snapshot.

```sh
python3 analysis/revision1_evidence_bridge_20261126/claim_matrix_r2/import_fig4_r05_crosswalk.py \
  --base-r1-path /scratch16/pclancy3/yi/MigrationBench_data_archive/revision1_claim_matrix_snapshots/20261126/claim_matrix_r1.sqlite \
  --r4-path /scratch16/pclancy3/yi/MigrationBench_data_archive/revision1_lineage_overlays/20261125_r4/migrationbench_r1_lineage_overlay_20261125_r4.sqlite \
  --output /scratch16/pclancy3/yi/MigrationBench_data_archive/revision1_claim_matrix_snapshots/20261126_r2/claim_matrix_r2.sqlite
python3 analysis/revision1_evidence_bridge_20261126/query_federated.py --query-id QCE05
```

The exact build output hash, row counts, source hashes, and remote query receipts are recorded in `snapshot_manifest.json` and `QCE_QUERY_VERIFICATION.json`. The four QCE01–QCE04 result row sets were compared to the pre-switch R1 federation baseline and are identical; the R2 file path/hash in connection metadata are expected to differ. QCE05 returned 30 rows from the same 11-source, query-only federation.
