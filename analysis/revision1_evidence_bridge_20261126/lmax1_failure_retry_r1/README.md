# Lmax1 G3 failure and follow-up evidence — R1

This additive SQLite source preserves collector-row evidence from the four user-owned Scratch16 campaign roots below. It keeps a diagnostic event grain separate from `learning_loop_review_r1`, which remains the seven-row family-summary index.

| Stage | Source rows | Recorded interpretation |
|---|---:|---|
| Tier 1 baseline | 54 | 10 `NOT_CONVERGED`, 18 `FINITE_DIAGNOSTIC`, 26 `CONVERGED`; leakage gate and publication boundaries remain in each source row. |
| Tier 1 retry | 30 planned | 20 `PRESENT_IN_COLLECTOR`, 10 explicitly `MISSING`; missing planned `climb_long` rows are not scientific failures. |
| Tier 1 perturbation | 20 | 20 collector rows with explicit source failure IDs. |
| Tier 2 diagnostic | 90 | 57 `CONVERGED`, 30 `FINITE_DIAGNOSTIC`, 3 `NOT_CONVERGED`; all 90 retain `PENDING_G3_11`. |

There are 50 explicit `source_record_id` edges: 30 retry rows and 20 perturbation rows link to the ten Tier 1 `NOT_CONVERGED` source records. Tier 2 remains a separate diagnostic expansion; this index invents no parent links for it.

## Provenance and limits

- `source_manifest.json` pins 25 compact source files by Rockfish absolute path, size, and SHA-256. They are compact acceptance-scope JSON, model/path registries, collector/failure/job TSVs; total source bytes are 254,936.
- `sources/` contains those same compact pinned input files for deterministic database rebuilds. Large run outputs, trajectories, models, and manifests were not imported, read, or rehashed here.
- All stages retain their own acceptance scope and model/path registry hashes. Tier 1/retry/perturb share path registry SHA-256 `cf5ab5c9…5b2ce53e`; Tier 2 has a separate registry SHA-256 `3376cf23…db6819`.
- Rows are collector metadata and diagnostic status, not accepted paper results. `COMPLETED` scheduler/run status is not scientific acceptance. No detector decision, selected training set, next checkpoint, or measured active-learning gain is evidenced.
- The derived SQLite snapshot is stored in the personal Rockfish archive, not committed to Git. The tracked source package, schema, importer, queries and manifest provide the reproducible interface.

## Rebuild and query

```sh
python3 analysis/revision1_evidence_bridge_20261126/lmax1_failure_retry_r1/import_lmax1_failure_retry.py \
  --output /tmp/lmax1_failure_retry_r1.sqlite \
  --archive-path /scratch16/pclancy3/yi/MigrationBench_data_archive/lmax1_g3_failure_retry_snapshots/20261126/lmax1_failure_retry_r1.sqlite
python3 analysis/revision1_evidence_bridge_20261126/query_federated.py --query-id QLM01
```

QLM01 summarizes status by stage; QLM02 lists source failures with planned follow-up counts; QLM03 shows explicit event-level links; QLM04 lists missing retry variants; QLM05 summarizes Tier 2 gate states; QLM06 returns pinned source files; QLM07 shows campaign hierarchy and acceptance-scope pins. The federated launcher opens databases read-only and enables `query_only`.
