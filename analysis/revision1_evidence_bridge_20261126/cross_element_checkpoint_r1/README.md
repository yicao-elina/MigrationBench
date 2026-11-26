# Cross-element checkpoint inventory SQL sidecar

This immutable sidecar indexes the 3,027-file metadata map for the three cross-element MACE training runs under `/scratch16/pclancy3/yi`. It records 3,000 checkpoint paths (epochs 0–999 for seeds 123/456/789), six model exports, three compiled exports, and supporting manifests/logs/metrics. It stores no model or checkpoint bytes.

The 2,997 intermediate checkpoints at epochs 0–998 remain `HOLD_PENDING_USER_POLICY`; their payload SHA-256 was not computed, so the SQL never treats path/size/inode as content identity. The three epoch-999 checkpoints and final exports have the exact hashes recorded in the pinned source map. Scheduler outcomes are historical `sacct` observations captured at 2026-11-25T19:34:21Z, not live state. The sidecar does not change scientific status or retention policy.

## Build

```sh
python3 analysis/revision1_evidence_bridge_20261126/cross_element_checkpoint_r1/import_checkpoint_inventory.py \
  --source-dir /path/to/cross_element_checkpoint_retention_20261125 \
  --output /scratch16/pclancy3/yi/MigrationBench_data_archive/training_checkpoint_inventory_snapshots/20261125/cross_element_checkpoint_inventory_r1.sqlite
```

The importer checks the source bundle's SHA-256 file, validates row counts and exact source columns, imports inventory metadata and the three cited scheduler outcomes, and checks SQLite integrity. It does not open, hash, move, or delete checkpoint payloads.

## Federated queries

- `QCK01`: source file inventory.
- `QCK02`: counts and byte totals by seed/class/disposition/hash status.
- `QCK03`: per-seed checkpoint coverage and held/final counts.
- `QCK04`: exact hash intersections with DUAL-X and MigrationBench R4 model tables; identity only.
- `QCK05`: captured scheduler status.
- `QCK06`: source pins and scope.

The path/size inventory is a navigation index. Retention decisions and scientific reuse require additional run lineage, exact content hashes, and manuscript/task context.
