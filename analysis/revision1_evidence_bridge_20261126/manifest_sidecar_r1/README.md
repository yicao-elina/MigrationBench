# MigrationBench Revision 1 manifest metadata staging sidecar R1

**Capture:** 2026-11-26  
**Canonical MigrationBench main:** `cf6c2b1649ba993d079d0eadf8478775a81736d2`  
**Source scan:** read-only inventory captured 2026-11-25; source scan SHA-256 `9ec3d8272eb60e2afbc1661cf6d0b92ce0697764bc79e74c645929809c64f486`.

## Scope and handling

The sidecar covers only the user's personal Rockfish directories:

- `/scratch16/pclancy3/yi/revision1_migrationbench_inputs/`
- `/scratch16/pclancy3/yi/revision1_migrationbench_runs/`

It is a **staging SQLite sidecar**. The importer never opens or modifies the production 4 GB provenance catalog and does not change the MigrationBench repository, source manifests, or experimental payloads. Outside-root paths from manifests are retained as literal candidate values only; no files or directories outside these two roots were touched.

## Central catalog compatibility

`schema.sql` mirrors the core central catalog tables (`project`, `storage_root`, `storage_path`, `artifact`, `artifact_location`) from `analysis/rockfish_skipjack_migration_audit_20261121/phase2_schema/001_provenance_catalog.sql`; it adds sidecar-specific scan, manifest observation, control-field, reference-candidate, and revalidation tables. Stable source paths and hashes are separate from manifest/run IDs. Root IDs are staging keys and must be mapped against the live catalog before any future merge. No central merge is included or implied.

The original scan selector was regular-file basenames containing `manifest` or `provenance`; it found **1,866 metadata files**: 494 input and 1,372 run. The sidecar stores one manifest observation per path, preserving each frozen Nov 25 SHA-256 and byte size. The previous headline 466/1,534 count uses an unresolved different selector and remains a documented discrepancy.

## Fresh Rockfish path revalidation

On 2026-11-26, a read-only Python 3.6.8 process on Rockfish login02 rechecked 2,322 unique declared personal-root paths. The final symlink was never followed.

| Recheck result | Paths | Meaning |
|---|---:|---|
| `METADATA_SIZE_SHA_MATCH` | 1,866 | Every selected input/run metadata file currently matches the Nov 25 size and SHA-256. |
| `TARGET_SIZE_SHA_MATCH` | 440 | Candidate referenced file matches its previously recorded size and small-file SHA-256. |
| `TARGET_SIZE_MATCH_SHA_NOT_RECHECKED` | 13 | File size matches; these larger candidate files were not rehashed. |
| `PRESENT_DIRECTORY` | 3 | Candidate path resolves to a directory; no file artifact was invented. |
| Missing, unreadable, hash error, scope escape | 0 | No target path failed the current personal-root recheck. |

The reference table retains **9,825 path-reference occurrences**. Of these, 1,092 literal path occurrences point under the two target roots (1,088 file-candidate occurrences and four directory observations in the frozen source scan; current target map resolves 700 unique paths). The 440 + 13 + 3 target-only recheck rows correspond to the 456 unique reference-only paths after paths already represented as metadata are deduplicated. Other references remain outside-scope or relative literals and were not resolved by scanning.

## SQL row grain and status boundary

- `manifest_observation`: 1,866 rows, one per indexed manifest/provenance file path.
- `manifest_control_observation`: 1,174 rows, one per manifest with job/retry/attempt/Slurm/scheduler-like fields.
- `manifest_control_field`: 3,367 scalar field values, preserving raw field paths and values. All are `UNVERIFIED`; they are not `slurm_job` records and do not establish scheduler state.
- `manifest_path_reference`: 9,825 rows, one per path-candidate occurrence; duplicates are retained and no fuzzy path resolution is introduced.
- `path_revalidation`: 2,322 unique paths with frozen expected identity and current stat/hash observation.

Manifest and target artifacts are `PROVENANCE_PARTIAL`. The sidecar intentionally contains no `workflow_run`, `slurm_job`, `run_artifact`, `paper_claim`, or `claim_evidence` rows. Thus directory names, job IDs, field values, or shared SHA tokens do not become scientific run/claim links.

## Build and integrity

The isolated importer requires the frozen input TSV/JSON exports, `REVALIDATION_TARGETS.tsv`, the Rockfish `REVALIDATION_RESULTS.tsv` and receipt, and `schema.sql`:

```sh
python3 import_revision1_manifest_sidecar.py \
  --source-dir source \
  --targets REVALIDATION_TARGETS.tsv \
  --revalidation-results REVALIDATION_RESULTS.tsv \
  --revalidation-receipt REVALIDATION_RECEIPT.json \
  --schema schema.sql \
  --output-db revision1_manifest_sidecar_r1.sqlite \
  --output-manifest SIDECAR_MANIFEST.json
```

The importer refuses overwrite, pins source scan/index/reference hashes, validates the Rockfish receipt, checks exact expected row grains, runs SQLite `quick_check` and foreign-key validation, and records its hashes and counts in the sidecar manifest. The built staging database reports `quick_check=ok` and zero foreign-key violations. Queries Q1–Q7 are in `queries.sql`.

## Merge gate

This package is deliberately not merged into the central catalog. A later merge should run against a new copy of the central snapshot, map staging roots to registered storage roots, detect artifact-key/path collisions, compare all rows additively, and retain the existing statuses. The job/retry field values must remain unverified until independently joined to scheduler records and validated run identity. No manuscript claim/result link is justified by this inventory alone.
