# R03 Conda coverage reconciliation — 2026-11-23

## Coverage and method

This is a metadata-only reconciliation of `/data/pclancy3/yi/.conda`. It does
not read file payloads, compute new hashes, alter environments, open the shared
SQLite catalog, or delete anything. It joins the completed R01 full-tree
baseline, the R02 partial refresh, four isolated R03 per-environment scans,
and exact receipts for the two earlier Conda-cache removals. R03 did not run a
single full `.conda` walk; its complete coverage is established by the
path-set union and exact reconciliation below.

- R01 completed at 2026-11-22 18:33 UTC and recorded 689747 paths under
  `.conda`: 69119 directories, 598922 regular files, and 21706
  symlinks.
- R02 started at 2026-11-23 01:23 UTC and recorded 488492 `.conda`
  paths before the broader `/data/pclancy3` scan ended without a completion
  marker. R02 covers the complete current `.conda/pkgs` entry set and seven of
  eleven `.conda/envs` roots; four environment roots were incomplete.
- R03 scanned only those four missing environment roots and emitted explicit
  `root_started`/`root_complete` markers with zero errors. Counts below are
  children; the separately recorded root identity adds one path per root.

| Root | entries | dirs | files | symlinks | errors | elapsed (s) |
|---|---:|---:|---:|---:|---:|---:|
| `automatbench` | 39906 | 4160 | 34592 | 1154 | 0 | 110.35 |
| `deepmd2` | 39534 | 5343 | 32317 | 1874 | 0 | 123.40 |
| `qe-dev` | 17588 | 1268 | 15173 | 1147 | 0 | 46.03 |
| `sklearn-env` | 108751 | 11373 | 95382 | 1996 | 0 | 299.33 |

The four completed R03 child counts match R01 exactly after excluding each
environment root itself. The union of R02 and R03 covers 689599
current paths against 689599 expected paths: 0 uncovered
and 0 unexpected; 0 receipt-deleted paths
were still observed. R02/R03 overlapping identities and baseline
metadata were compared; see the machine-readable manifest for exact field
differences. The only expected baseline change is the `.conda/pkgs` directory
metadata after removal of children.

## Conda cleanup reconciliation

The R01 baseline was taken before the two earlier cache cleanups. The later
receipts account for exactly 148 paths: 42 index-cache files,
the 88 files and 2 symlinks plus 15 directories in the guarded unused
`libstdcxx-14.2.0-h8f9b012_2` package-tree removal, and the now-missing
`.conda/pkgs/cache` directory. R02 confirms the package-cache path set is
exactly baseline minus those recorded removals. The index-cache event remains
`PARTIAL` because it lacks pre-delete hashes and a contemporaneous queue
guard; it is not upgraded by this reconciliation.

Current reconciled `.conda` totals are 689599 paths: 69103
directories, 598792 files, and 21704 symlinks. Regular files
sum to 45716514629 logical bytes. Allocated path bytes from metadata
are 57142035456 (hardlink aliases can be counted more than once); the
unique-inode metadata estimate is 45664584192 bytes.

## Large paths and disposition

The reconciled tree contains 75 regular files larger than 100 MB,
totalling 23561512281 logical bytes; 12 exceed 500 MiB. The TSV
records each path, current owner where observed, inode/link count, allocated
path bytes, and disposition. The large files are installed Torch/TensorFlow,
CUDA, and related environment libraries or package-cache payloads. They remain
held because size alone does not establish an unused package or reproducible
replacement. The previously verified 574,565,016-byte `libcublasLt.so.11`
hardlink/duplicate group is still not a safe deletion candidate; removing a
single hardlink name would not release inode blocks while the other link
remains. No new safe Conda cleanup candidate was found and no R03 file was
deleted.

## Evidence and limits

- Full baseline and R02 partial stream remain unchanged under `phase2_data/`.
- Each R03 root archive, status JSON, and stderr file is preserved separately;
  the scanner source is `scan_conda_env_roots_r03.py`.
- Cleanup source evidence remains in `conda_index_cache_removed_manifest_20261122.tsv`,
  `conda_index_cache_removed_verification_20261122.txt`, and
  `conda_package_cleanup_execution_receipt_20261122.json`.
- These scans cover `.conda` only. They do not complete the whole
  `/data/pclancy3` tree or resolve the mount-wide quota discrepancy.
