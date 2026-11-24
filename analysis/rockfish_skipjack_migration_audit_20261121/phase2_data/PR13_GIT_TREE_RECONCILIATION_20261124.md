# PR #13 Git tree reconciliation

## Scope and method

Compared all 1,384 unique rows in the captured `FILE_PROVENANCE_MANIFEST.tsv` against the package subtree at bundle commit `07b9521796aa8c44d291d379b32b0d4b12a5823a` and PR-head commit `2a40d7a171acb32e3352572eff67f06ab8634286` in the Rockfish MigrationBench repository. Git path presence and byte size came from `git ls-tree -r -l`; for present blobs, SHA-256 was recomputed from `git cat-file --batch` bytes. Git blob OIDs below are Git SHA-1 object IDs, distinct from the manifest SHA-256 values.

Captured source manifest SHA-256: `2d5f746d9b09aedd2682b5cecdd59b2bd662a3da65c0e746d4625c0f3732bc34` (266231 bytes). The captured manifest and source capture files were left unchanged. The full row-by-row comparison is [pr13_git_tree_reconciliation_20261124.json](pr13_git_tree_reconciliation_20261124.json), SHA-256 `7f4ceb1f21ef1a611fe0aaa7808f4e5d0655a67de26a95a5d47d428cfe9fc3c6`.

## Results

- 1,384 unique manifest rows; no duplicate paths.
- 48 rows resolve to files in both pinned Git trees. Of these, 47 match the manifest byte count and SHA-256.
- `README.md` is the only manifest row whose declaration disagrees with a file present in both trees.
- 1,336 rows, all role `raw_evidence`, are absent from both Git trees. Their declared Git status is preserved, and the separate Rockfish archive below contains all 1,336 with verified size and SHA-256.
- No manifest path changed between the bundle and PR-head trees. Both trees contain 62 blobs in the package subtree; 14 tracked files are not listed in the 1,384-row manifest.

### README discrepancy

| Evidence | Bytes | SHA-256 / Git blob OID |
|---|---:|---|
| Manifest row `README.md` | 1,477 | SHA-256 `fcc5419f83cca28a7354d9ac9ecfb6c755b4c71259194347f23e212a559ad204` |
| Bundle commit Git blob | 2,216 | SHA-256 `05ad3cc1eacdcb921fb3c3923ed5db68ceca3fa4229af06200b42900a7dbe9d0`; blob OID `4903df26e4ea7d595dea267c009adfe8e4fc6cb4` |
| PR-head Git blob | 2,216 | SHA-256 `05ad3cc1eacdcb921fb3c3923ed5db68ceca3fa4229af06200b42900a7dbe9d0`; blob OID `4903df26e4ea7d595dea267c009adfe8e4fc6cb4` |

The bundle and PR head point to the same README blob. Preserve the manifest row as a stale declaration and index the current README bytes as a separate verified Git artifact.

### Tracked files not listed in the manifest

- `paper_handoff/MIGRATIONBENCH_INTEGRATION_MAP.md` (5331 bytes; SHA-256 `a4bc88225387bdb30c05f141fe0996ad07774a77f42c343e2f5a403c674fa677`)
- `paper_handoff/OVERLEAF_HANDOFF.md` (2688 bytes; SHA-256 `a2ae8e94c61b875c34520eb91a062aafcb7f367b0bde5abe58e77a053195462a`)
- `paper_handoff/OVERLEAF_HANDOFF_MANIFEST.tsv` (3140 bytes; SHA-256 `91271c8cc2da626e033833efcc5d969cf18986c9fbf1c7fe2df572c5028de370`)
- `paper_handoff/PAPER_CLAIM_EVIDENCE_MATRIX.tsv` (2746 bytes; SHA-256 `c31d81b1f48286668a07d82ea8c233cc65305f7458d4e0e59541947a39b03de7`)
- `paper_handoff/PAPER_RELEASE_MANIFEST.json` (2358 bytes; SHA-256 `be788edf2e8a5baff1c7a05e6328288ebacf77827fa8a3e84ad43981b7c56bfe`)
- `paper_handoff/PR_BODY.md` (1971 bytes; SHA-256 `2ee67c4087087408518c5005c47b5aa3bac7904d74c8ea01fbd536544e05c8b9`)
- `raw_evidence_bundle.tar.gz` (3237382 bytes; SHA-256 `4930f34a9935d1e368a49c3081a8a1185e3418a2d30761a0a911660d9dfad7be`)
- `scripts/__pycache__/build_attribution_package.cpython-311.pyc` (41154 bytes; SHA-256 `b30a481549482888c951db7787b796ccb912d1cbacc995647bacaea54b7f3777`)
- `scripts/__pycache__/integrate_source_geometry_ensemble.cpython-311.pyc` (18143 bytes; SHA-256 `ccf2abdea73d86d2d62bf31b5b0b0ae596c45c51c9737d983fe2ca4d07560b85`)
- `scripts/__pycache__/render_publication_figures.cpython-311.pyc` (36177 bytes; SHA-256 `c7b73b8ea921bb7ee464010ef973d2a06cc9363bb0a4527b64adf2887ede65a6`)
- `scripts/__pycache__/write_package_provenance.cpython-311.pyc` (5092 bytes; SHA-256 `9448d56cd8ac3b76d5cc865afb013e47f40b8c82c6320f9f3150910533608560`)
- `scripts/build_overleaf_handoff.py` (5515 bytes; SHA-256 `f535d21d1bcdc06521fe7f4538173ecce35ac636fae32400ae87d36975bfb7e9`)
- `tables/FILE_PROVENANCE_MANIFEST.tsv` (266231 bytes; SHA-256 `2d5f746d9b09aedd2682b5cecdd59b2bd662a3da65c0e746d4625c0f3732bc34`)
- `tables/PACKAGE_PROVENANCE_MANIFEST.json` (372 bytes; SHA-256 `721b558a460fd2022ab870241fef2f29465de3ad5baabad19c8d0bcf3e9a5c23`)

The importer preserves each row as a source-manifest declaration with its Git-tree status, and records Rockfish archive content as a separate evidence source and storage location. It creates no Git pointer for the 1,336 paths that are absent from both Git trees.


## Rockfish raw archive

The 1,336 `raw_evidence` files are now organized under `/scratch16/pclancy3/yi/MigrationBench_data_archive/revision1_pr13_endpoint47_raw_evidence/`, catalog root ID `RF-SCRATCH16-YI`. The archive preserves each source-relative `raw/...` path. Independent live verification compared the archive manifest with the unchanged source manifest and recomputed the size and SHA-256 of every Rockfish file: **1,336/1,336 verified, 8,942,992 bytes, zero mismatches**. These are `PRESENT` Rockfish artifacts; their separate Git status remains absent from both pinned trees.

`ARCHIVE_MANIFEST.tsv` has 1,337 data rows: 1,336 raw files plus one `source_bundle.tar.gz` row. Its SHA-256 is `438b2255565a8d8fe29620f655fecf742768d526fcbe5df4512e45b7c4bfb142`; `VERIFICATION_REPORT.json` SHA-256 is `a6edb301e70e7c696187b301e2d7cde1c96260d8359d28299e5bf37110b12cdf`; archive `README.md` SHA-256 is `5601cc2d3b921e5dfe20ed9823f9b2e946fa7424a059c6d4bb35e7bea6489891`. The source bundle is 3,237,382 bytes, SHA-256 `4930f34a9935d1e368a49c3081a8a1185e3418a2d30761a0a911660d9dfad7be`, Git blob OID `b7f65ae69e7a18afd3a555d259568ffc716d7a6e` at commit `2a40d7a171acb32e3352572eff67f06ab8634286`. The archive manifest, verification report, and README are captured separately in `phase2_data/pr13_raw_archive_capture_20261124/`; the original PR13 file-manifest capture remains unchanged. The generated `.pyc` excluded by the archive receipt is not part of the 1,336 files.
