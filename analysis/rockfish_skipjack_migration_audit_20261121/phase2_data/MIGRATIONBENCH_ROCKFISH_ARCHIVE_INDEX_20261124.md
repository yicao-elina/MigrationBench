# MigrationBench Rockfish archive index — 2026-11-24

Primary organized root:

/scratch16/pclancy3/yi/MigrationBench_data_archive/

This index groups already verified, transferable evidence bundles without
claiming that every project-owned /data or /scratch4 path has been copied.
The SQLite provenance catalog is the lookup source for paper claims, release
items, hashes, and artifact locations. Its query cookbook is
analysis/rockfish_skipjack_migration_audit_20261121/phase2_schema/catalog_queries.sql
plus pr_candidate_release_catalog_queries_20261124.sql.

## MigrationBench archive bundles

| Rockfish path | Contents | Verified scope |
|---|---|---|
| revision1_git_assets_20261123/ | 14 large Git-index assets: Blender scenes, renders, reproducibility arrays, and story video | 405,789,182 logical bytes; per-file SHA-256 recorded in git_large_assets_manifest_20261123.tsv |
| revision1_git_large_assets_20261124/ | Four MACE checkpoints and the CANVA-P03 editable SVG | 5 payloads, 167,458,328 logical bytes; full SHA-256; Git blob IDs for the four tracked source-tree items |
| revision1_pr13_endpoint47_raw_evidence/ | PR #13 endpoint-47 raw evidence, preserving the source raw/ layout | 1,336 files, 8,942,992 bytes, checked against the source manifest |
| revision1_md_fig6_table_s8_20261124/ | Six grouped-A2 MD bundles: six coordinate segments, six source/final checkpoints, six per-seed MACE weights, one shared 2050-atom input, and run manifests/status/metrics/ledgers | 49 payload files, 2,320,577,507 bytes; segments 1,917,017,196 bytes, models 78,571,126 bytes, shared input 320,656,648 bytes; one run reached 1,000,000 and five stopped at earlier scientific thresholds; all file SHA-256 verified; no model hash duplicates the four-checkpoint archive |
| revision1_restart_cleanup_20261123/ | Cleanup receipts, scheduler evidence, coordinate/log hashes, and post-delete scan | Audit evidence only; deleted restart payloads are not archived |
| storage_audits/ | Exact-path deletion receipts and bounded inventory evidence | Metadata and manifests; no broad false-positive projwfc/PDOS files removed |
| provenance_catalog_snapshots/20261124/ | Read-only SQLite provenance catalog snapshot | See sibling DB_SNAPSHOT_MANIFEST.json for size, SHA-256 and integrity checks |

## Related DUAL-X archive

The X-FORCE/DUAL-X data archive is kept beside the MigrationBench archive:

/scratch16/pclancy3/yi/X-FORCE_data_archive/

Its rockfish_main_snapshot_20261123/ holds the verified source snapshot and
31 Git-index TIFF assets; dualx_research_data_20261123/ holds selected
research/model evidence. X-FORCE uses its own consolidated main repository
candidate and PR #1; it is not merged into MigrationBench.

## Original calculation data and transfer boundary

Most QE inputs, outputs, XML, charge densities, coordinate paths, MD trajectories,
and partner data remain in their original calculation trees and have not been
bulk-moved into this archive. The selective six-run grouped-A2 MD bundle above
is tied to the Revision 1 Fig. 6 / Table S8 evidence index; it does not represent
all 15 model conditions or all MD run roots. Other source scopes are uneven:
some run families have full manifests and deletion receipts, while older Daily
DFT trees and inaccessible /data and /scratch4 branches do not. Use the SQL
inventory and provenance queries to select exact run families and stable source
paths before a later Globus package. Do not treat a directory name, file size,
or JOB DONE alone as a sufficient migration manifest.

No original /data model copies were deleted during model centralization.
The archive is organized for later transfer; the personal Rockfish quota did
not provide a new after-cleanup reading in this audit.
