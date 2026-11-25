# Fresh metadata inventory: `/data/pclancy3/yi`

Captured 2026-11-25 10:35:13 UTC from Rockfish `login02`. Scope was exactly `/data/pclancy3/yi`; the walk used `lstat`-style metadata and did not follow symlinks, hash content, read file contents, or delete/modify Rockfish files. Traversal completed in 237.24 seconds with no stat errors.

## Current totals and ownership

| Measure | Current observation |
|---|---:|
| Regular-file paths | 673,601 |
| Directories below root | 79,149 |
| Symlinks | 22,992 |
| Total descendant entries | 775,742 |
| Regular-file logical bytes (path sum) | 257,588,816,651 |
| Allocated bytes (path sum) | 270,846,348,800 |
| Regular files uid/gid | uid 3901 / gid 3001 for every observed regular-file path |
| Stat/traversal errors | 0 |

There are 150,024 in-tree hardlink inode groups spanning 317,716 regular-file paths. 150,019 groups are within `.conda`; five are outside it. They are shared inode paths rather than independent byte-for-byte copies. A path-sum adjustment estimates about 12.89 GB of repeated allocated blocks inside the tree, but this is not recoverable-space evidence: links outside this scoped tree and consumers are not known. Do not remove links based on this inventory alone.

## Largest files and comparison to R04

There are 31 current files over 500 MiB, totaling 59,142,123,982 logical bytes. They match all 31 rows in the November 23 R04 large-file inventory by path, size, allocated bytes, device/inode, link count, and recorded mtime where R04 had an mtime. No new or missing >500 MiB path was observed. The full current list is in `large_over_500MiB_current.tsv`.

Largest paths include:

| Bytes | Path | Initial interpretation |
|---:|---|---|
| 10,832,901,604 | `flare-data/1-Cr-Sb2Te3/1.2_bands/1.vcrelax/Sb2Te3/atomic_proj.xml` | QE projection output; two in-tree hardlinks |
| 9,448,365,054 | `flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/X-FORCE/.git/objects/pack/pack-261424fcac029ceb31a391c4bc87ac5457e35664.pack` | Git object pack; repository integrity/ref review required |
| 4,825,525,031 | `flare-data/1-Cr-Sb2Te3/1.pwrelax/2.Sb2Te3/5.soc+optb86b/bands/atomic_proj.xml` | QE projection output; two in-tree hardlinks |
| 4,447,818,716 | `flare-data/1-Cr-Sb2Te3/1.pwrelax/2.Sb2Te3/2.vdw_corr/222/bands_before/atomic_proj.xml` | QE projection output; two in-tree hardlinks |
| 3,774,766,882 | `flare-data/1-Cr-Sb2Te3/1.pwrelax/3.Cr-Sb2Te3/*new_vdw_relaxed/antisite/1-Sb10/bands/atomic_proj.xml` | QE projection output; two in-tree hardlinks |
| 3,577,772,883 | `flare-data/1-Cr-Sb2Te3/4.lmp/Rippling/2L_octo_Cr2_v2/0806-NEB/3-ft-multi_T/neb_optimization.traj` | NEB trajectory; run/acceptance lineage needs review |
| 2,022,132,456 | `flare-data/1-Cr-Sb2Te3/3.neb/hpc_collaboration_20260306/mlff_datasets/farhan_perovskites_maidmf__20260310_162437__tr80_va10_te10__s42/merged/all.extxyz` | Curated training dataset; collaboration manifest/use chain needs review |
| 1,618,487,990 | `flare-data/1-Cr-Sb2Te3/3.neb/hpc_collaboration_20260306/mlff_datasets/farhan_perovskites_maidmf__20260310_162437__tr80_va10_te10__s42/splits/train.xyz` | Training split; manifest/model-use chain needs review |
| 876,284,922 | `other/AI/results/checkpoint-2500/optimizer.pt` | Optimizer checkpoint; training lineage needs review |
| 742,302,539 | `mylammps/.git/objects/pack/pack-a21eb80142455c386b435c37a9b9543ec5865d19.pack` | Git object pack; repository integrity/ref review required |

The current inventory also contains multi-gigabyte CUDA/TensorFlow runtime libraries in installed environments. The catalog marks them as installed runtime libraries or package-cache files; they need environment/package-use review, not size-only deletion.

## Existing provenance/catalog coverage

The existing local provenance catalog was queried read-only, restricted to `storage_root.root_id='RF-DATA-YI'`. All 200 paths in `top_200_files.tsv` had catalog artifact/location records. Catalog records include existing SHA-256 values from earlier audits; this refresh did not compute hashes. Among the top 200, 32 have a `run_artifact` link; none have direct recipe-input, claim-evidence, figure-asset, or cleanup-item links. Catalog retention classes for those 200 are: 14 `PUBLICATION_EVIDENCE`, 13 `PERMANENT_SOURCE`, 2 `REPRODUCIBLE_INPUT`, 2 `CACHE`, and 169 `UNCLASSIFIED`. This leaves many scientifically relevant large files only partially mapped; it does not establish that their evidence package is complete.

`manifest_structured_files.tsv` is a name/extension-based candidate list for JSON/CSV/TSV/YAML and paths containing “manifest”; it is not a parsed manifest inventory. It contains 501 files at or below 100 MiB and one structured file above 100 MiB. The largest structured candidate is a 108,917,375-byte CSV of positions. No file contents were opened to infer manifest membership.

The filename heuristic found 20 `wfc`/`mix`-like matches, all tiny generic SciPy fixture files or LAMMPS examples (largest 324,924 bytes); no QE WFC or mixing-restart target was identified.

## Ownership and job boundary

All regular-file metadata observed under the specified root reports uid 3901 and gid 3001. No other owner appeared in this scoped walk. A live user-scoped Slurm queue check could not be performed because `squeue` is absent from the Rockfish login-shell PATH (`squeue: command not found`). Therefore no candidate is cleared for cleanup by this inventory; recheck the owner’s current job/workdir overlap using a working Slurm client before any cleanup review.

## Evidence files

- `inventory.json`: full traversal summary, top 200 files, all in-tree hardlink group/path metadata, structured-file candidates, WFC/mix-name candidates, and errors.
- `top_200_files.tsv`: top 200 regular-file paths by logical size.
- `large_over_500MiB_current.tsv`: all 31 paths over 500 MiB.
- `in_tree_hardlink_paths.tsv`: all paths in in-tree multi-link inode groups.
- `top_200_with_catalog_refs.tsv`: catalog classification and direct link counts for top 200.
- `manifest_structured_files.tsv`: heuristic structured/manifest-like path candidates.

Prior comparison source: `../yi_full_inventory_20261123_r04/yi_large_files_over_100MB_current_r04.tsv`.
