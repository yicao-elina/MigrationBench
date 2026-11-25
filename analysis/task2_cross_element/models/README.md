# Model checkpoint locations

The four model checkpoints used by this campaign are stored in the Rockfish archive:

/scratch16/pclancy3/yi/MigrationBench_data_archive/revision1_git_large_assets_20261124/models/

Their sizes, SHA-256 digests, source paths, Git blob identities where tracked, and verification records are in the sibling archive's ARCHIVE_MANIFEST.tsv. Set MIGRATIONBENCH_MODEL_DIR or MIGRATIONBENCH_MODEL_ARCHIVE to the directory containing the files when running loader scripts. The default points to the Rockfish archive.

| Canonical ID | File | SHA-256 | Git blob | Status |
|---|---|---|---|---|
| MACE-CHECKPOINT-SHA256-e618ad582b842399 | Foundation-MACE-matpes-pbe-omat-ft.model | e618ad582b84239905b9c3b77ce6e9ce111b0ecd1533223a1a6aac7a696b8aa0 | 4a701e1def120a16ad3e0e3d7dbdaa0881d72fa1 | archived, tracked in source snapshot |
| MACE-CHECKPOINT-SHA256-6829d16e68bdd123 | FT-600K.model | 6829d16e68bdd1238ee4226363f76ceac936d9e7325c9282283d72093e2bb88a | 671a4bba6fd517a2f3a303b9dfa012a8fa47a9d6 | archived, tracked in source snapshot |
| MACE-CHECKPOINT-SHA256-f61082f629b1f03d | FT_Multi_T.model | f61082f629b1f03db12e9b2f34a525790979d831ada3c895cae48eed18cf7588 | 384bb324503d8340ecd70006acf858f6fabfffef | archived, tracked in source snapshot |
| MACE-CHECKPOINT-SHA256-0d83b39b8ce81a7b | Scratch_Multi_T.model | 0d83b39b8ce81a7b7a4a26e5a3dfe51645aee6ca73180d72da4007d701b70fcf | 8ff3cce4fb68342a32853cf3f11c3e698a7be381 | archived, tracked in source snapshot |

The four canonical IDs use content SHA-256, independent of historic Foundation/FT/Scratch labels. Exact checkpoint identity is verified; training lineage remains PROVENANCE_PARTIAL because the source-checkpoint hash and exact launch/training manifest are not registered. The /data source copies were retained. Historic smoke JSON records retain paths observed during those earlier runs; they are not rewritten as if the old runs used this archive. Health-gate and DFT-gate limitations remain in force.
