# MigrationBench large asset archive on Rockfish

The following large binary assets were removed from the current Git tree after
their Git-index blobs were copied and SHA-256 verified in the Rockfish archive:

`/scratch16/pclancy3/yi/MigrationBench_data_archive/revision1_git_assets_20261123/`

The archive has 14 path records totaling 405,789,182 logical bytes. Its
`git_large_assets_manifest_20261123.tsv` records each source path, archive path,
size, SHA-256, Git blob ID, source branch, and pre-removal merge heads. The
manifest SHA-256 is
`2b783520d47a0219f9676f09f6dc5e6377d538e830da243a00128e4d26bf594c`.

| Category | Paths | Logical bytes | Examples |
|---|---:|---:|---|
| Blender project files | 9 | 247,250,552 | Scene and source blend files for the Cr migration render set |
| High-resolution renders | 2 | 53,150,077 | 4500×3000 and 3600 px PNG renders |
| Reproducibility arrays | 2 | 87,218,368 | `fig6_current_input.npz` in the as-found and camera-ready trees; identical SHA-256 |
| Rendered media | 1 | 18,170,185 | Migration story MP4 |

The archived set was selected from the staged Git index at branch
`codex/revision1-main-consolidation-20261123`, HEAD
`8fd83b65676de7c35d1cc6fe646af017565e8a89`, with release merge head
`f924ca6c11236c4b75469e290a0abff884c56c02`. Full-file SHA-256 verification
passed for all 14 archive destinations. No WFC or mixing restart files are in
this archive. Smaller PDF/SVG publication figures and source scripts remain in
Git.

To restore the archived files to their original relative paths on Rockfish:

```bash
python3 - <<'PY'
import csv
from pathlib import Path
import shutil

archive = Path('/scratch16/pclancy3/yi/MigrationBench_data_archive/revision1_git_assets_20261123')
repo = Path('/scratch16/pclancy3/yi/MigrationBench')
with (archive / 'git_large_assets_manifest_20261123.tsv').open(newline='') as f:
    for row in csv.DictReader(f, delimiter='\t'):
        src = archive / row['archive_rel']
        dst = repo / row['source_rel']
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
PY
```

The local SQLite provenance catalog indexes each archived path and its checksum
under project `MIGRATIONBENCH`; the project query file is
`analysis/rockfish_skipjack_migration_audit_20261121/phase2_schema/catalog_queries.sql`.
