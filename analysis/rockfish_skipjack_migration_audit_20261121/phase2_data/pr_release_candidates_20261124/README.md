# MigrationBench PR candidate source captures

Read-only snapshots used by `phase2_schema/import_pr10_pr12_pr13_candidate_releases_20261124.py`.

The 19 source files are captured from GitHub PR heads pinned in
`CAPTURE_MANIFEST.tsv`: PR #10 `51cb06f71732e46cd10eb92cee1f0c5f8f4833c9`, PR #12
`0505125b728d59cfbd36d45646f5ea58e0568ebc`, and PR #13
`2a40d7a171acb32e3352572eff67f06ab8634286`. `CAPTURE_MANIFEST.tsv` records each
capture's SHA-256, byte size, exact source URL, branch, commit, and capture time.

These are metadata, claim matrices, and release/file manifests. The importer
records package file hashes and pinned GitHub paths as catalog evidence; it
does not copy model checkpoints, trajectory payloads, or PR package trees.
The PR #12 selection remains `no` for FIG-057/058/059. The PR #10 DFT handoff
remains pending/reconstructed, and PR #13 remains review-only with Overleaf
handoff pending. None of these candidates changes the submitted release row.
