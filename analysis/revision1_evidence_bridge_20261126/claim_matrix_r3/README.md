# Claim-matrix R3: manuscript source authority

R3 is an additive copy of the pinned R2 claim-matrix snapshot. It retains the 639 submitted-release claim rows, 20 existing metadata rows, and 30 Fig. 4/R05 summary links, then adds exact manuscript-version identities, a row-level submitted-ledger ↔ PR #10 candidate-ledger delta, and the full 47-file PR #10 reconciliation against current main. R1 and R2 files remain unchanged.

PR #10 is closed without a merge commit, but its head is an ancestor of current main. Full byte comparison finds 44 of 47 PR files at the same path and hash; the three changed manuscript/source-ledger versions remain byte-identical in a candidate snapshot directory. The experimental artifacts and workflow code therefore already have a mapped mainline path and need no duplicate merge. The current live Overleaf server tip has not been captured or hash-compared; R3 records that unknown rather than treating the GitHub candidate as current server state. Mainline integration does not establish scientific validation.

## Database and queries

The immutable R3 SQLite file is archived at:

`/scratch16/pclancy3/yi/MigrationBench_data_archive/revision1_claim_matrix_snapshots/20261126_r3/claim_matrix_r3.sqlite`

Rebuild from the archived R2 source in the canonical checkout with:

```sh
python3 analysis/revision1_evidence_bridge_20261126/claim_matrix_r3/import_source_authority.py \
  /scratch16/pclancy3/yi/MigrationBench_data_archive/revision1_claim_matrix_snapshots/20261126_r2/claim_matrix_r2.sqlite \
  paper/revision1 \
  paper/revision1_candidates/pr10_overleaf_candidate_20261123/source/paper/revision1 \
  paper/revision1_candidates/pr10_overleaf_candidate_20261123/README.md \
  analysis/revision1_evidence_bridge_20261126/claim_matrix_r3/PR10_FULL_FILE_RECONCILIATION.json \
  /tmp/claim_matrix_r3.sqlite
```

Rockfish confirmed a 1,056,768-byte file with mode `0600`, SHA-256 `33d1370a51e5c105f05206d54a649968002089d9269c28a93159050b0a082511`, `quick_check=ok`, and zero foreign-key violations. The archive copy receipt is [ROCKFISH_ARCHIVE_RECEIPT.json](ROCKFISH_ARCHIVE_RECEIPT.json). Read-only QCE01–QCE09 results ran with SQLite 3.26.0 and `query_only=true`; row counts and output hashes are in [QCE_QUERY_VERIFICATION_R3.json](QCE_QUERY_VERIFICATION_R3.json), and the nine JSON outputs are retained in Rockfish `MigrationBench_data_archive/storage_audits/claim_matrix_r3_validation_20261126/`.

The federated `claim_matrix` source alias points to R3 after the merge and remote sync. QCE01–QCE04 retain the R1 query text and row grains; QCE05 retains the R2 Fig. 4/R05 summary query. New queries are:

- **QCE06** — submitted release, closed-unmerged candidate and unverified live Overleaf source states, with exact file paths, byte counts and SHA-256 identities.
- **QCE07** — full row-level differences between the two claim ledgers.
- **QCE08** — counts of candidate-only, submitted-only and altered ledger rows.
- **QCE09** — all 47 PR #10 file hashes, current paths, and reconciliation status.

Run the standard read-only bridge from the repository root:

```sh
python3 analysis/revision1_evidence_bridge_20261126/query_federated.py --query-id QCE06
python3 analysis/revision1_evidence_bridge_20261126/query_federated.py --query-id QCE07
python3 analysis/revision1_evidence_bridge_20261126/query_federated.py --query-id QCE08
python3 analysis/revision1_evidence_bridge_20261126/query_federated.py --query-id QCE09
```

## Interpretation boundary

QCE06–QCE09 index source identity, textual claim-ledger differences, and PR file integration only. They do not validate candidate claims or select them for publication. Keep the PR #10 manuscript snapshot separate from the submitted release until the author identifies the authoritative current manuscript source and resolves the scientific/evidence changes.

The detailed source comparison and unresolved source-of-record decision are in [MANUSCRIPT_SOURCE_AUTHORITY_AUDIT_R3.md](MANUSCRIPT_SOURCE_AUTHORITY_AUDIT_R3.md). Exact source, database, and script identities are in `snapshot_manifest.json` and `SHA256SUMS`.
