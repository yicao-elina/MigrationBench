# MD15 attempt-lineage audit sidecar R1

This is a **model-grain audit** for the 15 submitted Revision 1 MD status rows. It records evidence found in a targeted read-only scan; it does not assert one definitive attempt per model.

## Source and archive pins

- R02 status table SHA-256: `89f32a3e5f6587e8ca3652e3b65ca6249d4891601e46fe74fa08ac336c0dd9c3`.
- Attempt-lineage TSV SHA-256: `b2dc4b6ae0936b3b338ae1e212a37510239e43e58250c8a33a39174516f446cc`.
- The source scan README and remote stat/hash receipt are preserved beside the TSV. Their source hashes are stored in SQLite metadata.
- The SQLite archive is pinned in `snapshot_manifest.json`; it is a separate sixth read-only source (`md15a`), distinct from the 15-row model/reuse source (`mb15`).

## Evidence summary

- 6 FT models have registered full-run directories with manifests, continuation ledgers, status, metrics, and segments.
- 3 Foundation models have terminal status/metrics and multi-segment chains.
- 6 Scratch models have smoke evidence and candidate Slurm IDs, but no registered full-run directory, full-run continuation ledger, or 1 ns trajectory at the inspected paths.
- 14 Scratch16 checkpoint files with expected digests matched. The MPA-0 checkpoint lives under `/data` and was not inspected in this scoped audit.
- No attempt UUID or raw trajectory SHA-256 was recorded in the inspected producer manifests. Candidate scheduler IDs remain candidates recovered from scoped logs, not definitive job-to-attempt foreign keys.

## Tables and queries

- `model_attempt_audit`: one row per model; copies the entire input row as canonical JSON and preserves R02 labels.
- `evidence_artifact`: path-level evidence, status, stat size, and only hashes directly observed in scope.
- `scheduler_job_candidate`: candidate job IDs with an explicit non-definitive linkage status.
- `trajectory_segment`: parsed start step and reported stat size; SHA remains explicitly unrecorded.
- `missing_link`: missing-evidence text preserved from the source TSV.

Run `QMA01`–`QMA07` through `query_federated.py`. These queries expose status rows, artifact paths, candidate IDs, segment order, missing links, counts, and snapshot pins. Source IDs remain local to this sidecar.

## Scope

The scan inspected only paths explicitly named by the R02 status table and its registries/status/run manifests, plus immediate children of their explicitly named Scratch16 directories. It did not traverse `/data/pclancy3/yi`, `/scratch4/pclancy3/yi`, or another user's directory. `/data` is therefore not covered by the scan; MPA-0's checkpoint hash is not checked.
