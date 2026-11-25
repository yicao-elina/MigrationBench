# QE binary identity and Slurm timestamp audit — 2026-11-23

Read-only; no database, calculation inputs, project files, or scheduler state was changed.

## QE 7.2 binary identity

The full machine-readable shell output is in `qe_binary_identity_20261123.txt`.

| Binary | `readlink -e` | Mode | Bytes | SHA-256 |
|---|---|---:|---:|---|
| `pw.x` | `/data/apps/extern/qe/7.2/bin/pw.x` | `0755` | 32,383,312 | `e893e4835199422ac09218709bab451b6d47620fdfaa873d135c582da028cb81` |
| `pp.x` | `/data/apps/extern/qe/7.2/bin/pp.x` | `0755` | 25,618,952 | `ab617735f7d8ea62870a2886a8afcfab4fd342a6a56221dec63c8036dbc6c328` |

Both are regular files owned by UID 1218/GID 1002, link count one. These are the exact paths invoked in the retained Slurm scripts. Hash reads completed without a metadata change during the read.

## `sacct` timestamp interpretation

- The original `qe_candidate_sacct_20261123.psv` has no timezone field, and there is no capture-command receipt beside it proving that `TZ=UTC` was set. Its Start/End values must be described as unzoned raw `sacct` timestamps.
- The parent-provided `qe_candidate_sacct_utc_20261123.psv` has a `TZ=UTC` capture context. Comparing by job/step ID gives 12/12 matching rows, and every Start and End is exactly **14,400 seconds (+04:00)** later than the original PSV.
- The three original Slurm stdout logs end with explicit `EDT` on July 13/16, 2025, consistent with the +04:00 conversion. This supports interpreting the original values as Eastern daylight-time wall-clock values for those jobs; the original capture itself remains unproven as to timezone.
- At the later read, Rockfish `date +%Z %z` returned `EST -0500`; `TZ=UTC date` returned `UTC +0000`. This is the current system timezone and does not establish the shell environment at the time the original PSV was captured.
- A direct current `sacct` invocation was unavailable on the remote login PATH (even after sourcing `/data/apps/go.sh`), so this audit did not run a second live `sacct` query. The `qe_sacct_timezone_check_20261123.txt` file preserves that failed command attempt and the `date` outputs.

The audit therefore leaves the original PSV timestamps labeled `TIMEZONE_UNSPECIFIED`; use the UTC recapture for UTC calculations and preserve the Slurm logs' explicit EDT as separate contemporaneous evidence.
