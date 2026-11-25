# FT-600K random-reproduction retraining audit

Audit date: 2026-10-24 (America/Los_Angeles)

Campaign: `ft_600k_random_reproduction_20261016`

Remote root: `/scratch16/pclancy3/yi/ft_600k_random_reproduction_20261016`

## Outcome

The retraining did not succeed. The submitted array was not left pending: both
array elements started on 2026-10-18 and failed before training began. The
failure is an environment/bootstrap failure, not a model-quality result.

Because no compiled checkpoint or preflight record was produced, the repaired
seed234/seed345 comparison remains invalid and no 1 ns MD job was submitted.

## Scheduler and filesystem evidence

The local submission record reported only the state at submission time
(`PENDING`), so it is superseded for final status by the Rockfish accounting
record below.

| array task | training seed | Slurm execution ID | state | exit | elapsed | start | end | node |
|---|---:|---:|---|---|---:|---|---|---|
| `32121744_0` | 234 | `32151298` | `FAILED` | `1:0` | 00:00:03 | 2026-10-18 06:35:16 | 2026-10-18 06:35:19 | gpu04 |
| `32121744_1` | 345 | `32121744` | `FAILED` | `1:0` | 00:00:01 | 2026-10-18 06:35:28 | 2026-10-18 06:35:29 | gpu04 |

The two `.err` files both report the same root cause at script line 28:

`/data/pclancy3/yi/.conda/etc/profile.d/conda.sh: No such file or directory`

The remote root contains six files: the manifest, submission script, four
logs, and an empty `models/` directory. There is no checkpoint, `preflight.txt`,
or `model_hash.txt` for either seed. The scheduler's `extern` steps completed,
but the batch steps failed; that does not count as training success.

## Root-cause check

Read-only checks on Rockfish found:

- `/data/pclancy3/yi/.conda/envs/mace/bin/python` is present and imports
  `torch 2.7.0+cu118`, `ase 3.25.0`, and `mace 0.3.12`.
- `/data/pclancy3/yi/.conda/envs/mace/bin/mace_run_train` is present.
- The two paths attempted by line 28 for `conda.sh` are absent.

The next retraining submission therefore needs a launch repair that uses the
existing environment directly or sources a verified conda installation. It
must create a new, non-overwriting attempt record; the failed attempt and its
logs remain historical evidence.

## Publication boundary

The FT-600K three-seed comparison remains `EXCLUDE_FROM_PUBLICATION` for the
repaired seed234/seed345 rows. Do not merge these rows with grouped-split
FT-600K models. The failure is retained as a `FAILURE_MODE` of the retraining
launch, not converted into a scientific training result.

## Evidence files

Small remote evidence files were copied additively to
`remote_evidence/`. Their SHA256 values and the scheduler query are recorded
in `AUDIT_MANIFEST.json` and `scheduler_evidence.txt`.
