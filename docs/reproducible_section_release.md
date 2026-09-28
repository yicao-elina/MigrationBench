# Reproducible section release protocol

Each completed manuscript section should be released as a small, auditable
change set. A section is ready for GitHub only when its scripts, compact data,
configuration, and provenance can be rerun without access to an unrecorded
cluster directory.

## Required contents

- an executable script or Snakemake target under `scripts/`, `src/`, or
  `workflow/`;
- the smallest processed/compact data needed to reproduce the section, with
  raw or heavyweight source data referenced by an external accession/path;
- a config or manifest recording model/checkpoint identity, input paths, seeds,
  scheduler job IDs, software environment, and protocol parameters;
- a short section-level README describing the command, expected outputs, and
  scientific acceptance gates;
- SHA256 entries in the release registry for every committed data or result
  artifact used by a manuscript number.

## Acceptance sequence

1. Run the section from a clean checkout using the documented environment.
2. Run the narrowest section test and the repository-wide `make test`.
3. Run `make audit` and update the relevant registry/claim ledger only from
   measured outputs.
4. Rebuild the section outputs in a fresh output directory and compare hashes.
5. Inspect `git diff --check`, confirm no credentials, caches, virtualenvs,
   model checkpoints, QE `outdir` trees, or unreviewed scheduler state are
   included.
6. Commit the code, compact data, manifest, and documentation together in one
   branch/PR. Keep failed attempts in an evidence directory or history note;
   do not silently replace them with a later successful run.

## Data boundary

Git stores compact, license-compatible, provenance-complete inputs and outputs.
Large trajectories, wavefunctions, charge densities, complete QE save trees,
and model checkpoints remain in the approved archive/cluster location, with a
stable external identifier and checksum recorded in the manifest. A file's
presence or a completed Slurm job is not by itself a scientific result: the
section must report the measured acceptance metrics and their evaluation
protocol.
