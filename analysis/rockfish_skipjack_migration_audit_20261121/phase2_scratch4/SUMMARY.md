# Rockfish `/scratch4` user tree — phase 2 inventory

**Scope:** `/scratch4/pclancy3/yi` only. Read-only inspection; no remote files
were moved, edited, or deleted.

## Inventory snapshot

- Full metadata walk: 2,886 regular files, 55 directories, no symlinks or
  traversal errors; 5,642,682,016 logical bytes. `du -sh` reported 5.3G
  allocated/reported usage for this tree.
- `2.Cr-Sb2Te3`: 2,844 files / 425,753,425 logical bytes. It contains the
  previously scoped phonon and Raman work: QE inputs and logs, 1,500+ small
  displacement/input files, scripts, XML and charge-density outputs, plots,
  and result tables. No WFC restart files over 100 MB were found here.
- `3.WSe2`: 42 files / 5,216,928,591 logical bytes. Three completed QE
  variable-cell relaxations account for 5,045,101,744 bytes in eight `wfc*.dat`
  restart files (six files are in the 54/108-atom runs; the largest single
  file is 1,364,047,632 bytes). The remaining 13 files in `0.diff-layer`
  include CIF inputs and Python/notebook generation scripts.
- No `.git` repositories or run manifests were found under this tree.
- The six WSe2 XML files are exact-content pairs within each run: the top-level
  `WSe2_vc_relax_z.xml` and `.save/data-file-schema.xml` hashes match, but they
  have separate inodes. They are small retained outputs; no XML was removed.

## WSe2 restart review

Each WSe2 run has its QE input, exact submission script, captured Slurm log,
`relax.out`, top-level XML, `.save/data-file-schema.xml`, and charge density.
The script invokes QE 7.2 with 48 MPI ranks and each `relax.out` contains SCF
convergence, BFGS completion, a final enthalpy, and `JOB DONE`. The pseudopotential
files referenced by the inputs exist at `/home/ycao73/pseudo/`; their paths and
SHA-256 hashes are recorded with the run lineage.

Date-bounded accounting on the configured Slurm service recorded jobs
19076231, 19075950, and 19075952 as `COMPLETED 0:0` on 2024-12-20/21. The
records were checked at 2026-11-22 18:21:53 UTC. These IDs were reused for
different jobs in 2026, so date-bounded `sacct` is required. At 18:24:13 UTC,
none of the three IDs was in the current queue. A text/source search under
`/scratch4/pclancy3/yi` found no downstream use of the WSe2 restart paths
beyond the WSe2 input-generation and run scripts. An additional exact-path and
prefix search over `/scratch16/pclancy3/yi/FLARE/wse2`,
`/scratch16/pclancy3/yi/FLARE/lammps/wse2`, and
`/data/pclancy3/yi/flare-data/2-WSe2-growth` found no matching source or
manifest references. This bounded result is recorded in
`wse2_external_reference_search_20261122.txt`; it does not rule out consumers
outside those WSe2-specific trees.

The eight WFC files are therefore **cleanup candidates** after their rows are
ingested into the project SQL registry and the parent audit confirms no external
consumer. Their sizes, modes, mtimes, and pre-cleanup SHA-256 hashes are in
`candidate_wfc_manifest.tsv`. No cleanup was performed in this inventory
subtask.

## Evidence files

- `inventory_scratch4.py` and `scratch4_file_inventory.tsv`: repeatable
  metadata-only inventory and per-path results (type, logical bytes, mtime,
  mode, extension, traversal errors).
- `directory_summary.tsv`, `large_files_over_100MB.tsv`, and
  `extension_summary.tsv`: hierarchical file/byte totals, all eight files over
  100 MB, and extension-level size totals.
- `wse2_run_lineage.tsv`: one row per WSe2 run, with Slurm state, input/output
  and pseudo paths/hashes, module/executable details, and reproduction command.
- `candidate_wfc_manifest.tsv`: one row per proposed WFC cleanup artifact.
- `wse2_reproduction_evidence_sha256.txt` and
  `wse2_wfc_sha256_predelete.txt`: input/script/log/output/pseudo hashes and
  deletion-candidate pre-cleanup hashes. Exact local copies of the three
  submitted scripts are under `sbatch_scripts/`; their local hashes were checked
  against the Rockfish script hashes.
- Historical quota and scheduler captures are not included in this
  scope-limited package; the current inventory and cleanup decision need their
  own path-scoped preflight.
- `downstream_reference_search.txt` and
  `wse2_external_reference_search_20261122.txt`: bounded textual search evidence.

## Method and limits

The scan used Python `os.scandir`/`lstat` metadata across the full user tree;
it did not read file contents except for SHA-256 checks on the eight WFC
candidates and the listed WSe2 run evidence/pseudopotentials. The downstream
reference search excluded binary and generated output classes and only covered
`/scratch4/pclancy3/yi`; it cannot rule out references elsewhere on Rockfish.
No deletion decision is based on scheduler `COMPLETED` alone: input, output,
script, log, XML, charge-density, and pseudo lineage were also checked. The
absence of a per-run manifest and the lack of a paper-claim link mean the
scientific/publication status remains unadjudicated.
