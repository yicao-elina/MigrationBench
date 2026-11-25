# Discovery DFT campaigns — organized package (2026-10-03)

Built from the Reviewer-Response-Tracker's open DFT needs + the proven working Discovery QE examples
(`2L_octo_Cr1` relax, `2L/NEB` climbing-image NEB) that the user confirmed as successful (`JOB DONE`).
All packages below have been **re-verified non-magnetic** (`grep -iE "nspin|magnetization"` returns nothing)
after the 2026-10-02 Phase 3 magnetism incident (see
`analysis/phase3_dft_validation_20261002/status_20261002.md`).

Local staging root: `analysis/discovery_dft_campaigns_20261003/`

## Category 1 — `01_endpoint_relax_47/` (reviewer A1, 47-endpoint provenance campaign)

Rockfish source: `/scratch16/pclancy3/yi/revision1_migrationbench_inputs/endpoint_dft_qe72_nonmag_campaign_20260930/`
(the 2026-09-30 non-magnetic rebuild referenced in the tracker).

- **10/47 endpoints have a clean, verified non-magnetic `relax.in` ready**: `041 042 044 051 052 056 057 065 066 067`.
  Converted to Discovery form in `01_endpoint_relax_47/discovery_ready/unique_<ID>/` (pseudo_dir swapped to
  `/home/ycao73/ONCVPseudoPack/SG15_v1.2/UPF/PBE/`, `calculation='relax'`, fixed cell, `ion_dynamics='bfgs'`,
  matches the proven `2L_octo_Cr1.in` style — single-GPU, `-N 1 --ntasks-per-node=4 --cpus-per-task=2 --mem=96G
  --gres=gpu:1`, 71h walltime). Each folder has its own `relax_unique_<ID>.in` + `sbatch_relax_unique_<ID>.slurm`;
  the Slurm script aborts before submission if it finds `nspin`/`magnetization` in the input.
- **37/47 endpoints do NOT yet have a prepared non-magnetic input** (they were part of the original 47-endpoint
  list but were never rebuilt after the 2026-09-30 magnetic→non-magnetic switch, or were never submitted at all).
  Generating these needs the original MACE-relaxed geometry for each `unique_ID` (candidate source is
  `/scratch16/pclancy3/yi/MigrationBench_collaborators/processed_endpoints.extxyz` +
  `endpoint_candidates.csv`, not yet pulled/converted in this pass) — **flagged as follow-up, not done here**.
- **⚠️ Cleanup flag (found, not yet acted on)**: every one of the 10 prepared "nonmag" directories still contains
  a leftover magnetic `relax.source.in` (looks intentional — kept as before/after provenance), and
  `unique_041`/`unique_042` additionally have stale magnetic `relax.cont01.in`/`relax.cont02.in` continuation
  files. **Verified the actual submitted jobs (`nm_dft_unique_041_qe72_31409080`, `..._31409069`) only ever used
  the clean `relax.in`** — no contaminated result exists. But the stale `cont*.in` files are a landmine for any
  future walltime-continuation resubmission that blindly copies "the last cont file" — recommend deleting or
  renaming them (e.g. `*.in.DO_NOT_SUBMIT_magnetic_leftover`) on Rockfish before anyone continues those two jobs.

## Category 2 — `02_neb_pathway_1-4_in_gap/` (reviewer A1, QE-NEB production for the in-gap pathway)

Source: `/Users/alina/Project/26MIgrationBench/cluster/mb_qe81n4_direct_pf_s42_30791514/` (local mirror of
Rockfish `revision1_migrationbench_runs/`), which is the pathway `1-4` in-gap production NEB that the tracker's
A1 row calls "归档 QE NEB 0.336 eV 未达收敛标准" — it only completed **1 iteration** before hitting its 6h Slurm
walltime (`neb.out`: `WARNING: scf convergence NOT achieved on image 1` then clean shutdown at `max_seconds`).

- **Already non-magnetic** (verified: no `nspin`/`magnetization` in `neb.in` or any `pw_*.in`).
- **`pseudo_dir` already points at the Discovery path** (`/home/ycao73/ONCVPseudoPack/...`) in the source file —
  this input appears to have been prepared with Discovery in mind already, or the path happens to coincide; not
  changed further.
- **Bumped `max_seconds` from `19800` (5.5h) to `171000` (~47.5h)** to match the walltime budget that the proven
  `2L_octo_Cr1`/`2L/NEB` examples actually used — the previous attempt's single-iteration result is consistent
  with simply running out of time, not a convergence failure in the physics.
- **Slurm script is a direct adaptation of your working `2L/NEB/sbatch-script.slurm`** (same resources: `-N 1
  --ntasks-per-node=16 --cpus-per-task=4 --mem=400G --gres=gpu:4 --partition=azgpuh200 -A pclancy3_lab`, same
  `singularity exec --nv --bind $(pwd):/input ... neb.x -inp /input/neb.in` pattern), just retargeted paths/job
  name and with the same pre-submit magnetism guard as category 1.
- **⚠️ Needs your decision before submitting**: this `neb.in` uses `degauss = 0.00500`, but the endpoint-relax
  and Phase 3 single-point inputs all use `degauss = 0.0146997236`. Different smearing width changes the
  electronic structure and is not automatically comparable across calculation types — confirm whether this is
  an intentional NEB-specific choice (tighter smearing for better saddle-point resolution) or should be
  unified with the rest of the campaign before this goes into a reviewer-facing barrier number.
- `qe_restart_manifest_PREVIOUS_RUN.json` kept for provenance only (describes the prior 6h attempt), not used by
  the new Slurm script.

## Category 3 — Phase 3 81-atom single points (already covered)

See `analysis/phase3_dft_validation_20261002/qe_inputs_81_discovery/` and
`Compute/Discovery Quickstart - MigrationBench DFT 执行手册.md` in the vault — not duplicated here.

## What's NOT in this pass (explicitly out of scope, not silently skipped)

- The 37 endpoints without a prepared non-magnetic `relax.in` (category 1).
- Any QE-NEB production for the other pathways (`1-2`, `1-3`, `1-5`, `1-7`) — only `1-4` (in-gap) had an
  existing, reusable production input; the others would need the same kind of input-build pass this NEB
  package got, starting from their respective MACE-NEB image geometries.
- Actual submission to Discovery — nothing in this package has been `sbatch`'d yet.
