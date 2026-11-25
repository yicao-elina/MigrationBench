# Foundation-model NEB coverage audit (2026-10-03)

Answers two questions: (1) can the 61-atom pathway family be organized into the same 3 categories
(endpoint relax / NEB / single-point validation) built for the 81-atom MD-trajectory system on
2026-10-02 (`analysis/discovery_dft_campaigns_20261003/`)? (2) did the 10-foundation-model NEB sweep
actually finish for pathways 1-2…1-8, and where are the results?

## Q2 answer: foundation-model × pathway completion status

Checked live on Rockfish (`/scratch16/pclancy3/yi/foundation_grouped_benchmark_20261001/outputs/`),
all 50+50 result JSONs pulled and parsed — see `foundation_neb_coverage_summary.csv` (100 rows).

| Family | Paths covered | Models | Status | Rockfish path | Local mirror |
|---|---|---:|---|---|---|
| 61-atom (original) | 1-2, 1-3, 1-4, 1-5, 1-7 | 10/10 | **COMPLETE, 50/50 finite** | `.../foundation_grouped_benchmark_20261001/outputs/neb/` | `neb_61/` (this dir) |
| 61-atom (gap) | **1-6, 1-8** | 0/10 | **NOT RUN** — no `data/1-6.extxyz` / `data/1-8.extxyz` exists yet in this pipeline | — | — |
| 81-atom In-QL diffusion (`neb_1`…`neb_5`) | all 5 | 10/10 | **COMPLETE, 50/50 finite — NEW FINDING, not previously logged in the vault** | `.../foundation_grouped_benchmark_20261001/outputs/neb_81/` | `neb_81/` (this dir) |

The 81-atom result is new: `outputs/neb_81/` has mtime 2026-10-02 (a day after the "Foundation grouped
benchmark closeout" note that's in the Reviewer-Response-Tracker, which only covered the 61-atom 5
paths). It directly closes one item of the Data-Assets §7a TODO ("81-atom：用当前
Foundation/OMAT-FT/Scratch pipeline 对 neb_1–neb_5 做统一 fixed-path evaluation") that was still
listed as open.

No analysis/summary table or figure exists yet for `neb_81` (unlike the 61-atom family, which already
has `all_paths_barrier_results.csv`/`all_paths_publication_summary.md` from the earlier closeout) — the
raw 50 per-model JSONs are complete and finite but have not been turned into a barrier comparison table.

## Q1 answer: does the 61-atom family split into the same 3 categories?

**No, not 1:1.** The 81-atom categories existed because that system has three genuinely different
calculation types in flight (a 47-point defect survey needing relaxation, an NEB pathway needing DFT
convergence, and MD-trajectory snapshots needing single-point validation). The 61-atom system only has
one calculation type — NEB — and it turns out that type is **already DFT-complete for all 7 pathways**,
including 1-6 and 1-8:

- Checked `/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.neb/1.vdw_corr_DFT_D3/{1-2..1-8}/neb.out` directly.
  All 7 have historical QE-NEB output from **November 2024** (not currently running — `squeue` confirms
  nothing active). `1-8/neb.out` ends in a clean `JOB DONE` (hit its step budget and terminated
  normally); the other six end mid-iteration-log (no `JOB DONE` marker), but the directory-level
  `neb_summary.csv` reports converged forward/reverse barriers for **all seven** paths including 1-6
  (`1.026035 eV`) and 1-8 (`1.755620 eV`) — these are the same numbers already recorded in
  `MigrationBench-Data-Assets.md` §7a, now re-confirmed against the live source file (copied here as
  `dft_reference_neb_summary_61atom_all7paths.csv`).
- So there is **no missing DFT-NEB work** for 1-6/1-8, and therefore **no analog of the "NEB production"
  Discovery package** needed — unlike the 81-atom pathway 1-4, which genuinely stalled after 1 iteration
  and needed a real walltime-fixed resubmission.
- There is also **no analog of "endpoint relax"** — the 61-atom pathways don't have a defect-survey
  companion dataset the way the 81-atom system has the 47-endpoint campaign.
- The only real gap is the cheap part: **foundation-model (MACE) evaluation of the existing 1-6/1-8
  geometry**, which is a CPU/short-GPU inference task (same scale as the other 50 that already ran),
  not a DFT/Discovery job.

### Why 1-6/1-8 foundation evaluation wasn't just run now

`neb_foundation_matrix.slurm` (the script that produced the 50 complete 61-atom results) reads
`data/$pid.extxyz` per path. `data/1-6.extxyz` and `data/1-8.extxyz` don't exist. The only converter
found in the pathway directories is `.../3.neb/1.vdw_corr_DFT_D3/1-2/neb2xyz.py`, but its output format
does **not match** the clean format actually used in `data/1-2.extxyz` (no `energy=`/`forces` fields,
just `pathway=/image_index=` + bare positions) — and critically, `neb2xyz.py` writes **placeholder
all-zero forces** (`generate_fake_forces()`) into its output. Blindly running it and feeding the result
into anything resembling a training set would reproduce exactly the C8 "NEB exporter fabricates zero
forces" risk that was already audited and closed elsewhere in this project. The real generator for the
clean `data/*.extxyz` format used by the other 5 paths wasn't located in this benchmark's `scripts/`
directory — **did not improvise a replacement in this pass**, to avoid a repeat of the Scratch-5%
`--E0s` / Phase-3-magnetism pattern of running something before its interface is verified.

**Recommended next step (not done here)**: find/confirm the actual script that built `data/1-2..1-7.extxyz`,
reuse it verbatim for 1-6/1-8 from `.../3.neb/1.vdw_corr_DFT_D3/1-6/sb2te3.xyz` and `.../1-8/sb2te3.xyz`,
then this is a same-day turnaround (20 cheap evaluations, same script as the other 50).
