# Handoff: From-scratch MD failure reproduction, monitoring, attribution, and publication package

## Mission

Complete a provenance-first, publication-ready study of the three registered
from-scratch failure branches in Cr-Sb2Te3 MD. Reproduce each branch exactly
under its original protocol, monitor every trajectory until its scientific
instability threshold is reached, then ensure that each training-seed/model
branch has three independent MD runs total: the original run plus two exact
reproductions. Preserve every failure and every checkpoint. Do not reinterpret
an unstable trajectory as a stable transport run.

The scientific question is:

> Does each model branch reproducibly leave its training-distribution support
> before a force/geometry/temperature runaway, and what chemical or local
> configuration change marks the precursor/tipping region?

The analysis must distinguish a reproducible model-specific PES runaway from a
physical phase transition. Descriptor novelty is evidence of extrapolation,
not proof of causality.

## Live campaign and paths

All remote paths are on Rockfish under user `ycao73`.

Original protected campaign:

`/scratch16/pclancy3/yi/mace_from_scratch_ablation_MD_20261005`

Current exact-reproduction campaign:

`/scratch16/pclancy3/yi/mace_from_scratch_reproduce_20261011`

Current reproduce array submitted as Job `31891286_[0-2]`; task mapping comes
from `registry/selected_1ns.tsv`:

| array task | training seed | family | model |
|---:|---:|---|---|
| 0 | 345 | `baseline` | `baseline_seed_345_stagetwo_compiled.model` |
| 1 | 123 | `ablation_lmax_2` | `ablation_lmax_2_stagetwo_compiled.model` |
| 2 | 234 | `ablation_lmax_2` | `ablation_lmax_2_seed_234_stagetwo_compiled.model` |

The current array has a concurrency cap of two GPU tasks. This is deliberate:
do not raise concurrency into `MaxGRESPerAccount` failure territory.

The analysis dependency job is `31891326`, output root:

`/scratch16/pclancy3/yi/mace_from_scratch_failure_attribution_20261011`

Local analysis source root:

`analysis/from_scratch_failure_attribution_20261011/`

Important files:

- `diagnose_from_scratch.py`: full trajectory, snapshot, SOAP/PCA/kNN,
  chemistry-feature, and optional counterfactual force-disagreement audit.
- `plot_failure_attribution.py`: publication figures.
- `write_failure_report.py`: evidence-bounded Results/Discussion text.
- `run_attribution.slurm`: CPU dependency job.
- `README.md`: scientific and plotting contract.

## Locked MD protocol

Use the original runner and exact model/input/seed pairing. Do not switch
families or silently substitute model files.

- input: `inputs/common_initial.xyz`, 2050 atoms, full PBC
- input SHA256: `bb42c493fb0c997af3957a89dfde03e3e5a67ebcfd37f5eced8f722dad546161`
- ensemble: Langevin NVT
- target temperature: 600 K
- timestep: 1 fs
- friction: 0.01 fs^-1
- save interval: every 200 steps = 0.2 ps
- MD seed: `20261005`
- target: 1,000,000 steps
- scientific stop gate: `T > 5000 K` OR `max |F| > 100 eV/A` OR `d_min < 0.5 A` OR non-finite values
- preserve original model, input, software environment, and initial momenta
- no fresh Maxwell-Boltzmann reinitialization

The original runner writes the threshold frame before stopping. Treat
`status.json.last_step` and the first bad row in `metrics.csv` as the saved
threshold-onset frame, not automatically as a physical transition frame.

## Existing onset evidence

Original selected branches already show:

| training seed | family | original onset | trigger evidence |
|---:|---|---:|---|
| 123 | `ablation_lmax_2` | 795.6 ps | `T=7733.6 K`, `maxF=180.88 eV/A`, `d_min=0.476 A` |
| 234 | `ablation_lmax_2` | 52.4 ps | `maxF=134.23 eV/A`; `T=2094.6 K`, `d_min=0.905 A` |
| 345 | `baseline` | 83.0 ps | `maxF=108.70 eV/A`; `T=820.9 K`, `d_min=0.695 A` |

These three are not three replicas of one identical neural-network
parameterization: seed123/234 are `ablation_lmax_2`, while seed345 is
`baseline`. Preserve that distinction in every table and figure.

There is also an older, separate main-campaign From-scratch model that first
crossed its gate at 462.2 ps. Do not merge that 462.2 ps result into these
three seed-replica rows; retain it as a separate parent-model reference.

## Required replication policy

The desired final design is three MD runs per registered branch:

1. original protected run in `mace_from_scratch_ablation_MD_20261005`;
2. current reproduce in `mace_from_scratch_reproduce_20261011`;
3. a new non-overwriting reproduce in a new root, for example
   `/scratch16/pclancy3/yi/mace_from_scratch_reproduce_20261011_r02`.

Do not launch the third array into an existing root. Before submitting it:

- verify the current second-run array has terminal `status.json` for all three;
- verify each run stopped at a scientific threshold or completed to target;
- preserve its logs, metrics, checkpoint, trajectory, and manifest;
- verify no target run directory already exists;
- copy the exact runner, selected registry, and input without editing model
  files;
- use the same model SHA256, input SHA256, model family, training seed, MD
  seed, NVT parameters, and 1,000,000-step target;
- submit with GPU concurrency no greater than two;
- record the third-array Job ID and task mapping in a replica ledger.

If a running replicate reaches the threshold, the runner should stop itself at
the next saved frame. If a task is still running after a threshold frame is
written, terminate only that task safely after preserving its checkpoint and
status; do not cancel healthy sibling tasks. Never continue from a failed
threshold checkpoint as if it were stable.

If the current runner exits nonzero after a scientific threshold, that is an
expected scientific failure, not a scheduler failure. Distinguish it from
OOM, missing model, filesystem, dependency, or environment failures.

## Monitoring protocol

At every check, read live `squeue`, `sacct`, logs, and each run's
`status.json`, `metrics.csv`, `run_manifest.json`, `checkpoint.extxyz`, and
trajectory size/mtime. Record:

- job/array task and node;
- state, exit code, elapsed time, and MaxRSS;
- last step and time in ps;
- whether the first bad frame is temperature, force, minimum distance, or a
  combination;
- checkpoint presence and SHA256;
- 2050 atoms, PBC, fixed cell, species order, finite positions/momenta/forces;
- whether a third replicate is eligible for submission.

Quietly ignore healthy unchanged progress. Notify on a threshold stop, job
failure, OOM, checkpoint stall, provenance mismatch, dependency problem,
replicate completion, third-run submission, or scientific anomaly.

## Full attribution analysis

Run the audit on both original and reproduction runs, and after the third run
on all three replicates. Use stable and failure windows separately:

- `stable_early`: 5–15 ps;
- `stable_pre`: 30–15 ps before onset;
- `precursor`: 5 ps before onset through onset;
- `threshold`: exact first bad saved frame.

For each window and seed/model branch, extract:

### Thermodynamic and dynamical features

- potential energy, kinetic energy, total energy if available;
- temperature and temperature derivative;
- maximum and RMS force, force quantiles, top-force atom/species;
- minimum pair distance and species-pair identity;
- Cr displacement, velocity, and local kinetic-energy tail;
- fixed-cell/PBC validation.

### Chemical and local-configuration features

- Cr–Te, Cr–Sb, and Cr–Cr coordination at 2.8, 3.2, 3.6, and 4.0 Å;
- minimum Cr–Te, Cr–Sb, Cr–Cr, and Sb–Te distances;
- Cr-centered bond-angle mean and dispersion;
- local environment around the top-force atom;
- species-resolved RDF/coordination summaries where computationally feasible;
- identify whether the precursor is a Cr hop, close contact, coordination loss,
  angle collapse, or a more global rearrangement.

### Configuration-distribution attribution

Use the established NEB SOAP convention:

`species=[Cr,Sb,Te], r_cut=5 Å, n_max=4, l_max=4, average=inner, sigma=0.5 Å`.

Compare MD snapshots against:

- actual from-scratch training structures;
- validation structures;
- test structures;
- available NEB images and existing `neb_soap_features.csv` files.

Fit standardized PCA on training descriptors only. Report nearest-neighbour
distances to train/validation/test/NEB clouds and train-support q95/q99 gates.
Do not call this calibrated uncertainty. Save every onset-neighbouring
snapshot as an extxyz artifact.

### Counterfactual model disagreement

For selected stable-pre, precursor, and threshold snapshots, evaluate the
same configuration with all three exact model files on CPU/GPU as available.
Report pairwise RMS/max force disagreement and ensemble force spread. Do not
compare raw absolute energies across model families without an explicit
within-model normalization.

### Optional DFT follow-up

The analysis must prepare, but not silently submit, DFT single-point inputs
for the most informative stable/precursor/threshold snapshots. Prioritize
snapshots that are (i) outside training support, (ii) closer to NEB than
training data, or (iii) associated with a force-disagreement jump. Any DFT
claim requires its own convergence/provenance gate.

## Required publication figures

Use the local `scientific-plot-atlas` conventions: Arial, bold tick labels,
heavy spines, no decorative grids, explicit thresholds, and reproducible
PNG/SVG/PDF outputs. Keep each panel independently interpretable.

### Main Figure 1: exact reproduction and onset

Three rows (training seeds) × three columns:

1. temperature versus time relative to onset;
2. maximum force versus time relative to onset;
3. minimum pair distance versus time relative to onset.

Overlay original, reproduce-1, and reproduce-2 with separate line styles;
draw the scientific threshold and onset at zero. Show the onset spread in a
compact table or annotation. Do not smooth away the saved threshold frame.

### Main Figure 2: failure-distribution attribution

Four panels:

1. PCA SOAP embedding of training/validation/test/NEB references and MD
   stable/precursor/threshold snapshots;
2. nearest training/test/NEB descriptor distance versus time to onset;
3. stable versus precursor/threshold novelty distributions;
4. Cr-centered coordination, angle, and minimum-bond features.

### Main Figure 3 or Supplementary Figure: model disagreement

Force-disagreement trajectories and stable/precursor/threshold distributions,
with one panel per seed. Mark whether disagreement rises before or only after
the instability threshold.

### Supplementary snapshot atlas

For each seed, show stable-pre, precursor, threshold, and if available first
post-threshold frames with consistent atom coloring and labels. The exact
extxyz source path, step, time, model SHA256, and descriptor distances must be
in the machine-readable manifest.

## Final deliverables

Under the local analysis directory, produce:

- `replica_ledger.csv` with original/reproduce-1/reproduce-2 provenance;
- `run_summary.csv` and `frame_features.csv`;
- `force_disagreement.csv`;
- `diagnostic_manifest.json` with script/data/model/input hashes;
- onset-neighbouring extxyz snapshots;
- main and supplementary figures in PNG/SVG/PDF;
- `RESULTS_DISCUSSION.md` with exact onset values, recurrence spread,
  trigger classification, distribution-distance evidence, and caveats;
- LaTeX-ready summary table and a short manuscript paragraph.

The final conclusion must use evidence-bounded language. It may say that a
branch is reproducibly unstable or that precursor configurations are outside
the observed training support. It may not claim a unique physical tipping
point or causal training-data defect without matching DFT/force evidence.
