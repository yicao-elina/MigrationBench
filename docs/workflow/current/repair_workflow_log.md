# NEB Repair Workflow Log

This file records every nontrivial NEB-image repair or preconditioning branch. A repair is a proposal-generation step only; manuscript-grade values must still come from accepted QE NEB runs.

Machine-readable companion log: `/Users/alina/Project/26MIgrationBench/state/workflow_events.jsonl`. Use `scripts/migrationbench/append_workflow_event.py` to append events with config, metrics, decisions, and checksums.

## Acceptance Policy

A repaired path may proceed to MACE/QE only if all of the following are recorded:

- source images, source checksum or upstream manifest
- repair mode and all repair parameters
- before/after geometry metrics: minimum pair distance, migrant-host minimum distance, maximum all-atom step, maximum migrant step, total migrant path length
- post-repair MACE metrics: barrier proxy, per-image force, per-image energy change from input to output
- QE preflight metrics: parsed image count, path length sanity, first activation table if available, max image error, memory/time status
- branch decision: accept for continuation, continue with caution, quarantine, or rerun from different seed/config

## 2026-09-11: 1-6 MACE Path Repair Test

### Motivation

The first MACE-derived QE refinement of path `1-6` was cancelled and quarantined. QE parsed the images but produced a nonphysical first activation table: forward barrier `340.825195 eV`, max image error `923.019648 eV/A`, with an earlier path-length overflow. This indicates bad initialization rather than ordinary slow NEB convergence.

### Source Branch

- Historical source images: `/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.neb/1.vdw_corr_DFT_D3/1-6/sb2te3.xyz`
- First MACE branch: `/scratch16/pclancy3/yi/revision1_migrationbench_runs/mb_mlff_cpu_1_6_s42_30791347/mlff_1-6_s42/mlff_neb_images.extxyz`
- First MACE model: `/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE-multihead_600K/finetuned_MACE_multihead0804_compiled.model`
- First bad QE branch: `/scratch16/pclancy3/yi/revision1_migrationbench_runs/mb_qe1_6_mace_r1_s42_30791362`
- Local synchronized repair files: `/Users/alina/Project/26MIgrationBench/cluster/repair_tests_20260911_020649/`

### Repair Configs Tested

| Repair id | Mode | Remote output | Geometry decision |
|---|---|---|---|
| `1-6_mace_smooth-host` | smooth host atoms against endpoints and repel short contacts | `/scratch16/pclancy3/yi/revision1_migrationbench_runs/repair_tests_20260911_020649/1-6_mace_smooth-host.extxyz` | not preferred: minimum pair distance was pushed to the hard threshold |
| `1-6_mace_smooth-all` | smooth all atoms between endpoints and repel short contacts | `/scratch16/pclancy3/yi/revision1_migrationbench_runs/repair_tests_20260911_020649/1-6_mace_smooth-all.extxyz` | pass |
| `1-6_mace_idpp` | endpoint-consistent IDPP-style interpolation plus short-contact repair | `/scratch16/pclancy3/yi/revision1_migrationbench_runs/repair_tests_20260911_020649/1-6_mace_idpp.extxyz` | pass and selected for second MACE test |

### Geometry Results

| Repair id | Max migrant step | Total migrant path | Minimum pair distance | Decision |
|---|---:|---:|---:|---|
| `smooth-host` | `2.276 A` | `6.702 A` | `1.700 A` | reject/low priority |
| `smooth-all` | `0.678 A` | `2.713 A` | `2.305 A` | acceptable |
| `idpp` | `0.678 A` | `2.713 A` | `2.305 A` | selected |

### Submitted Follow-Up Jobs

| Job id | Job name | Branch | Walltime | Purpose | Current status |
|---|---|---|---|---|---|
| `30791499` | `mb_mlff_cpu_1_6_repair_idpp_s42` | repaired `idpp` images -> MACE NEB | `02:00:00` | test whether repair gives stable MLFF forces/energies | completed; quarantined for QE initialization |
| `30791500` | `mb_qe16_direct_pf_s42` | direct historical images -> QE preflight | `06:00:00` | control branch that bypasses bad MACE images | running; initial QE path length is sane |


### 30791499 Result

The repaired `idpp` geometry passed the geometric sanity check, but the second MACE NEB still caused large internal-image energy drops. Therefore this branch is useful as diagnostic evidence but should not be converted into QE input.

| Metric | Value |
|---|---:|
| Barrier proxy | `0.452755 eV` |
| Image 1 relaxation delta | `-17.104943 eV` |
| Image 2 relaxation delta | `-18.940209 eV` |
| Image 3 relaxation delta | `-17.315661 eV` |
| Max internal final force | `0.335540 eV/A` |

Decision: `quarantine_for_qe_initialization`. The failure mode is not only close contact; the MLFF relaxes intermediate images into a much lower-energy basin inconsistent with a small local path refinement.

### 30791500 Initial QE Preflight Signal

The direct historical branch bypasses the bad MACE images and has a sane initial QE path length: `5.8928 bohr`, inter-image distance `1.4732 bohr`, 5 images parsed, 2 MPI ranks, about `131298 MiB` available memory at start. It is currently the preferred `1-6` continuation branch if the first activation table remains physical.

### Required Next Parse

When `30791499` finishes, parse `mlff_neb_manifest.json` and record:

- barrier proxy
- maximum internal force
- maximum absolute relaxation-energy change per image
- whether any image collapses by more than a few eV relative to its input image

When `30791500` writes enough QE output, parse `neb.out` and record:

- number of images parsed
- first activation table, if present
- max image error
- whether path length/energy is physical
- decision for 24h/48h restart continuation

## 2026-09-11: 1-7 MACE Branch Quarantine and Direct QE Recovery

### Motivation

The long QE run `30791312` (`mb_qe17_mace_r1_s42`) from MACE-relaxed images produced a nonphysical first activation table: activation energies overflowed, image 2/3 energies were roughly `1700 eV` above the endpoint scale, and image errors overflowed. The job was cancelled rather than continued.

### Geometry Diagnosis

| Source | Geometry result | Decision |
|---|---|---|
| MACE-relaxed `1-7` images `/scratch16/pclancy3/yi/revision1_migrationbench_runs/mb_mlff_1_7c_s42_30763474/mlff_1-7_cont_s42/mlff_neb_images.extxyz` | warning: max all-atom MIC step `5.047 A`; migrant path itself `3.457 A` | bad QE initializer |
| Historical `1-7` images `/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.neb/1.vdw_corr_DFT_D3/1-7/sb2te3.xyz` | pass: max migrant step `1.097 A`, total path `2.733 A`, minimum pair distance `2.535 A` | preferred recovery source |

### Recovery Branch

Generated direct-historical QE input at `/scratch16/pclancy3/yi/revision1_migrationbench_inputs/mb_qe1_7_direct_hist_s42` using an engine template stripped from the historical `pw_1.in`, so coordinates come only from the audited image path. Submitted Slurm job `30791508` (`mb_qe17_direct_pf_s42`) with 2 MPI ranks, `160G`, `06:00:00` walltime, and restart mode `from_scratch`.

Decision: wait for the first activation table. If physical but unfinished, continue with a 24h or 48h restart segment using the previous run directory as parent.

## 2026-09-11: 81_neb_4 MACE Branch Quarantine and Direct QE Recovery

The MACE-derived `81_neb_4` QE preflight `30791512` was cancelled because QE reported `initial path length = *******` at startup. The source MACE images passed the current local geometry thresholds, so this failure is tracked as a QE path-length/PBC representation failure rather than a simple close-contact failure.

Historical `81_neb_4` images are much cleaner: max migrant/all-atom step `0.268 A`, total path `2.302 A`, minimum pair distance `2.373 A`. A direct-historical QE input was generated at `/scratch16/pclancy3/yi/revision1_migrationbench_inputs/mb_qe81_neb_4_direct_hist_s42`, and preflight job `30791514` (`mb_qe81n4_direct_pf_s42`) was submitted.

Decision: quarantine `30791512`; monitor `30791514` to first activation table.
