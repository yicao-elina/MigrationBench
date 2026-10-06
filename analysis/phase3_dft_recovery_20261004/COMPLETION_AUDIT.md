# MigrationBench completion audit — 2026-10-04 11:07 EDT

This audit preserves the original scope. `PASS` means current artifacts prove
the requirement; `PENDING` means the requirement is not yet scientifically
achieved.

| Requirement | Status | Evidence |
|---|---|---|
| Plot Atlas repaired and two Phase-2 figures rendered | PASS | `phase2_handoff_md_20261002/audit_20261003/figures/plot_env.json`, SVG/PNG/PDF outputs, Plot Atlas smoke tests |
| First divergence step saved per canonical run | PASS | `phase2_handoff_md_20261002/audit_20261003/results/first_divergence_by_run.csv/.json` |
| Frame-aligned force, velocity, temperature, energy, distance, displacement and coordination metrics | PASS | Canonical `phase2_handoff_md_20261002/audit_20261003/results/frame_aligned_metrics.csv` plus additive `phase2_context_handoff_20261004/frame_aligned_metrics.csv` (1616 rows, 8/8 complete) |
| First anomalous atom and local Cr/Sb/Te neighborhood | PASS | Canonical `first_anomaly_local_neighborhood.jsonl` plus context `phase2_context_handoff_20261004/context_anomaly_local_neighborhood.json`; 462200 is explicitly marked source-boundary/nonphysical |
| Canonical and context trajectory first-frame state matches inputs | PASS | Canonical hash audit plus `phase2_context_handoff_20261004/audit_results.json`: 8/8 context runs, source/input SHA256 and position/momentum/cell/PBC/species checks pass |
| Phase-1 0.629 Å frame excluded from formal DFT/diffusion statistics | PASS | `phase1_static_force_comparison_20261002/Results.md`, Phase-2 `PAPER_TEXT.md` |
| NVE, dt 0.5/1/2 fs, zero/weak/current Langevin sensitivity | PASS | `phase2_integrator_sensitivity_20261003/SENSITIVITY_RESULTS.md`, 48/48 completed runs |
| Normal, bounded-shift and runaway DFT input geometries selected | PASS | `phase3_dft_recovery_20261004/qe_inputs_reconstructed/`, all active cases pass local minimum-distance gate |
| Same local geometries evaluated by all four MACE models | PASS | `mace_results/mace_reconstructed.json`, 16/16 GPU evaluations, job 31590216 |
| Within-model ΔE and force-response table/figure | PASS | `mace_within_model_comparison.csv`, `figures/mace_reconstruction_diagnostic.{png,svg,pdf}` |
| QE finite energies and forces for the four DFT cases | PENDING | Force-enabled smoke `31649829_[0]` passed; formal tasks 1 and 2 of `31662332_[0-3]` passed the complete gate; tasks 0 and 3 failed before QE on `c372` and are being retried as `31673546_[0,3]`; prior resource/transport failures remain rejected |
| Explicit SCF convergence, `JOB DONE`, input hashes and DFT-vs-MACE table | PENDING | Two formal cases are accepted, but normal and runaway-scratch still require successful retry; `make_dft_vs_mace_table.py` remains withheld |
| Formal DFT validation gate | PENDING / 2 accepted | Formal QE gate is partially complete; no accepted four-case identity-preserving DFT result exists yet |

## Current completion decision

The Phase-2/P1 dynamical-stability evidence chain is complete and paper-ready,
including the retained 460000/462200 context snapshots.
The full objective is not complete because the DFT single-point acceptance and
DFT-vs-MACE comparison remain unverified. The goal must remain active; no
formal DFT causal claim is permitted.
