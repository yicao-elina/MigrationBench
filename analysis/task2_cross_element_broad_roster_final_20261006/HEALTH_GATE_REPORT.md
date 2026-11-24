# Task 2 three-seed final-model audit

Created: 2026-10-06T02:33:18.571179+00:00
Namespace: `task2_cross_element_broad_roster_final_20261006`

## Final-model gate

Seeds 123, 456, and 789 completed 1,000 requested epochs. Each has exactly one epoch-999 checkpoint, an exported model whose 90 tensors match the checkpoint exactly (max absolute difference 0), and a compiled model. A100 validation job `31658285` loaded Foundation plus all three epoch-999 models, producing 1,200 finite profile rows, 120 summaries, and zero calculation failures.

## DFT boundary

The ten archived QE/NEB paths remain `dft_scientific_success=false` because the retained NEB outputs contain SCF convergence warnings. The `<0.1 eV` last-two-table screen and QE curves are labelled provisional references only. Model values are non-equilibrium snapshot diagnostics; no DFT error, barrier accuracy, or transfer superiority is released.

## Delivered tables

- `results/seed_validation_metrics.csv`
- `results/table_1_dft_quality.csv`
- `results/broad_roster/qe_neb_path_profiles.csv`
- `results/table_2_model_element_status.csv`
- `results/broad_roster/table_3_diagnostic_profile_summary.csv`
- `results/table_4_seed_path_stability.csv`
- `results/table_5_imagewise_seed_stability.csv`
- `results/table_6_perturbation_sensitivity.csv`
