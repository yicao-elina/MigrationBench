# Task 2 health-gate report: Cr-trained / Scratch cross-element generalization

Created: 2026-10-02T07:39:18.237078+00:00
Namespace: `task2_cross_element`

## Gate result

The read-only audit found 10 NEB directories: 4 Cu, 3 Li, and 3 Na. All have 10 images, 81 atoms, matching endpoint element order, and the same periodic cell. However, all 10 outputs contain `WARNING: scf convergence NOT achieved` for at least one image. Therefore DFT scientific success is **0/10**; `JOB DONE` is retained as runtime evidence only.

No cross-element error, barrier, force, basin, seed, or transfer-gain claim is released. Table files are header-only schemas, and figures remain placeholder-only by design.

## Provisional non-equilibrium snapshot run

At the user's requested exploratory level, the archived NEB coordinates were also treated as non-equilibrium snapshots. The `sb2te3.path` coordinates were converted from bohr to Å before MACE evaluation. Foundation-MACE completed 300 finite single-point evaluations: 10 paths × 10 images × 3 variants (`base`, `+0.20 Å x`, and `[-0.10, +0.14, +0.10] Å` on the dopant). All 300 rows remain `scientific_success=false` because the DFT source gate failed. The `<0.1 eV` last-two-completed-image-table screen passes for 8/10 base paths; it does not override SCF incompleteness or large residual image force errors.

The exact outer-iteration and last-two-table audit is in `results/provisional_neb_iteration_audit.csv`; Foundation profiles are in `results/provisional_mace_profiles.csv` and summarized in `results/provisional_mace_summary.csv`.

Electronic SCF `iteration #` counts were recovered from retained per-image `PW.out` files for 6/10 paths and are summarized in `results/pw_scf_iteration_inventory.csv`. For the other 4 paths those per-image files are absent; the outer NEB log still records the number of image-level self-consistency calls in the final block, but it does not expose the internal electronic iteration count.

## Model health gate

Foundation loaded on CPU and evaluated finite energy/forces for Cr/Cu/Li/Na/Sb/Te; its Cu diagnostic path profile completed for all 10 images. The Cr-specific FT model loaded and exposes Cr/Sb/Te, but its compiled forward schema is incompatible with the installed MACE calculator (`compute_edge_forces` keyword); Cu/Li/Na are element-unsupported by model metadata. Cr-specific Scratch and Multi-element FT remain load-blocked by a CUDA-backend serialization/runtime error in the current local environment. These are environment/model-interface statuses, not scientific performance failures.

## Evidence boundary

The Foundation profile is a diagnostic smoke only because its source DFT path failed the DFT gate. Relative energies are therefore not released as a model-comparison result. Transfer Gain and overfitting/degradation language are intentionally withheld until DFT-gated, identity-matched metrics plus training-element, embedding, and distribution evidence exist.
